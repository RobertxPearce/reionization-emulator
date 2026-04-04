# -----------------------------------------------------------------------------
# Ray Tune utilities for tuning the 4-parameter emulator.
#
# _make_loaders(): Build train/validation dataloaders from numpy arrays
# _prepare_arrays(): Fit standardizers on train only and transform train/val
# arrays
# resolve_device(): Helper function for resolving devices.
# train_four_param_tune(): Ray Tune trainable for one trial
# default_param_space(): Default Ray Tune search space for the deterministic
# 4-parameter emulator
# run_tune_four_param(): Launch Ray Tune for the 4-parameter emulator and
# return the ResultGrid
#
# Robert Pearce
# -----------------------------------------------------------------------------

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray import air, tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, TensorDataset

from ..data.normalization import fit_standardizer, transform_standardizer
from .builders import build_four_param_model, build_optimizer
from .metrics import mean_relative_error, rmse
from .train_loop import evaluate_metrics, train_one_epoch


def _make_loaders(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/validation dataloaders from numpy arrays.
    """
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(Y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(Y_val.astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _prepare_arrays(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    normalize_X: bool,
    normalize_Y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Fit standardizers on train only and transform train/val arrays.
    """
    x_norm = None
    y_norm = None

    if normalize_X:
        x_norm = fit_standardizer(X_train)
        X_train = transform_standardizer(X_train, x_norm)
        X_val = transform_standardizer(X_val, x_norm)

    if normalize_Y:
        y_norm = fit_standardizer(Y_train)
        Y_train = transform_standardizer(Y_train, y_norm)
        Y_val = transform_standardizer(Y_val, y_norm)

    norms = {"X": x_norm, "Y": y_norm}
    return X_train, Y_train, X_val, Y_val, norms


def resolve_device(device: str = "auto") -> torch.device:
    """
    Helper function for resolving devices.
    """
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_four_param_tune(
    config: dict,
    *,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> None:
    """
    Ray Tune trainable for one trial.
    """
    device = resolve_device(config.get("device", "auto"))

    X_train_p, Y_train_p, X_val_p, Y_val_p, norms = _prepare_arrays(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        normalize_X=config.get("normalize_X", True),
        normalize_Y=config.get("normalize_Y", False),
    )

    train_loader, val_loader = _make_loaders(
        X_train_p,
        Y_train_p,
        X_val_p,
        Y_val_p,
        batch_size=config["batch_size"],
    )

    model = build_four_param_model(
        {
            "input_dim": X_train_p.shape[1],
            "output_dim": Y_train_p.shape[1],
            "hidden_dim": config["hidden_dim"],
            "num_hidden_layers": config["num_hidden_layers"],
            "activation": config["activation"],
        }
    ).to(device)

    optimizer = build_optimizer(
        model,
        {
            "optimizer": config.get("optimizer", "adamw"),
            "lr": config["lr"],
            "weight_decay": config.get("weight_decay", 0.0),
        },
    )

    loss_fn = torch.nn.MSELoss()

    eval_metrics = {
        "rmse": rmse,
        "relative_error": mean_relative_error,
    }

    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    patience = config.get("early_stopping_patience")

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gradient_clipping=config.get("gradient_clipping"),
        )

        val_result = evaluate_metrics(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            metrics=eval_metrics,
        )
        val_loss = val_result["loss"]

        report_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_result["rmse"],
            "val_relative_error": val_result["relative_error"],
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            report_metrics["best_val_loss"] = best_val_loss
            report_metrics["best_epoch"] = best_epoch

            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_dir = Path(tmpdir)
                torch.save(model.state_dict(), ckpt_dir / "model.pt")
                torch.save(
                    {
                        "config": config,
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "norms": norms,
                    },
                    ckpt_dir / "metadata.pt",
                )
                checkpoint = Checkpoint.from_directory(str(ckpt_dir))
                tune.report(report_metrics, checkpoint=checkpoint)
        else:
            bad_epochs += 1
            tune.report(report_metrics)

        if patience is not None and bad_epochs >= patience:
            break


def default_param_space() -> dict:
    """
    Default Ray Tune search space for the deterministic 4-parameter emulator.
    """
    return {
        "hidden_dim": tune.choice([16, 32, 64, 128, 256]),
        "num_hidden_layers": tune.choice([1, 2, 3, 4]),
        "activation": tune.choice(["relu", "gelu", "silu", "tanh"]),
        "optimizer": tune.choice(["adam", "adamw"]),
        "lr": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-8, 1e-4),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": 200,
        "early_stopping_patience": 20,
        "gradient_clipping": tune.choice([None, 1.0, 5.0]),
        "normalize_X": True,
        "normalize_Y": False,
        "device": "auto",
    }


def run_tune_four_param(
    *,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    param_space: dict | None = None,
    num_samples: int = 40,
    max_concurrent_trials: int = 4,
    device: str = "auto",
    storage_path: str | None = None,
    experiment_name: str = "train_four_param_tune",
):
    """
    Launch Ray Tune for the 4-parameter emulator and return the ResultGrid.
    """
    if param_space is None:
        param_space = default_param_space()
    else:
        param_space = dict(param_space)

    param_space["device"] = device
    max_t = int(param_space.get("epochs", 200))
    if max_t < 1:
        raise ValueError("param_space['epochs'] must be at least 1")

    use_gpu = device == "cuda" or (device == "auto" and torch.cuda.is_available())

    scheduler = ASHAScheduler(
        max_t=max_t,
        grace_period=min(15, max_t),
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_four_param_tune,
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_val,
                Y_val=Y_val,
            ),
            resources={"cpu": 2, "gpu": 1 if use_gpu else 0},
        ),
        param_space=param_space,
        run_config=air.RunConfig(
            name=experiment_name,
            storage_path=storage_path,
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
        ),
    )

    return tuner.fit()


# -----------------------------
#         END OF FILE
# -----------------------------
