# Hyperparameter Tuning

The *hyperparameter tuning* module integrates Ray Tune with the deterministic four-parameter emulator training workflow. It searches over model architecture, optimizer, learning-rate, normalization, and training settings for `FourParamEmulator`, then returns Ray Tune results that can be inspected to choose a final configuration.

## What This Module Does

- Provides a default Ray Tune search space for `FourParamEmulator`
- Trains one trial at a time with train/validation arrays
- Fits optional normalizers using training arrays only
- Reports validation loss, RMSE, relative error, and best-trial metadata
- Saves checkpoints when a trial reaches a new best validation loss
- Launches a full Ray Tune search and returns the resulting `ResultGrid`

This module specifically handles this step in the workflow: Tune Hyperparameters.

The current tuning helper is scoped to `FourParamEmulator`. `MCDropoutEmulator` is part of the stable model and training API, but it is trained with `fit(..., evaluation="evaluate_mc_metrics")` rather than `run_tune_four_param(...)`.

## When To Use It

Use tuning when you want to compare many model and optimizer settings systematically. It is useful after the preprocessing pipeline is stable and you are ready to improve the baseline emulator.

Tuning is optional. If you are testing the package, debugging data preparation, or training a quick baseline, start with `FourParamEmulator` and `fit(...)` instead. If you need dropout-based predictive-spread estimates, use `MCDropoutEmulator` with the regular training API. Ray Tune runs many training trials and can take significantly longer than a single training run.

## What You Need Before Tuning

Before calling the tuning helpers, you need:

- Training arrays `X_train` and `Y_train`.
- Validation arrays `X_val` and `Y_val`.
- Arrays with shapes compatible with `FourParamEmulator`: by default `X.shape[1] == 4` and `Y.shape[1] == 5`.
- Ray Tune installed. The package declares `ray[tune]>=2.0` as a dependency.
- Enough local disk space for Ray Tune trial artifacts and checkpoints.

You can get the arrays from a condensed HDF5 file with `load_training_arrays(...)`.

## Typical Workflow

```python
from pathlib import Path

from reionemu import load_training_arrays, run_tune_four_param

h5_path = Path("path/to/condensed.h5")
X, Y, ell = load_training_arrays(h5_path)

split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]

results = run_tune_four_param(
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    num_samples=20,
    max_concurrent_trials=2,
    device="cpu",
    storage_path="ray_results",
    experiment_name="four_param_search",
)

best = results.get_best_result(metric="val_loss", mode="min")

print(best.config)
print(best.metrics["best_val_loss"])
```

## default_param_space

`default_param_space` returns the default Ray Tune search space for the deterministic four-parameter emulator.

### Main Entry Point

```python
def default_param_space() -> dict:
```

### Default Search Space

```python
{
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
```

### Typical Usage

```python
from ray import tune

from reionemu import default_param_space, run_tune_four_param

param_space = default_param_space()
param_space["hidden_dim"] = tune.choice([32, 64])
param_space["epochs"] = 100

results = run_tune_four_param(
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    param_space=param_space,
    num_samples=10,
)
```

## resolve_device

`resolve_device` converts a device string into a `torch.device`.

### Main Entry Point

```python
def resolve_device(device: str = "auto") -> torch.device:
```

If `device` is not `"auto"`, the function returns `torch.device(device)`. If `device="auto"`, it chooses:

1. `"cuda"` when CUDA is available.
2. `"mps"` when Apple Silicon MPS is available.
3. `"cpu"` otherwise.

This helper is used inside tuning trials, but it can also be useful in scripts.

## train_four_param_tune

`train_four_param_tune` is the Ray Tune trainable for one trial. Most users do not call it directly; `run_tune_four_param(...)` wraps it with resources, parameters, scheduling, and result handling.

### Main Entry Point

```python
def train_four_param_tune(
    config: dict,
    *,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> None:
```

### What Happens In One Trial

For each trial, the trainable:

- Resolves the requested device.
- Optionally fits `X` and `Y` normalizers on `X_train` and `Y_train`.
- Transforms train and validation arrays using training-only statistics.
- Builds train and validation `DataLoader` objects.
- Builds a `FourParamEmulator` using the trial config.
- Builds an Adam or AdamW optimizer.
- Trains for up to `config["epochs"]` epochs.
- Reports `train_loss`, `val_loss`, `val_rmse`, `val_relative_error`, `best_val_loss`, and `best_epoch`.
- Saves a checkpoint whenever validation loss improves.
- Stops early if `early_stopping_patience` is set and validation loss does not improve for that many epochs.

### Trial Checkpoints

When a trial reaches a new best validation loss, the checkpoint contains:

- `model.pt`: The model state dictionary.
- `metadata.pt`: The trial config, best validation loss, best epoch, and fitted normalizers.

These checkpoints are managed by Ray Tune and are usually accessed through the best result object returned by `run_tune_four_param(...)`.

## run_tune_four_param

`run_tune_four_param` is the main user-facing tuning entry point. It launches Ray Tune with an ASHA scheduler and returns a Ray Tune `ResultGrid`.

### Main Entry Point

```python
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
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| X_train | `np.ndarray` | *Required* | Training input array |
| Y_train | `np.ndarray` | *Required* | Training target array |
| X_val | `np.ndarray` | *Required* | Validation input array |
| Y_val | `np.ndarray` | *Required* | Validation target array |
| param_space | `dict | None` | `None` | Ray Tune search space; defaults to `default_param_space()` |
| num_samples | `int` | `40` | Number of hyperparameter configurations to sample |
| max_concurrent_trials | `int` | `4` | Maximum number of trials running at once |
| device | `str` | `"auto"` | Device used by each trial |
| storage_path | `str | None` | `None` | Ray Tune output directory |
| experiment_name | `str` | `"train_four_param_tune"` | Name for the Ray Tune experiment |

### Scheduler And Resources

The function uses `ASHAScheduler` with:

- `max_t` equal to `param_space["epochs"]`.
- `grace_period` equal to `min(15, max_t)`.
- `reduction_factor=2`.

Each trial requests:

- `2` CPUs.
- `1` GPU only when `device="cuda"` or `device="auto"` resolves to CUDA.

### Returns

`run_tune_four_param` returns Ray Tune's `ResultGrid`.

Common operations include:

```python
best = results.get_best_result(metric="val_loss", mode="min")

print(best.config)
print(best.metrics)
print(best.checkpoint)
```

The tuning objective is validation loss, minimized with `metric="val_loss"` and `mode="min"`.

### Typical Usage With A Custom Search Space

```python
from ray import tune

from reionemu import run_tune_four_param

param_space = {
    "hidden_dim": tune.choice([32, 64, 128]),
    "num_hidden_layers": tune.choice([2, 3]),
    "activation": tune.choice(["relu", "gelu", "silu"]),
    "optimizer": tune.choice(["adamw"]),
    "lr": tune.loguniform(3e-4, 3e-3),
    "weight_decay": tune.loguniform(1e-8, 1e-4),
    "batch_size": tune.choice([32, 64]),
    "epochs": 150,
    "early_stopping_patience": 15,
    "gradient_clipping": tune.choice([None, 1.0]),
    "normalize_X": True,
    "normalize_Y": False,
}

results = run_tune_four_param(
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    param_space=param_space,
    num_samples=20,
    max_concurrent_trials=2,
    device="cpu",
    storage_path="ray_results",
    experiment_name="four_param_search",
)

best = results.get_best_result(metric="val_loss", mode="min")
print(best.config)
```

## Using The Best Result

After tuning, use the best config to train a final model with the standard training API or inspect the checkpoint saved by Ray Tune.

### Train A Final Model From The Best Config

```python
import torch

from reionemu import FitConfig, build_four_param_model, build_optimizer, fit

best = results.get_best_result(metric="val_loss", mode="min")
best_config = best.config

model = build_four_param_model(best_config)
optimizer = build_optimizer(model, best_config)

history = fit(
    model,
    loaders["train"],
    loaders["val"],
    optimizer,
    torch.nn.MSELoss(),
    config=FitConfig(
        epochs=best_config["epochs"],
        device="cpu",
        early_stopping_patience=best_config.get("early_stopping_patience"),
        gradient_clipping=best_config.get("gradient_clipping"),
    ),
)
```

### Inspect A Best Checkpoint

```python
import torch

best = results.get_best_result(metric="val_loss", mode="min")

with best.checkpoint.as_directory() as checkpoint_dir:
    metadata = torch.load(f"{checkpoint_dir}/metadata.pt")

print(metadata["best_val_loss"])
print(metadata["best_epoch"])
```

## Reproducibility Notes

The tuning helpers use train/validation arrays exactly as passed to the function. If you want a reproducible split, create that split with a fixed seed before calling `run_tune_four_param(...)`.

The current default search space does not set a PyTorch random seed inside each trial. Ray Tune will track the sampled hyperparameters and metrics, but exact training curves can still vary across runs because of model initialization, dataloader shuffling, and hardware-level nondeterminism.

## Common Issues

- **Ray stores outputs in an unexpected place**: Pass `storage_path="ray_results"` or another explicit directory.
- **Trials are too slow**: Reduce `num_samples`, reduce `epochs`, or lower `max_concurrent_trials`.
- **GPU is not used**: Pass `device="cuda"` and confirm CUDA is available in the active PyTorch environment.
- **`param_space["epochs"]` is invalid**: `epochs` must be at least `1`.
- **Tuning feels unnecessary**: Use `FourParamEmulator` with `fit(...)` for a quick deterministic baseline before launching Ray Tune, or use `MCDropoutEmulator` with `evaluate_mc_metrics` when dropout-based predictive spread is the goal.
