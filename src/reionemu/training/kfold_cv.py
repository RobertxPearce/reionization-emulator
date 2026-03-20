# -----------------------------------------------------------------------------
# K-Fold Cross Validation utilities for reionemu training.
#
# KFoldConfig: Container for k-fold configuration
# _validate_k(): Ensure k is valid
# _make_kfold_splits(): Create shuffled index folds
# kfold_cross_validate(): Run k-fold CV using user-provided model/optimizer
# builders
#
# Robert Pearce
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from ..data.dataloaders import DataLoaderConfig, load_training_arrays
from ..data.normalization import fit_standardizer, transform_standardizer
from .train_loop import FitConfig, fit


@dataclass
class KFoldConfig:
    """
    K-Fold configuration.

    k: Number of folds
    seed: Random seed for reproducibility
    return_histories: If True, return full loss curves for each fold
    """

    k: int = 5
    seed: int = 42
    return_histories: bool = False


def _validate_k(n_samples: int, k: int) -> None:
    """
    Ensure k is valid for number of samples.
    """
    # K must be at least 2
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    # Need enough samples to split
    if n_samples < k:
        raise ValueError(f"Need n_samples >= k. Got n_samples={n_samples}, k={k}")


def _make_kfold_splits(n_samples: int, *, k: int, seed: int) -> List[np.ndarray]:
    """
    Create shuffled index folds.

    return: List of length k where each entry is an index array for the validation fold
    """
    # Validate k relative to dataset size
    _validate_k(n_samples, k)

    # Create reproducible random generator
    rng = np.random.default_rng(seed)

    # Permute indices once
    perm = rng.permutation(n_samples)

    # Split permuted indices into k folds
    folds = np.array_split(perm, k)

    # Ensure dtype is int64 for indexing
    return [np.asarray(f, dtype=np.int64) for f in folds]


def kfold_cross_validate(
    h5_path: Path,
    *,
    model_builder: Callable[[], torch.nn.Module],
    optimizer_builder: Callable[[torch.nn.Module], torch.optim.Optimizer],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    kfold_config: KFoldConfig = KFoldConfig(),
    dl_config: DataLoaderConfig = DataLoaderConfig(),
    fit_config: FitConfig = FitConfig(),
) -> Dict[str, object]:
    """
    Run K-Fold cross validation on the condensed HDF5 training dataset.

    h5_path: Path to condensed HDF5 file
    model_builder: Function that returns a fresh model instance (once per fold)
    optimizer_builder: Function that takes model and returns optimizer (once per fold)
    loss_fn: Loss function
    kfold_config: KFoldConfig containing k and seed
    dl_config: DataLoaderConfig controlling batch size, shuffling, and normalization
    fit_config: FitConfig for epochs/device/early stopping

    return:
        dict with:
            "ell": np.ndarray of ell bin centers
            "fold_best_val": list of best validation loss per fold
            "mean_best_val": mean of best val losses
            "std_best_val": std of best val losses
            "models": list of trained model instances, one per fold
            "norms": list of dicts per fold, each with keys "X" and "Y" containing
                     the fitted standardizer for that fold (None if norm disabled)
            "val_indices": list of np.ndarray of validation indices per fold
            "histories": (optional) list of history dicts for each fold
    """
    # Load raw arrays from HDF5
    X, Y, ell = load_training_arrays(Path(h5_path))

    # Create folds (each fold is the validation indices for that fold)
    folds = _make_kfold_splits(
        n_samples=len(X), k=kfold_config.k, seed=kfold_config.seed
    )

    # Store best val loss for each fold
    fold_best_val: List[float] = []

    # Optionally store full histories
    histories: List[Dict[str, list]] = []

    # Fold models, norms and val indicies for error %
    fold_models: List[torch.nn.Module] = []
    fold_norms: List[Dict] = []
    fold_val_indices: List[np.ndarray] = []

    # Loop over folds
    for fold_id in range(kfold_config.k):
        # Validation indices for this fold
        val_idx = folds[fold_id]

        # Training indices are all indices not in val_idx
        train_idx = np.concatenate(
            [folds[i] for i in range(kfold_config.k) if i != fold_id]
        )

        x_norm = None
        y_norm = None

        # Fit normalization on train only
        if dl_config.normalize_X:
            x_norm = fit_standardizer(X[train_idx])
            Xn = transform_standardizer(X, x_norm).astype(np.float32)
        else:
            Xn = X.astype(np.float32)

        if dl_config.normalize_Y:
            y_norm = fit_standardizer(Y[train_idx])
            Yn = transform_standardizer(Y, y_norm).astype(np.float32)
        else:
            Yn = Y.astype(np.float32)

        # Convert arrays into PyTorch dataset
        dataset = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Yn))

        # Create subset loaders
        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()),
            batch_size=dl_config.batch_size,
            shuffle=dl_config.shuffle_train,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx.tolist()),
            batch_size=dl_config.batch_size,
            shuffle=False,
        )

        # Fresh model + optimizer per fold
        model = model_builder()
        optimizer = optimizer_builder(model)

        # Print progress
        print(
            f"\n=== Fold {fold_id + 1}/{kfold_config.k} | "
            f"train={len(train_idx)} val={len(val_idx)} ==="
        )

        # Train the fold
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=fit_config,
        )

        # Save trained model
        fold_models.append(model)

        # Save norms for this fold
        fold_norms.append(
            {
                "X": x_norm if dl_config.normalize_X else None,
                "Y": y_norm if dl_config.normalize_Y else None,
            }
        )

        # Save val indices
        fold_val_indices.append(val_idx)

        # Save best validation loss achieved in this fold
        best_val = float(np.min(np.asarray(history["val_loss"], dtype=np.float64)))
        fold_best_val.append(best_val)
        print(f"Fold {fold_id + 1} best val: {best_val:.6f}")

        # Save full history if requested
        if kfold_config.return_histories:
            histories.append(history)

    # Summarize fold results
    vals = np.asarray(fold_best_val, dtype=np.float64)

    result: Dict[str, object] = {
        "ell": ell,
        "fold_best_val": fold_best_val,
        "mean_best_val": float(vals.mean()),
        "std_best_val": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "models": fold_models,
        "norms": fold_norms,
        "val_indices": fold_val_indices,
    }

    # Attach histories if requested
    if kfold_config.return_histories:
        result["histories"] = histories

    # Return summary
    return result


# -----------------------------
#         END OF FILE
# -----------------------------
