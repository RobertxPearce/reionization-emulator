# ------------------------------------------------------------------------------------------
# DataLoader utilities for reionemu training fraction-based splits.
#
# load_training_arrays(): Load X, Y, ell from condensed HDF5 file
# _validate_split(): Ensure split dictionary is valid
# _make_fraction_splits(): Create shuffled index arrays based on fraction specification
# make_dataloaders(): Create PyTorch DataLoaders using fraction-based splits
#
# Robert Pearce
# ------------------------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset

from .normalization import Normalizer, fit_standardizer, transform_standardizer


@dataclass
class DataLoaderConfig:
    """
    Data loader configuration.
    
    batch_size: Number of samples per batch during training
    seed: Random seed for reproducibility
    shuffle_train: Whether to shuffle training data each epoch
    normalize_X: Whether to normalize input features X
    normalize_Y: Whether to normalize target features Y
    """
    batch_size: int = 32
    seed: int = 42
    shuffle_train: bool = True
    normalize_X: bool = True
    normalize_Y: bool = False
    
    

def _validate_training_arrays(X: np.ndarray, Y: np.ndarray, ell: np.ndarray) -> None:
    """
    Sanity checks on loaded X, Y, ell before constructing dataloaders.
    Raises ValueError with an informative message if any check fails.
    """
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2D (n_samples, n_params), got ndim={X.ndim} and shape={X.shape}."
        )
    if Y.ndim != 2:
        raise ValueError(
            f"Y must be 2D (n_samples, n_bins), got ndim={Y.ndim} and shape={Y.shape}."
        )
    if ell.ndim != 1:
        raise ValueError(
            f"ell must be 1D (n_bins,), got ndim={ell.ndim} and shape={ell.shape}."
        )
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(
            f"X and Y must have the same number of samples: X.shape[0]={n}, Y.shape[0]={Y.shape[0]}."
        )
    n_bins = ell.shape[0]
    if Y.shape[1] != n_bins:
        raise ValueError(
            f"Y number of bins must match ell length: Y.shape[1]={Y.shape[1]}, len(ell)={n_bins}."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains non-finite values (NaN or inf). Cannot create dataloaders."
        )
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "Y contains non-finite values (NaN or inf). Cannot create dataloaders."
        )
    if not np.all(np.isfinite(ell)):
        raise ValueError(
            "ell contains non-finite values (NaN or inf). Cannot create dataloaders."
        )


def load_training_arrays(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load X, Y, ell from condensed HDF5 file.

    h5_path: Path to condensed HDF5 file.

    return: X (N, num_params), Y (N, num_bins), ell (num_bins,)
    """
    h5_path = Path(h5_path).expanduser().resolve()

    with h5py.File(h5_path, "r") as f:
        X = f["training"]["X"][...].astype(np.float32)
        Y = f["training"]["Y"][...].astype(np.float32)
        ell = f["training"]["ell"][...].astype(np.float32)

    _validate_training_arrays(X, Y, ell)
    return X, Y, ell


def _validate_split(split: Dict[str, float]) -> None:
    """
    Ensure split dictionary is valid.
    """
    # The split dictionary must contain a training split
    if "train" not in split:
        raise ValueError("Split dictionary must contain 'train'")
    
    # Sum the fraction splits
    total = float(sum(split.values()))
    # Check if fractions sum to 1.0
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split dictionary fractions must sum to 1.0 got: {total}")
    
    # Ensure no negative fractions
    for name, fraction in split.items():
        if fraction < 0.0:
            raise ValueError(f"Split dictionary fractions must be positive got: {fraction}")


def _make_fraction_splits(n_samples: int, split: Dict[str, float], *, seed: int) -> Dict[str, np.ndarray]:
    """
    Create shuffled index arrays based on fraction specification.
    
    return: Dict[split name, NumPy array of indices
    """
    # Create reproducible random generator
    rng = np.random.default_rng(seed)
    
    # Randomly permute indices instead of X and Y directly
    permuted_indices = rng.permutation(n_samples)
    
    # Dictionary to hold final index splits {"train": np.array, "val": np.array, "test": np.array}
    split_indices: Dict[str, np.ndarray] = {}
    
    # Track the start of indices to allocate samples to splits
    start = 0
    
    # Preserve insertion order of user-provided split dictionary
    keys = list(split.keys())
    
    # Loop through each split name and assign it a slice of the shuffled index array
    for i, name in enumerate(keys):
        # Last split gets remainder
        if i == len(keys) - 1:
            end = n_samples
        else:
            # Compute number of samples for this split
            end = start + int(round(split[name] * n_samples))
        
        # Slice permuted index array
        split_indices[name] = permuted_indices[start:end]
        
        # Move tracker forward
        start = end
    return split_indices


def make_dataloaders(h5_path: Path,
                     *,
                     split: Dict[str, float] = {"train": 0.8, "val": 0.2},
                     config: DataLoaderConfig = DataLoaderConfig()
                     ) -> Tuple[Dict[str, DataLoader], Dict[str, Optional[Normalizer]], np.ndarray]:
    """
    Create PyTorch DataLoaders using fraction-based splits.
    
    return:
        loader: dict of DataLoader objects keyed by split name
        norms: dict containing fitted normalizers
        ell: ell bin centers
    """
    # Validate split fractions
    _validate_split(split)

    # Load raw arrays from HDF5 (includes sanity checks on X, Y, ell)
    X, Y, ell = load_training_arrays(h5_path)
    
    # Generate shuffled index splits
    split_indices = _make_fraction_splits(n_samples=len(X), split=split, seed=config.seed)
    
    # Extract training indices
    train_idx = split_indices["train"]
    
    # Initialize X and Y normalization
    X_norm = None
    Y_norm = None
    
    # Fit normalizer on training inputs only
    if config.normalize_X:
        X_norm = fit_standardizer(X[train_idx])
    # Fit normalizer on training targets
    if config.normalize_Y:
        Y_norm = fit_standardizer(Y[train_idx])
    
    # Apply normalization to full dataset using training statistics
    if X_norm is not None:
        X = transform_standardizer(X, X_norm).astype(np.float32)
    if Y_norm is not None:
        Y = transform_standardizer(Y, Y_norm).astype(np.float32)
    
    # Convert NumPy arrays into PyTorch tensors
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    
    # Dictionary to store final DataLoaders
    loaders: Dict[str, DataLoader] = {}
    
    # Create DataLoader for each split
    for name, indices in split_indices.items():
        # Create subset corresponding to this split
        subset = Subset(dataset, indices.tolist())
        
        # Training split may shuffle batches
        loaders[name] = DataLoader(subset, batch_size=config.batch_size, shuffle=(name == "train" and config.shuffle_train))
        
    # Return loaders, fitted normalizers, and ell bins
    return loaders, {"X": X_norm, "Y": Y_norm}, ell
    
#-----------------------------
#         END OF FILE
#-----------------------------