# Data Loading

The *data loading* module reads the training arrays written by the simulation I/O workflow and prepares them for PyTorch training. It is the bridge between the processed HDF5 dataset and the model/training utilities.

## What This Module Does

- Loads `X`, `Y`, and `ell` arrays from the condensed HDF5 `/training` group
- Validates that loaded arrays have compatible shapes and finite values
- Creates reproducible, fraction-based train/validation/test splits
- Builds PyTorch `DataLoader` objects for each split
- Optionally fits and applies feature-wise normalization using training split statistics only

This module specifically handles this step in the workflow: Load Training Data -> Create DataLoaders.

## When To Use It

Use this module after the simulation I/O workflow has produced a condensed HDF5 file with a `/training` group. If your file only contains `/sims` and `/cl`, run `build_and_write_training(...)` first.

The expected HDF5 layout is:

```
training:
 ['X', 'Y', 'ell', 'param_names', 'sim_ids']
```

The required arrays are:

- `training/X`: Input parameter matrix with shape `(n_samples, n_parameters)`.
- `training/Y`: Target spectrum matrix with shape `(n_samples, n_ell_bins)`.
- `training/ell`: Reference multipole bin centers with shape `(n_ell_bins,)`.

## Typical Workflow

```python
from pathlib import Path

from reionemu import DataLoaderConfig, load_training_arrays, make_dataloaders

condensed_h5 = Path("path/to/condensed.h5")

X, Y, ell = load_training_arrays(condensed_h5)

config = DataLoaderConfig(
    batch_size=32,
    seed=42,
    shuffle_train=True,
    normalize_X=True,
    normalize_Y=False,
)

loaders, normalizers, ell = make_dataloaders(
    condensed_h5,
    split={"train": 0.8, "val": 0.2},
    config=config,
)

train_loader = loaders["train"]
val_loader = loaders["val"]
X_normalizer = normalizers["X"]
Y_normalizer = normalizers["Y"]
```

## Load Training Arrays

`load_training_arrays` is the lowest-level public helper in this module. It reads the arrays from `/training`, casts them to `float32`, validates them, and returns NumPy arrays.

### Purpose

Use this function when you want direct access to the training arrays without constructing PyTorch loaders.

### Main Entry Point

```python
def load_training_arrays(
    h5_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

| Parameter | Type   | Default    | Description                                      |
|:----------|:-------|:-----------|:-------------------------------------------------|
| h5_path   | `Path` | *Required* | Path to the condensed HDF5 file containing `/training` |

### Returns

This function returns a tuple:

| Return | Type         | Shape                            | Description                         |
|:-------|:-------------|:---------------------------------|:------------------------------------|
| `X`    | `np.ndarray` | `(n_samples, n_parameters)`      | Input simulation parameters         |
| `Y`    | `np.ndarray` | `(n_samples, n_ell_bins)`        | Target spectra                      |
| `ell`  | `np.ndarray` | `(n_ell_bins,)`                  | Multipole bin centers for `Y`       |

All three arrays are returned as `float32`.

### Validation

Before returning, the function verifies that:

- `X` and `Y` are two-dimensional.
- `ell` is one-dimensional.
- `X` and `Y` contain the same number of samples.
- The second dimension of `Y` matches the length of `ell`.
- `X`, `Y`, and `ell` contain only finite values.

If any check fails, a `ValueError` is raised with a description of the failed condition.

### Typical Usage

```python
from pathlib import Path

from reionemu import load_training_arrays

condensed_h5 = Path("path/to/condensed.h5")

X, Y, ell = load_training_arrays(condensed_h5)

print(X.shape)
print(Y.shape)
print(ell.shape)
```

## Make DataLoaders

`make_dataloaders` constructs PyTorch `DataLoader` objects from the HDF5 training arrays. It supports any split names provided in the split dictionary, as long as one of them is named `"train"`.

### Purpose

Use this function when training or evaluating models with PyTorch. It handles array loading, validation, reproducible sample splitting, optional normalization, tensor conversion, and loader construction.

### Configuration

```python
@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    seed: int = 42
    shuffle_train: bool = True
    normalize_X: bool = True
    normalize_Y: bool = False
```

- **batch_size**: Number of samples per batch in each `DataLoader`.
- **seed**: Random seed used when assigning samples to splits.
- **shuffle_train**: Whether the training loader shuffles batches each epoch.
- **normalize_X**: Whether to standardize input parameters.
- **normalize_Y**: Whether to standardize target spectra.

### Main Entry Point

```python
def make_dataloaders(
    h5_path: Path,
    *,
    split: Dict[str, float] = {"train": 0.8, "val": 0.2},
    config: DataLoaderConfig = DataLoaderConfig(),
) -> Tuple[Dict[str, DataLoader], Dict[str, Optional[Normalizer]], np.ndarray]:
```

| Parameter | Type                 | Default                         | Description                                      |
|:----------|:---------------------|:--------------------------------|:-------------------------------------------------|
| h5_path   | `Path`               | *Required*                      | Path to the condensed HDF5 file containing `/training` |
| split     | `Dict[str, float]`   | `{"train": 0.8, "val": 0.2}`   | Fraction-based split definition                  |
| config    | `DataLoaderConfig`   | Defaults                        | DataLoader and normalization configuration       |

### Split Rules

The `split` dictionary must:

- Include a `"train"` key.
- Sum to `1.0`.
- Contain no negative fractions.

The split order follows the insertion order of the dictionary. For each split except the last, the number of samples is computed with `round(fraction * n_samples)`. The final split receives the remaining samples so that every sample is assigned exactly once.

For example:

```python
split = {"train": 0.7, "val": 0.15, "test": 0.15}
```

returns loaders with keys `"train"`, `"val"`, and `"test"`.

### Returns

This function returns a tuple:

| Return        | Type                                 | Description                                      |
|:--------------|:-------------------------------------|:-------------------------------------------------|
| `loaders`     | `Dict[str, DataLoader]`              | PyTorch loaders keyed by split name              |
| `normalizers` | `Dict[str, Optional[Normalizer]]`    | Fitted normalizers for `"X"` and `"Y"`, or `None` |
| `ell`         | `np.ndarray`                         | Multipole bin centers loaded from the HDF5 file  |

Each loader returns batches of `(X_batch, Y_batch)` tensors.

### Typical Usage

```python
from pathlib import Path

from reionemu import DataLoaderConfig, make_dataloaders

condensed_h5 = Path("path/to/condensed.h5")

loaders, normalizers, ell = make_dataloaders(
    condensed_h5,
    split={"train": 0.7, "val": 0.15, "test": 0.15},
    config=DataLoaderConfig(
        batch_size=64,
        seed=123,
        shuffle_train=True,
        normalize_X=True,
        normalize_Y=True,
    ),
)

for X_batch, Y_batch in loaders["train"]:
    print(X_batch.shape, Y_batch.shape)
    break
```

## Normalization

The `Normalizer` dataclass stores feature-wise mean and standard deviation arrays.

```python
@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
```

When `normalize_X=True`, `make_dataloaders` fits a normalizer on the training rows of `X` and applies it to the full dataset before constructing split loaders. When `normalize_Y=True`, the same process is applied to `Y`.

Normalization is computed feature-wise along `axis=0`:

```python
X_standardized = (X - normalizer.mean) / normalizer.std
```

If a feature has zero standard deviation in the training split, its stored standard deviation is replaced with `1.0` to avoid division by zero.

### Why Training-Only Statistics Matter

Validation and test samples should not influence preprocessing statistics. Fitting the normalizers on the training split only keeps validation and test metrics from seeing information outside the training data.

### Using Normalizers After Prediction

If `normalize_Y=True`, model outputs are in standardized target space. Convert predictions back to the original target scale before plotting or interpreting spectra:

```python
from reionemu.data.normalization import inverse_transform_standardizer

Y_pred_original = inverse_transform_standardizer(
    Y_pred_standardized,
    normalizers["Y"],
)
```

## Common Issues

- **`KeyError: 'training'`**: The HDF5 file does not contain a `/training` group. Run `build_and_write_training(...)` first.
- **Split fractions do not sum to `1.0`**: Adjust the split dictionary, for example `{"train": 0.8, "val": 0.2}`.
- **`X` and `Y` have mismatched sample counts**: Rebuild the `/training` group from a consistent set of simulations.
- **Non-finite values found**: Inspect the upstream `/cl` products and the target transform used by `BuildXYConfig`.
