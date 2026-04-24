# Training

The *training* module contains the PyTorch utilities used to fit emulator models, evaluate validation performance, compute regression metrics, build optimizers, and run k-fold cross-validation. It is designed to work with the dataloaders produced by the data loading module and the models exposed by the models module.

## What This Module Does

- Trains models for one epoch or many epochs
- Evaluates models with loss functions and optional metrics
- Supports early stopping and optional gradient clipping
- Supports MC-dropout evaluation for predictive-spread summaries
- Provides optimizer and model-builder helpers for config-driven workflows
- Runs k-fold cross-validation from a condensed HDF5 training dataset

This module specifically handles these steps in the workflow: Train Model -> Evaluate Model -> Cross-Validate Model.

## When To Use It

Use this module after you have a model, dataloaders, optimizer, and loss function. For most workflows, the recommended path is:

1. Build dataloaders with `make_dataloaders(...)`.
2. Instantiate `FourParamEmulator` or `MCDropoutEmulator`.
3. Create an optimizer and loss function.
4. Configure training with `FitConfig`.
5. Call `fit(...)`.

## Typical Workflow

```python
from pathlib import Path

import torch

from reionemu import (
    DataLoaderConfig,
    FitConfig,
    FourParamEmulator,
    fit,
    make_dataloaders,
    mean_relative_error,
    rmse,
)

h5_path = Path("path/to/condensed.h5")

loaders, normalizers, ell = make_dataloaders(
    h5_path,
    split={"train": 0.8, "val": 0.2},
    config=DataLoaderConfig(batch_size=32, seed=42),
)

model = FourParamEmulator()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

history = fit(
    model=model,
    train_loader=loaders["train"],
    val_loader=loaders["val"],
    optimizer=optimizer,
    loss_fn=loss_fn,
    config=FitConfig(
        epochs=100,
        device="cpu",
        early_stopping_patience=15,
        gradient_clipping=1.0,
    ),
    metrics={
        "rmse": rmse,
        "relative_error": mean_relative_error,
    },
)

print(history["val_loss"])
```

## FitConfig

`FitConfig` controls the multi-epoch training loop.

### Configuration

```python
@dataclass
class FitConfig:
    epochs: int = 200
    device: str = "cpu"
    early_stopping_patience: Optional[int] = None
    gradient_clipping: Optional[float] = None
```

- **epochs**: Maximum number of training epochs.
- **device**: Torch device string, such as `"cpu"`, `"cuda"`, or `"mps"`.
- **early_stopping_patience**: If set, stop after this many consecutive epochs without validation-loss improvement.
- **gradient_clipping**: If set, clip gradient norm to this value during training.

### Early Stopping Behavior

When `early_stopping_patience` is set, `fit(...)` tracks the best validation loss. If validation loss stops improving for the configured number of epochs, training stops early. After training ends, the model weights are restored to the best validation-loss state seen during that run.

## fit

`fit` is the main multi-epoch training entry point. It calls `train_one_epoch(...)` on the training loader, evaluates on the validation loader after each epoch, prints epoch progress, and returns a history dictionary.

### Main Entry Point

```python
def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    config: FitConfig,
    metrics: Optional[Dict[str, Callable]] = None,
    evaluation: str = "evaluate_metrics",
    n_mc_samples: int = 100,
) -> Dict[str, list]:
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| model | `torch.nn.Module` | *Required* | Model to train |
| train_loader | `DataLoader` | *Required* | Training batches of `(X_batch, Y_batch)` |
| val_loader | `DataLoader` | *Required* | Validation batches of `(X_batch, Y_batch)` |
| optimizer | `torch.optim.Optimizer` | *Required* | Optimizer used for parameter updates |
| loss_fn | `Callable` | *Required* | Loss function, commonly `torch.nn.MSELoss()` |
| config | `FitConfig` | *Required* | Epoch, device, early stopping, and gradient clipping settings |
| metrics | `Dict[str, Callable]` | `None` | Optional validation metrics computed each epoch |
| evaluation | `str` | `"evaluate_metrics"` | Evaluation path: `"evaluate_metrics"` or `"evaluate_mc_metrics"` |
| n_mc_samples | `int` | `100` | Number of stochastic passes for MC-dropout evaluation |

### Returns

`fit` returns a dictionary of lists. The base keys are:

- `train_loss`: Average training loss for each epoch.
- `val_loss`: Average validation loss for each epoch.

If extra metrics are provided, each metric is returned as `val_<metric_name>`. For example, a metric named `"rmse"` is stored under `history["val_rmse"]`.

When `evaluation="evaluate_mc_metrics"`, the history also includes:

- `val_mean_predictive_std`: Mean predictive standard deviation from stochastic MC-dropout samples.

### Typical Usage

```python
import torch

from reionemu import FitConfig, FourParamEmulator, fit, rmse

model = FourParamEmulator()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

history = fit(
    model,
    loaders["train"],
    loaders["val"],
    optimizer,
    loss_fn,
    config=FitConfig(epochs=50, device="cpu"),
    metrics={"rmse": rmse},
)
```

## train_one_epoch

`train_one_epoch` is a lower-level helper used by `fit`. Call it directly when you need to write a custom training loop.

### Main Entry Point

```python
def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    gradient_clipping=None,
) -> float:
```

### Behavior

This function:

- Sets the model to training mode with `model.train()`.
- Moves each batch to the requested device.
- Runs forward pass, loss, backward pass, and optimizer step.
- Optionally clips gradient norm.
- Returns the average loss over the dataset.

## Evaluation Helpers

The training module provides evaluation helpers for deterministic models and MC-dropout models.

### evaluate

```python
def evaluate(model, loader, loss_fn, device) -> float:
```

`evaluate` returns only the average loss over a loader. It is useful when you do not need extra metrics.

### evaluate_metrics

```python
def evaluate_metrics(
    model,
    loader,
    loss_fn,
    device,
    metrics: dict | None = None,
) -> dict:
```

`evaluate_metrics` returns a dictionary with `"loss"` plus any metrics provided in the `metrics` dictionary. Each metric function should accept `(pred, target)` and return a scalar tensor.

```python
from reionemu import evaluate_metrics, mean_relative_error, rmse

result = evaluate_metrics(
    model,
    loaders["val"],
    torch.nn.MSELoss(),
    device="cpu",
    metrics={
        "rmse": rmse,
        "relative_error": mean_relative_error,
    },
)

print(result)
```

### evaluate_mc_metrics

```python
def evaluate_mc_metrics(
    model,
    loader,
    loss_fn,
    device,
    metrics: dict | None = None,
    n_mc_samples: int = 100,
) -> dict:
```

`evaluate_mc_metrics` is intended for `MCDropoutEmulator`. It keeps the model in evaluation mode, re-enables only dropout layers, performs `n_mc_samples` stochastic forward passes per batch, and computes loss and metrics on the predictive mean.

It returns:

- `loss`: Average loss computed from the predictive mean.
- Any extra metrics passed in the `metrics` dictionary.
- `mean_predictive_std`: Average predictive standard deviation across stochastic passes.

`n_mc_samples` must be at least `2`.

### MC-Dropout Training Example

```python
import torch

from reionemu import FitConfig, MCDropoutEmulator, fit, rmse

model = MCDropoutEmulator(dropout_rate=0.2)

history = fit(
    model,
    loaders["train"],
    loaders["val"],
    torch.optim.AdamW(model.parameters(), lr=1e-3),
    torch.nn.MSELoss(),
    config=FitConfig(epochs=100, device="cpu"),
    metrics={"rmse": rmse},
    evaluation="evaluate_mc_metrics",
    n_mc_samples=50,
)

print(history["val_mean_predictive_std"])
```

## Metrics

The package includes three basic regression metrics.

### mse

```python
def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
```

Returns mean squared error.

### rmse

```python
def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
```

Returns root mean squared error.

### mean_relative_error

```python
def mean_relative_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
```

Returns the mean absolute relative error, using `target.abs() + eps` in the denominator to avoid division by zero.

## Model And Optimizer Builders

The builder helpers create models and optimizers from dictionaries. These are used by the Ray Tune integration, but they are also useful when you want a config-driven training script.

### build_four_param_model

```python
def build_four_param_model(config: dict) -> torch.nn.Module:
```

Builds a `FourParamEmulator` from `input_dim`, `output_dim`, `hidden_dim`, `num_hidden_layers`, and `activation`.

### build_mc_dropout_model

```python
def build_mc_dropout_model(config: dict) -> torch.nn.Module:
```

Builds an `MCDropoutEmulator` from the same keys as `build_four_param_model`, plus optional `dropout_rate`.

### build_optimizer

```python
def build_optimizer(
    model: torch.nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
```

Builds either `torch.optim.Adam` or `torch.optim.AdamW`.

Supported config keys are:

- `optimizer`: `"adam"` or `"adamw"`; defaults to `"adamw"`.
- `lr`: Learning rate.
- `weight_decay`: Weight decay; defaults to `0.0`.

### Typical Usage

```python
from reionemu import build_four_param_model, build_optimizer

config = {
    "hidden_dim": 64,
    "num_hidden_layers": 3,
    "activation": "silu",
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 1e-6,
}

model = build_four_param_model(config)
optimizer = build_optimizer(model, config)
```

## Cross-Validation

K-fold cross-validation trains a fresh model on each fold and reports validation-loss summaries across folds. Use this when you want a more robust estimate of model performance than a single train/validation split.

### KFoldConfig

```python
@dataclass
class KFoldConfig:
    k: int = 5
    seed: int = 42
    return_histories: bool = False
```

- **k**: Number of folds. Must be at least `2` and no larger than the number of samples.
- **seed**: Random seed used to shuffle samples before splitting into folds.
- **return_histories**: Whether to include full training histories for every fold in the return object.

### kfold_cross_validate

```python
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
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| h5_path | `Path` | *Required* | Condensed HDF5 file containing `/training` |
| model_builder | `Callable` | *Required* | Function returning a fresh model for each fold |
| optimizer_builder | `Callable` | *Required* | Function accepting a model and returning a fresh optimizer |
| loss_fn | `Callable` | *Required* | Loss function used during training |
| kfold_config | `KFoldConfig` | Defaults | Fold count, seed, and history-return settings |
| dl_config | `DataLoaderConfig` | Defaults | Batch size, shuffling, and normalization settings |
| fit_config | `FitConfig` | Defaults | Epoch, device, early stopping, and gradient clipping settings |

### Returns

The result dictionary contains:

- `ell`: Multipole bin centers loaded from the HDF5 file.
- `fold_best_val`: Best validation loss for each fold.
- `mean_best_val`: Mean of the best fold validation losses.
- `std_best_val`: Sample standard deviation of the best fold validation losses.
- `models`: Trained model instance for each fold.
- `norms`: Per-fold normalizers for `"X"` and `"Y"`.
- `val_indices`: Validation indices used in each fold.
- `histories`: Full per-fold histories, only when `return_histories=True`.

### Typical Usage

```python
from pathlib import Path

import torch

from reionemu import (
    DataLoaderConfig,
    FitConfig,
    FourParamEmulator,
    KFoldConfig,
    kfold_cross_validate,
)

result = kfold_cross_validate(
    Path("path/to/condensed.h5"),
    model_builder=lambda: FourParamEmulator(),
    optimizer_builder=lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3),
    loss_fn=torch.nn.MSELoss(),
    kfold_config=KFoldConfig(k=5, seed=42, return_histories=True),
    dl_config=DataLoaderConfig(batch_size=32, normalize_X=True, normalize_Y=False),
    fit_config=FitConfig(epochs=100, device="cpu", early_stopping_patience=15),
)

print(result["mean_best_val"])
print(result["std_best_val"])
```

## Common Issues

- **Device errors**: Make sure `FitConfig.device` is available on your machine. Use `"cpu"` for the most portable option.
- **Validation loss is unstable**: Try a smaller learning rate, enable gradient clipping, or use early stopping.
- **Cross-validation reuses weights**: Make sure `model_builder` returns a new model instance every time it is called.
- **MC-dropout uncertainty is missing**: Use `MCDropoutEmulator` with `evaluation="evaluate_mc_metrics"`.
