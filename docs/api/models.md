# Models

The *models* module contains the PyTorch emulator architectures used to map reionization parameters to binned kSZ angular power spectrum targets. These models are intentionally small multilayer perceptrons so they can be trained quickly during experiments and used as fast surrogate models after training.

## What This Module Does

- Provides stable neural network classes for four-parameter emulation
- Supports configurable hidden width, hidden depth, and activation function
- Provides a dropout-based variant for uncertainty-oriented experiments
- Provides `predict_mc` for Monte Carlo dropout prediction in physical units
- Separates stable public models from proof-of-concept experimental variants

This module specifically handles this step in the workflow: Instantiate Model.

## When To Use It

Use this module after your training arrays or dataloaders are ready and you need a concrete PyTorch model to train. For most workflows, start with `FourParamEmulator`. Use `MCDropoutEmulator` when you want stochastic dropout predictions at evaluation time to estimate predictive spread.

The stable model inputs are four reionization parameters in the order used by the default training dataset:

```python
("zmean_zre", "alpha_zre", "kb_zre", "b0_zre")
```

The default output is a five-bin target spectrum, matching the default `BuildXYConfig` and `ClConfig` workflow.

## Typical Workflow

```python
import torch

from reionemu import FourParamEmulator, MCDropoutEmulator

model = FourParamEmulator()
dropout_model = MCDropoutEmulator(dropout_rate=0.1)

X_batch = torch.randn(32, 4)
Y_pred = model(X_batch)
Y_pred_dropout = dropout_model(X_batch)

print(Y_pred.shape)
print(Y_pred_dropout.shape)
```

## Which Model Should I Start With?

Start with `FourParamEmulator` unless you specifically need stochastic dropout behavior. It is the stable baseline architecture used by the training and tuning helpers.

| Model | Stability | Main Use |
|:------|:----------|:---------|
| `FourParamEmulator` | Stable public API | Default deterministic emulator |
| `MCDropoutEmulator` | Stable public API | Dropout-based predictive spread experiments |
| `reionemu.models.experimental.*` | Experimental | Proof-of-concept architecture comparisons |

## FourParamEmulator

`FourParamEmulator` is the default deterministic multilayer perceptron for predicting the binned kSZ target spectrum from four reionization parameters.

### Purpose

Use this model for standard training, validation, cross-validation, and hyperparameter tuning runs. It produces one deterministic prediction per input batch.

### Constructor

```python
class FourParamEmulator(
    input_dim: int = 4,
    output_dim: int = 5,
    hidden_dim: int = 20,
    num_hidden_layers: int = 2,
    activation: str = "relu",
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| input_dim | `int` | `4` | Number of input features per sample |
| output_dim | `int` | `5` | Number of predicted spectrum bins |
| hidden_dim | `int` | `20` | Width of each hidden layer |
| num_hidden_layers | `int` | `2` | Number of hidden layers; must be at least `1` |
| activation | `str` | `"relu"` | Hidden activation name |

Supported activation names are `"relu"`, `"gelu"`, `"silu"`, `"tanh"`, and `"sigmoid"`.

### Input And Output

- Input: `torch.Tensor` with shape `(N, input_dim)`.
- Default input meaning: `(zmean_zre, alpha_zre, kb_zre, b0_zre)`.
- Output: `torch.Tensor` with shape `(N, output_dim)`.
- Default output meaning: five binned target values from the training dataset, usually transformed `D_ell` values if the default simulation I/O configuration was used.

### Default Architecture

With default settings, the model is:

```text
4 -> 20 -> 20 -> 5
```

with ReLU activations between linear layers.

### Typical Usage

```python
import torch

from reionemu import FourParamEmulator

model = FourParamEmulator(
    input_dim=4,
    output_dim=5,
    hidden_dim=20,
    num_hidden_layers=2,
    activation="relu",
)

xb = torch.randn(16, 4)
pred = model(xb)

print(pred.shape)
```

## MCDropoutEmulator

`MCDropoutEmulator` uses the same configurable MLP pattern as `FourParamEmulator`, but inserts dropout after each hidden activation. This makes it useful for Monte Carlo dropout evaluation, where repeated stochastic forward passes provide a compact uncertainty summary.

### Purpose

Use this model when you want to train an emulator with dropout and evaluate it with `evaluate_mc_metrics(...)` or `fit(..., evaluation="evaluate_mc_metrics")`.

### Constructor

```python
class MCDropoutEmulator(
    input_dim: int = 4,
    output_dim: int = 5,
    hidden_dim: int = 20,
    num_hidden_layers: int = 2,
    activation: str = "relu",
    dropout_rate: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| input_dim | `int` | `4` | Number of input features per sample |
| output_dim | `int` | `5` | Number of predicted spectrum bins |
| hidden_dim | `int` | `20` | Width of each hidden layer |
| num_hidden_layers | `int` | `2` | Number of hidden layers; must be at least `1` |
| activation | `str` | `"relu"` | Hidden activation name |
| dropout_rate | `float` | `0.1` | Dropout probability; must satisfy `0.0 <= p < 1.0` |

### Input And Output

- Input: `torch.Tensor` with shape `(N, input_dim)`.
- Output: `torch.Tensor` with shape `(N, output_dim)`.
- During normal evaluation with `model.eval()`, dropout is disabled.
- During MC-dropout evaluation, dropout layers are re-enabled while the rest of the model remains in evaluation mode.

### Typical Usage

```python
import torch

from reionemu import MCDropoutEmulator

model = MCDropoutEmulator(
    hidden_dim=32,
    num_hidden_layers=3,
    activation="gelu",
    dropout_rate=0.2,
)

xb = torch.randn(16, 4)
pred = model(xb)

print(pred.shape)
```

## MC-Dropout Prediction

A plain forward pass through a trained `MCDropoutEmulator` is deterministic, because `model.eval()` disables dropout. `predict_mc` performs the full Monte Carlo dropout prediction instead: it re-enables dropout, runs repeated stochastic forward passes, undoes the target normalization, and returns both the predictive mean and its spread in physical units.

Use this when you need predictions and uncertainties outside the training loop, for example when evaluating a saved model on a held-out set or when using the emulator as a likelihood inside parameter inference. Inside the training loop, prefer `evaluate_mc_metrics(...)` or `fit(..., evaluation="evaluate_mc_metrics")`.

### predict_mc

```python
def predict_mc(
    params,
    model: torch.nn.Module,
    X_mean=None,
    X_std=None,
    Y_mean=None,
    Y_std=None,
    *,
    normalize_X: bool = True,
    normalize_Y: bool = False,
    n_mc_samples: int = 100,
    device: str = "cpu",
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| params | `array-like` | required | Raw parameter values with shape `(n_params,)` or `(N, n_params)` |
| model | `torch.nn.Module` | required | Trained model containing dropout layers |
| X_mean, X_std | `array-like` | `None` | Input normalizer statistics; required when `normalize_X` is `True` |
| Y_mean, Y_std | `array-like` | `None` | Target normalizer statistics; required when `normalize_Y` is `True` |
| normalize_X | `bool` | `True` | Standardize inputs before the forward passes |
| normalize_Y | `bool` | `False` | Undo target standardization on the draws |
| n_mc_samples | `int` | `100` | Number of stochastic forward passes |
| device | `str` | `"cpu"` | Torch device string to run the passes on |

The function returns four values:

| Return value | Shape | Description |
|:-------------|:------|:------------|
| pred_mean | `(..., output_dim)` | Predictive mean in physical units |
| pred_std | `(..., output_dim)` | Predictive standard deviation in physical units, `ddof=1` |
| samples_dl | `(n_mc_samples, ..., output_dim)` | Individual draws in physical units |
| samples_log | `(n_mc_samples, ..., output_dim)` | The same draws before exponentiating |

The model is trained on `log(D_ell)`, so `predict_mc` exponentiates the draws before averaging. The mean and standard deviation are therefore computed in physical units rather than in log space, and `pred_mean` is not the exponential of the mean log prediction.

!!! warning "Assumes `y_transform="ln"`"

    The final exponentiation assumes the training targets were built with
    `BuildXYConfig(y_transform="ln")`, which is the default. The other supported
    options are `"log10"` and `"none"` (see [Simulation I/O](simulation-io.md)), and
    with either of those `predict_mc` returns silently incorrect physical values —
    it will not raise. In that case use the returned `samples_log` and apply the
    matching inverse transform yourself. `physical_mean_relative_error` carries the
    same assumption.

### Choosing n_mc_samples

One dropout mask is drawn per forward pass and shared across every row of `params`. Predictions for different parameter points within a single call are therefore correlated, and the effective sample size for averaging is `n_mc_samples`, not `n_mc_samples * len(params)`. Increasing the number of parameter points does not reduce Monte Carlo noise; only increasing `n_mc_samples` does.

As a rough guide, the standard error on `pred_mean` scales as the per-bin coefficient of variation divided by the square root of `n_mc_samples`, while a stable `pred_std` requires roughly `1 + 1 / (2 * tol**2)` samples for a relative tolerance `tol`. Reporting the predictive spread usually drives the sample count higher than reporting the mean alone.

### Typical Usage

```python
import numpy as np

from reionemu import MCDropoutEmulator, predict_mc

model = MCDropoutEmulator(dropout_rate=0.1)

theta = np.array([[8.0, 0.5, 1.0, 0.45]], dtype=np.float32)

pred_mean, pred_std, samples_dl, samples_log = predict_mc(
    theta,
    model,
    X_mean=normalizers["X"].mean,
    X_std=normalizers["X"].std,
    normalize_X=True,
    n_mc_samples=200,
    device="cpu",
)

print(pred_mean.shape, pred_std.shape)
print(samples_dl.shape)
```

Because dropout is stochastic, repeated calls return different values. Seed torch immediately before the call when you need a reproducible number:

```python
import torch

torch.manual_seed(42)
pred_mean, pred_std, _, _ = predict_mc(theta, model, X_mean, X_std, n_mc_samples=200)
```

### enable_dropout_only

```python
def enable_dropout_only(model: torch.nn.Module) -> None:
```

Puts every `torch.nn.Dropout` layer back into training mode while leaving the rest of the model in evaluation mode. `predict_mc` calls this internally, so you only need it directly when writing a custom MC-dropout loop.

Keeping the rest of the model in evaluation mode is what makes MC dropout well defined: the stochasticity must come from dropout alone, not from other layers whose behavior differs between training and evaluation.

## Model Builders

The training layer includes helper functions that build models from configuration dictionaries. These are especially useful for Ray Tune workflows, where each trial receives a different `config`.

### build_four_param_model

```python
def build_four_param_model(config: dict) -> torch.nn.Module:
```

This function constructs a `FourParamEmulator` using:

- `input_dim`, defaulting to `4`
- `output_dim`, defaulting to `5`
- `hidden_dim`
- `num_hidden_layers`
- `activation`

### build_mc_dropout_model

```python
def build_mc_dropout_model(config: dict) -> torch.nn.Module:
```

This function constructs an `MCDropoutEmulator` using the same configuration keys as `build_four_param_model`, plus optional `dropout_rate`, which defaults to `0.1`.

### Typical Usage

```python
from reionemu import build_four_param_model, build_mc_dropout_model

config = {
    "input_dim": 4,
    "output_dim": 5,
    "hidden_dim": 64,
    "num_hidden_layers": 3,
    "activation": "silu",
}

model = build_four_param_model(config)

dropout_model = build_mc_dropout_model(
    {
        **config,
        "dropout_rate": 0.15,
    }
)
```

## Experimental Models

Experimental proof-of-concept architectures live under `reionemu.models.experimental`. They are useful for architecture comparisons and older notebook experiments, but they are not the recommended default API.

Available experimental classes are:

- `POCEmulatorThreeParams`: A three-input proof-of-concept model with architecture `3 -> 5 -> 5`.
- `POCEmulatorFourParamsV1`: A four-input proof-of-concept model with architecture `4 -> 20 -> 5`.
- `POCEmulatorFourParamsV2`: A four-input proof-of-concept model with architecture `4 -> 20 -> 20 -> 20 -> 5` and dropout.
- `POCEmulatorFourParamsV3`: A four-input proof-of-concept model with architecture `4 -> 20 -> 20 -> 5`.

Use these directly from the experimental namespace:

```python
from reionemu.models.experimental import POCEmulatorFourParamsV3

model = POCEmulatorFourParamsV3()
```

For production workflows, prefer `FourParamEmulator` or `MCDropoutEmulator`.

## Common Issues

- **Unknown activation function**: Use one of `"relu"`, `"gelu"`, `"silu"`, `"tanh"`, or `"sigmoid"`.
- **Shape mismatch during training**: Make sure `input_dim` matches `X.shape[1]` and `output_dim` matches `Y.shape[1]`.
- **MC dropout gives deterministic predictions**: Use `predict_mc(...)` outside the training loop, or `evaluate_mc_metrics(...)` / `fit(..., evaluation="evaluate_mc_metrics")` inside it; a plain `model.eval()` forward pass disables dropout.
- **`predict_mc` results change between runs**: This is expected, since dropout masks are redrawn on every call. Seed torch immediately before the call, or raise `n_mc_samples`, to reduce run-to-run variation.
- **`predict_mc` values look wrong by orders of magnitude**: Check `BuildXYConfig.y_transform`. `predict_mc` exponentiates with `exp`, so a dataset built with `"log10"` or `"none"` produces silently incorrect physical units.
- **`predict_mc` raises on `X_mean` or `X_std`**: These are required whenever `normalize_X` is `True`. Pass the fitted `Normalizer` statistics, or set `normalize_X=False` if your inputs are already standardized.
