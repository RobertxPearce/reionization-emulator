# Models

The *models* module contains the PyTorch emulator architectures used to map reionization parameters to binned kSZ angular power spectrum targets. These models are intentionally small multilayer perceptrons so they can be trained quickly during experiments and used as fast surrogate models after training.

## What This Module Does

- Provides stable neural network classes for four-parameter emulation
- Supports configurable hidden width, hidden depth, and activation function
- Provides a dropout-based variant for uncertainty-oriented experiments
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
- **MC dropout gives deterministic predictions**: Use `evaluate_mc_metrics(...)` or `fit(..., evaluation="evaluate_mc_metrics")`; a plain `model.eval()` forward pass disables dropout.
