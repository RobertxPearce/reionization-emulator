# -----------------------------------------------------------------------------
# Defines a configurable 4-parameter MC-dropout MLP used to predict
# the kSZ angular power spectrum from reionization parameters.
#
# Robert Pearce
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn

from .four_param_emulator import get_activation


class MCDropoutEmulator(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 5,
        hidden_dim: int = 20,
        num_hidden_layers: int = 2,
        activation: str = "relu",
        dropout_rate: float = 0.1,
    ):
        """
        Configurable MLP for predicting the binned kSZ angular power spectrum
        from reionization parameters using hidden-layer dropout.
        Input: Tensor of shape (N, 4) containing (zmean, alpha, kb, b0)
        Output: Tensor of shape (N, 5) containing log(D_ell) for 5 ell bins
        Default Architecture: 4 -> 20 -> 20 -> 5 with dropout after each
        hidden activation
        """
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError("Number of hidden layers must be at least 1")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0.0 <= p < 1.0")

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(p=dropout_rate),
        ]

        for _ in range(num_hidden_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    get_activation(activation),
                    nn.Dropout(p=dropout_rate),
                ]
            )

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def enable_dropout_only(model: nn.Module) -> None:
    """
    Put every Dropout layer back into training mode, leaving the rest of the
    model in eval mode.

    This is what makes MC dropout an inference-time method: the stochasticity
    must come from dropout.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predict_mc(
    params,
    model: nn.Module,
    X_mean=None,
    X_std=None,
    Y_mean=None,
    Y_std=None,
    *,
    normalize_X: bool = True,
    normalize_Y: bool = False,
    n_mc_samples: int = 100,
    device: str = "cpu",
):
    """
    Monte-Carlo dropout prediction in physical units.

    Runs `n_mc_samples` stochastic forward passes with dropout active, then
    exponentiates the draws.

    That final `exp` assumes the targets were built with
    `BuildXYConfig(y_transform="ln")`, which is the default. With
    `y_transform="log10"` or `"none"` this function returns silently incorrect
    physical values; exponentiate `samples_log` yourself with the matching
    inverse instead. The same assumption is baked into
    `physical_mean_relative_error`.

    params: (n_params,) or (N, n_params) array of raw parameter values
    model: trained MCDropoutEmulator
    X_mean, X_std: input normalizer statistics, required when normalize_X
    Y_mean, Y_std: target normalizer statistics, required when normalize_Y
    device: torch device string to run the forward passes on

    return: pred_mean, pred_std, samples_dl, samples_log
        pred_mean: predictive mean in physical units
        pred_std: predictive standard deviation in physical units (ddof=1)
        samples_dl: (n_mc_samples, ...) draws in physical units
        samples_log: the same draws before exponentiating
    """
    params = np.asarray(params, dtype=np.float32)

    if normalize_X:
        params = (params - X_mean) / X_std

    xb = torch.from_numpy(params.astype(np.float32)).to(device)

    model.eval()
    enable_dropout_only(model)
    with torch.no_grad():
        samples = torch.stack([model(xb) for _ in range(n_mc_samples)], dim=0)

    samples_log = samples.cpu().numpy()
    if normalize_Y:
        samples_log = samples_log * Y_std + Y_mean

    samples_dl = np.exp(samples_log)
    pred_mean = samples_dl.mean(axis=0)
    pred_std = samples_dl.std(axis=0, ddof=1)

    return pred_mean, pred_std, samples_dl, samples_log


# -----------------------------
#         END OF FILE
# -----------------------------
