# -----------------------------------------------------------------------------
# Defines a configurable 4-parameter MLP used to predict the kSZ angular
# power spectrum from reionization parameters.
#
# Robert Pearce
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Helper function to return the specified activation function.
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation function: {name}")


class FourParamEmulator(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 5,
        hidden_dim: int = 20,
        num_hidden_layers: int = 2,
        activation: str = "relu",
    ):
        """
        Configurable MLP for predicting the binned kSZ angular power spectrum
        from reionization parameters.
        Input: Tensor of shape (N, 4) containing (zmean, alpha, kb, b0)
        Output: Tensor of shape (N, 5) containing log(D_ell) for 5 ell bins
        Default Architecture: 4 -> 20 -> 20 -> 5 (ReLU)
        """
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError("Number of hidden layers must be at least 1")

        layers = [nn.Linear(input_dim, hidden_dim), get_activation(activation)]

        for _ in range(num_hidden_layers - 1):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), get_activation(activation)]
            )

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# -----------------------------
#         END OF FILE
# -----------------------------
