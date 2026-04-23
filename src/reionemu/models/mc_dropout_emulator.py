# -----------------------------------------------------------------------------
# Defines a configurable 4-parameter MC-dropout MLP used to predict
# the kSZ angular power spectrum from reionization parameters.
#
# Robert Pearce
# -----------------------------------------------------------------------------

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


# -----------------------------
#         END OF FILE
# -----------------------------
