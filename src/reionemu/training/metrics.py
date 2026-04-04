# -----------------------------------------------------------------------------
# Helper functions for calculating evaluation metrics mean square error,
# root mean squared error, and mean relative error.
#
# Robert Pearce
# -----------------------------------------------------------------------------

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mse(pred, target))


def mean_relative_error(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    return torch.mean(torch.abs((pred - target) / (target.abs() + eps)))


# -----------------------------
#         END OF FILE
# -----------------------------
