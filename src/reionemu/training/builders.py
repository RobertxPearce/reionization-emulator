# -----------------------------------------------------------------------------
# Helper functions for building the 4 param model and optimizer.
#
# build_four_param_model(): Helper function for building four-parameter
# emulator with Ray Tune specs.
# build_optimizer(): Helper function for building optimizer.
#
# Robert Pearce
# -----------------------------------------------------------------------------

import torch

from ..models.four_param_emulator import FourParamEmulator


def build_four_param_model(config: dict) -> torch.nn.Module:
    """
    Helper function for building the four-parameter emulator with
    Ray Tune specs.

    config: dict containing Ray Tune configuration specs

    return: Four-parameter emulator with Ray Tune specs
    """
    return FourParamEmulator(
        input_dim=config.get("input_dim", 4),
        output_dim=config.get("output_dim", 5),
        hidden_dim=config["hidden_dim"],
        num_hidden_layers=config["num_hidden_layers"],
        activation=config["activation"],
    )


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Helper function for building optimizer.

    model: torch.nn.Module
    config: dict containing Ray Tune configuration specs

    return: Optimizer with Ray Tune specs
    """
    optimizer_name = config.get("optimizer", "adamw").lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


# -----------------------------
#         END OF FILE
# -----------------------------
