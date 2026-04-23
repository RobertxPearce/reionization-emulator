# -----------------------------------------------------------------------------
# Training utilities for PyTorch models.
#
# FitConfig: Container for training configurations
# train_one_epoch(): Train the given model for one epoch
# evaluate(): Evaluate the given model for one epoch
# evaluate_metrics(): Calculates the average loss and optionally any
# additional scalar metrics provided in the "metrics" dictionary.
# fit(): Train for many epochs with optional early stopping
#
# Robert Pearce
# -----------------------------------------------------------------------------

from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class FitConfig:
    """
    Container for training configurations.

    epochs: How many passes through the training set
    device: Torch device string ("cpu", "cuda", "mps")
    early_stopping_patience: Optional stop if val loss doesn't improve for
    this many epochs
    gradient_clipping: Optional max norm for gradient clipping (default None)
    """

    epochs: int = 200
    device: str = "cpu"
    early_stopping_patience: Optional[int] = None
    gradient_clipping: Optional[float] = None


def train_one_epoch(
    model, loader, optimizer, loss_fn, device, gradient_clipping=None
) -> float:
    """
    Train the model for one epoch.

    model: The neural network to train
    loader: Dataloader giving batches (xb, yb)
    optimizer: Optimizer to use
    loss_fn: Loss function
    device: Which device to use
    gradient_clipping: Optional gradient clipping (default None)

    return: Average loss over loader
    """
    # Set model to training
    model.train()
    # Count to accumulate loss
    total = 0.0

    # Loop over batch
    for xb, yb in loader:
        # Move tensors to correct device
        xb = xb.to(device)
        yb = yb.to(device)

        # Reset gradients from last step
        optimizer.zero_grad(set_to_none=True)

        # Forward pass to produce prediction
        pred = model(xb)
        # Calculate loss
        loss = loss_fn(pred, yb)
        # Backward pass to compute gradients for every parameter
        loss.backward()

        # Set gradient clipping if set to prevent large updates
        if gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        # Update the weights using the gradients
        optimizer.step()

        # Calculate the total loss scaled by batch size
        total += loss.item() * xb.size(0)

    # Return the average loss over loader
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    """
    Evaluate the model for one epoch.

    model: The neural network to evaluate
    loader: Dataloader giving batches (xb, yb)
    loss_fn: Loss function

    return: Average loss over loader
    """
    # Put model in evaluation mode
    model.eval()
    # Count for accumulation loss
    total = 0.0

    # Loop over batch
    for xb, yb in loader:
        # Move tensors to correct device
        xb = xb.to(device)
        yb = yb.to(device)

        # Forward pass to produce prediction
        pred = model(xb)
        # Calculate loss
        loss = loss_fn(pred, yb)
        # Calculate the total loss scaled by batch size
        total += loss.item() * xb.size(0)
    # Return overage loss over the loader
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_metrics(
    model, loader, loss_fn, device, metrics: dict | None = None
) -> dict:
    """
    Evaluate the model over the full loader.

    Calculates the average loss and optionally any additional scalar metrics
    provided in the "metrics" dictionary.

    return: Dict {"loss": ... , "rmse": ... , "relative_error": ...}
    """
    # Put model in eval mode
    model.eval()
    # Counts for loss and examples
    total_loss = 0.0
    total_examples = 0

    # Initialize metric sums and names dict
    metric_sums = {}
    if metrics is None:
        metrics = {}

    # Loop over loader
    for xb, yb in loader:
        # Send tensors to device
        xb = xb.to(device)
        yb = yb.to(device)

        # Predict and calculate loss
        pred = model(xb)
        loss = loss_fn(pred, yb)

        # Update counts
        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        # Calculate metrics for batch
        for name, fn in metrics.items():
            value = fn(pred, yb).item()
            metric_sums[name] = metric_sums.get(name, 0.0) + value * batch_size

    # Calculate total loss
    result = {"loss": total_loss / total_examples}
    for name, total in metric_sums.items():
        result[name] = total / total_examples

    return result


def _enable_dropout_only(model: nn.Module) -> None:
    """
    Re-enable dropout layers while keeping the rest of the model in eval mode.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def evaluate_mc_metrics(
    model,
    loader,
    loss_fn,
    device,
    metrics: dict | None = None,
    n_mc_samples: int = 100,
) -> dict:
    """
    Evaluate the MC-dropout model over the full loader.

    Calculates the average loss and optionally any additional scalar metrics
    provided in the "metrics" dictionary. Metrics are computed on the
    predictive mean across stochastic dropout-enabled forward passes.

    return: Dict {"loss": ... , "rmse": ... , "mean_predictive_std": ...}
    """
    if n_mc_samples < 2:
        raise ValueError("n_mc_samples must be at least 2 for MC-dropout")

    model.eval()
    total_loss = 0.0
    total_examples = 0
    metric_sums = {}
    total_predictive_std = 0.0

    if metrics is None:
        metrics = {}

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        _enable_dropout_only(model)
        samples = torch.stack([model(xb) for _ in range(n_mc_samples)], dim=0)
        pred_mean = samples.mean(dim=0)
        pred_std = samples.std(dim=0, unbiased=True)
        loss = loss_fn(pred_mean, yb)

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_predictive_std += pred_std.mean().item() * batch_size

        for name, fn in metrics.items():
            value = fn(pred_mean, yb).item()
            metric_sums[name] = metric_sums.get(name, 0.0) + value * batch_size

    result = {"loss": total_loss / total_examples}
    for name, total in metric_sums.items():
        result[name] = total / total_examples
    result["mean_predictive_std"] = total_predictive_std / total_examples

    return result


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    config: FitConfig,
    metrics: Optional[
        Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    ] = None,
    evaluation: str = "evaluate_metrics",
    n_mc_samples: int = 100,
) -> Dict[str, list]:
    """
    Train model for many epochs with optional early stopping.
    Calls train_one_epoch() and evaluate_metrics().

    model: The neural network to train
    train_loader: Dataloader giving batches (xb, yb)
    val_loader: Dataloader giving batches (xb, yb)
    optimizer: Optimizer to use
    loss_fn: Loss function
    config: Configuration for epochs, device, early_stopping_patience,
        and gradient_clipping
    metrics: Optional dict of extra validation metrics, e.g.
        {"rmse": rmse, "relative_error": mean_relative_error}
    evaluation: String indicating which evaluation function to use
    n_mc_samples: Number of stochastic passes for MC-dropout evaluation

    return: Training and validation history
    """
    # Set device to configuration spec
    device = torch.device(config.device)

    # Move model to device
    model = model.to(device)

    # Initialize metrics dict
    if metrics is None:
        metrics = {}

    # Initialize dict for train and validation loss history
    history = {"train_loss": [], "val_loss": []}
    # Add optinal metrics
    for name in metrics:
        history[f"val_{name}"] = []
    if evaluation == "evaluate_mc_metrics":
        history["val_mean_predictive_std"] = []

    # Initialize variable for best validation seen so far
    best_val = float("inf")
    # Initialize variable for best states model weights
    best_state = None
    # Count for epochs since val improved
    bad_epochs = 0

    # Loop through epoch count
    for epoch in range(1, config.epochs + 1):
        # Train one epoch
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            gradient_clipping=config.gradient_clipping,
        )
        
        if evaluation == "evaluate_metrics":
            val_result = evaluate_metrics(
                model, val_loader, loss_fn, device, metrics=metrics
            )
        elif evaluation == "evaluate_mc_metrics":
            val_result = evaluate_mc_metrics(
                model,
                val_loader,
                loss_fn,
                device,
                metrics=metrics,
                n_mc_samples=n_mc_samples,
            )
        else:
            raise ValueError(
                "evaluation must be 'evaluate_metrics' or "
                "'evaluate_mc_metrics'"
            )

        val_loss = val_result["loss"]

        # Save training loss and validation loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for name in metrics:
            history[f"val_{name}"].append(val_result[name])
        if evaluation == "evaluate_mc_metrics":
            history["val_mean_predictive_std"].append(
                val_result["mean_predictive_std"]
            )

        # Print progress
        metric_parts = [f"train={train_loss:.6f}", f"val={val_loss:.6f}"]
        for name in metrics:
            metric_parts.append(f"{name}={val_result[name]:.6f}")
        if evaluation == "evaluate_mc_metrics":
            metric_parts.append(
                f"predictive_std={val_result['mean_predictive_std']:.6f}"
            )
        print(f"Epoch {epoch:03d}: " + ", ".join(metric_parts))

        # Check if early stopping was set
        if config.early_stopping_patience is not None:
            # If set compare current validation loss to current best
            if val_loss < best_val:
                # Save current validation loss
                best_val = val_loss
                # Save model state
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                # Reset counter
                bad_epochs = 0
            else:
                # Update counter
                bad_epochs += 1
                # Check if reached stopping patience
                if bad_epochs >= config.early_stopping_patience:
                    # Print that the early stopping patience was reached
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(patience={config.early_stopping_patience})"
                    )
                    break

    # Make sure best state seen so far is saved
    if best_state is not None:
        model.load_state_dict(best_state)

    # Return training loss and validation loss history
    return history


# -----------------------------
#         END OF FILE
# -----------------------------
