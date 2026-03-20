# -----------------------------------------------------------------------------
# Training utilities for PyTorch models.
#
# FitConfig: Container for training configurations
# train_one_epoch(): Train the given model for one epoch
# evaluate(): Evaluate the given model for one epoch
# fit(): Train for many epochs with optional early stopping
#
# Robert Pearce
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Optional

import torch


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


def fit(
    model, train_loader, val_loader, optimizer, loss_fn, config: FitConfig
) -> Dict[str, list]:
    """
    Train model for many epochs with optional early stopping.
    Calls train_one_epoch() and evaluate().

    model: The neural network to train
    train_loader: Dataloader giving batches (xb, yb)
    val_loader: Dataloader giving batches (xb, yb)
    optimizer: Optimizer to use
    loss_fn: Loss function
    config: Configuration for epochs, device, early_stopping_patience,
    and gradient_clipping

    return: Training and validation history
    """
    # Set device to configuration spec
    device = torch.device(config.device)

    # Move model to device
    model = model.to(device)

    # Initialize dict for train and validation loss history
    history = {"train_loss": [], "val_loss": []}

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
        # Validate the epoch
        val_loss = evaluate(model, val_loader, loss_fn, device)

        # Save training loss and validation loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Print progress
        print(f"Epoch {epoch:03d}: train={train_loss:.6f}, val={val_loss:.6f}")

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
