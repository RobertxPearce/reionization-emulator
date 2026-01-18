# ------------------------------------------------------------------------------------------
# Trains the emulator using the prepared dataset, performs normalization, and saves the
# trained model weights and normalization statistics.
# Robert Pearce
# ------------------------------------------------------------------------------------------

import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from emulator import ProofOfConceptEmulator


DATA_PATH = "../data/processed/emulator_dataset_v1.npz"
MODEL_PATH = "../models/v1/proof_of_concept.pt"
NORM_PATH = "../models/v1/norm"

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 200


def load_data():
    # Load dataset
    data = np.load(DATA_PATH)
    X = data["X"]       # Parameters with shape (N, 3)
    Y = data["Y"]       # log(D_ell) with shape (N, 5)
    ell = data["ell"]   # ell bins

    N = X.shape[0]
    print(f"Loaded {N} simulations")

    # Split dataset into train/val/test (70 / 10 / 20)
    rng = np.random.default_rng(seed=0)
    indices = rng.permutation(N)

    n_train = int(0.7 * N)
    n_val = int(0.10 * N)
    # Remainder goes into test
    n_test = N - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    print(f"Train Sims: {X_train.shape[0]}")
    print(f"Val Sims  : {X_val.shape[0]}")
    print(f"Test Sims : {X_test.shape[0]}")

    # Normalize inputs (z-score using TRAIN set only)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0 # Dont scale if param is const

    X_train_n = (X_train - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std

    # Create PyTorch datasets + loaders
    X_train_t = torch.tensor(X_train_n, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_n, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    # Group datasets
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    test_dataset = TensorDataset(X_test_t, Y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # test_loader is created here but not used in training; it's for later evaluation.
    return train_loader, val_loader, test_loader, X_mean, X_std, ell


def main():
    # Set the device
    if torch.backends.mps.is_available():
        # Use MacBook GPU
        device = torch.device("mps")
    else:
        # Use CPU
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Process and prepare data
    train_loader, val_loader, test_loader, X_mean, X_std, ell = load_data()

    # Set the model, loss, and optimizer
    model = ProofOfConceptEmulator().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train the model
    for epoch in range(1, EPOCHS + 1):
        # Set the model to train
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save mean/std for normalizing input
    os.makedirs(NORM_PATH, exist_ok=True)
    np.save(f"{NORM_PATH}/X_mean.npy", X_mean)
    np.save(f"{NORM_PATH}/X_std.npy", X_std)
    np.save(f"{NORM_PATH}/ell.npy", ell)


if __name__ == "__main__":
    main()


#-----------------------------
#         END OF FILE
#-----------------------------
