# ------------------------------------------------------------------------------------------
# Loads the trained model and normalization stats, evaluates performance on the test
# set, computes accuracy metrics, and plot/print true vs. predicted spectra.
# Robert Pearce
# ------------------------------------------------------------------------------------------

import numpy as np
import torch

from emulator import ProofOfConceptEmulator


MODEL_PATH = "../models/v1/proof_of_concept.pt"
NORM_PATH = "../models/v1/norm"
DATA_PATH = "../data/processed/emulator_dataset_v1.npz"


def load_model():
    """
    Load the trained model and normalization statistics.
    """
    # Select device and use MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load normalization parameters
    X_mean = np.load(f"{NORM_PATH}/X_mean.npy")
    X_std = np.load(f"{NORM_PATH}/X_std.npy")
    ell = np.load(f"{NORM_PATH}/ell.npy")

    # Create model and load weights
    model = ProofOfConceptEmulator().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, device, X_mean, X_std, ell


def predict(model, device, X_mean, X_std, params):
    """
    Predict Dl spectrum for a single parameter vector.

    params: array of shape (3,) = [zmean, alpha, kb]
    """
    params = np.array(params, dtype=np.float32)

    # Normalize inputs
    x_norm = (params - X_mean) / X_std

    # Convert to tensor
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        pred_log = model(x_tensor).cpu().numpy()  # shape (5,)

    # Convert from log(Dl) to Dl
    pred = np.exp(pred_log)

    return pred


def evaluate_on_test():
    """
    Reconstruct the same train/val/test split used in train.py,
    evaluate the model on the test set, and print aggregate metrics.
    """
    # Load full dataset
    data = np.load(DATA_PATH)
    X = data["X"]    # (N, 3)
    Y = data["Y"]    # (N, 5) = log(Dl)
    ell = data["ell"]

    N = X.shape[0]

    # Recreate the exact same split as in train.py
    rng = np.random.default_rng(seed=0)
    indices = rng.permutation(N)

    n_train = int(0.7 * N)
    n_val = int(0.10 * N)
    n_test = N - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Extract test set
    X_test = X[test_idx]
    Y_test = Y[test_idx]  # log(Dl)

    # Load model and normalization
    model, device, X_mean, X_std, ell_loaded = load_model()

    # Accumulators for metrics
    all_true = []
    all_pred = []

    for i in range(X_test.shape[0]):
        params = X_test[i]          # (3,)
        true_log = Y_test[i]        # (5,)
        true_dl = np.exp(true_log)  # Convert from log(Dl) to Dl

        pred_dl = predict(model, device, X_mean, X_std, params)

        all_true.append(true_dl)
        all_pred.append(pred_dl)

    all_true = np.stack(all_true, axis=0)  # (N_test, 5)
    all_pred = np.stack(all_pred, axis=0)  # (N_test, 5)

    # Compute metrics in Dl space
    diff = all_pred - all_true
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    mean_pct_error = np.mean(np.abs(diff) / all_true) * 100.0

    print(f"Test MSE        : {mse:.6f}")
    print(f"Test MAE        : {mae:.6f}")
    print(f"Test Mean % Err : {mean_pct_error:.3f}%")
    print(f"Test samples    : {X_test.shape[0]}")


def compare_to_true_test_sample(k=0):
    """
    Pick the k-th sample from the TEST set and compare true vs predicted Dl.
    """
    # Load full dataset
    data = np.load(DATA_PATH)
    X = data["X"]
    Y = data["Y"]
    ell = data["ell"]

    N = X.shape[0]

    # Rebuild the split
    rng = np.random.default_rng(seed=0)
    indices = rng.permutation(N)

    n_train = int(0.7 * N)
    n_val = int(0.10 * N)
    n_test = N - n_train - n_val

    test_idx = indices[n_train + n_val:]
    if k < 0 or k >= len(test_idx):
        raise ValueError(f"k must be between 0 and {len(test_idx) - 1}, got {k}")

    sim_index = test_idx[k]

    # Load model
    model, device, X_mean, X_std, ell_loaded = load_model()

    params = X[sim_index]
    true_log = Y[sim_index]
    true = np.exp(true_log)

    pred = predict(model, device, X_mean, X_std, params)

    print(f"Sim Index: {sim_index}")
    print("Parameters:", params)
    print("ell bins:", ell)
    print("True Dl:", true)
    print("Predicted Dl:", pred)


if __name__ == "__main__":
    # Aggregate evaluation on the held-out test set
    evaluate_on_test()

    # Inspect a single test example
    compare_to_true_test_sample(0)
