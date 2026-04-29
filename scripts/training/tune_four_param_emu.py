# -----------------------------------------------------------------------------
# Script for using reionemu Ray Tune and training utilities to tune the
# four-parameter model.
#
# Robert Pearce
# -----------------------------------------------------------------------------

from pathlib import Path

from reionemu import load_training_arrays, run_tune_four_param

X, Y, ell = load_training_arrays(
    Path(
        r"/Users/robertxpearce/Desktop/reionization-emulator/datasets/processed/TEST.h5"
    )
)

n = len(X)
split = int(0.8 * n)

X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

results = run_tune_four_param(
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    num_samples=40,
)

best = results.get_best_result(metric="val_loss", mode="min")
print("Best config:", best.config)
print("Best metrics:", best.metrics)

# -----------------------------
#         END OF FILE
# -----------------------------
