<p align="center">
  <img src="https://raw.githubusercontent.com/RobertxPearce/reionization-emulator/main/docs/assets/reionemu-logo.png" alt="reionemu logo" width="300">
</p>

# reionemu

A modular Python package for building machine-learning emulators of the kinetic Sunyaev-Zel'dovich (kSZ) angular power spectrum from kSZ 2LPT reionization simulations. It includes tools to condense simulation outputs, compute flat-sky power spectra, assemble training datasets, and train neural networks that predict binned rescaled kSZ power spectra from reionization parameters.

The goal is to learn a fast surrogate model that maps reionization parameters → binned kSZ power spectrum, enabling rapid exploration of cosmological parameter space without re-running expensive simulations.

---

## Installation

```bash
pip install reionemu
```

Or from source (editable):

```bash
git clone https://github.com/RobertxPearce/reionization-emulator.git
cd reionization-emulator
python -m pip install -e .
```

**Requirements:** Python 3.10+, NumPy, HDF5, PyTorch, and Ray Tune.

To run the test suite:

```bash
pip install -e ".[test]"   # or: pip install . pytest
pytest tests/ -v
```

---

## Quick start

After installing, you can load a processed HDF5 training dataset, create dataloaders, and train the baseline 4-parameter emulator:

```python
from pathlib import Path
import torch
import reionemu

# Path to a condensed HDF5 that already has /training (X, Y, ell)
h5_path = Path("path/to/condensed.h5")

# Dataloaders with train/val split and optional normalization
loaders, normalizers, ell = reionemu.make_dataloaders(
    h5_path,
    split={"train": 0.8, "val": 0.2},
    config=reionemu.DataLoaderConfig(batch_size=32, seed=42),
)

# Baseline 4-parameter model, optimizer, loss
model = reionemu.FourParamEmulator()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Train for a few epochs
history = reionemu.fit(
    model,
    loaders["train"],
    loaders["val"],
    optimizer,
    loss_fn,
    config=reionemu.FitConfig(epochs=10, device="cpu"),
)

# Validation loss per epoch
print(history["val_loss"])
```

If you want to tune the four-parameter architecture with Ray Tune before training a final model, you can work directly with the loaded arrays:

```python
from pathlib import Path

import reionemu
from ray import tune

h5_path = Path("path/to/condensed.h5")
X, Y, ell = reionemu.load_training_arrays(h5_path)

split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]

param_space = {
    "hidden_dim": tune.choice([20, 32, 64]),
    "num_hidden_layers": tune.choice([1, 2, 3]),
    "activation": tune.choice(["relu", "silu", "tanh"]),
    "optimizer": tune.choice(["adam", "adamw"]),
    "lr": tune.loguniform(3e-4, 2e-3),
    "weight_decay": tune.loguniform(1e-8, 1e-4),
    "batch_size": tune.choice([16, 32, 64]),
    "epochs": 150,
    "early_stopping_patience": tune.choice([10, 15]),
    "gradient_clipping": tune.choice([None, 0.5, 1.0]),
    "normalize_X": True,
    "normalize_Y": False,
}

results = reionemu.run_tune_four_param(
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    param_space=param_space,
    num_samples=20,
    max_concurrent_trials=2,
    device="cpu",
    storage_path="ray_results",
    experiment_name="four_param_search",
)

best = results.get_best_result(metric="val_loss", mode="min")
print(best.config)
print(best.metrics["best_val_loss"])
```

For a full pipeline example (condense → compute power spectra → build training data → tune/train/evaluate) and scientific context, see the **[reionemu package example notebook](docs/reionemu_package_example.ipynb)**.

---

## Scientific context

The kinetic Sunyaev-Zel'dovich (kSZ) effect arises from the scattering of CMB photons by free electrons with bulk motion, generating secondary temperature anisotropies. The kSZ angular power spectrum carries information about the timing, duration, and structure of reionization. This emulator provides a fast surrogate that maps reionization parameters (zmean_zre, alpha_zre, kb_zre, b0_zre) to binned, rescaled kSZ power spectra, making parameter-space exploration much faster than rerunning the full simulations.

---

## Repository structure

| Path                     | Description                                                                            |
|--------------------------|----------------------------------------------------------------------------------------|
| **`src/reionemu/`**      | Core library (pip-installable package)                                                 |
| `src/reionemu/simio/`    | Simulation I/O, power spectrum computation, training-array building                    |
| `src/reionemu/data/`     | Dataloaders, normalization                                                             |
| `src/reionemu/models/`   | Baseline and experimental emulator architectures                                       |
| `src/reionemu/training/` | Training loop, K-fold cross-validation, metrics, and model builders                    |
| `src/reionemu/tuning/`   | Ray Tune integration for hyperparameter search                                         |
| **`scripts/`**           | Dataset builder, HPC runners, sampling (environment-specific)                          |
| **`notebooks/`**         | Analysis and training examples                                                         |
| **`docs/`**              | Package example notebook                                                               |
| `data/`                  | Raw and processed data (not tracked)                                                   |
| `results/`               | Visualizations for simulation checks, parameter-space validation, and model evaluation |

The **core API** is in `src/reionemu/`. Scripts under `scripts/hpc/` and `scripts/sampling/` are for cluster and sampling workflows and may use machine-specific paths; the library itself is portable.

---

## Main public API

Import from the top-level package after `pip install reionemu`:

- **Simulation I/O:** `condense_sim_root`, `CondenseConfig`, `add_cl_to_condensed_h5`, `ClConfig`, `build_and_write_training`, `build_training_arrays`, `BuildXYConfig`, `BuildStats`, `CondenseStats`
- **Data:** `make_dataloaders`, `load_training_arrays`, `DataLoaderConfig`, `Normalizer`
- **Models:** `FourParamEmulator` (experimental variants live in `reionemu.models.experimental`)
- **Training:** `fit`, `FitConfig`, `train_one_epoch`, `evaluate`, `evaluate_metrics`, `kfold_cross_validate`, `KFoldConfig`
- **Training helpers:** `build_four_param_model`, `build_optimizer`, `mse`, `rmse`, `mean_relative_error`
- **Tuning:** `train_four_param_tune`, `default_param_space`, `run_tune_four_param`

See [src/README.md](src/README.md) for module-level documentation.

---

## Typical workflow

1. **Parameter sampling** - Latin Hypercube Sampling over the 4D reionization parameter space.
2. **Simulation (HPC)** - Run Zreion (or compatible) simulations; outputs per sim in HDF5.
3. **Dataset construction** - Use `condense_sim_root` → `add_cl_to_condensed_h5` → `build_and_write_training` to produce a single condensed HDF5 with `/sims` and `/training`.
4. **Hyperparameter search (optional)** - Use `load_training_arrays` and `run_tune_four_param` to search over model and optimizer settings with Ray Tune.
5. **Training and evaluation** - Use `make_dataloaders` and `fit` (or `kfold_cross_validate`) to train and evaluate the selected emulator configuration.

---

## Acknowledgments

This research is conducted in the LEADS Lab at the University of Nevada, Las Vegas, under [Dr. Paul La Plante](https://plaplant.github.io/), with computing resources from the Pittsburgh Supercomputing Center (Bridges-2).
