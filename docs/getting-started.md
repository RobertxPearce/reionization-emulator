# Getting Started

Use this page as the quickest path to a first successful run.

## Installation

**reionemu** can be installed via `pip` with:

```bash
pip install reionemu
```

or from source (editable):

```bash
git clone https://github.com/RobertxPearce/reionization-emulator.git
cd reionization-emulator
python -m pip install -e .
```

Requires Python 3.10+. Package dependencies include NumPy, `h5py`, PyTorch, and Ray Tune.

## Verify installation

To confirm the package imports correctly:

```python
import reionemu as remu

print(remu.__all__)
```

This should print the main public functions, config objects, and model classes exposed by the package.

## First run

The example below starts from a condensed HDF5 file that already contains a `/training` group with `X`, `Y`, and `ell`. If you are starting from raw simulation outputs instead, begin with the Simulation I/O workflow first.

After installing, you can load a prepared training dataset, create dataloaders, and train the baseline deterministic 4-parameter emulator:

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

artifact_dir = reionemu.save_artifact(
    "baseline_four_param",
    Path("artifacts"),
    dataset_path=h5_path,
    dataloader_config=reionemu.DataLoaderConfig(batch_size=32, seed=42),
    fit_config=reionemu.FitConfig(epochs=10, device="cpu"),
    model_config={"class_name": "FourParamEmulator"},
    optimizer_config={"name": "Adam", "lr": 1e-3},
    history=history,
    normalizers=normalizers,
    checkpoint=model.state_dict(),
)

print(artifact_dir)
```

If this runs successfully, you should see a validation-loss history printed at the end of the script and an `artifacts/baseline_four_param/` directory with `info.json`, `configs.json`, `results.json`, and optional binary sidecars.

If you want a dropout-based emulator that can be evaluated with Monte Carlo dropout, swap in `MCDropoutEmulator` and use the MC evaluation path:

```python
model = reionemu.MCDropoutEmulator(dropout_rate=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = reionemu.fit(
    model,
    loaders["train"],
    loaders["val"],
    optimizer,
    torch.nn.MSELoss(),
    config=reionemu.FitConfig(epochs=10, device="cpu"),
    evaluation="evaluate_mc_metrics",
    n_mc_samples=50,
)

print(history["val_mean_predictive_std"])
```

## Common pitfalls

- Make sure the input HDF5 file already contains a `/training` group before calling `make_dataloaders`.
- Use the Simulation I/O pipeline first if you only have raw simulation outputs.
- Confirm that your Python environment includes the package dependencies before running training code.
- Save normalizers and model weights as artifact sidecars rather than putting them directly into JSON.
