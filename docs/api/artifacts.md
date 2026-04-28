# Artifacts

The *artifact* module saves lightweight experiment records for reproducibility. It keeps the condensed HDF5 dataset as a data product and writes experiment metadata to a separate run directory.

## What This Module Does

- Creates an artifact directory for one experiment run
- Writes JSON files for run identity, configs, and results
- Records which HDF5 dataset was used without modifying the HDF5 file
- Saves `Normalizer` objects as a NumPy `.npz` sidecar
- Saves PyTorch checkpoints as `.pt` files
- Provides helpers for reading JSON and loading saved normalizers

This module specifically handles the final step in the workflow: Save Artifact.

## When To Use It

Use this module after preparing a dataset, training a model, evaluating a model, or running a tuning/cross-validation workflow. It is intended for simple local experiment tracking, not as a replacement for a full experiment database.

The recommended path is:

1. Prepare or load a condensed HDF5 training dataset.
2. Train or evaluate a model.
3. Call `save_artifact(...)` with the configs, results, normalizers, and checkpoint you want to keep.

## Output Layout

A typical artifact directory looks like this:

```text
artifacts/
    baseline_four_param/
        info.json
        configs.json
        results.json
        normalizers.npz
        model.pt
```

The JSON files are human-readable. The `.npz` and `.pt` files are sidecars for data that should not be forced into JSON.

## Typical Workflow

```python
from pathlib import Path

import torch

from reionemu import (
    DataLoaderConfig,
    FitConfig,
    FourParamEmulator,
    fit,
    make_dataloaders,
    save_artifact,
)

h5_path = Path("path/to/condensed.h5")

dataloader_config = DataLoaderConfig(batch_size=32, seed=42)
fit_config = FitConfig(epochs=100, device="cpu")

loaders, normalizers, ell = make_dataloaders(
    h5_path,
    split={"train": 0.8, "val": 0.2},
    config=dataloader_config,
)

model = FourParamEmulator()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

history = fit(
    model=model,
    train_loader=loaders["train"],
    val_loader=loaders["val"],
    optimizer=optimizer,
    loss_fn=torch.nn.MSELoss(),
    config=fit_config,
)

artifact_dir = save_artifact(
    "baseline_four_param",
    Path("artifacts"),
    dataset_path=h5_path,
    dataloader_config=dataloader_config,
    fit_config=fit_config,
    model_config={
        "class_name": "FourParamEmulator",
        "input_dim": 4,
        "output_dim": 5,
    },
    optimizer_config={
        "name": "AdamW",
        "lr": 1e-3,
    },
    history=history,
    normalizers=normalizers,
    checkpoint=model.state_dict(),
)

print(artifact_dir)
```

## save_artifact

`save_artifact` is the high-level entry point. It creates the artifact directory and writes all available artifact files.

```python
def save_artifact(
    name: str,
    root_dir: Path,
    *,
    dataset_path: Path | None = None,
    condense_config: Any = None,
    cl_config: Any = None,
    build_config: Any = None,
    dataloader_config: Any = None,
    fit_config: Any = None,
    kfold_config: Any = None,
    model_config: Mapping[str, Any] | None = None,
    optimizer_config: Mapping[str, Any] | None = None,
    tuning_config: Mapping[str, Any] | None = None,
    results_summary: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    history: Mapping[str, Any] | None = None,
    dataset_prep_stats: Mapping[str, Any] | None = None,
    normalizers: Mapping[str, Normalizer | None] | None = None,
    checkpoint: Any = None,
    description: str | None = None,
) -> Path:
```

### Main Arguments

| Parameter | Description |
|:----------|:------------|
| `name` | Run name and artifact directory name |
| `root_dir` | Parent directory where artifacts are stored |
| `dataset_path` | Optional path to the HDF5 dataset used by the experiment |
| `condense_config` | Optional `CondenseConfig` used to make the dataset |
| `cl_config` | Optional `ClConfig` used to compute spectra |
| `build_config` | Optional `BuildXYConfig` used to build training arrays |
| `dataloader_config` | Optional `DataLoaderConfig` used for dataloaders |
| `fit_config` | Optional `FitConfig` used for model training |
| `kfold_config` | Optional `KFoldConfig` used for cross-validation |
| `model_config` | Optional model architecture dictionary |
| `optimizer_config` | Optional optimizer dictionary |
| `tuning_config` | Optional tuning/search dictionary |
| `results_summary` | Optional high-level result summary |
| `metrics` | Optional scalar metrics dictionary |
| `history` | Optional training history dictionary |
| `dataset_prep_stats` | Optional preprocessing statistics |
| `normalizers` | Optional mapping such as `{"X": x_norm, "Y": y_norm}` |
| `checkpoint` | Optional PyTorch checkpoint or model `state_dict` |
| `description` | Optional human-readable run description |

### Returns

`save_artifact` returns the created artifact directory as a `Path`.

## JSON Files

### info.json

`info.json` is the manifest for the run. It records the run name, creation time, dataset summary, and the artifact files that were written.

```json
{
    "artifacts": {
        "configs": "configs.json",
        "model_checkpoint": "model.pt",
        "normalizers": "normalizers.npz",
        "results": "results.json"
    },
    "created_at": "2026-04-27T18:30:12.123456+00:00",
    "dataset": {
        "fingerprint": {
            "file_size_bytes": 123456789,
            "modified_at": "2026-04-27T17:10:05.000000+00:00",
            "path": "/path/to/condensed.h5"
        },
        "n_parameters": 4,
        "n_samples": 100,
        "n_targets": 5,
        "param_names": ["zmean_zre", "alpha_zre", "kb_zre", "b0_zre"],
        "path": "/path/to/condensed.h5"
    },
    "description": "Baseline four-parameter emulator",
    "experiment_name": "baseline_four_param",
    "run_id": "baseline_four_param",
    "schema_version": 1
}
```

### configs.json

`configs.json` stores choices made before the run.

```json
{
    "data_loading": {
        "dataloader": {
            "batch_size": 32,
            "normalize_X": true,
            "normalize_Y": false,
            "seed": 42,
            "shuffle_train": true
        }
    },
    "dataset_prep": {
        "build_xy": null,
        "cl": null,
        "condense": null
    },
    "kfold": null,
    "model": {
        "class_name": "FourParamEmulator",
        "input_dim": 4,
        "output_dim": 5
    },
    "optimizer": {
        "lr": 0.001,
        "name": "AdamW"
    },
    "schema_version": 1,
    "training": {
        "device": "cpu",
        "early_stopping_patience": null,
        "epochs": 100,
        "gradient_clipping": null
    },
    "tuning": null
}
```

### results.json

`results.json` stores outputs from the run.

```json
{
    "created_at": "2026-04-27T18:30:12.234567+00:00",
    "dataset_prep_stats": {},
    "history": {
        "train_loss": [0.6, 0.4, 0.25],
        "val_loss": [0.7, 0.5, 0.3]
    },
    "metrics": {
        "val_loss": 0.3,
        "val_rmse": 0.12
    },
    "schema_version": 1,
    "status": "completed",
    "summary": {
        "best_epoch": 47,
        "best_val_loss": 0.3
    }
}
```

## Lower-Level Helpers

Use these helpers when you want to control individual artifact files yourself.

| Helper | Purpose |
|:-------|:--------|
| `create_artifact_dir(name, root_dir)` | Create the run directory |
| `save_configs(artifact_dir, ...)` | Write `configs.json` |
| `save_results(artifact_dir, ...)` | Write `results.json` |
| `save_info(artifact_dir, ...)` | Write `info.json` |
| `save_normalizers(artifact_dir, normalizers)` | Write `normalizers.npz` |
| `load_normalizers(path)` | Read `normalizers.npz` |
| `save_model_checkpoint(artifact_dir, checkpoint)` | Write `model.pt` |
| `dataset_summary(h5_path)` | Summarize the HDF5 training dataset |
| `file_fingerprint(path)` | Record path, file size, and modified time |
| `read_json(path)` | Read an artifact JSON file |

## Notes

- The artifact system does not write into the condensed HDF5 file.
- JSON files are intended for readable metadata, configs, metrics, and histories.
- Normalizers are saved as `.npz` because they contain NumPy arrays.
- Model checkpoints are saved as `.pt` because they use PyTorch serialization.
- `dataset_summary(...)` reads the HDF5 file in read-only mode and records basic `/training` metadata when available.
