# API Overview

## Workflow

The package follows a natural workflow from raw simulation output to trained emulator. Each stage prepares data for the next one, so most users will move through the API in roughly the same order.

You typically begin by condensing simulation outputs into a single HDF5 file, then compute angular power spectra, build training arrays, prepare data loaders, instantiate a model, train or tune it, and save an experiment artifact for reproducibility. This page gives a high-level map of those stages and highlights the main entry points in each one.

Condense Simulation Output → Compute Power Spectra → Build Training Data → Tune Hyperparameters → Train Model → Evaluate Model → Save Artifact

Below is a guide to the API in the same order it is usually used in the pipeline.

## How to read this page

This page is a map of the package rather than a full reference. Use it to understand which layer of the API you need, then move to the dedicated API guide for function-by-function details, parameters, and examples.

The main stable entry points are exposed from the top-level `reionemu` package. Experimental models live under `reionemu.models.experimental` and are better treated as exploratory tools than the default public path.

## Simulation I/O

Simulation I/O contains the utilities that turn raw simulation products into the structured HDF5 data used throughout the rest of the package. In the typical workflow, this layer handles dataset condensation first, then adds computed power-spectrum products, and finally writes training-ready arrays into the same condensed file.

You should use this layer whenever you are starting from raw or partially processed simulation outputs. If you already have a condensed HDF5 file with power spectra and training data written to it, you can usually move on to the data loading and training APIs instead of calling these functions directly.

Main entry points in this layer include `CondenseConfig` and `condense_sim_root` for creating the condensed simulation file, `ClConfig` and `add_cl_to_condensed_h5` for computing and attaching angular power spectra, and `BuildXYConfig`, `build_training_arrays`, and `build_and_write_training` for producing ML-ready training data.

For deeper documentation on this stage, see [Simulation I/O](api/simulation-io.md).

## Data loading

The data loading layer is the bridge between prepared training data on disk and model-ready PyTorch objects in memory. It is responsible for reading training arrays, applying optional normalization, and constructing the loaders used during training and evaluation.

You should use this layer after your condensed HDF5 file already contains training data. In other words, this is the right entry point once preprocessing is complete and you are ready to start model development.

Main entry points in this layer include `load_training_arrays` for reading the prepared `X`, `Y`, and `ell` arrays, `DataLoaderConfig` for controlling loader behavior, `make_dataloaders` for constructing train and validation loaders, and `Normalizer` for working with feature or target normalization.

For deeper documentation on this stage, see [Data Loading](api/data-loading.md).

## Models

The models layer contains the emulator architectures that map reionization parameters to predicted kSZ power-spectrum targets. For most users, this is the point where the prepared training inputs meet a stable baseline model, with an optional MC-dropout variant available when predictive-spread estimates are useful.

You should use this layer once your data is ready for training and you need a concrete model instance. The recommended default public model is `FourParamEmulator`; `MCDropoutEmulator` is the stable dropout-based model for Monte Carlo dropout evaluation. Experimental variants live under `reionemu.models.experimental` and are best treated as optional research extensions.

Main entry points in this layer include `FourParamEmulator` for the stable deterministic baseline, `MCDropoutEmulator` for dropout-based predictive spread, `build_four_param_model` for config-driven deterministic model construction, and `build_mc_dropout_model` for config-driven MC-dropout model construction.

For deeper documentation on this stage, see [Models](api/models.md).

## Training

The training layer contains the utilities used to fit models, evaluate performance, and run validation workflows. It includes the standard multi-epoch training loop, lower-level helpers, evaluation metrics, and cross-validation support.

You should use this layer after you have dataloaders and a model instance ready. For most users, the default path is to configure training with `FitConfig`, call `fit`, and use `evaluate` or `evaluate_metrics` to inspect results. For `MCDropoutEmulator`, use `evaluate_mc_metrics` or `fit(..., evaluation="evaluate_mc_metrics")` when you want validation metrics based on stochastic dropout samples. Lower-level helpers such as `train_one_epoch` are useful when you need more control over the loop.

Main entry points in this layer include `FitConfig`, `fit`, `evaluate`, `evaluate_metrics`, `evaluate_mc_metrics`, `train_one_epoch`, `KFoldConfig`, `kfold_cross_validate`, and the regression metrics `mse`, `rmse`, `mean_relative_error`, and `physical_mean_relative_error`.

For deeper documentation on this stage, see [Training](api/training.md).

## Hyperparameter tuning

The hyperparameter tuning layer integrates Ray Tune with the deterministic four-parameter emulator training workflow. It helps you search over architecture and optimizer settings when you want to go beyond a fixed baseline configuration.

You should use this layer after you already have prepared train and validation arrays. Tuning is optional for many workflows, especially when you are just validating the pipeline or training a baseline model, but it becomes useful when you want to systematically compare configurations and select a stronger final deterministic setup. `MCDropoutEmulator` is trained through the regular training API; the current Ray Tune helper is scoped to `FourParamEmulator`.

Main entry points in this layer include `default_param_space` for a starting search space, `train_four_param_tune` for the per-trial training logic, and `run_tune_four_param` for launching and managing a full tuning run.

For deeper documentation on this stage, see [Hyperparameter Tuning](api/hyperparameter-tuning.md).

## Experiment artifacts

The artifact layer records experiment metadata without modifying the condensed HDF5 dataset. It writes JSON files for run identity, configuration choices, and results, plus optional sidecar files for objects that are better saved in binary formats.

Use this layer after building a dataset, training a model, or running a validation workflow. The main entry point is `save_artifact`, which creates a run directory containing `info.json`, `configs.json`, `results.json`, and optional files such as `normalizers.npz` and `model.pt`. Lower-level helpers such as `save_configs`, `save_results`, `save_info`, `save_normalizers`, `load_normalizers`, `save_model_checkpoint`, `dataset_summary`, `file_fingerprint`, and `read_json` are available when you want to write or read individual pieces.

For deeper documentation on this stage, see [Artifacts](api/artifacts.md).

## Typical top-level imports

Most users can stay at the package top level for the stable public API:

```python
import reionemu

model = reionemu.FourParamEmulator()
# Or, for MC-dropout evaluation:
# model = reionemu.MCDropoutEmulator(dropout_rate=0.1)
loaders, normalizers, ell = reionemu.make_dataloaders(h5_path)
history = reionemu.fit(...)
artifact_dir = reionemu.save_artifact("baseline_four_param", "artifacts", history=history)
```

If you are exploring internals, experimental variants, or implementation details, you may also work directly with submodules such as `reionemu.simio`, `reionemu.training`, `reionemu.tuning`, `reionemu.artifact`, or `reionemu.models.experimental`.
