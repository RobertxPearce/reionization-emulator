# Emulator Library (`reionemu`)

Core Python package for the reionization emulator: condensing simulation outputs, computing kSZ power spectra, building ML-ready training arrays, and training networks.

---

## Public API

Install with `pip install reionemu` (or `pip install -e .` from the repo root), then import from the top level:

```python
import reionemu

# Simulation I/O and training-array building
reionemu.condense_sim_root(...)
reionemu.CondenseConfig
reionemu.add_cl_to_condensed_h5(...)
reionemu.ClConfig
reionemu.build_and_write_training(...)
reionemu.build_training_arrays(...)
reionemu.BuildXYConfig
reionemu.BuildStats
reionemu.CondenseStats

# Data loaders and normalization
reionemu.make_dataloaders(...)
reionemu.load_training_arrays(...)
reionemu.DataLoaderConfig
reionemu.Normalizer

# Models (baseline + experimental)
reionemu.FourParamEmulator
reionemu.models.experimental.POCEmulatorThreeParams

# Training loops, metrics, builders, and tuning
reionemu.fit(...)
reionemu.FitConfig
reionemu.train_one_epoch(...)
reionemu.evaluate(...)
reionemu.evaluate_metrics(...)
reionemu.kfold_cross_validate(...)
reionemu.KFoldConfig
reionemu.build_four_param_model(...)
reionemu.build_optimizer(...)
reionemu.mse(...)
reionemu.rmse(...)
reionemu.mean_relative_error(...)
reionemu.train_four_param_tune(...)
reionemu.default_param_space(...)
reionemu.run_tune_four_param(...)
```

Experimental POC architectures live in `reionemu.models.experimental`.

---

## Modules

### simio/

Simulation I/O and preprocessing.

- **condense_h5.py** — Build a single condensed HDF5 from per-simulation directories. Uses `CondenseConfig`; returns a dict with `written`, `skipped`, and skip breakdown. Optional `progress_callback(completed, total)`. Condensed structure is as follows:  

    ```
    Top-Level:
        ['sims']
    sims:
        ['sim0'], ['sim1'], ['sim2'], ... , ['sim<n>']
    sim<n>:
        [params], [output], [cl]
    params:
        ['alpha_zre', 'b0_zre', 'kb_zre', 'zmean_zre']
    output:
        ['ksz_map', 'Tcmb0', 'theta_max_ksz', 'pk_tt', 'tau', 'xmval_list', 'zval_list']
    ```

- **compute_cl.py** — Compute flat-sky angular power spectrum from each sim’s kSZ map, write `/cl` (ell, cl_ksz, dcl, dl_ksz) into the condensed HDF5. Uses `ClConfig`. Optional `progress_callback`. Condensed structure is as follows:  

    ```
    Top-Level:
        ['sims']
    sims:
        ['sim0'], ['sim1'], ['sim2'], ... , ['sim<n>']
    sim<n>:
        [params], [output], [cl]
    params:
        ['alpha_zre', 'b0_zre', 'kb_zre', 'zmean_zre']
    output:
        ['ksz_map', 'Tcmb0', 'theta_max_ksz', 'pk_tt', 'tau', 'xmval_list', 'zval_list']
    cl:
        ['cl_ksz', 'dcl', 'dl_ksz', 'ell']
    ```

- **build_xy.py** — Build ML arrays (X, Y, ell) from condensed HDF5 and write `/training`. Uses `BuildXYConfig`. Returns `BuildStats` (processed/skipped counts). Skip reasons: missing params, missing cl, inconsistent ell, non-finite values. Condensed structure is as follows:  

    ```
    Top-Level:
        ['sims', 'training']
    sims:
        ['sim0'], ['sim1'], ['sim2'], ... , ['sim<n>']
    sim<n>:
        [params], [output], [cl]
    params:
        ['alpha_zre', 'b0_zre', 'kb_zre', 'zmean_zre']
    output:
        ['ksz_map', 'Tcmb0', 'theta_max_ksz', 'pk_tt', 'tau', 'xmval_list', 'zval_list']
    cl:
        ['cl_ksz', 'dcl', 'dl_ksz', 'ell']
    training:
        ['X', 'Y', 'ell', 'param_names', 'sim_ids']
    ```

### data/

- **normalization.py** — `Normalizer`, `fit_standardizer`, `transform_standardizer`, `inverse_transform_standardizer`.
  - `Normalizer(mean, std)` Container
  - `fit_standardizer()` Computes mean/std over `axis=0` and guards `std==0`
  - `transform_standardizer()` and `inverse_transform_standardizer()` apply/undo scaling
- **dataloaders.py** — `load_training_arrays(h5_path)`, `make_dataloaders(h5_path, split=..., config=DataLoaderConfig())`. Validates X, Y, ell (shapes and finite values) before building loaders.

### models/

- **four_param_emulator.py** — `FourParamEmulator`: 4 → 20 → 20 → 5 (ReLU), 5 spectrum bins.
- **experimental/** — POC variants: `POCEmulatorFourParamsV1/V2/V3`, `POCEmulatorThreeParams`. Import from `reionemu.models.experimental`.

### training/

- **train_loop.py** — `FitConfig`, `train_one_epoch`, `evaluate`, `fit(model, train_loader, val_loader, optimizer, loss_fn, config)`.
  - `FitConfig` (epochs, device, optional early stopping patience, optional gradient clipping)
  - `fit()` trains for many epochs, prints losses each epoch, supports early stopping and restores best weights when used
- **kfold_cv.py** — `KFoldConfig`, `kfold_cross_validate(h5_path, model_builder=..., ...)`.
- **builders.py** — `build_four_param_model(config)`, `build_optimizer(model, config)`.
- **metrics.py** — `mse(pred, target)`, `rmse(pred, target)`, `mean_relative_error(pred, target)`.

### tuning/

- **four_param.py** — Ray Tune helpers: `resolve_device`, `train_four_param_tune`, `default_param_space`, `run_tune_four_param`.

---

## Typical flow

1. Condense raw sim outputs → one HDF5: `condense_sim_root(...)`
2. Compute spectra and write `/cl`: `add_cl_to_condensed_h5(...)`
3. Build and write `/training`: `build_and_write_training(...)`
4. Create loaders and train: `make_dataloaders(...)`, `run_tune_four_param`, then `fit(...)` or `kfold_cross_validate(...)`
