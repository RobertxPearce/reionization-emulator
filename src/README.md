# Emulator Library

Core Python package for the reionization emulator: condensing simulation outputs, computing kSZ power spectra, building ML-ready training arrays, and training networks.

---

## Modules

### simio/
Simulation I/O and preprocessing utilities.

- **[condense_h5.py](simio/condense_h5.py)**  
  Builds a single condensed HDF5 from per-simulation directories by extracting arrays/scalars, and validating required fields. Condensed structure is as follows:  
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

- **[compute_cl.py](simio/compute_cl.py)**  
  Computes the flat-sky angular power spectrum from each sim’s `ksz_map` (converted to $\mu$K using `Tcmb0`), applies a Hann window + normalization, removes map mean/NaNs, FFTs to 2D power, bins to full-res then rebins to `nbins` after dropping low-$\ell$ bins below `ell_cut`, and writes into the condensed HDF5. Condensed structure is as follows:  
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

- **[build_xy.py](simio/build_xy.py)**  
  Builds ML-ready arrays from the condensed HDF5 and writes them back into the condensed HDF5 file:
  - `X`: Ordered parameter matrix (default `("zmean_zre","alpha_zre","kb_zre","b0_zre")`)
  - `Y`: Target spectrum (default `dl_ksz`) with optional transform (`none`, `log10`, `ln`) and `eps` added for stability
  - Enforces consistent `ell` binning across sims and writes reproducibility metadata.
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

---

### data/
Training data helpers.

- **[normalization.py](data/normalization.py)**  
  Simple feature-wise standardization utilities:
  - `Normalizer(mean, std)` Container
  - `fit_standardizer()` Computes mean/std over `axis=0` and guards `std==0`
  - `transform_standardizer()` and `inverse_transform_standardizer()` apply/undo scaling

---

### models/
Baseline PyTorch model definitions.

- **[poc_three_params.py](models/poc_three_params.py)**  
  POC MLP: 3 $\rightarrow$ 5 $\rightarrow$ 5 with GELU, predicting 5 spectrum bins

- **[poc_four_params.py](models/poc_four_params.py)**  
  POC MLP: 4 $\rightarrow$ 5 $\rightarrow$ 5 with GELU, predicting 5 spectrum bins

---

### training/
Reusable training loop.

- **[train_loop.py](training/train_loop.py)**  
  Minimal PyTorch training utilities:
  - `FitConfig` (epochs, device, optional early stopping patience, optional gradient clipping)
  - `train_one_epoch()`, `evaluate()`
  - `fit()` trains for many epochs, prints losses each epoch, supports early stopping and restores best weights when used

---

## Typical Flow

1. **Condense raw sim outputs $\rightarrow$ one HDF5**  
   `simio/condense_h5.py`

2. **Compute spectra and write `/cl` into the condensed HDF5**  
   `simio/compute_cl.py`

3. **Build training arrays and write `/training` into the same HDF5**  
   `simio/build_xy.py`

4. **Standardize inputs/targets + create loaders**  
   `data/normalization.py`

5. **Train model**  
   `models/` + `training/train_loop.py`