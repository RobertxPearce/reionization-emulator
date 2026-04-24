# Simulation I/O

The *simulation I/O* module processes the raw simulation output and provides a uniform, reduced file structure. These methods are required for utilizing the rest of the package, since all later methods expect this specific data format and layout.

## What This Module Does

- Condenses raw per-simulation HDF5 outputs into a single structured HDF5 file
- Computes flat-sky kSZ angular power spectra and writes them into the condensed file
- Builds ML-ready training arrays used by the rest of the package

This module specifically handles these steps in the workflow: Condense Simulation Output → Compute Power Spectra → Build Training Data.

## When To Use It

Use this module when you are starting from raw or partially processed simulation output. If you already have a condensed HDF5 file with `/cl` and `/training` written, you can usually move on to the data loading and training modules instead.

The raw kSZ 2LPT simulation produces data that is unnecessary for the purpose of emulating the kSZ angular power spectrum. The raw data is also split into two HDF5 files per simulation (*obs_grids* and *pk_arrays*), so this can be compacted into a single HDF5 file. By removing the unnecessary information and condensing to a single HDF5 file, the overall file size is reduced by 75%, making it easier to store, compress, and transfer.

The kSZ angular power spectrum is also not computed by the simulation code. Therefore, the next step in the process is computing the kSZ angular power spectrum and inserting it into a subdirectory of the condensed HDF5.

The last step in preparing the raw simulation is building the training arrays to allow for quick execution of training scripts. These training arrays store the input parameters (*zmean_zre*, *alpha_zre*, *kb_zre*, *b0_zre*), targets (*binned kSZ angular power spectrum*), and reference *ell* bins. There are important decisions that must be made when preparing the training data, such as whether the target should be the angular power spectrum or rescaled angular power spectrum, and whether the target should be transformed. Completing this step and saving it into the file allows for multiple different datasets for training that can be easily identified.

There are configuration classes for each of these steps, allowing the settings to be saved for reproducibility of experiments. Below are in-depth explanations of the main methods and configurations.

## Typical Workflow

```python
from pathlib import Path

from reionemu import (
    BuildXYConfig,
    ClConfig,
    CondenseConfig,
    add_cl_to_condensed_h5,
    build_and_write_training,
    condense_sim_root,
)

raw_sim_root = Path("path/to/raw/simulations")
condensed_h5 = Path("path/to/condensed.h5")

condense_sim_root(
    sim_root=raw_sim_root,
    out_path=condensed_h5,
    config=CondenseConfig(),
)

add_cl_to_condensed_h5(
    condensed_h5,
    config=ClConfig(),
)

build_and_write_training(
    condensed_h5,
    config=BuildXYConfig(),
)
```

## Condense H5

This is the first step in the workflow that extracts the data, condenses it into a single HDF5, and verifies the payloads. Both per-simulation HDF5 files are read. From the *pk_arrays* file, `pk_tt`, `xmval_list`, `zval_list`, `alpha_zre`, `b0_zre`, `kb_zre`, `zmean_zre`, and `tau` are extracted. From *obs_grids*, `ksz_map`, `Tcmb0`, and `theta_max_ksz` are extracted.

### Purpose

Use this step to convert raw per-simulation output into the condensed HDF5 layout expected by the rest of the pipeline.

### Configuration

This dataclass provides options for overwrite control and an extra validation step.

```python
@dataclass(frozen=True)
class CondenseConfig:
    overwrite: bool = True
    require_obs_and_pk: bool = True
```

- **overwrite**: When True, an existing condensed HDF5 file at `out_path` will be replaced. When False, `condense_sim_root` raises a `FileExistsError` if `out_path` already exists. This option does not append to an existing file or skip already-written simulations.
- **require_obs_and_pk**: When True, simulations missing either an `obs_grids` file or a `pk_arrays` file are counted as missing-file skips before reading. When False, the function attempts to read whichever files are present, but the payload is still validated before writing, so simulations missing required fields from either file will be skipped as validation errors. For most cases this should remain True.

### Main Entry Point

This is the method that orchestrates the process of extracting the necessary data and condensing it into a single HDF5 file.

```python
def condense_sim_root(
    sim_root: Path,
    out_path: Path,
    *,
    config: CondenseConfig = CondenseConfig(),
    sim_prefix: str = "sim",
    file_description: str = "Condensed simulation outputs for kSZ 2LPT emulator.",
    version: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> CondenseStats:
```

| Parameter         | Type             | Default                                                | Description                                               |
|:------------------|:-----------------|:-------------------------------------------------------|:----------------------------------------------------------|
| sim_root          | `Path`           | *Required*                                             | Path to the raw kSZ 2LPT simulation output                |
| out_path          | `Path`           | *Required*                                             | Path to the condensed output .h5 file                     |
| config            | `CondenseConfig` | Defaults                                               | Configuration dataclass                                   |
| sim_prefix        | `str`            | "sim"                                                  | Subfolder name prefix to include                          |
| file_description  | `str`            | "Condensed simulation outputs for kSZ 2LPT emulator."  | Description of the output file                            |
| version           | `int`            | 1                                                      | Versioning option                                         |
| progress_callback | `Callable`       | Optional                                               | Optional callable(completed, total) called after each sim |

### Returns

The method returns a *CondenseStats* object containing the number of sims written, sims skipped due to missing `obs_grids` or `pk_arrays`, sims skipped due to a read error, and a catch-all for sims skipped due to a validation error. There is also a property that returns the total sims skipped. The class is outlined below.

```python
@dataclass(frozen=True)
class CondenseStats:
    written: int
    skipped_missing_obs_pk: int
    skipped_read_error: int
    skipped_validation_error: int

    @property
    def skipped_total(self) -> int:
        return (
            self.skipped_missing_obs_pk
            + self.skipped_read_error
            + self.skipped_validation_error
        )
```

After `condense_sim_root`, the file will have a structure like:

```
Top-Level:
 ['sims']
 
sims:
 ['sim0', 'sim1', ... ,  'sim<n>']
sim<n>:
 ['output', 'params']
params:
 ['alpha_zre', 'b0_zre', 'kb_zre', 'zmean_zre']
output:
 ['Tcmb0', 'ksz_map', 'pk_tt', 'tau', 'theta_max_ksz', 'xmval_list', 'zval_list']
```

### Written Products

- `params/alpha_zre`, `params/b0_zre`, `params/kb_zre`, `params/zmean_zre`: The scalar reionization parameters used later as the input features for training.
- `output/ksz_map`: The kSZ map used later to compute the angular power spectrum.
- `output/Tcmb0`: The CMB temperature metadata used in the map-to-microkelvin conversion.
- `output/theta_max_ksz`: The angular size metadata used when computing the flat-sky multipole grid.
- `output/pk_tt`, `output/xmval_list`, `output/zval_list`, `output/tau`: Additional simulation products and metadata preserved in the condensed file for traceability and later use.

### Typical Usage

```python
from pathlib import Path
from reionemu import CondenseConfig, condense_sim_root

stats = condense_sim_root(
    sim_root=Path("path/to/raw/simulations"),
    out_path=Path("path/to/condensed.h5"),
    config=CondenseConfig(overwrite=True, require_obs_and_pk=True),
)

print(stats.written, stats.skipped_total)
```

## Compute kSZ Angular Power Spectrum

The kSZ 2LPT simulation does not compute the kSZ angular power spectrum, but it does provide the kSZ map (`ksz_map`) and the metadata needed to compute it. A flat-sky angular power spectrum is computed following a standard method.

### Purpose

Use this step after condensation when you want to attach binned power-spectrum products to each simulation in the condensed file.

### Configuration

A dataclass is used to adjust how the computation is done.

```python
@dataclass(frozen=True)
class ClConfig:
    nbins: int = 5
    ell_cut: float = 1000.0
    overwrite: bool = True
    sims_group: str = "sims"
```

This dataclass provides four options, two of which are especially important in the calculation of the angular power spectrum.

- **nbins:** Controls the number of bins used when binning `ell`
- **ell_cut:** Controls the minimum `ell` retained. Full-resolution bins with centers `< ell_cut` are discarded.
- **overwrite:** If True, any existing `/cl` directory will be overwritten.
- **sims_group:** Name of the top-level sims group in the HDF5 file.

### Main Entry Point

This is the main method used to compute the angular power spectrum from the condensed HDF5 file and save the results back into the same file.

```python
def add_cl_to_condensed_h5(
    h5_path: Path,
    *,
    config: ClConfig = ClConfig(),
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
```

| Parameter         | Type       | Default      | Description                                               |
|:------------------|:-----------|:-------------|:----------------------------------------------------------|
| h5_path           | `Path`     | *Required*   | Path to the condensed `.h5` file                          |
| config            | `ClConfig` | Defaults     | Configuration dataclass for power-spectrum computation    |
| progress_callback | `Callable` | Optional     | Optional callable `(completed, total)` called after each sim |

### Returns

This method writes the computed angular power spectrum products into each simulation's `/cl` group in the condensed HDF5 file. It returns the number of simulations that were updated.


```
Top-Level:
 ['sims']
 
sims:
 ['sim0', 'sim1', ... ,  'sim<n>']
sim<n>:
 ['cl', 'output', 'params']
params:
 ['alpha_zre', 'b0_zre', 'kb_zre', 'zmean_zre']
output:
 ['Tcmb0', 'ksz_map', 'pk_tt', 'tau', 'theta_max_ksz', 'xmval_list', 'zval_list']
cl:
 ['cl_ksz', 'dcl', 'dl_ksz', 'ell']
```

### Written Products

- `cl/ell`: The final binned multipole centers kept after the `ell_cut` filtering and any coarse rebinning.
- `cl/cl_ksz`: The binned `C_ell` values computed from the flat-sky power spectrum of the kSZ map.
- `cl/dl_ksz`: The corresponding `D_ell` values, computed from `C_ell` using `D_ell = ell * (ell + 1) * C_ell / (2 * pi)`.
- `cl/dcl`: An uncertainty estimate per bin based on the number of Fourier modes contributing to that bin.

The code first builds a full-resolution spectrum, removes bins below `ell_cut`, and then rebins the remaining high-`ell` part into `nbins` coarse bins if needed. For the full-resolution bins, `dcl` is computed as `cl / sqrt(counts)`, where `counts` is the number of Fourier modes in the bin. After coarse rebinning, the same idea is applied using the total mode counts in each coarse bin. The final `cl` and `dcl` values are then corrected by the Hann-window normalization factor.

### Typical Usage

```python
from pathlib import Path
from reionemu import ClConfig, add_cl_to_condensed_h5

updated = add_cl_to_condensed_h5(
    Path("path/to/condensed.h5"),
    config=ClConfig(nbins=5, ell_cut=1000.0, overwrite=True),
)

print(updated)
```

## Build Training Data

This is the last main step in the simulation I/O workflow. After the condensed file has the `/cl` group for each simulation, the next step is building the training arrays used by the emulator training code. This step reads the simulation parameters and the selected power-spectrum product, checks that the `ell` bins are consistent across simulations, applies an optional target transform, and writes the result into a `/training` group.

This step is important because it converts the per-simulation structure into the ML-ready layout expected by the rest of the package. Instead of reading values from each sim one by one during training, the package can load a single set of arrays for inputs, targets, reference `ell` bins, parameter names, and simulation ids.

### Purpose

Use this step when you want to convert the per-simulation condensed layout into the array-based format used by the training and evaluation code.

### Configuration

This dataclass controls which groups are read, which target is used, and whether the target should be transformed before training.

```python
@dataclass(frozen=True)
class BuildXYConfig:
    sims_group: str = "sims"
    params_group: str = "params"
    cl_group: str = "cl"
    param_names: Tuple[str, ...] = ("zmean_zre", "alpha_zre", "kb_zre", "b0_zre")
    y_source: str = "dl_ksz"
    y_transform: str = "ln"
    eps: float = 1e-30
```

- **sims_group:** Name of the top-level sims group in the HDF5 file.
- **params_group:** Name of the parameter subgroup under each simulation.
- **cl_group:** Name of the power-spectrum subgroup under each simulation.
- **param_names:** Ordered parameter names used to construct the input matrix `X`.
- **y_source:** Which power-spectrum product should be used as the training target. This is typically `dl_ksz` or `cl_ksz`.
- **y_transform:** Optional transform applied to the target values. The current options are `none`, `log10`, and `ln`.
- **eps:** Small constant added before a logarithm is applied, mainly to avoid issues around zero.

### Main Entry Point

This is the main method used to construct the in-memory training arrays from the condensed HDF5 file.

```python
def build_training_arrays(
    h5_path: Path,
    *,
    config: BuildXYConfig = BuildXYConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, BuildStats]:
```

| Parameter | Type            | Default    | Description                                            |
|:----------|:----------------|:-----------|:-------------------------------------------------------|
| h5_path   | `Path`          | *Required* | Path to the condensed `.h5` file                       |
| config    | `BuildXYConfig` | Defaults   | Configuration dataclass for building the training data |

### Returns

The method returns the input matrix `X`, target matrix `Y`, reference `ell` bin centers, simulation ids, parameter names, and a `BuildStats` object. This allows the arrays to be inspected or modified before they are written to disk.

```python
@dataclass(frozen=True)
class BuildStats:
    total_sims: int
    processed: int
    skipped_missing_params: int
    skipped_missing_cl: int
    skipped_inconsistent_ell: int
    skipped_non_finite: int
```

This stats object keeps track of how many simulations were processed and how many were skipped for different reasons. This is useful because it makes it easier to detect if a large part of the dataset was excluded due to missing parameters, missing power-spectrum products, inconsistent `ell` bins, or non-finite values.

### Write Entry Point

Once the arrays have been built, they can be written into the condensed HDF5 file using the convenience method below.

```python
def build_and_write_training(
    h5_path: Path,
    *,
    config: BuildXYConfig = BuildXYConfig(),
    overwrite: bool = True
) -> int:
```

| Parameter | Type            | Default    | Description                                                  |
|:----------|:----------------|:-----------|:-------------------------------------------------------------|
| h5_path   | `Path`          | *Required* | Path to the condensed `.h5` file                             |
| config    | `BuildXYConfig` | Defaults   | Configuration dataclass for building the training data       |
| overwrite | `bool`          | `True`     | If True, an existing `/training` group will be overwritten   |

### Returns

This method returns the number of simulations included in the written training dataset. Unlike `build_training_arrays`, this method writes the results directly into the `/training` group of the condensed HDF5 file.

After `build_and_write_training`, the training group will have a structure like:

```text
training:
 ['X', 'Y', 'ell', 'param_names', 'sim_ids']
X:
 [array([zmean_zre, alpha_zre, kb_zre, b0_zre]), ...]
Y:
 [array([...]), ...]
ell:
 [np.float64(...), np.float64(...), ...]
param_names:
 [b'zmean_zre', b'alpha_zre', b'kb_zre', b'b0_zre']
sim_ids:
 [b'sim0', b'sim1', b'sim2', ... , b'sim<n>']
```

### Written Products

- `training/X`: The input feature matrix built from `BuildXYConfig.param_names`.
- `training/Y`: The target matrix built from `BuildXYConfig.y_source`, with the optional transform from `BuildXYConfig.y_transform` applied.
- `training/ell`: The reference `ell` bin centers matching the columns of `Y`.
- `training/param_names`: The ordered parameter names used to build the columns of `X`.
- `training/sim_ids`: The simulation ids included in the final dataset.

The `/training` group also stores metadata attributes describing the build configuration, including `y_source`, `y_transform`, `eps`, the number of samples, the number of parameters, the ell bin count, and the skip statistics when they are available.

### Typical Usage

```python
from pathlib import Path
from reionemu import BuildXYConfig, build_and_write_training

count = build_and_write_training(
    Path("path/to/condensed.h5"),
    config=BuildXYConfig(y_source="dl_ksz", y_transform="ln"),
    overwrite=True,
)

print(count)
```

## Notes

In practice, the full simulation I/O workflow is usually run in this order:

1. `condense_sim_root`
2. `add_cl_to_condensed_h5`
3. `build_and_write_training`

At the end of these steps, the condensed HDF5 file contains the raw outputs needed for traceability, the computed power-spectrum products, and the final training arrays used by the rest of the package.

Another useful point to keep in mind is that this stage is where many data-quality issues will first show up clearly. Missing files, missing groups, inconsistent `ell` bins, and non-finite values will all surface here, so this part of the workflow is also an important validation step before training begins.
