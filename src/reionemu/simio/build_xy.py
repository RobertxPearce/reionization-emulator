# -----------------------------------------------------------------------------
# Utilities to build machine learning ready arrays (X, Y, ell) from the
# condensed simulation HDF5 file.
#
# Reads:
#   /sims/sim<n>/params/{alpha_zre, b0_zre, kb_zre, zmean_zre}
#   /sims/sim<n>/cl/ell
#   /sims/sim<n>/cl/dl_ksz
#
# Writes:
#   /training/X             : Parameters per simulation
#   /training/Y             : Binned log(d_ell) per simulation
#   /training/ell           : Ell bin centers
#   /training/sim_ids       : Simulation ids
#   /training/param_names   : Parameter names used in dataset
#
# _read_scalar(): Read scalars from HDF5
# _apply_y_transform(): Apply optional transformation to target Y
# build_training_arrays(): Construct machine learning arrays from condensed
# HDF5
# write_training_to_h5(): Write training data to condensed HDF5 file
# build_and_write_training(): Build and write training data to HDF5 file
#
# Robert Pearce
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class BuildStats:
    """
    Counts of processed and skipped simulations when building training arrays.
    """

    total_sims: int
    processed: int
    skipped_missing_params: int
    skipped_missing_cl: int
    skipped_inconsistent_ell: int
    skipped_non_finite: int

    @property
    def skipped_total(self) -> int:
        return (
            self.skipped_missing_params
            + self.skipped_missing_cl
            + self.skipped_inconsistent_ell
            + self.skipped_non_finite
        )


@dataclass(frozen=True)
class BuildXYConfig:
    """
    Configuration for building the training dataset.

    sims_group: Name of top-level group containing simulations
    params_group: Subgroup under each simulation containing parameter scalars
    cl_group: Subgroup under each simulation containing power spectra data
    param_names: Ordered tuple of parameter names to construct X matrix
    y_source: Which power spectrum product to use as target
    y_transform: Optional transformation applied to the power spectrum
    eps: Small value added before logarithm to avoid log(0)
    """

    sims_group: str = "sims"
    params_group: str = "params"
    cl_group: str = "cl"
    param_names: Tuple[str, ...] = ("zmean_zre", "alpha_zre", "kb_zre", "b0_zre")
    y_source: str = "dl_ksz"  # "dl_ksz" or "cl_ksz"
    y_transform: str = "ln"  # "none", "log10", "ln"
    eps: float = 1e-30


def _read_scalar(ds) -> float:
    """
    Read scalar from HDF5 file.

    return: Scalar read from HDF5 file
    """
    return float(np.asarray(ds[()]).squeeze())


def _apply_y_transform(y: np.ndarray, mode: str, eps: float) -> np.ndarray:
    """
    Apply optional transformation to target Y.

    returns: Y with no transformation, log10 or ln
    """
    if mode == "none":
        return y
    if mode == "log10":
        return np.log10(np.maximum(y, 0.0) + eps)
    if mode == "ln":
        return np.log(np.maximum(y, 0.0) + eps)
    raise ValueError(f"Unknown y_transform='{mode}'")


def build_training_arrays(
    h5_path: Path,
    *,
    config: BuildXYConfig = BuildXYConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, BuildStats]:
    """
    Construct machine learning arrays from condensed HDF5.

    return: X parameter matrix
            Y target matrix
            ell bin centers
            sim_ids
            param_names
            BuildStats
    """
    # Expand and resolve file path
    h5_path = Path(h5_path).expanduser().resolve()

    # Initialize training arrays and skip counters
    X_rows = []
    Y_rows = []
    sim_ids = []
    ell_ref = None
    n_missing_params = 0
    n_missing_cl = 0
    n_inconsistent_ell = 0
    n_non_finite = 0
    with h5py.File(h5_path, "r") as f:
        sims = f[config.sims_group]
        sim_names = sorted(sims.keys())
        total_sims = len(sim_names)

        for sim_name in sim_names:
            sim_grp = sims[sim_name]

            if config.params_group not in sim_grp:
                n_missing_params += 1
                continue
            if config.cl_group not in sim_grp:
                n_missing_cl += 1
                continue

            params_grp = sim_grp[config.params_group]
            cl_grp = sim_grp[config.cl_group]

            try:
                x = np.array(
                    [_read_scalar(params_grp[p]) for p in config.param_names],
                    dtype=np.float64,
                )
            except KeyError:
                n_missing_params += 1
                continue

            try:
                ell = np.asarray(cl_grp["ell"][()], dtype=np.float64)
                y_raw = np.asarray(cl_grp[config.y_source][()], dtype=np.float64)
            except KeyError:
                n_missing_cl += 1
                continue

            if ell.ndim != 1 or y_raw.ndim != 1:
                n_missing_cl += 1
                continue
            if len(ell) != len(y_raw):
                n_missing_cl += 1
                continue

            if ell_ref is None:
                ell_ref = ell.copy()
            else:
                if not np.allclose(ell, ell_ref, rtol=1e-12, atol=1e-12):
                    n_inconsistent_ell += 1
                    continue

            y = _apply_y_transform(y_raw, config.y_transform, config.eps)

            if not np.all(np.isfinite(x)):
                n_non_finite += 1
                continue
            if not np.all(np.isfinite(y)):
                n_non_finite += 1
                continue

            X_rows.append(x)
            Y_rows.append(y)
            sim_ids.append(sim_name)

    stats = BuildStats(
        total_sims=total_sims,
        processed=len(X_rows),
        skipped_missing_params=n_missing_params,
        skipped_missing_cl=n_missing_cl,
        skipped_inconsistent_ell=n_inconsistent_ell,
        skipped_non_finite=n_non_finite,
    )

    if not X_rows:
        raise RuntimeError(
            f"No valid simulations found when building training arrays. Stats: {stats}"
        )

    X = np.vstack(X_rows).astype(np.float32)
    Y = np.vstack(Y_rows).astype(np.float32)
    sim_ids_out = np.array(sim_ids, dtype=object)
    param_names_out = np.array(config.param_names, dtype=object)

    return X, Y, ell_ref, sim_ids_out, param_names_out, stats


def write_training_to_h5(
    h5_path: Path,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    ell: np.ndarray,
    sim_ids: Sequence[str],
    param_names: Sequence[str],
    config: BuildXYConfig,
    overwrite: bool = True,
    stats: Optional[BuildStats] = None,
) -> int:
    """
    Write training data to condensed HDF5 file.

    stats: If provided, store build statistics as training group attributes.
    """
    h5_path = Path(h5_path).expanduser().resolve()

    with h5py.File(h5_path, "r+") as f:
        if "training" in f:
            if overwrite:
                del f["training"]
            else:
                raise FileExistsError(f"Training data already exists in {h5_path}")

        training_grp = f.create_group("training")

        # Store core arrays
        training_grp.create_dataset("X", data=X)
        training_grp.create_dataset("Y", data=Y)
        training_grp.create_dataset("ell", data=ell)

        # Store string arrays
        training_grp.create_dataset("sim_ids", data=np.array(sim_ids, dtype=object))
        training_grp.create_dataset(
            "param_names", data=np.array(param_names, dtype=object)
        )

        # Store metadata for reproducibility
        training_grp.attrs["y_source"] = config.y_source
        training_grp.attrs["y_transform"] = config.y_transform
        training_grp.attrs["eps"] = config.eps
        training_grp.attrs["number_of_samples"] = X.shape[0]
        training_grp.attrs["number_of_parameters"] = X.shape[1]
        training_grp.attrs["ell_bin_count"] = Y.shape[1]

        if stats is not None:
            training_grp.attrs["build_total_sims"] = stats.total_sims
            training_grp.attrs["build_processed"] = stats.processed
            training_grp.attrs["build_skipped_missing_params"] = (
                stats.skipped_missing_params
            )
            training_grp.attrs["build_skipped_missing_cl"] = stats.skipped_missing_cl
            training_grp.attrs["build_skipped_inconsistent_ell"] = (
                stats.skipped_inconsistent_ell
            )
            training_grp.attrs["build_skipped_non_finite"] = stats.skipped_non_finite

    return X.shape[0]


def build_and_write_training(
    h5_path: Path, *, config: BuildXYConfig = BuildXYConfig(), overwrite: bool = True
) -> int:
    """
    Build and write training data to HDF5 file.
        1) Read simulations from sims
        2) Build X, Y, ell arrays (with skip stats)
        3) Write them into condensed HDF5 /training

    return: Number of simulations included in training set
    """
    X, Y, ell_ref, sim_ids, param_names, stats = build_training_arrays(
        h5_path, config=config
    )

    count = write_training_to_h5(
        h5_path,
        X=X,
        Y=Y,
        ell=ell_ref,
        sim_ids=sim_ids,
        param_names=param_names,
        config=config,
        overwrite=overwrite,
        stats=stats,
    )

    return count


# -----------------------------
#         END OF FILE
# -----------------------------
