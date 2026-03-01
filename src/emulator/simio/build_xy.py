# ------------------------------------------------------------------------------------------
# Utilities to build machine learning ready arrays (X, Y, ell) from the condensed
# simulation HDF5 file.
#
# Reads:
#   /sims/sim<n>/params/{alpha_zre, b0_zre, kb_zre, zmean_zre}
#   /sims/sim<n>/cl/ell
#   /sims/sim<n>/cl/dl_ksz
#
# Writes:
#   /training/X             : Parameters per simulation
#   /training/Y             : Binned log(d_ell) per simulation
#   /training/ell           : Number of ell bins
#   /training/sim_ids       : Simulation ids
#   /training/param_names   : Parameter names used in dataset
#
# _read_scalar(): Read scalars from HDF5
# _apply_y_transform(): Apply optional transformation to target Y
# build_training_arrays(): Construct machine learning arrays from condensed HDF5
# write_training_to_h5(): Write training data to condensed HDF5 file
# build_and_write_training(): Build and write training data to HDF5 file
#
# Robert Pearce
# ------------------------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import h5py


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
    y_source: str = "dl_ksz"    # "dl_ksz" or "cl_ksz"
    y_transform: str = "ln"   # "none", "log10", "ln"
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


def build_training_arrays(h5_path: Path,
                          *,
                          config: BuildXYConfig = BuildXYConfig(),
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct machine learning arrays from condensed HDF5.
    
    return: X parameter matrix, Y target matrix, ell bin centers, sim_ids simulation ids, param_names in X order
    """
    # Expand and resolve file path
    h5_path = Path(h5_path).expanduser().resolve()
    
    # Initialize training arrays
    X_rows = []
    Y_rows = []
    sim_ids = []
    ell_ref = None
    
    # Open condensed HDF5 file in read mode
    with h5py.File(h5_path, "r") as f:
        # Set top-level path
        sims = f[config.sims_group]
        
        # Loop through sims in group
        for sim_name in sorted(sims.keys()):
            # Set path for current sim group
            sim_grp = sims[sim_name]
            
            # Ensure required groups exists
            if config.params_group not in sim_grp:
                continue
            if config.cl_group not in sim_grp:
                continue
                
            # Set path for params and cl
            params_grp = sim_grp[config.params_group]
            cl_grp = sim_grp[config.cl_group]
            
            # Construct the X row (parameters)
            try:
                x = np.array([_read_scalar(params_grp[p]) for p in config.param_names], dtype=np.float64)
            except KeyError:
                # Skip sims missing required parameter
                continue
            
            # Construct Y row (power spectrum)
            try:
                ell = np.asarray(cl_grp["ell"][()], dtype=np.float64)
                y_raw = np.asarray(cl_grp[config.y_source][()], dtype=np.float64)
            except KeyError:
                # Skip sims missing required parameters
                continue
                
            # Ensure consistent ell binning across all sims
            if ell_ref is None:
                ell_ref = ell
            else:
                if not np.allclose(ell, ell_ref, rtol=1e-12, atol=1e-12):
                    # Skip if binning is inconsistent
                    continue
            
            # Apply optional target transformation
            y = _apply_y_transform(y_raw, config.y_transform, config.eps)
            
            # Check for non-finite values
            if not np.all(np.isfinite(x)):
                continue
            if not np.all(np.isfinite(y)):
                continue
            
            # Append parameters to X row and c_ell to Y and sim id to list
            X_rows.append(x)
            Y_rows.append(y)
            sim_ids.append(sim_name)
            
    # Final stack into arrays
    if not X_rows:
        raise RuntimeError(f"No valid simulations found when building training arrays")
    
    X = np.vstack(X_rows)
    Y = np.vstack(Y_rows)
    
    sim_ids_out = np.array(sim_ids, dtype=object)
    param_names_out = np.array(config.param_names, dtype=object)
    
    return X, Y, ell_ref, sim_ids_out, param_names_out


def write_training_to_h5(h5_path: Path,
                         *,
                         X: np.ndarray,
                         Y: np.ndarray,
                         ell: np.ndarray,
                         sim_ids: Sequence[str],
                         param_names: Sequence[str],
                         config: BuildXYConfig,
                         overwrite: bool = True) -> int:
    """
    Write training data to condensed HDF5 file.
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
        training_grp.create_dataset("param_names", data=np.array(param_names, dtype=object))
        
        # Store metadata for reproducibility
        training_grp.attrs["y_source"] = config.y_source
        training_grp.attrs["y_transform"] = config.y_transform
        training_grp.attrs["eps"] = config.eps
        training_grp.attrs["number_of_samples"] = X.shape[0]
        training_grp.attrs["number_of_parameters"] = X.shape[1]
        training_grp.attrs["ell_bin_count"] = Y.shape[1]

    return X.shape[0]


def build_and_write_training(h5_path: Path,
                             *,
                             config: BuildXYConfig = BuildXYConfig(),
                             overwrite: bool = True) -> int:
    """
    Build and write training data to HDF5 file.
        1) Read simulations from sims
        2) Build X, Y, ell arrays
        3) Write them into condensed HDF5 /training
        
    return: Number of simulations included in training set
    """
    # Build the training arrays
    X, Y, ell_ref, sim_ids, param_names = build_training_arrays(h5_path, config=config)
    
    # Write training arrays to HDF5
    count = write_training_to_h5(h5_path, X=X, Y=Y, ell=ell_ref, sim_ids=sim_ids, param_names=param_names, config=config, overwrite=overwrite)
    
    # Return number of simulations
    return count

#-----------------------------
#         END OF FILE
#-----------------------------