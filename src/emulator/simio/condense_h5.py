# ------------------------------------------------------------------------------------------
# Utilities for condensing simulation HDF5 outputs into smaller HDF5 files to prepare for
# angular spectrum calculation and NN training datasets.
#
# Pipeline:
#   raw per-sim HDF5 outputs -> one condensed HDF5 with consistent layout
#
# find_obs_and_pk_files(): Locate OBS Grids and PK Arrays files inside a sim directory
# _read_header_scalar(): Helper function for reading scalars from header dataset
# read_pk_fields(): Open the PK Array file and extract arrays and scalars
# read_obs_fields(): Open the OBS Grids file and extract arrays and scalars
# validate_payload(): Sanity check that one sim has all required fields
# write_sim(): Write one sim payload into the condensed output file
# condense_sim_root(): Orchestrate the full condense over sim<n>/ directories
#
# Robert Pearce
# ------------------------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import h5py


@dataclass(frozen=True)
class CondenseConfig:
    """
    Configuration for condensing simulation output.

    overwrite: Overwrite output file if it exists
    require_obs_and_pk: If True, skip sims missing either OBS or PK files
    """
    overwrite: bool = True
    require_obs_and_pk: bool = True
    
    
def find_obs_and_pk_files(sim_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate OBS Grids and PK Arrays files inside a sim directory.
    
    return: (obs_file, pk_file) paths from a single simulation directory
    """
    # Path for obs and pk files
    obs_file: Optional[Path] = None
    pk_file: Optional[Path] = None
    
    # Build list of files with extensions .h5 or .hdf5 sorted
    for fp in sorted(list(sim_dir.glob("*.h5")) + list(sim_dir.glob("*.hdf5"))):
        try:
            # Open files in read mode
            with h5py.File(fp, "r") as f:
                # Check if OBS file by seeing if it contains the kSZ map
                if obs_file is None and "data/ksz_map" in f:
                    obs_file = fp
                # Check if PK file by seeing if it contains pk_tt
                if pk_file is None and "data/pk_tt" in f:
                    pk_file = fp
        except OSError:
            # If file cant be opened, ignore and continue
            continue
    return obs_file, pk_file


def _read_header_scalar(group: h5py.Group, key: str, *, default: float = float("nan")) -> float:
    """
    Helper function for reading scalars from header dataset.
    
    group: HDF5 group that contains scalar datasets (f["header"])
    key: name of the dataset inside that group
    default: value to return if key is not in group
    
    returns: Scalar value or NaN if missing
    """
    # Check if name is not in header
    if key not in group:
        # If not return NaN
        return default
    # Read the scalar
    v = group[key][()]
    # Return the scalar (squeeze if array)
    return float(np.asarray(v).squeeze())


def read_pk_fields(pk_path: Path) -> Dict[str, object]:
    """
    Open the PK Array file and extract:
        Arrays: pk_tt, xmval_list, zval_list
        Scalars: alpha_zre, b0_zre, kb_zre, zmean_zre, tau
        
    pk_path: Path to the PK Arrays file
    
    return: Dict with arrays and scalars from the PK arrays file
    """
    with h5py.File(pk_path, "r") as f:
        pk_tt = f["data/pk_tt"][()]
        xm = f["data/xmval_list"][()]
        zz = f["data/zval_list"][()]
        
        # Scalar params/outputs live in header
        hdr = f["header"]
        
        return {
            "pk_tt": pk_tt,
            "xmval_list": xm,
            "zval_list": zz,
            "alpha_zre": _read_header_scalar(hdr, "alpha_zre"),
            "b0_zre": _read_header_scalar(hdr, "b0_zre"),
            "kb_zre": _read_header_scalar(hdr, "kb_zre"),
            "zmean_zre": _read_header_scalar(hdr, "zmean_zre"),
            "tau": _read_header_scalar(hdr, "tau"),
        }


def read_obs_fields(obs_path: Path) -> Dict[str, object]:
    """
    Open the OBS Grids file and extract:
        Arrays: ksz_map
        Scalars: Tcmb0, theta_max_ksz
        
    obs_path: Path to the OBS Grids file
    
    return: Dict with arrays and scalars from the PK arrays file
    """
    with h5py.File(obs_path, "r") as f:
        ksz = f["data/ksz_map"][()]
        Tcmb0 = _read_header_scalar(f["header"], "Tcmb0")
        theta_max = _read_header_scalar(f["header"], "theta_max_ksz")
        
        return {
            "ksz_map": ksz,
            "Tcmb0": Tcmb0,
            "theta_max_ksz": theta_max,
        }


def validate_payload(sim_name: str, payload: Dict[str, object]) -> None:
    """
    Validate the data for one simulation.
    
    sim_name: The name of the simulation
    payload: Dict of names and values
    
    raises: ValueError if required fields are missing or arrays are empty
    """
    required = [
        # Parameters
        "alpha_zre", "b0_zre", "kb_zre", "zmean_zre",
        # Scalar output
        "tau",
        # Array outputs
        "pk_tt", "xmval_list", "zval_list",
        # Map metadata
        "ksz_map", "Tcmb0", "theta_max_ksz",
    ]
    
    # Ensure all required keys exists
    for k in required:
        if k not in payload or payload[k] is None:
            raise ValueError(f"{sim_name}: missing required field '{k}'")
    # Ensure arrays are non-empty
    if np.asarray(payload["ksz_map"]).size == 0:
        raise ValueError(f"{sim_name}: kSZ map is empty")
    if np.asarray(payload["pk_tt"]).size == 0:
        raise ValueError(f"{sim_name}: pk_tt is empty")


def write_sim(sim_group: h5py.Group, payload: Dict[str, object]) -> None:
    """
    Write one simulation into the condensed HDF5 file under /sims/sim<n>/...
    Creates: params group and output group
    
    sim_group: h5py.Group to write
    payload: Dict of names and values
    
    return: None
    """
    gp = sim_group.create_group("params")
    go = sim_group.create_group("output")

    # Create dataset and write params
    gp.create_dataset("alpha_zre", data=float(payload["alpha_zre"]))
    gp.create_dataset("b0_zre", data=float(payload["b0_zre"]))
    gp.create_dataset("kb_zre", data=float(payload["kb_zre"]))
    gp.create_dataset("zmean_zre", data=float(payload["zmean_zre"]))

    # Create dataset and write outputs
    go.create_dataset("ksz_map", data=np.asarray(payload["ksz_map"]))
    go.create_dataset("Tcmb0", data=float(payload["Tcmb0"]))
    go.create_dataset("theta_max_ksz", data=float(payload["theta_max_ksz"]))

    go.create_dataset("pk_tt", data=np.asarray(payload["pk_tt"]))
    go.create_dataset("xmval_list", data=np.asarray(payload["xmval_list"]))
    go.create_dataset("zval_list", data=np.asarray(payload["zval_list"]))

    go.create_dataset("tau", data=float(payload["tau"]))


def condense_sim_root(sim_root: Path,
                      out_path: Path,
                      *,
                      config: CondenseConfig = CondenseConfig(),
                      sim_prefix: str = "sim",
                      file_description: str = "Condensed simulation outputs for reionization emulator.",
                      version: int = 1,
                      ) -> Dict[str, int]:
    """
    Condense all sim<n>/ directories under sim_root into one output HDF5 file.
    
    sim_root: Directory containing sim<n>/ subfolders
    out_path: Path to the condensed output .h5 file
    config: CondenseConfig (overwrite, require_obs_and_pk)
    sim_prefix: Subfolder name prefix to include (default "sim")
    file_description: Description of the output file
    version: Version of the output file
    
    returns: Dict of number of simulations written and skipped
    """
    # Expand and resolve to absolute paths
    sim_root = sim_root.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    
    # Check that input directory exists
    if not sim_root.exists():
        raise FileNotFoundError(f"{sim_root} does not exist")
    
    # Collect all sim directories (sim0, sim1, ... , sim<n>)
    sim_dirs = sorted([d for d in sim_root.iterdir() if d.is_dir() and d.name.startswith(sim_prefix)])
    
    if not sim_dirs:
        raise RuntimeError(f"No '{sim_prefix}*' directories found in: '{sim_root}'")
    
    # Prevent overwrite if overwrite=false
    if out_path.exists() and not config.overwrite:
        raise FileExistsError(f"Output exists and overwrite=False: '{out_path}'")
    
    # Counts for written and skipped sims
    written = 0
    skipped = 0
    
    # Create output HDF5
    with h5py.File(out_path, "w") as fout:
        # Write file metadata
        fout.attrs["source_root"] = str(sim_root)
        fout.attrs["description"] = file_description
        fout.attrs["version"] = int(version)
        
        # Top-level group containing all simulations
        sims_group = fout.create_group("sims")
        
        # Loop over each sim directory
        for sim_dir in sim_dirs:
            sim_name = sim_dir.name
            
            # Find which HDF5 files correspond to OBS and PK data
            obs_file, pk_file = find_obs_and_pk_files(sim_dir)
            
            # Check if both files are required
            if config.require_obs_and_pk and (obs_file is None or pk_file is None):
                skipped += 1
                continue
            
            # Try and read fields skip sim if files are incorrect or keys are missing
            try:
                pk = read_pk_fields(pk_file) if pk_file else {}
                obs = read_obs_fields(obs_file) if obs_file else {}
            except (OSError, KeyError):
                skipped += 1
                continue
            
            # Merge payloads into one dict
            payload = {**pk, **obs}
            
            # Validate required keys
            try:
                validate_payload(sim_name, payload)
            except ValueError:
                skipped += 1
                continue
            
            # Create /sims/sim<n> group and write
            g = sims_group.create_group(sim_name)
            write_sim(g, payload)
            written += 1

    return {"written": written, "skipped": skipped}

#-----------------------------
#         END OF FILE
#-----------------------------