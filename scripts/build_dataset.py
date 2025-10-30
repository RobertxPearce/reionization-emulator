#-------------------------------------------------------------------------------------------
# Build data set for emulator. Extract ksz_map, pk_tt, xmval_list, zval_list, alpha_zre, 
# kb_zre, and zmean_zre into HDF5
# Robert Pearce
#-------------------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import h5py

# ============================================================================
#                              Constant Paths
# ============================================================================
# Path to parent sim directory
SIM_DIR = Path(r"/Users/robertxpearce/Desktop/reionization-emulator/data/raw/sims_v5")
# Output File
OUTPUT_FILE = Path(r"/Users/robertxpearce/Desktop/reionization-emulator/data/processed/proc_sims_v5.h5")


# ----------------------------------------------------------------------------
#                               File discovery
# ----------------------------------------------------------------------------
def find_files(sim_dir: Path):
    """
    Return (obs_file, pk_file) within single simulation directory (sim<n>)
    """
    obs_file = None # Path to OBS Grids file
    pk_file = None  # Path to PK Arrays file

    # Build list of files with extensions .h5 or .hdf5 sorted
    for fp in sorted(list(sim_dir.glob("*.h5")) + list(sim_dir.glob("*.hdf5"))):
        try:
            # Open file in read mode
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


# ----------------------------------------------------------------------------
#                               Readers
# ----------------------------------------------------------------------------
def _read_header_scalar(hdr, name: str):
    """
    Read scalar from header dataset; return NaN if missing.
    """
    # Check if name is in header
    if name in hdr:
        # Read the data
        v = hdr[name][()]
        # Return the data (squeeze if array)
        return np.asarray(v).squeeze()
    # Return nan if missing
    return np.nan


def read_pk_fields(path: Path):
    """
    Open a PK Array file and extract:
        Arrays: pk_tt, xmval_list, zval_list
        Scalars: alpha_zre, b0_zre, kb_zre, zmean_zre, tau
    """
    # Open the PK HDF5 file in read mode
    with h5py.File(path, "r") as f:
        # Extract the arrays from the data group (preserve shape)
        pk_tt = f["data/pk_tt"][()]
        xm    = f["data/xmval_list"][()]
        zz    = f["data/zval_list"][()]
        hdr   = f["header"]
        # Reference the header group and read the params
        alpha = _read_header_scalar(hdr, "alpha_zre")
        b0    = _read_header_scalar(hdr, "b0_zre")
        kb    = _read_header_scalar(hdr, "kb_zre")
        zmean = _read_header_scalar(hdr, "zmean_zre")
        tau   = _read_header_scalar(hdr, "tau")
    # Return a dict with arrays and scalar parameters
    return dict(
        pk_tt=pk_tt,
        xmval_list=xm,
        zval_list=zz,
        alpha_zre=alpha,
        b0_zre=b0,
        kb_zre=kb,
        zmean_zre=zmean,
        tau=tau,
    )


def read_obs_fields(path: Path):
    """
    From OBS GRIDS:
        Array: ksz_map
    """
    with h5py.File(path, "r") as f:
        ksz = f["data/ksz_map"][()]
    return dict(ksz_map=ksz)


# ----------------------------------------------------------------------------
#                                   Writer
# ----------------------------------------------------------------------------
def write_sim_group(g: h5py.Group, payload: dict):
    """
    Write one simulation into group g
    """
    # Arrays
    g.create_dataset("ksz_map",    data=payload["ksz_map"],    compression="gzip", shuffle=True)
    g.create_dataset("pk_tt",      data=payload["pk_tt"],      compression="gzip", shuffle=True)
    g.create_dataset("xmval_list", data=payload["xmval_list"], compression="gzip", shuffle=True)
    g.create_dataset("zval_list",  data=payload["zval_list"],  compression="gzip", shuffle=True)
    # Scalars
    g.create_dataset("alpha_zre",  data=float(payload["alpha_zre"]))
    g.create_dataset("b0_zre",     data=float(payload["b0_zre"]))
    g.create_dataset("kb_zre",     data=float(payload["kb_zre"]))
    g.create_dataset("zmean_zre",  data=float(payload["zmean_zre"]))
    g.create_dataset("tau",        data=float(payload["tau"]))


# ----------------------------------------------------------------------------
#                               Validation
# ----------------------------------------------------------------------------
def validate_all_fields(sim_name: str, payload: dict):
    """
    Ensure all required fields exist and arrays are non-empty
    """
    # List of keys needed
    required = [
        "ksz_map", "pk_tt", "xmval_list", "zval_list",
        "alpha_zre", "b0_zre", "kb_zre", "zmean_zre", "tau",
    ]
    # Loops over each required key
    for key in required:
        if key not in payload or payload[key] is None:
            raise ValueError(f"{sim_name}: missing required field '{key}'.")
    if hasattr(payload["pk_tt"], "size") and payload["pk_tt"].size == 0:
        raise ValueError(f"{sim_name}: pk_tt is empty.")
    if hasattr(payload["ksz_map"], "size") and payload["ksz_map"].size == 0:
        raise ValueError(f"{sim_name}: ksz_map is empty.")


# ----------------------------------------------------------------------------
#                                   Main
# ----------------------------------------------------------------------------
def main():
    # Resolve to absolute paths
    root = SIM_DIR.expanduser().resolve()
    out_path = OUTPUT_FILE.expanduser().resolve()

    # Check if input directory exists
    if not root.exists():
        raise SystemExit(f"Input folder not found: {root}")
    # Check if output directory exists
    if not out_path.parent.exists():
        raise SystemExit(f"Output folder does not exist: {out_path.parent}")

    # Find simulation subdirectories with names that start with "sim"
    sim_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("sim")])
    # If none found fail
    if not sim_dirs:
        raise SystemExit(f"No sim* directories found in {root}")

    # Counters for reporting how many sims were written or skipped
    num_skipped = 0
    num_written = 0

    # Open the HDF5 output file
    with h5py.File(out_path, "w") as fout:
        # Record metadata
        fout.attrs["source_root"] = str(root)
        fout.attrs["description"] = "Processed ksz_2lpt simulation output for reionization emulator."
        fout.attrs["version"] = 1

        # Create top-level group to hold all simulation subgroups
        sims = fout.create_group("sims")

        # Loop over each simulation directory (sim0, sim1, ...)
        for sim_dir in sim_dirs:
            # Sim name ("sim0")
            sim_name = sim_dir.name

            # Locate the OBS Grids and PK Arrays files inside the sim folder
            obs_file, pk_file = find_files(sim_dir)

            # If either is missing, warn adn skip the simulation
            if pk_file is None or obs_file is None:
                print(f"[WARN] {sim_name}: missing obs or pk file; skipping.")
                num_skipped += 1
                continue

            try:
                # Read data from PK Arrays (arrays + scalars) and OBS Grids (array)
                pk_payload  = read_pk_fields(pk_file)
                obs_payload = read_obs_fields(obs_file)
            except KeyError as e:
                # If any data is missing skip the sim
                print(f"[WARN] {sim_name}: missing expected dataset {e}; skipping.")
                num_skipped += 1
                continue

            #
            payload = {
                **pk_payload,
                "ksz_map": obs_payload["ksz_map"],
                "xmval_list": pk_payload["xmval_list"],
                "zval_list":  pk_payload["zval_list"],
            }

            try:
                # Run basic test before writing
                validate_all_fields(sim_name, payload)
            except ValueError as e:
                # If it fails warn and skip sim
                print(f"[WARN] {e}")
                num_skipped += 1
                continue

            # Create the subgroup and write the dataset inside
            g = sims.create_group(sim_name)
            write_sim_group(g, payload)
            num_written += 1

    # Print summary once closed
    print(f"[OK] Wrote {out_path} with {num_written} simulations. Skipped: {num_skipped}")


if __name__ == "__main__":
    main()

#-----------------------------
#         END OF FILE
#-----------------------------


