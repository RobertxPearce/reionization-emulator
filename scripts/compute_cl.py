# ------------------------------------------------------------------------------------------
# Compute angular power spectrum (C_ell) from each kSZ map in the processed HDF5 file.
# Writes results back into the same file under /sims/sim<N>/cl.
# Input:  ksz_map is deltaT/T (dimensionless)
# Output: C_ell and D_ell (both in microkelvin^2)
# Author: Robert Pearce
# ------------------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import h5py

# =============================================================================
#                              CONSTANTS
# =============================================================================
SIM_DIR = Path(r"/Users/robertxpearce/Desktop/reionization-emulator/data/raw/sims_v5")
PROC_H5 = Path(r"/Users/robertxpearce/Desktop/reionization-emulator/data/processed/proc_sims_v5.h5")
NBINS = 50  # Number of ell bins for the azimuthal averaging


# -----------------------------------------------------------------------------
#                              FILE DISCOVERY
# -----------------------------------------------------------------------------
def find_files(sim_dir: Path):
    """
    Finds the OBS and PK files inside a given simulation directory (sim<N>).

    OBS file: contains "data/ksz_map"
    PK  file: contains "data/pk_tt"
    """
    obs_file, pk_file = None, None
    for fp in sorted(list(sim_dir.glob("*.h5")) + list(sim_dir.glob("*.hdf5"))):
        try:
            with h5py.File(fp, "r") as f:
                if obs_file is None and "data/ksz_map" in f:
                    obs_file = fp
                if pk_file is None and "data/pk_tt" in f:
                    pk_file = fp
        except OSError:
            continue
    return obs_file, pk_file


# -----------------------------------------------------------------------------
#                      READ Tcmb0, kSZ MAP, THETA_MAX
# -----------------------------------------------------------------------------
def read_tcmb0(obs_file: Path) -> float:
    """
    Read the CMB temperature (Tcmb0, in Kelvin) from the header.
    """
    with h5py.File(obs_file, "r") as f:
        tcmb0 = f["header/Tcmb0"][()]
    return float(np.asarray(tcmb0).squeeze())


def read_ksz_map(obs_file: Path) -> np.ndarray:
    """
    Read the kSZ map from the data group.
    """
    with h5py.File(obs_file, "r") as f:
        ksz = f["data/ksz_map"][()]
    return np.asarray(ksz)


def read_theta_max_rad(obs_file: Path) -> float:
    """
    Read theta_max_ksz (field-of-view size) in radians.
    """
    with h5py.File(obs_file, "r") as f:
        th = f["header/theta_max_ksz"][()]
    th = float(np.asarray(th).squeeze())
    return th


# -----------------------------------------------------------------------------
#                 CONVERT deltaT/T TO MICROKELVIN
# -----------------------------------------------------------------------------
def to_microkelvin(ksz_map_dt_over_t: np.ndarray, tcmb0_K: float) -> np.ndarray:
    """
    Convert deltaT/T to microkelvin (uK).
    Multiply by the CMB temperature (Tcmb0) and 1e6.
    """
    return (ksz_map_dt_over_t * tcmb0_K) * 1e6


# -----------------------------------------------------------------------------
#         COMPUTE FLAT-SKY ANGULAR POWER SPECTRUM (C_ell)
# -----------------------------------------------------------------------------
def compute_cl_flat_sky(map_uK: np.ndarray, theta_max_rad: float, nbins: int):
    """
    Compute the flat-sky angular power spectrum from a 2D temperature map.

    Inputs:
        map_uK        - 2D array (N x N) of temperature in microkelvin
        theta_max_rad - total angular size of the map in radians
        nbins         - number of ell bins for averaging

    Returns:
        ell_centers   - center of each ell bin
        cl            - angular power spectrum (C_ell) in microkelvin^2
        dcl           - simple mode-count uncertainty
        meta          - dictionary of info about this calculation
    """
    # Ensure map is square
    T = np.array(map_uK, dtype=np.float64, copy=True)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("ksz_map must be a square 2D array (N x N).")
    N = T.shape[0]

    # Remove the mean to avoid a spike at ell = 0
    T -= np.nanmean(T)
    T = np.nan_to_num(T, copy=False)

    # Compute pixel size and total area
    dtheta = theta_max_rad / N
    area = theta_max_rad * theta_max_rad

    # Compute FFT frequencies and convert to multipoles (ell = 2 * pi * f)
    fx = np.fft.fftfreq(N, d=dtheta)
    fy = np.fft.fftfreq(N, d=dtheta)
    lx = 2.0 * np.pi * fx
    ly = 2.0 * np.pi * fy
    Lx, Ly = np.meshgrid(lx, ly, indexing="xy")
    ell_2d = np.sqrt(Lx**2 + Ly**2)

    # Perform 2D FFT and normalize
    T_tilde = np.fft.fft2(T) * (dtheta**2)
    P2D = (T_tilde * np.conj(T_tilde)).real / area  # microK^2

    # Define ell range and bins
    ell_min = 2.0 * np.pi / theta_max_rad
    ell_max = ell_min * (N / 2.0)
    edges = np.linspace(ell_min, ell_max, nbins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    cl = np.empty(nbins, dtype=np.float64)
    dcl = np.empty(nbins, dtype=np.float64)
    counts = np.empty(nbins, dtype=np.int64)

    # Flatten arrays for binning
    flat_ell = ell_2d.ravel()
    flat_P = P2D.ravel()

    # Ignore the zero mode
    mask_nonzero = flat_ell > 0
    flat_ell = flat_ell[mask_nonzero]
    flat_P = flat_P[mask_nonzero]

    # Bin the power spectrum radially in ell-space
    inds = np.digitize(flat_ell, edges) - 1
    for i in range(nbins):
        sel = inds == i
        counts[i] = np.count_nonzero(sel)
        if counts[i] > 0:
            cl[i] = np.mean(flat_P[sel])
            dcl[i] = cl[i] / np.sqrt(counts[i])  # simple uncertainty estimate
        else:
            cl[i] = np.nan
            dcl[i] = np.nan

    # Save extra info
    meta = dict(
        N=int(N),
        theta_max_rad=float(theta_max_rad),
        dtheta_rad=float(dtheta),
        area_rad2=float(area),
        ell_min=float(ell_min),
        ell_max=float(ell_max),
        nbins=int(nbins),
        nmodes=counts,
        units_cl="microK^2",
        note="Flat-sky FFT estimator: C_ell = <|FFT(T)*dtheta^2|^2>/Area, T in microK."
    )

    return centers, cl, dcl, meta


# -----------------------------------------------------------------------------
#               COMPUTE D_ELL = ell*(ell+1)*C_ELL/(2*pi)
# -----------------------------------------------------------------------------
def compute_dl(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """
    Convert C_ell to D_ell.

    D_ell = ell * (ell + 1) * C_ell / (2 * pi)
    Units are microK^2 if C_ell is microK^2.
    """
    ell = np.asarray(ell, dtype=np.float64)
    cl = np.asarray(cl, dtype=np.float64)
    return ell * (ell + 1.0) * cl / (2.0 * np.pi)


# -----------------------------------------------------------------------------
#                                   MAIN
# -----------------------------------------------------------------------------
def main():
    # Resolve file paths
    root = SIM_DIR.expanduser().resolve()
    proc_path = PROC_H5.expanduser().resolve()

    # Check if paths exist
    if not root.exists():
        raise SystemExit(f"Input folder not found: {root}")
    if not proc_path.exists():
        raise SystemExit(f"Processed HDF5 not found: {proc_path}")

    # Find all subdirectories that start with "sim"
    sim_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("sim")])
    if not sim_dirs:
        raise SystemExit(f"No sim* directories found in {root}")

    with h5py.File(proc_path, "r+") as proc:
        if "sims" not in proc:
            raise SystemExit("Processed file is missing top-level group '/sims'")
        sims_grp = proc["sims"]

        # Loop over each simulation folder
        for sim_dir in sim_dirs:
            sim_name = sim_dir.name
            print(f"Processing {sim_name} ...")

            # Skip if simulation is not in processed file
            if sim_name not in sims_grp:
                print(f"[SKIP] {sim_name}: not present in processed file")
                continue

            cl_group_path = f"/sims/{sim_name}/cl"

            # Remove existing /cl group if it exists
            if cl_group_path in proc:
                del proc[cl_group_path]
                print(f"[REPLACE] {sim_name}: existing /cl group removed")

            # Find the original OBS file that contains the kSZ map
            obs_file, _ = find_files(sim_dir)
            if obs_file is None:
                print(f"[SKIP] {sim_name}: OBS file with data/ksz_map not found")
                continue

            # Read data from the OBS file
            tcmb0 = read_tcmb0(obs_file)
            theta_max_rad = read_theta_max_rad(obs_file)
            ksz = read_ksz_map(obs_file)

            # Convert deltaT/T to microkelvin
            map_uK = to_microkelvin(ksz, tcmb0)

            # Compute flat-sky angular power spectrum
            ell, cl_ksz, dcl, meta = compute_cl_flat_sky(map_uK, theta_max_rad, NBINS)

            # Compute D_ell from C_ell
            dl_ksz = compute_dl(ell, cl_ksz)

            # Create new group in HDF5 and store results
            grp = proc.create_group(cl_group_path)
            grp.create_dataset("ell", data=ell)
            grp.create_dataset("cl_ksz", data=cl_ksz)
            grp.create_dataset("dcl", data=dcl)
            grp.create_dataset("dl_ksz", data=dl_ksz)

            # Store metadata for reproducibility
            m = grp.create_group("metadata")
            m.attrs["Tcmb0_K"] = float(tcmb0)
            m.attrs["theta_max_rad"] = float(theta_max_rad)
            m.attrs["Npix"] = int(map_uK.shape[0])
            m.attrs["units_cl"] = "microK^2"
            m.attrs["units_dl"] = "microK^2"

            for k, v in meta.items():
                try:
                    m.attrs[k] = v
                except TypeError:
                    m.create_dataset(k, data=v)

            print(f"[OK] {sim_name}: wrote /cl (ell, cl_ksz, dcl, dl_ksz)")

    print("[OK] All done.")


if __name__ == "__main__":
    main()
