# ------------------------------------------------------------------------------------------
# Compute angular power spectrum (C_ell) from each kSZ map in the processed HDF5 file.
# Writes results back into the same file under /sims/sim<n>/cl.
# Robert Pearce
# ------------------------------------------------------------------------------------------

import h5py
from pathlib import Path
import numpy as np
from powerbox.tools import get_power


SIM_DIR = Path(r"/Users/robertxpearce/Desktop/reionization-emulator/data/raw/sims_v5")
PROC_H5 = Path(r"/Users/robertxpearce/Desktop/reionization-emulator/data/processed/proc_sims_v5.h5")
NBINS = 15  # Number of ell bins for averaging


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


def to_microkelvin(ksz_map_dt_over_t: np.ndarray, tcmb0_K: float) -> np.ndarray:
    """
    Convert deltaT/T to microkelvin (uK).
    Multiply by the CMB temperature (Tcmb0) and 1e6.
    """
    return (ksz_map_dt_over_t * tcmb0_K) * 1e6


def compute_cl_flat_sky(map_uK: np.ndarray, theta_max_rad: float, nbins: int):
    """
    Compute the flat-sky angular power spectrum from a 2D temperature map.
    """

    # --------------------------------------------------------------
    # Input
    # --------------------------------------------------------------

    # Make a float64 copy of the map and read its size
    T = np.array(map_uK, dtype=np.float64, copy=True)
    N = T.shape[0]

    # --------------------------------------------------------------
    # Apply Windowing
    #
    # The kSZ map has "sharp" edges and so when Fourier Transform
    # is applied there will be a large variance as beyond the image
    # the value will be 0.
    #
    # The Hann window function will reduce the overall power. A
    # normalization factor is computed to undo the loss.
    # --------------------------------------------------------------

    # Apply a 2D window function before FFT
    window = np.hanning(N)
    window_2d = np.outer(window, window)
    T *= window_2d

    # Normalize factor for the window
    window_norm = np.sum(window_2d**2) / (N**2)

    # --------------------------------------------------------------
    # Remove the Map Mean and NaNs
    #
    # The mean is (ell = 0) and can cause a spike at ell = 0 and
    # when averaging the ell = 0 will be in the smaller ell bins.
    # --------------------------------------------------------------

    # Remove the mean to avoid a spike at ell = 0
    T -= np.nanmean(T)
    # Replace any NaNs with zeros
    T = np.nan_to_num(T, copy=False)

    # --------------------------------------------------------------
    # Angular Pixel Size and Area
    #
    # Fourier transforms measure frequency per unit distance.
    # Since the kSZ map is in angles the pixels and area must be
    # converted to radians so the FFT gives frequencies per radian.
    # --------------------------------------------------------------

    # Compute the pixel size in radians per pixel
    dtheta = theta_max_rad / N
    # Compute the total patch area
    area = theta_max_rad * theta_max_rad

    # --------------------------------------------------------------
    # Build the Fourier (multipole) Grid
    #
    # --------------------------------------------------------------

    # Compute Fast Fourier Transform frequencies and convert to multipoles (ell = 2 * pi * f)
    fx = np.fft.fftfreq(N, d=dtheta)    # Returns frequency values corresponding to the x-axis
    fy = np.fft.fftfreq(N, d=dtheta)    # Returns frequency values corresponding to the y-axis

    # Convert from frequency to multipole
    lx = 2.0 * np.pi * fx   # l = 2*pi*fx
    ly = 2.0 * np.pi * fy   # l = 2*pi*fy

    # Combine into a 2D grid
    Lx, Ly = np.meshgrid(lx, ly, indexing="xy") # 2D arrays representing lx and ly

    ell_2d = np.sqrt(Lx**2 + Ly**2)             # Compute the total angular wave number

    # --------------------------------------------------------------
    # FFT to Raw 2D Power
    #
    # Numpys fft2() computes a discrete sum. The FFT needs to be
    # multiplied by dtheta^2 to approximate the integral. Without
    # this the power spectrum would have the wrong amplitude since
    # each pixel will be assumed to have an area of 1.
    #
    #
    # --------------------------------------------------------------

    # Perform 2D FFT and normalize
    T_tilde = np.fft.fft2(T) * (dtheta**2)          # Compute the 2D FFT and normalize

    P2D = (T_tilde * np.conj(T_tilde)).real / area  # Get power for each angular mode and normalize to uK^2

    # --------------------------------------------------------------
    # Define the ell Range and Bins
    #
    # The min and max will set the largest (min) and smallest (max)
    # structure the map can fit.
    # ell_min features that can be seen (the whole map)
    # ell_max tiniest features we can see (pixels size)
    # --------------------------------------------------------------

    # Define ell range and bins
    ell_min = 2.0 * np.pi / theta_max_rad               # Compute smallest ell (large-scale features)
    ell_max = ell_min * (N / 2.0)                       # Compute largest ell (small-scale structures)

    # --------------------------------------------------------------
    # Create the bin edges and centers
    #
    # --------------------------------------------------------------

    edges = np.linspace(ell_min, ell_max, nbins + 1)    # Evenly spaced boundaries from ell_min to ell_max
    centers = 0.5 * (edges[1:] + edges[:-1])            # Midpoint of each bin

    # --------------------------------------------------------------
    # Create the bin edges and centers
    #
    # cl[i] is the average pawer (Cl) in bin i
    # dcl[i] the estimated uncertainty for that bin
    # counts[i] how many Fourier pixels (modes) landed in that bin
    # --------------------------------------------------------------

    cl = np.empty(nbins, dtype=np.float64)      # Initialize array for the averaged c_ell in each bin
    dcl = np.empty(nbins, dtype=np.float64)     # Initialize array for the uncertainty for each c_ell
    counts = np.empty(nbins, dtype=np.int64)    # Initialize array for how many pixels fell into each bin

    # --------------------------------------------------------------
    # Flatten
    #
    # ell_2d contains the ell value for every pixel
    # P2D contains the corresponding power value for that pixel
    # Flattening them creates two 1D arrays that represent the ell
    # and power for a specific pixel at i.
    # --------------------------------------------------------------

    # Flatten arrays for binning
    flat_ell = ell_2d.ravel()
    flat_P = P2D.ravel()

    # --------------------------------------------------------------
    # Drop the Zero Mode
    #
    # Create a mask that only keeps elements greater than 0. This
    # will ensure only meaningful modes that represent actual
    # fluctuations in the map are present.
    # --------------------------------------------------------------

    # Ignore the zero mode
    mask_nonzero = flat_ell > 0
    flat_ell = flat_ell[mask_nonzero]
    flat_P = flat_P[mask_nonzero]

    # --------------------------------------------------------------
    # Bin the Power Spectrum in ell-Space
    #
    # --------------------------------------------------------------

    # Bin the power spectrum radially in ell-space
    inds = np.digitize(flat_ell, edges) - 1     # Use digitize to look at every ell and determine which bin it belongs to
    for i in range(nbins):                      # Loop through each ell bin
        sel = inds == i                         # Select only the Fourier modes belonging to bin i
        counts[i] = np.count_nonzero(sel)       # Count how many pixels fell into bin i
        if counts[i] > 0:
            cl[i] = np.mean(flat_P[sel])        # Select all power values whos ell fall in that bin and compute their average
            dcl[i] = cl[i] / np.sqrt(counts[i]) # Compute the standard error of the mean
        else:
            cl[i] = np.nan                      # If no modes fell into bin put NaN
            dcl[i] = np.nan                     # If no modes fell into bin put NaN

    # --------------------------------------------------------------
    # Correct for the Window Function
    #
    # The Hann window will reduce the overall power. Dividing by
    # the normalization factor will undo this reduction in power.
    # --------------------------------------------------------------

    cl = cl / window_norm
    dcl = dcl / window_norm

    return centers, cl, dcl


def compute_cl_powerbox(map_uK: np.ndarray, theta_max_rad: float, nbins: int):
    """
    Flat-sky C_ell using powerbox.get_power for a 2D temperature map (uK).
    """
    # Get the map size
    N = map_uK.shape[0]

    # Compute the angular size per pixel
    dtheta = theta_max_rad / N  # radians per pixel

    # Call Powerbox get_power() function to compute the power spectrum
    Pk, k = get_power(
        map_uK,                     # Input map
        boxlength=theta_max_rad,    # The total size of the map in radians
        bins=nbins,                 # Number of bins
        log_bins=False,             # Use linearly spaced bins
        bin_ave=True,               # Report the average k in each bin instead of bin edges
        ignore_zero_mode=True,      # Exclude the DC component
        bins_upto_boxlen=True       # Ensure bins span up to the box smallest dimension
    )

    # Powerbox output k is approximately equal to ell
    ell = k.copy()
    # Copy the power spectrum
    cl  = Pk.copy()
    # Initialize the uncertainty array with NaNs to match manual calculation format
    dcl = np.full_like(cl, np.nan)

    return ell, cl, dcl


def compute_dl(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """
    Convert C_ell to D_ell.

    D_ell = ell * (ell + 1) * C_ell / (2 * pi)
    Units are microK^2 if C_ell is microK^2.
    """
    ell = np.asarray(ell, dtype=np.float64)
    cl = np.asarray(cl, dtype=np.float64)
    return ell * (ell + 1.0) * cl / (2.0 * np.pi)


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

        # Choose to use manual or powerbox computation
        # choice = input("Manual Calculation (1) or PowerBox (2): ")
        # if choice != "1" and choice != "2":
        #     raise SystemExit(f"Invalid choice {choice}")

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

            # if choice == "1":
            #     # Compute flat-sky angular power spectrum using manual calculations
            #     ell, cl_ksz, dcl = compute_cl_flat_sky(map_uK, theta_max_rad, NBINS)
            # elif choice == "2":
            #     # Compute flat-sky angular power spectrum using powerbox
            #     ell, cl_ksz, dcl = compute_cl_powerbox(map_uK, theta_max_rad, NBINS)

            # Compute flat-sky angular power spectrum using manual calculations
            ell, cl_ksz, dcl = compute_cl_flat_sky(map_uK, theta_max_rad, NBINS)

            # Compute flat-sky angular power spectrum using powerbox
            # ell, cl_ksz, dcl = compute_cl_powerbox(map_uK, theta_max_rad, NBINS)

            # Compute D_ell from C_ell
            dl_ksz = compute_dl(ell, cl_ksz)

            # Create new group in HDF5 and store results
            grp = proc.create_group(cl_group_path)
            grp.create_dataset("ell", data=ell)
            grp.create_dataset("cl_ksz", data=cl_ksz)
            grp.create_dataset("dcl", data=dcl)
            grp.create_dataset("dl_ksz", data=dl_ksz)

            print(f"[OK] {sim_name}: wrote /cl (ell, cl_ksz, dcl, dl_ksz)")

    print("[OK] All done.")


if __name__ == "__main__":
    main()

#-----------------------------
#         END OF FILE
#-----------------------------
