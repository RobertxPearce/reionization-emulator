# ------------------------------------------------------------------------------------------
# Utilities to compute angular power spectrum (flat-sky) from kSZ map and write into
# condensed HDF5.
#
# Write:
#   /sims/sim<n>/cl/ell      : ell bin centers
#   /sims/sim<n>/cl/cl_ksz   : Binned C_ell values
#   /sims/sim<n>/cl/dl_ksz   : Binned D_ell values
#   /sims/sim<n>/cl/dcl      : Uncertainty estimate per bin
#
# Assumptions:
#   Flat-sky approximation
#   kSZ map is square 2D (NxN)
#
# to_microkelvin(): Convert deltaT/T to microkelvin (uK)
# compute_cl_flat_sky(): Compute the flat-sky angular power spectrum from a kSZ map
# compute_dl(): Convert C_ell to D_ell (D_ell = ell * (ell + 1) * C_ell / (2 * pi))
# add_cl_to_condensed_h5(): Compute and write C_ell and D_ell for every sim
#
# Robert Pearce
# ------------------------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import h5py


@dataclass(frozen=True)
class ClConfig:
    """
    Configuration for power spectrum computation.
    
    nbins: Number of ell bins to output
    ell_cut: Minimal ell retained, full-res bins with centers < ell_cut are discarded
    overwrite: If true, overwrite existing /cl group
    sims_group: Name of the top-level sims group in the HDF5 file
    """
    nbins: int = 5
    ell_cut: float = 1000.0
    overwrite: bool = True
    sims_group: str = "sims"


def to_microkelvin(ksz_map_dt_over_t: np.ndarray, tcmb0_K: float) -> np.ndarray:
    """
    Convert deltaT/T to microkelvin (uK).
    map_uK = (deltaT/T * Tcmb0) *1e6
    
    ksz_map_dt_over_t: kSZ map
    tcmb0_K: CMB temperature
    
    return: kSZ map converted to microkelvin (uK)
    """
    return (ksz_map_dt_over_t * tcmb0_K) * 1e6


def compute_cl_flat_sky(map_uK: np.ndarray,
                        theta_max_rad: float,
                        nbins: int,
                        ell_cut: float,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the flat-sky angular power spectrum from a kSZ map.
    
    Steps:
        1) Compute full-resolution spectrum with nbins_full = N//2 bins
        2) Drop bins with ell < ell_cut
        3) Rebin remaining high-ell bins into nbins coarse bins
        4) Correct for Hann window
    
    return: centers, cl, dcl
    """
    # --------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------
    map_uK = np.asarray(map_uK, dtype=np.float64)
    if map_uK.ndim != 2:
        raise ValueError(
            f"kSZ map must be 2D, got ndim={map_uK.ndim}. "
            "Expected shape (N, N) for a square map."
        )
    if map_uK.shape[0] != map_uK.shape[1]:
        raise ValueError(
            f"kSZ map must be square (N, N), got shape {map_uK.shape}. "
            "Flat-sky power spectrum expects a square map."
        )
    if theta_max_rad <= 0:
        raise ValueError(
            f"theta_max_rad must be positive, got {theta_max_rad}."
        )

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
    # --------------------------------------------------------------

    # Perform 2D FFT and normalize
    T_tilde = np.fft.fft2(T) * (dtheta**2)          # Compute the 2D FFT and normalize
    P2D = (T_tilde * np.conj(T_tilde)).real / area  # Get power for each angular mode and normalize to uK^2
    
    # --------------------------------------------------------------
    # Define the ell Range and Full-Resolution Bins
    #
    # The min and max will set the largest (min) and smallest (max)
    # structure the map can fit.
    # ell_min features that can be seen (the whole map)
    # ell_max tiniest features we can see (pixels size)
    # --------------------------------------------------------------

    # Define ell range and bins
    ell_min = 2.0 * np.pi / theta_max_rad   # Compute smallest ell (large-scale features)
    ell_max = ell_min * (N / 2.0)           # Compute largest ell (small-scale structures)

    nbins_full = N // 2                     # Full-resolution bins

    # --------------------------------------------------------------
    # Create the full-res bin edges and centers
    # --------------------------------------------------------------

    edges = np.linspace(ell_min, ell_max, nbins_full + 1)  # Evenly spaced boundaries from ell_min to ell_max
    centers = 0.5 * (edges[1:] + edges[:-1])               # Midpoint of each full-res bin
    
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
    # Initialize cl, dcl, counts
    # --------------------------------------------------------------

    cl = np.empty(nbins_full, dtype=np.float64)      # Full-res averaged C_ell in each bin
    dcl = np.empty(nbins_full, dtype=np.float64)     # Full-res uncertainties
    counts = np.empty(nbins_full, dtype=np.int64)    # How many modes per full-res bin
    
    # --------------------------------------------------------------
    # Full resolution binning: N/2 ell-bins
    # --------------------------------------------------------------

    inds = np.digitize(flat_ell, edges) - 1     # Use digitize to look at every ell and determine which bin it belongs too

    for i in range(nbins_full):                 # Loop through each full-res ell bin
        sel = inds == i                         # Select only the Fourier modes belonging to bin i
        counts[i] = np.count_nonzero(sel)       # Count how many pixels fell into bin i
        if counts[i] > 0:
            cl[i] = np.mean(flat_P[sel])        # Average power in this full-res bin
            dcl[i] = cl[i] / np.sqrt(counts[i]) # Standard error of the mean
        else:
            cl[i] = np.nan                      # If no modes fell into bin put NaN
            dcl[i] = np.nan
            
    # --------------------------------------------------------------
    # Cut at ell_cut and rebin into nbins coarse bins
    #   1) Keep only centers >= ell_cut
    #   2) Re-aggregate into nbins bins over that high-ell range
    # --------------------------------------------------------------

    # Keep only high-ell part of the full-resolution spectrum
    mask_high = centers >= ell_cut      # Boolean array containing if each entry is >= ell_cut
    ell_high = centers[mask_high]       # Keep only ell bin centers >= ell_cut
    cl_high = cl[mask_high]             # Keep only the cl for ell bin centers >= ell_cut
    dcl_high = dcl[mask_high]           # Keep only the dl for ell bin centers >= ell_cut
    counts_high = counts[mask_high]     # Keep the number of Fourier modes contributing to each high ell bin

    # If nbins >= number of available high-ell bins, just return the high-res part
    if nbins >= len(ell_high):
        centers = ell_high
        cl = cl_high
        dcl = dcl_high
    else:
        # New coarse bin edges and centers over the high-ell range
        edges_coarse = np.linspace(ell_high[0], ell_high[-1], nbins + 1)
        centers_coarse = 0.5 * (edges_coarse[1:] + edges_coarse[:-1])

        cl_coarse = np.empty(nbins, dtype=np.float64)
        dcl_coarse = np.empty(nbins, dtype=np.float64)
        counts_coarse = np.empty(nbins, dtype=np.int64)

        # Which coarse bin each high-ell bin belongs to
        inds_coarse = np.digitize(ell_high, edges_coarse) - 1

        for j in range(nbins):
            sel2 = inds_coarse == j
            # Total number of Fourier modes contributing to this coarse bin
            w = counts_high[sel2]
            counts_coarse[j] = w.sum()
            if counts_coarse[j] > 0:
                # Weighted average of cl_high using counts as weights
                cl_coarse[j] = np.average(cl_high[sel2], weights=w)
                dcl_coarse[j] = cl_coarse[j] / np.sqrt(counts_coarse[j])
            else:
                cl_coarse[j] = np.nan
                dcl_coarse[j] = np.nan

        # Save to centers, cl and dcl
        centers = centers_coarse
        cl = cl_coarse
        dcl = dcl_coarse

    # --------------------------------------------------------------
    # Correct for the Window Function
    #
    # The Hann window will reduce the overall power. Dividing by
    # the normalization factor will undo this reduction in power.
    # --------------------------------------------------------------

    cl = cl / window_norm
    dcl = dcl / window_norm

    return centers, cl, dcl


def compute_dl(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """
    Convert C_ell to D_ell: D_ell = ell * (ell + 1) * C_ell / (2 * pi).
    Units are microK^2 if C_ell is microK^2.
    
    return: dl
    """
    ell = np.asarray(ell, dtype=np.float64)
    cl = np.asarray(cl, dtype=np.float64)
    return ell * (ell + 1.0) * cl / (2.0 * np.pi)


def add_cl_to_condensed_h5(h5_path: Path,
                           *,
                           config: ClConfig = ClConfig(),
                           progress_callback: Optional[Callable[[int, int], None]] = None,
                           ) -> int:
    """
    Compute and write /cl for every sim in a condensed HDF5 file.

    For each sim:
        read:
            /sims/sim<n>/output/ksz_map
            /sims/sim<n>/output/Tcmb0
            /sims/sim<n>/output/theta_max_ksz
        writes:
            /sims/sim<n>/cl/ell
            /sims/sim<n>/cl/cl_ksz
            /sims/sim<n>/cl/dcl
            /sims/sim<n>/cl/dl_ksz

    progress_callback: Optional callable(completed, total) for progress bars

    return: Number of simulations updated
    """
    h5_path = Path(h5_path).expanduser().resolve()

    with h5py.File(h5_path, "r+") as proc:
        if config.sims_group not in proc:
            raise KeyError(
                f"Missing top-level group '{config.sims_group}' in {h5_path}"
            )

        sims_grp = proc[config.sims_group]
        sim_names = sorted(sims_grp.keys())
        total = len(sim_names)
        updated = 0

        for i, sim_name in enumerate(sim_names):
            sim_grp = sims_grp[sim_name]
            out_grp = sim_grp["output"]

            ksz = np.asarray(out_grp["ksz_map"][()])
            tcmb0 = float(np.asarray(out_grp["Tcmb0"][()]).squeeze())
            theta_max_ksz = float(np.asarray(out_grp["theta_max_ksz"][()]).squeeze())

            # Validate kSZ map shape before compute
            if ksz.ndim != 2:
                raise ValueError(
                    f"Sim '{sim_name}': ksz_map must be 2D, got ndim={ksz.ndim}. "
                    "Expected (N, N) for flat-sky power spectrum."
                )
            if ksz.shape[0] != ksz.shape[1]:
                raise ValueError(
                    f"Sim '{sim_name}': ksz_map must be square, got shape {ksz.shape}."
                )

            cl_group_path = f"/{config.sims_group}/{sim_name}/cl"
            if cl_group_path in proc:
                if config.overwrite:
                    del proc[cl_group_path]
                else:
                    if progress_callback is not None:
                        progress_callback(i + 1, total)
                    continue

            map_uK = to_microkelvin(ksz, tcmb0)
            ell, cl_ksz, dcl = compute_cl_flat_sky(
                map_uK, theta_max_ksz, config.nbins, config.ell_cut
            )
            dl_ksz = compute_dl(ell, cl_ksz)

            # Sanity check output shapes
            if len(ell) != len(cl_ksz) or len(cl_ksz) != len(dl_ksz):
                raise ValueError(
                    f"Sim '{sim_name}': inconsistent CL output lengths "
                    f"ell={len(ell)}, cl_ksz={len(cl_ksz)}, dl_ksz={len(dl_ksz)}"
                )

            grp = proc.create_group(cl_group_path)
            grp.create_dataset("ell", data=ell)
            grp.create_dataset("cl_ksz", data=cl_ksz)
            grp.create_dataset("dcl", data=dcl)
            grp.create_dataset("dl_ksz", data=dl_ksz)

            updated += 1

            if progress_callback is not None:
                progress_callback(i + 1, total)

    return updated

#-----------------------------
#         END OF FILE
#-----------------------------