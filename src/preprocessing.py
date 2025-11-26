# ------------------------------------------------------------------------------------------
# Loads processed simulation data, extracts parameters and spectra, normalizes them, and
# builds the final emulator dataset (params, log(d_ell), ell).
# Robert Pearce
# ------------------------------------------------------------------------------------------

import h5py
import numpy as np
from pathlib import Path


# =============================================================================
#                              CONSTANTS
# =============================================================================
H5_PATH = "../data/processed/proc_sims_v5.h5"
OUTPUT_PATH = "../data/processed/emulator_dataset_v1.npz"


# -----------------------------------------------------------------------------
#                              File Discovery
# -----------------------------------------------------------------------------
def get_sim_ids(h5_file):
    """
    Return sorted simulation IDs ['sim0', ... , 'simN']
    """
    sims_grp = h5_file["sims"]
    try:
        return sorted(sims_grp.keys(), key=lambda s: int(s.replace("sim", "")))
    except:
        return sorted(sims_grp.keys())


# -----------------------------------------------------------------------------
#                              File Discovery
# -----------------------------------------------------------------------------
def main():
    print(f"Loading: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        sim_ids = get_sim_ids(f)

        # Load ell grid from first simulation
        first = sim_ids[0]
        ell_ref = f["sims"][first]["cl"]["ell"][()]
        N_bins = len(ell_ref)
        # Print bin count
        print(f"Spectrum has {N_bins} bins")

        # Prepare containers for sim count, input parameters, and output d_ell
        N = len(sim_ids)
        X = np.zeros((N, 4), dtype=np.float64)
        Y = np.zeros((N, N_bins), dtype=np.float64)

        # Loop over sims and save params and d_ell
        for i, sid in enumerate(sim_ids):
            sim = f["sims"][sid]

            # Load parameters
            alpha = float(sim["params"]["alpha_zre"][()])
            kb = float(sim["params"]["kb_zre"][()])
            zmean = float(sim["params"]["zmean_zre"][()])
            b0 = float(sim["params"]["b0_zre"][()])

            X[i] = [zmean, alpha, kb, b0]

            # Load spectrum
            ell = sim["cl"]["ell"][()]
            dl = sim["cl"]["dl_ksz"][()]

            # Check ell matches ref
            if not np.array_equal(ell, ell_ref):
                raise RuntimeError(f"ell grids do not match for sim: {sid}")

            # Clip and take the log of d_ell
            clipped_dl = np.clip(dl, 1e-12, None)
            Y[i] = np.log(clipped_dl)

    # Save to npz dataset
    out = Path(OUTPUT_PATH)
    np.savez(OUTPUT_PATH, X = X, Y = Y, ell = ell_ref, sim_ids = np.array(sim_ids))

    print("Completed:")
    print("X Shape: ", X.shape)
    print("Y Shape: ", Y.shape)
    print("ell shape: ", ell_ref.shape)


if __name__ == "__main__":
    main()


#-----------------------------
#         END OF FILE
#-----------------------------
