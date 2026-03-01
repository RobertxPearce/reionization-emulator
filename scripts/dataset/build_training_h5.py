# ------------------------------------------------------------------------------------------
# Script to condense simulation output into single HDF5 file with training dataset for
# emulator.
#
# Robert Pearce
# ------------------------------------------------------------------------------------------

from pathlib import Path

from emulator.simio.condense_h5 import condense_sim_root, CondenseConfig
from emulator.simio.compute_cl import add_cl_to_condensed_h5, ClConfig
from emulator.simio.build_xy import build_and_write_training, BuildXYConfig


def main():
    raw_sim_root = Path(r"/Users/robertxpearce/Desktop/reionization-emulator/data/raw/sims_v6")
    condensed_h5 = Path(r"/Users/robertxpearce/Desktop/reionization-emulator/data/processed/condensed_v6.h5")

    print("1) Condensing raw simulation outputs")
    stats = condense_sim_root(sim_root=raw_sim_root, out_path=condensed_h5, config=CondenseConfig(overwrite=True, require_obs_and_pk=True))
    print("Done condensing raw simulation outputs")
    
    print("2) Computing C_ell / D_ell and writing into /cl")
    n_updated = add_cl_to_condensed_h5(condensed_h5, config=ClConfig(nbins=5, ell_cut=1000.0, overwrite=True, sims_group="sims"))
    print(f"Done computing C_ell / D_ell and updated: {n_updated} sims")
    
    print("3) Building and writing training data")
    n_train = build_and_write_training(condensed_h5, config=BuildXYConfig(sims_group="sims", params_group="params", cl_group="cl", param_names=("zmean_zre", "alpha_zre", "kb_zre", "b0_zre"), y_source="dl_ksz", y_transform="ln", eps=1e-30))
    print(f"Done building and writing training data updated: {n_updated} sims")
    
    print("Finished building condensed dataset with /cl and /training")
    print(f"Output: {condensed_h5}")
    
if __name__ == "__main__":
    main()

#-----------------------------
#         END OF FILE
#-----------------------------