# -----------------------------------------------------------------------------
# Script to condense simulation output into single HDF5 file with training dataset for
# reionemu.
#
# Robert Pearce
# -----------------------------------------------------------------------------

from pathlib import Path

from reionemu.simio import (
    BuildXYConfig,
    ClConfig,
    CondenseConfig,
    add_cl_to_condensed_h5,
    build_and_write_training,
    condense_sim_root,
)


def _progress_print(step: str):
    """
    Return a progress_callback that prints step name and percentage.
    """

    def callback(completed: int, total: int):
        pct = 100.0 * completed / total if total else 0
        print(f"\r{step} {completed}/{total} ({pct:.1f}%)", end="", flush=True)

    return callback


def main():
    raw_sim_root = Path(
        r"/Users/robertxpearce/Desktop/reionization-emulator/data/raw/sims_v6"
    )
    condensed_h5 = Path(
        r"/Users/robertxpearce/Desktop/reionization-emulator/data/processed/TEST.h5"
    )

    print("1) Condensing raw simulation outputs")
    stats = condense_sim_root(
        sim_root=raw_sim_root,
        out_path=condensed_h5,
        config=CondenseConfig(overwrite=True, require_obs_and_pk=True),
        progress_callback=_progress_print("Condensing"),
    )
    print()
    print(f"   Written: {stats.written}, skipped: {stats.skipped_total}")

    print("2) Computing C_ell / D_ell and writing into /cl")
    n_updated = add_cl_to_condensed_h5(
        condensed_h5,
        config=ClConfig(nbins=5, ell_cut=1000.0, overwrite=True, sims_group="sims"),
        progress_callback=_progress_print("CL computation"),
    )
    print()
    print(f"   Updated: {n_updated} sims")

    print("3) Building and writing training data")
    n_train = build_and_write_training(
        condensed_h5,
        config=BuildXYConfig(
            sims_group="sims",
            params_group="params",
            cl_group="cl",
            param_names=("zmean_zre", "alpha_zre", "kb_zre", "b0_zre"),
            y_source="dl_ksz",
            y_transform="ln",
            eps=1e-30,
        ),
    )
    print(f"   Training samples: {n_train}")

    print("Finished building condensed dataset with /cl and /training")
    print(f"Output: {condensed_h5}")


if __name__ == "__main__":
    main()

# -----------------------------
#         END OF FILE
# -----------------------------
