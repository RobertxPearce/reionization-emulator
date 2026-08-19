"""
Configuration file for notebooks generating publication results/figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reionemu import (
    DataLoaderConfig,
    FitConfig,
    KFoldConfig,
)

# ------------------
# Device
# ------------------

device = "cpu"

# ------------------
# Experiment
# ------------------

SEED = 42

N_MC_PLOT = 201
N_MC_EVAL = 201
EPOCHS = 1000
EARLY_STOP = 150

PARAM_BOUNDS = np.array([
    [7.0, 9.0],
    [0.10, 0.90],
    [0.10, 2.0],
    [0.10, 0.80],
])

param_labels = [r"$\bar{z}$", r"$\alpha$", r"$k_b$", r"$b_0$"]
param_order = ["zmean_zre", "alpha_zre", "kb_zre", "b0_zre"]

# ------------------
# MCMC
# ------------------

N_DIM = 4
N_WALKERS = 32
N_STEPS = 30_000

DISCARD = 1000
THIN = 100

# ------------------
# Package Configs
# ------------------

SPLIT = {"train": 0.70, "val": 0.10, "test": 0.20}

dlcfg = DataLoaderConfig(
    batch_size=32,
    seed=SEED,
    shuffle_train=True,
    normalize_X=True,
    normalize_Y=False,
)

kf_fitcfg = FitConfig(
    epochs=EPOCHS,
    device=device,
    early_stopping_patience=EARLY_STOP,
    gradient_clipping=None,
)

kfcfg = KFoldConfig(
    k=5,
    seed=SEED,
    return_histories=True,
)

fitcfg = FitConfig(
    epochs=EPOCHS,
    device=device,
    early_stopping_patience=EARLY_STOP,
    gradient_clipping=None,
    seed=SEED,
)

# ------------------
# Paths
# ------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = REPO_ROOT / "datasets/raw/sims_v6"
H5_PATH = REPO_ROOT / "datasets/processed/publication_condensed_v6.h5"

EXPERIMENT_NAME = f"n_mc_{N_MC_PLOT}_epochs_{EPOCHS}_early_stop_{EARLY_STOP}"

# Large binaries
CKPT_DIR = REPO_ROOT / "checkpoints/pearce_2026_reionemu_mc_dropout" / f"mc_dropout_{EXPERIMENT_NAME}"
MODEL_PATH = CKPT_DIR / "model.pt"
NORM_PATH = CKPT_DIR / "norm"

# Tracked run outputs
RUN_DIR = REPO_ROOT / "results/pearce_2026_reionemu_mc_dropout" / EXPERIMENT_NAME
FIG_DIR = RUN_DIR
RECORDS_DIR = RUN_DIR / "records"

ARTIFACT_DIR = REPO_ROOT / "artifacts/pearce_2026_reionemu_mc_dropout"
ARTIFACT_NAME = f"mc_dropout_experiment_{EXPERIMENT_NAME}"
ARTIFACT_PATH = ARTIFACT_DIR / ARTIFACT_NAME

# ------------------
# Records
# ------------------

BUILD_INFO_PATH = RECORDS_DIR / "dataset_build.json"
HELDOUT_INDICES_PATH = RECORDS_DIR / "heldout_indices.npy"
TRAIN_RESULTS_PATH = RECORDS_DIR / "train_results.json"
KFOLD_SUMMARY_PATH = RECORDS_DIR / "kfold_summary.json"
HELDOUT_METRICS_PATH = RECORDS_DIR / "heldout_metrics.json"
HELDOUT_PREDICTIONS_PATH = RECORDS_DIR / "heldout_predictions.npz"
MCMC_SINGLE_PATH = CKPT_DIR / "mcmc_single.npz"


def make_dirs():
    """
    Create the output directories for this experiment.

    Kept out of module scope so that importing config has no filesystem side
    effects; notebooks that write figures call this once during setup.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------
# Plotting
# ------------------

plt.style.use("default")

plt.rcParams.update({
    "figure.figsize": (3.35, 2.5),
    "font.size": 9,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})
