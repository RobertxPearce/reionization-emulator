# ------------------------------------------------------------------------------------------
# Public API for the reionemu package.
# ------------------------------------------------------------------------------------------

from .simio.condense_h5 import condense_sim_root, CondenseConfig, CondenseStats
from .simio.compute_cl import add_cl_to_condensed_h5, ClConfig
from .simio.build_xy import (
    build_and_write_training,
    build_training_arrays,
    BuildXYConfig,
    BuildStats,
)

from .data.dataloaders import (
    make_dataloaders,
    load_training_arrays,
    DataLoaderConfig,
)
from .data.normalization import Normalizer

from .models.four_param_emulator import FourParamEmulator
from .models.three_param_emulator import ThreeParamEmulator

from .training.train_loop import fit, FitConfig
from .training.kfold_cv import kfold_cross_validate, KFoldConfig

__all__ = [
    # simio
    "condense_sim_root",
    "CondenseConfig",
    "CondenseStats",
    "add_cl_to_condensed_h5",
    "ClConfig",
    "build_and_write_training",
    "build_training_arrays",
    "BuildXYConfig",
    "BuildStats",
    # data
    "make_dataloaders",
    "load_training_arrays",
    "DataLoaderConfig",
    "Normalizer",
    # models
    "FourParamEmulator",
    "ThreeParamEmulator",
    # training
    "fit",
    "FitConfig",
    "kfold_cross_validate",
    "KFoldConfig",
]