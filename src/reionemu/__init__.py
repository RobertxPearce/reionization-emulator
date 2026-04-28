# -----------------------------------------------------------------------------
# Public API for the reionemu package.
# -----------------------------------------------------------------------------

from .artifact import (
    create_artifact_dir,
    dataset_summary,
    file_fingerprint,
    load_normalizers,
    read_json,
    save_artifact,
    save_configs,
    save_info,
    save_model_checkpoint,
    save_normalizers,
    save_results,
)
from .data.dataloaders import (
    DataLoaderConfig,
    load_training_arrays,
    make_dataloaders,
)
from .data.normalization import Normalizer
from .models.four_param_emulator import FourParamEmulator
from .models.mc_dropout_emulator import MCDropoutEmulator
from .simio.build_xy import (
    BuildStats,
    BuildXYConfig,
    build_and_write_training,
    build_training_arrays,
)
from .simio.compute_cl import ClConfig, add_cl_to_condensed_h5
from .simio.condense_h5 import CondenseConfig, CondenseStats, condense_sim_root
from .training.builders import (
    build_four_param_model,
    build_mc_dropout_model,
    build_optimizer,
)
from .training.kfold_cv import KFoldConfig, kfold_cross_validate
from .training.metrics import (
    mean_relative_error,
    mse,
    physical_mean_relative_error,
    rmse,
)
from .training.train_loop import (
    FitConfig,
    evaluate,
    evaluate_mc_metrics,
    evaluate_metrics,
    fit,
    train_one_epoch,
)

try:
    from .tuning import (
        default_param_space,
        run_tune_four_param,
        train_four_param_tune,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ray":
        raise

    def _missing_ray_tuning(*args, _exc=exc, **kwargs):
        raise ModuleNotFoundError(
            "Ray Tune is required for reionemu tuning utilities. "
            "Install it with `pip install 'ray[tune]>=2.0'` or "
            "`pip install -e '.[dev]'`."
        ) from _exc

    train_four_param_tune = _missing_ray_tuning
    default_param_space = _missing_ray_tuning
    run_tune_four_param = _missing_ray_tuning

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
    # artifact
    "create_artifact_dir",
    "dataset_summary",
    "file_fingerprint",
    "load_normalizers",
    "read_json",
    "save_artifact",
    "save_configs",
    "save_info",
    "save_model_checkpoint",
    "save_normalizers",
    "save_results",
    # models
    "FourParamEmulator",
    "MCDropoutEmulator",
    # training loops
    "train_one_epoch",
    "evaluate",
    "evaluate_mc_metrics",
    "evaluate_metrics",
    "fit",
    "FitConfig",
    # training metrics
    "mse",
    "rmse",
    "mean_relative_error",
    "physical_mean_relative_error",
    # builders
    "build_four_param_model",
    "build_mc_dropout_model",
    "build_optimizer",
    # cross-validation
    "kfold_cross_validate",
    "KFoldConfig",
    # ray tune
    "train_four_param_tune",
    "default_param_space",
    "run_tune_four_param",
]
