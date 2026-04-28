# -----------------------------------------------------------------------------
# Utilities for saving experiment artifacts.
#
# The artifact directory keeps JSON metadata/configs next to binary sidecars
# such as PyTorch checkpoints and NumPy normalizer files.
#
# Robert Pearce
# -----------------------------------------------------------------------------

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from ..data.normalization import Normalizer

SCHEMA_VERSION = 1


def _utc_timestamp() -> str:
    """
    Return an ISO-8601 UTC timestamp for artifact metadata.
    """
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """
    Convert common scientific Python objects into JSON-safe values.
    """
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """
    Write a JSON file with stable indentation.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=4, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    """
    Load one artifact JSON file.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def file_fingerprint(path: Path) -> dict[str, Any]:
    """
    Return basic file identity metadata.
    """
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "file_size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def dataset_summary(h5_path: Path) -> dict[str, Any]:
    """
    Summarize the condensed HDF5 dataset used by an experiment.
    """
    h5_path = Path(h5_path).expanduser().resolve()
    summary: dict[str, Any] = {
        "path": str(h5_path),
        "fingerprint": file_fingerprint(h5_path),
    }

    with h5py.File(h5_path, "r") as f:
        if "training" not in f:
            return summary

        training = f["training"]
        if "X" in training:
            summary["n_samples"] = int(training["X"].shape[0])
            summary["n_parameters"] = int(training["X"].shape[1])
        if "Y" in training:
            summary["n_targets"] = int(training["Y"].shape[1])
        if "param_names" in training:
            param_names = training["param_names"][()]
            summary["param_names"] = [
                p.decode("utf-8") if isinstance(p, bytes) else str(p)
                for p in param_names
            ]
        if "ell" in training:
            summary["ell"] = np.asarray(training["ell"][()]).tolist()

        for key in ("y_source", "y_transform", "eps"):
            if key in training.attrs:
                summary[key] = _json_safe(training.attrs[key])

    return summary


def create_artifact_dir(
    name: str,
    root_dir: Path,
) -> Path:
    """
    Create and return the directory for one experiment artifact.
    """
    artifact_dir = Path(root_dir).expanduser().resolve() / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def save_configs(
    artifact_dir: Path,
    *,
    condense_config: Any = None,
    cl_config: Any = None,
    build_config: Any = None,
    dataloader_config: Any = None,
    fit_config: Any = None,
    kfold_config: Any = None,
    model_config: Mapping[str, Any] | None = None,
    optimizer_config: Mapping[str, Any] | None = None,
    tuning_config: Mapping[str, Any] | None = None,
) -> Path:
    """
    Save experiment configuration choices to configs.json.
    """
    path = Path(artifact_dir) / "configs.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_prep": {
            "condense": condense_config,
            "cl": cl_config,
            "build_xy": build_config,
        },
        "data_loading": {
            "dataloader": dataloader_config,
        },
        "model": model_config,
        "optimizer": optimizer_config,
        "training": fit_config,
        "kfold": kfold_config,
        "tuning": tuning_config,
    }
    _write_json(path, payload)
    return path


def save_results(
    artifact_dir: Path,
    *,
    summary: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    history: Mapping[str, Any] | None = None,
    dataset_prep_stats: Mapping[str, Any] | None = None,
    status: str = "completed",
) -> Path:
    """
    Save experiment outcomes to results.json.
    """
    path = Path(artifact_dir) / "results.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_at": _utc_timestamp(),
        "summary": summary or {},
        "metrics": metrics or {},
        "history": history or {},
        "dataset_prep_stats": dataset_prep_stats or {},
    }
    _write_json(path, payload)
    return path


def save_normalizers(
    artifact_dir: Path,
    normalizers: Mapping[str, Normalizer | None],
    *,
    filename: str = "normalizers.npz",
) -> Path | None:
    """
    Save Normalizer objects to a NumPy sidecar file.
    """
    arrays = {}
    for name, normalizer in normalizers.items():
        if normalizer is None:
            continue
        arrays[f"{name}_mean"] = np.asarray(normalizer.mean)
        arrays[f"{name}_std"] = np.asarray(normalizer.std)

    if not arrays:
        return None

    path = Path(artifact_dir) / filename
    np.savez(path, **arrays)
    return path


def load_normalizers(path: Path) -> dict[str, Normalizer]:
    """
    Load normalizers saved by save_normalizers().
    """
    normalizers: dict[str, Normalizer] = {}
    with np.load(Path(path)) as data:
        names = sorted(k.removesuffix("_mean") for k in data if k.endswith("_mean"))
        for name in names:
            normalizers[name] = Normalizer(
                mean=np.asarray(data[f"{name}_mean"]),
                std=np.asarray(data[f"{name}_std"]),
            )
    return normalizers


def save_model_checkpoint(
    artifact_dir: Path,
    checkpoint: Any,
    *,
    filename: str = "model.pt",
) -> Path:
    """
    Save a PyTorch model, state_dict, or checkpoint dictionary.
    """
    path = Path(artifact_dir) / filename
    torch.save(checkpoint, path)
    return path


def save_info(
    artifact_dir: Path,
    *,
    run_id: str,
    experiment_name: str | None = None,
    description: str | None = None,
    dataset_path: Path | None = None,
    artifacts: Mapping[str, str | None] | None = None,
) -> Path:
    """
    Save the top-level artifact manifest to info.json.
    """
    path = Path(artifact_dir) / "info.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_name": experiment_name or run_id,
        "description": description,
        "created_at": _utc_timestamp(),
        "dataset": dataset_summary(dataset_path) if dataset_path is not None else None,
        "artifacts": {k: v for k, v in (artifacts or {}).items() if v is not None},
    }
    _write_json(path, payload)
    return path


def save_artifact(
    name: str,
    root_dir: Path,
    *,
    dataset_path: Path | None = None,
    condense_config: Any = None,
    cl_config: Any = None,
    build_config: Any = None,
    dataloader_config: Any = None,
    fit_config: Any = None,
    kfold_config: Any = None,
    model_config: Mapping[str, Any] | None = None,
    optimizer_config: Mapping[str, Any] | None = None,
    tuning_config: Mapping[str, Any] | None = None,
    results_summary: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    history: Mapping[str, Any] | None = None,
    dataset_prep_stats: Mapping[str, Any] | None = None,
    normalizers: Mapping[str, Normalizer | None] | None = None,
    checkpoint: Any = None,
    description: str | None = None,
) -> Path:
    """
    Save a complete experiment artifact directory.
    """
    artifact_dir = create_artifact_dir(name, root_dir)

    config_path = save_configs(
        artifact_dir,
        condense_config=condense_config,
        cl_config=cl_config,
        build_config=build_config,
        dataloader_config=dataloader_config,
        fit_config=fit_config,
        kfold_config=kfold_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        tuning_config=tuning_config,
    )
    result_path = save_results(
        artifact_dir,
        summary=results_summary,
        metrics=metrics,
        history=history,
        dataset_prep_stats=dataset_prep_stats,
    )

    normalizer_path = None
    if normalizers is not None:
        normalizer_path = save_normalizers(artifact_dir, normalizers)

    checkpoint_path = None
    if checkpoint is not None:
        checkpoint_path = save_model_checkpoint(artifact_dir, checkpoint)

    save_info(
        artifact_dir,
        run_id=name,
        experiment_name=name,
        description=description,
        dataset_path=dataset_path,
        artifacts={
            "configs": config_path.name,
            "results": result_path.name,
            "normalizers": normalizer_path.name if normalizer_path else None,
            "model_checkpoint": checkpoint_path.name if checkpoint_path else None,
        },
    )

    return artifact_dir


# -----------------------------
#         END OF FILE
# -----------------------------
