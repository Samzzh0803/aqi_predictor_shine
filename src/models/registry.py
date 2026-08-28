"""Hopsworks-backed model registry for the AQI forecaster."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from tensorflow import keras

from src.data.open_meteo import OpenMeteoClientError

MODEL_NAME = "pearls_aqi_forecaster"
CERT_FOLDER = str(Path("data") / ".hopsworks_certs")
_MANIFEST_FILENAME = "extra_manifest.json"

_model_registry_cache: Any = None


@dataclass(frozen=True)
class RegisteredModelVersion:
    """Metadata for one registered Hopsworks model version."""

    model_name: str
    version: int
    model_type: str
    metrics: dict[str, float]
    feature_list: list[str]
    trained_at: str
    data_start: str
    data_end: str
    artifact_paths: dict[str, str]
    backend: str = "hopsworks_model_registry"


def register_model_version(
    *,
    model_name: str,
    model_type: str,
    fitted_models: dict[str, Any],
    metrics: dict[str, float],
    feature_list: list[str],
    data_start: str,
    data_end: str,
    shap_artifact_paths: dict[str, str],
) -> RegisteredModelVersion:
    """Persist one Hopsworks Model Registry version with metadata and serialized models."""

    mr = _get_model_registry()

    with tempfile.TemporaryDirectory() as tmp:
        version_dir = Path(tmp)
        artifact_filenames: dict[str, str] = {}
        for artifact_name, model in fitted_models.items():
            artifact_filenames.update(_dump_model_artifact(version_dir, artifact_name, model))
        for shap_name, shap_source_path in shap_artifact_paths.items():
            source = Path(shap_source_path)
            if not source.exists():
                continue
            destination_filename = f"shap_{shap_name}{source.suffix}"
            shutil.copyfile(source, version_dir / destination_filename)
            artifact_filenames[f"shap_{shap_name}"] = destination_filename

        numeric_metrics = {key: float(value) for key, value in metrics.items()}
        extra_metadata = {
            "model_type": model_type,
            "feature_list": feature_list,
            "data_start": data_start,
            "data_end": data_end,
            "artifact_filenames": artifact_filenames,
        }
        # Hopsworks' `description` column has a short length limit that the full
        # metadata (feature list + artifact filenames) can exceed. Ship the full
        # metadata as a small uploaded file instead, and keep `description` tiny.
        (version_dir / _MANIFEST_FILENAME).write_text(json.dumps(extra_metadata), encoding="utf-8")

        hw_model = mr.python.create_model(
            name=model_name,
            metrics=numeric_metrics,
            description=f"model_type={model_type}",
        )
        hw_model.save(str(version_dir))

    return RegisteredModelVersion(
        model_name=model_name,
        version=hw_model.version,
        model_type=model_type,
        metrics=numeric_metrics,
        feature_list=feature_list,
        trained_at=_epoch_ms_to_isoformat(hw_model.created),
        data_start=data_start,
        data_end=data_end,
        artifact_paths=artifact_filenames,
    )


def get_champion(model_name: str = MODEL_NAME) -> RegisteredModelVersion:
    """Return the registered version with the best validation mean MAE."""

    versions = list_registered_versions(model_name=model_name)
    if not versions:
        raise OpenMeteoClientError(f"No registered models found for {model_name}")
    return min(
        versions,
        key=lambda item: (
            float(item.metrics.get("selection_mae_mean", item.metrics["mae_mean"])),
            float(item.metrics.get("selection_mae_std", 0.0)),
            float(item.metrics["mae_mean"]),
            int(item.version),
        ),
    )


def list_registered_versions(model_name: str = MODEL_NAME) -> list[RegisteredModelVersion]:
    """Load all registered Hopsworks versions for a model name."""

    mr = _get_model_registry()
    try:
        hw_models = mr.get_models(model_name)
    except Exception:  # noqa: BLE001 - Hopsworks raises its own exception types for "not found"
        return []

    versions: list[RegisteredModelVersion] = []
    for hw_model in hw_models:
        extra_metadata = _load_manifest(hw_model)
        versions.append(
            RegisteredModelVersion(
                model_name=model_name,
                version=hw_model.version,
                model_type=extra_metadata.get("model_type", "unknown"),
                metrics=dict(hw_model.training_metrics or {}),
                feature_list=extra_metadata.get("feature_list", []),
                trained_at=_epoch_ms_to_isoformat(hw_model.created),
                data_start=extra_metadata.get("data_start", ""),
                data_end=extra_metadata.get("data_end", ""),
                artifact_paths=extra_metadata.get("artifact_filenames", {}),
            )
        )
    return versions


def _load_manifest(hw_model: Any) -> dict[str, Any]:
    download_dir = Path(hw_model.download())
    manifest_path = download_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_registered_models(version: RegisteredModelVersion) -> dict[str, Any]:
    """Download and load serialized model artifacts for one registered version."""

    mr = _get_model_registry()
    hw_model = mr.get_model(version.model_name, version=version.version)
    download_dir = Path(hw_model.download())

    loaded: dict[str, Any] = {}
    for artifact_name, filename in version.artifact_paths.items():
        if artifact_name.startswith("shap_"):
            continue
        path = download_dir / filename
        if not path.exists():
            raise OpenMeteoClientError(f"Registered model artifact is missing after download: {path}")
        if path.suffix == ".keras":
            scaler_path = download_dir / f"{artifact_name}_scaler.joblib"
            if not scaler_path.exists():
                raise OpenMeteoClientError(
                    f"TensorFlow registry artifact is missing its scaler file: {scaler_path}"
                )
            loaded[artifact_name] = {
                "model": keras.models.load_model(path),
                "scaler": joblib.load(scaler_path),
            }
            continue
        if filename.endswith("_scaler.joblib"):
            continue
        loaded[artifact_name] = joblib.load(path)
    return loaded


def _dump_model_artifact(version_dir: Path, artifact_name: str, model: Any) -> dict[str, str]:
    if isinstance(model, dict) and {"model", "scaler"}.issubset(model.keys()):
        keras_filename = f"{artifact_name}.keras"
        scaler_filename = f"{artifact_name}_scaler.joblib"
        keras_model = model["model"]
        if not isinstance(keras_model, keras.Model):
            raise OpenMeteoClientError("TensorFlow registry artifact is missing a keras.Model instance")
        keras_model.save(version_dir / keras_filename)
        joblib.dump(model["scaler"], version_dir / scaler_filename)
        return {artifact_name: keras_filename}

    filename = f"{artifact_name}.joblib"
    joblib.dump(model, version_dir / filename)
    return {artifact_name: filename}


def _epoch_ms_to_isoformat(epoch_ms: int | None) -> str:
    if epoch_ms is None:
        return datetime.now(UTC).isoformat()
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).isoformat()


def _get_model_registry() -> Any:
    """Log in to Hopsworks (once per process) and return the project's model registry."""

    global _model_registry_cache
    if _model_registry_cache is not None:
        return _model_registry_cache

    from dotenv import load_dotenv

    load_dotenv()

    import hopsworks
    import hopsworks_common.connection as hw_connection

    def _patched_provide_project(self: Any, name: str | None = None) -> None:
        from hopsworks_common import client

        _client = client._get_instance()
        if name:
            self._project = name
            if _client._is_external():
                _client._provide_project(name)
        if _client._project_name:
            self._project = _client._project_name
        if not self._project:
            return

        from hsfs import engine

        engine._get_instance()
        if self._variable_api._get_data_science_profile_enabled():
            try:
                self._model_serving_api._load_default_configuration()
            except Exception:  # noqa: BLE001 - free-tier keys lack SERVING scope; this is expected
                pass

    hw_connection.Connection._provide_project = _patched_provide_project

    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    if not api_key or not project_name:
        raise OpenMeteoClientError(
            "HOPSWORKS_API_KEY and HOPSWORKS_PROJECT must be set to use the model registry"
        )

    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name,
        cert_folder=CERT_FOLDER,
    )
    _model_registry_cache = project.get_model_registry()
    return _model_registry_cache
