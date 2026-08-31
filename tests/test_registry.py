"""Tests for the Hopsworks-backed model registry, using a fake Hopsworks client."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models import registry as registry_module
from src.models.registry import (
    get_champion,
    list_registered_versions,
    load_registered_models,
    register_model_version,
)


class _FakeHwModel:
    def __init__(self, name: str, version: int, metrics: dict[str, float], description: str) -> None:
        self.name = name
        self.version = version
        self.training_metrics = metrics
        self.description = description
        self.created = 1735689600000
        self._saved_dir: Path | None = None

    def save(self, model_dir: str) -> None:
        self._saved_dir = Path(model_dir)
        # Hopsworks moves files out of the source dir; snapshot contents before the
        # real TemporaryDirectory context manager cleans it up.
        self._files = {path.name: path.read_bytes() for path in self._saved_dir.iterdir()}

    def download(self) -> str:
        import tempfile

        download_dir = Path(tempfile.mkdtemp())
        for name, content in self._files.items():
            (download_dir / name).write_bytes(content)
        return str(download_dir)

    def delete(self) -> None:
        pass


class _FakePythonNamespace:
    def __init__(self, models: dict[str, list[_FakeHwModel]]) -> None:
        self._models = models

    def create_model(self, name: str, metrics: dict[str, float], description: str) -> _FakeHwModel:
        version = len(self._models.get(name, [])) + 1
        model = _FakeHwModel(name, version, metrics, description)
        self._models.setdefault(name, []).append(model)
        return model


class FakeModelRegistry:
    """In-memory stand-in for a Hopsworks ModelRegistry, scoped to one test."""

    def __init__(self) -> None:
        self._models: dict[str, list[_FakeHwModel]] = {}
        self.python = _FakePythonNamespace(self._models)

    def get_models(self, name: str) -> list[_FakeHwModel]:
        return list(self._models.get(name, []))

    def get_model(self, name: str, version: int) -> _FakeHwModel:
        for model in self._models.get(name, []):
            if model.version == version:
                return model
        raise KeyError(f"No model {name} version {version}")


@pytest.fixture()
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> FakeModelRegistry:
    registry = FakeModelRegistry()
    monkeypatch.setattr(registry_module, "_get_model_registry", lambda: registry)
    return registry


def test_register_model_version_uses_validation_metric_for_champion(fake_registry: FakeModelRegistry) -> None:
    register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="ridge",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 9.0, "selection_mae_mean": 8.0, "selection_mae_std": 0.3},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={},
    )
    champion = register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="hist_gradient_boosting",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 9.5, "selection_mae_mean": 7.5, "selection_mae_std": 0.2},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={},
    )

    versions = list_registered_versions("pearls_aqi_forecaster")

    assert len(versions) == 2
    assert get_champion("pearls_aqi_forecaster").version == champion.version
    assert get_champion("pearls_aqi_forecaster").model_type == "hist_gradient_boosting"


def test_register_model_version_serializes_tensorflow_bundle(fake_registry: FakeModelRegistry) -> None:
    from src.pipelines.train import _build_mlp_model

    scaler = StandardScaler().fit(np.array([[0.0], [1.0], [2.0]]))
    model = _build_mlp_model(input_dim=1)

    version = register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="tensorflow_mlp",
        fitted_models={"all_horizons": {"model": model, "scaler": scaler}},
        metrics={"mae_mean": 9.0, "selection_mae_mean": 8.0, "selection_mae_std": 0.3},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={},
    )

    loaded = load_registered_models(version)

    assert "all_horizons" in loaded
    assert "model" in loaded["all_horizons"]
    assert "scaler" in loaded["all_horizons"]


def test_register_model_version_attaches_shap_artifacts(
    tmp_path: Path,
    fake_registry: FakeModelRegistry,
) -> None:
    shap_csv = tmp_path / "importance.csv"
    shap_csv.write_text("feature,mean_abs_shap\nus_aqi,1.0\n", encoding="utf-8")

    version = register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="ridge",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 9.0, "selection_mae_mean": 8.0},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={"target_aqi_day1_importance_csv": str(shap_csv)},
    )

    assert "shap_target_aqi_day1_importance_csv" in version.artifact_paths
    # Non-model artifacts must not be treated as loadable model files.
    loaded = load_registered_models(version)
    assert "shap_target_aqi_day1_importance_csv" not in loaded


def test_get_champion_ignores_lower_mae_version_from_a_different_city(
    fake_registry: FakeModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared model_name can hold versions for multiple cities; champion selection must not
    cross city boundaries just because one city's MAE happens to be numerically lower."""

    def _city(city_id: str) -> object:
        return type("City", (), {"city_id": city_id})()

    monkeypatch.setattr(registry_module, "load_city_config", lambda: _city("lahore"))
    lahore_version = register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="ridge",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 16.0, "selection_mae_mean": 16.0, "selection_mae_std": 1.0},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={},
    )

    monkeypatch.setattr(registry_module, "load_city_config", lambda: _city("karachi"))
    karachi_version = register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="hist_gradient_boosting",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 7.0, "selection_mae_mean": 7.0, "selection_mae_std": 0.5},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={},
    )

    assert len(list_registered_versions("pearls_aqi_forecaster")) == 2

    monkeypatch.setattr(registry_module, "load_city_config", lambda: _city("lahore"))
    assert get_champion("pearls_aqi_forecaster").version == lahore_version.version

    monkeypatch.setattr(registry_module, "load_city_config", lambda: _city("karachi"))
    assert get_champion("pearls_aqi_forecaster").version == karachi_version.version


def test_get_champion_raises_when_no_versions_match_configured_city(
    fake_registry: FakeModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.open_meteo import OpenMeteoClientError

    monkeypatch.setattr(
        registry_module, "load_city_config", lambda: type("City", (), {"city_id": "lahore"})()
    )
    register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="ridge",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 16.0, "selection_mae_mean": 16.0, "selection_mae_std": 1.0},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={},
    )

    monkeypatch.setattr(
        registry_module, "load_city_config", lambda: type("City", (), {"city_id": "karachi"})()
    )
    with pytest.raises(OpenMeteoClientError):
        get_champion("pearls_aqi_forecaster")


def test_get_champion_raises_when_no_versions_registered(fake_registry: FakeModelRegistry) -> None:
    from src.data.open_meteo import OpenMeteoClientError

    with pytest.raises(OpenMeteoClientError, match="No registered models found"):
        get_champion("pearls_aqi_forecaster_missing")


def test_list_registered_versions_skips_version_when_manifest_download_fails(
    fake_registry: FakeModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, "load_city_config", lambda: type("City", (), {"city_id": "karachi"})())

    broken = register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="ridge",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 10.0, "selection_mae_mean": 10.0, "selection_mae_std": 1.0},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={},
    )
    healthy = register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="hist_gradient_boosting",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 7.0, "selection_mae_mean": 7.0, "selection_mae_std": 0.5},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={},
    )
    fake_registry.get_model("pearls_aqi_forecaster", broken.version).download = lambda: (_ for _ in ()).throw(
        RuntimeError("artifact download failed")
    )

    versions = list_registered_versions("pearls_aqi_forecaster")

    assert [version.version for version in versions] == [healthy.version]
    assert get_champion("pearls_aqi_forecaster").version == healthy.version
