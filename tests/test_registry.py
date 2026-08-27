"""Tests for the local fallback model registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.registry import get_champion, list_registered_versions, register_model_version
from src.pipelines.train import _build_mlp_model


def test_register_model_version_uses_validation_metric_for_champion(tmp_path: Path) -> None:
    shap_path = tmp_path / "shap.csv"
    shap_path.write_text("feature,mean_abs_shap\nus_aqi,1.0\n", encoding="utf-8")

    register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="ridge",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 9.0, "selection_mae_mean": 8.0, "selection_mae_std": 0.3},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={"day1_csv": str(shap_path)},
        registry_root=tmp_path,
    )
    champion = register_model_version(
        model_name="pearls_aqi_forecaster",
        model_type="hist_gradient_boosting",
        fitted_models={"target_aqi_day1": Ridge()},
        metrics={"mae_mean": 9.5, "selection_mae_mean": 7.5, "selection_mae_std": 0.2},
        feature_list=["us_aqi"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-01-02T00:00:00+00:00",
        shap_artifact_paths={"day1_csv": str(shap_path)},
        registry_root=tmp_path,
    )

    versions = list_registered_versions("pearls_aqi_forecaster", registry_root=tmp_path)

    assert len(versions) == 2
    assert get_champion("pearls_aqi_forecaster", registry_root=tmp_path).version == champion.version


def test_register_model_version_serializes_tensorflow_bundle(tmp_path: Path) -> None:
    shap_path = tmp_path / "shap.csv"
    shap_path.write_text("feature,mean_abs_shap\nus_aqi,1.0\n", encoding="utf-8")
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
        shap_artifact_paths={"day1_csv": str(shap_path)},
        registry_root=tmp_path,
    )

    assert Path(version.artifact_paths["all_horizons_model"]).exists()
    assert Path(version.artifact_paths["all_horizons_scaler"]).exists()
