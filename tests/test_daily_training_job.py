"""Day 8 tests for automated daily training promotion decisions."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.registry import RegisteredModelVersion
from src.pipelines.train import Day5Artifacts, ModelRunResult, run_daily_training_job


def test_daily_training_registers_candidate_when_no_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = RegisteredModelVersion(
        model_name="pearls_aqi_forecaster",
        version=3,
        model_type="ridge",
        metrics={"mae_mean": 8.5, "selection_mae_mean": 8.0, "selection_mae_std": 0.2},
        feature_list=["feature_a"],
        trained_at="2026-08-29T00:00:00+00:00",
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-08-29T00:00:00+00:00",
        artifact_paths={},
    )
    monkeypatch.setattr("src.pipelines.train._safe_get_existing_champion", lambda: None)
    monkeypatch.setattr("src.pipelines.train.run_day5_pipeline", lambda register_in_local_registry: _day5_artifacts())
    monkeypatch.setattr("src.pipelines.train.register_model_version", lambda **kwargs: registered)

    decision = run_daily_training_job()

    assert decision.promoted is True
    assert decision.registered_version == registered
    assert decision.incumbent_version is None


def test_daily_training_registers_candidate_when_it_beats_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incumbent = RegisteredModelVersion(
        model_name="pearls_aqi_forecaster",
        version=2,
        model_type="hist_gradient_boosting",
        metrics={"mae_mean": 9.0, "selection_mae_mean": 8.8, "selection_mae_std": 0.3},
        feature_list=["feature_a"],
        trained_at="2026-08-28T00:00:00+00:00",
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-08-28T00:00:00+00:00",
        artifact_paths={},
    )
    registered = RegisteredModelVersion(
        model_name="pearls_aqi_forecaster",
        version=3,
        model_type="ridge",
        metrics={"mae_mean": 8.5, "selection_mae_mean": 8.0, "selection_mae_std": 0.2},
        feature_list=["feature_a"],
        trained_at="2026-08-29T00:00:00+00:00",
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-08-29T00:00:00+00:00",
        artifact_paths={},
    )
    monkeypatch.setattr("src.pipelines.train._safe_get_existing_champion", lambda: incumbent)
    monkeypatch.setattr("src.pipelines.train.run_day5_pipeline", lambda register_in_local_registry: _day5_artifacts())
    monkeypatch.setattr("src.pipelines.train.register_model_version", lambda **kwargs: registered)

    decision = run_daily_training_job()

    assert decision.promoted is True
    assert decision.registered_version == registered
    assert decision.incumbent_version == 2


def test_daily_training_skips_registration_when_candidate_is_worse_than_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incumbent = RegisteredModelVersion(
        model_name="pearls_aqi_forecaster",
        version=2,
        model_type="hist_gradient_boosting",
        metrics={"mae_mean": 7.0, "selection_mae_mean": 6.8, "selection_mae_std": 0.1},
        feature_list=["feature_a"],
        trained_at="2026-08-28T00:00:00+00:00",
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-08-28T00:00:00+00:00",
        artifact_paths={},
    )
    monkeypatch.setattr("src.pipelines.train._safe_get_existing_champion", lambda: incumbent)
    monkeypatch.setattr("src.pipelines.train.run_day5_pipeline", lambda register_in_local_registry: _day5_artifacts())

    called = {"value": False}

    def _register(**kwargs):
        called["value"] = True
        raise AssertionError("register_model_version should not be called for a worse candidate")

    monkeypatch.setattr("src.pipelines.train.register_model_version", _register)

    decision = run_daily_training_job()

    assert decision.promoted is False
    assert decision.registered_version is None
    assert decision.incumbent_version == 2
    assert called["value"] is False


def _day5_artifacts() -> Day5Artifacts:
    comparison = pd.DataFrame(
        [
            {
                "model": "ridge",
                "mae_day1": 8.0,
                "mae_day2": 8.5,
                "mae_day3": 9.0,
                "mae_mean": 8.5,
                "rmse_mean": 9.1,
                "r2_mean": 0.5,
                "selection_mae_mean": 8.0,
                "selection_mae_std": 0.2,
            }
        ]
    )
    champion_result = ModelRunResult(
        model_name="ridge",
        fitted_models={"target_aqi_day1": object()},
        best_params={},
        metrics={
            "mae_day1": 8.0,
            "mae_day2": 8.5,
            "mae_day3": 9.0,
            "mae_mean": 8.5,
            "rmse_mean": 9.1,
            "r2_mean": 0.5,
        },
    )
    return Day5Artifacts(
        comparison=comparison,
        rolling_validation_raw=pd.DataFrame(),
        rolling_validation_summary=pd.DataFrame(),
        top_two_models=["ridge", "random_forest"],
        champion_name="ridge",
        champion_result=champion_result,
        feature_columns=["feature_a"],
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-08-29T00:00:00+00:00",
        shap_artifact_paths={},
        registered_version=None,
    )
