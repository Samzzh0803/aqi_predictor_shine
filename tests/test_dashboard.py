"""Day 7 dashboard helper tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.open_meteo import OpenMeteoClientError


def test_load_day5_summary_reads_json(tmp_path: Path) -> None:
    from dashboard.app import _load_day5_summary

    summary_path = tmp_path / "day5_summary.json"
    summary_path.write_text(json.dumps({"champion_name": "ridge"}), encoding="utf-8")

    payload = _load_day5_summary(summary_path)

    assert payload["champion_name"] == "ridge"


def test_load_shap_importance_reads_top_rows(tmp_path: Path) -> None:
    from dashboard.app import _load_shap_importance

    shap_dir = tmp_path
    pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d", "e", "f"],
            "mean_abs_shap": [6, 5, 4, 3, 2, 1],
        }
    ).to_csv(shap_dir / "ridge_target_aqi_day1_importance.csv", index=False)

    frame = _load_shap_importance("ridge", shap_dir)

    assert len(frame) == 5
    assert frame.iloc[0]["feature"] == "a"


def test_load_current_conditions_reads_latest_feature_row(monkeypatch: pytest.MonkeyPatch) -> None:
    from dashboard.app import _load_current_conditions

    frame = pd.DataFrame(
        {
            "city_id": ["lahore", "lahore"],
            "event_time": pd.to_datetime(
                ["2026-08-27T22:00:00+00:00", "2026-08-27T23:00:00+00:00"],
                utc=True,
            ),
            "pm2_5": [100.0, 101.0],
            "pm10": [120.0, 121.0],
            "ozone": [10.0, 11.0],
            "nitrogen_dioxide": [20.0, 21.0],
            "sulphur_dioxide": [5.0, 6.0],
            "carbon_monoxide": [400.0, 410.0],
            "relative_humidity_2m": [70.0, 71.0],
            "wind_speed_10m": [4.0, 4.5],
        }
    )
    monkeypatch.setattr("dashboard.app.load_features", lambda: frame)

    payload = _load_current_conditions()

    assert payload["event_time"] == "2026-08-27T23:00:00+00:00"
    assert payload["pm2_5"] == 101.0
    assert payload["wind_speed_10m"] == 4.5


def test_load_dashboard_payload_includes_current_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    from dashboard.app import load_dashboard_payload

    responses = {
        "http://localhost:8000/forecast": {"city": "Lahore"},
        "http://localhost:8000/model-info": {"model_type": "ridge"},
        "http://localhost:8000/history": {"points": []},
    }
    monkeypatch.setattr(
        "dashboard.app._get_json",
        lambda url, params=None: responses[url],
    )
    monkeypatch.setattr("dashboard.app._load_day5_summary", lambda: {"champion_name": "ridge"})
    monkeypatch.setattr(
        "dashboard.app._load_shap_importance",
        lambda model_type: pd.DataFrame({"feature": ["a"], "mean_abs_shap": [1.0]}),
    )
    monkeypatch.setattr(
        "dashboard.app._load_current_conditions",
        lambda: {"event_time": "2026-08-27T23:00:00+00:00", "pm2_5": 101.0},
    )

    load_dashboard_payload.clear()
    payload = load_dashboard_payload("http://localhost:8000")

    assert payload.current_conditions["pm2_5"] == 101.0


def test_load_dashboard_payload_surfaces_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from dashboard.app import load_dashboard_payload

    monkeypatch.setattr(
        "dashboard.app._get_json",
        lambda url, params=None: (_ for _ in ()).throw(OpenMeteoClientError("API down")),
    )
    # load_dashboard_payload runs its API calls and the direct Hopsworks read
    # concurrently (see the ThreadPoolExecutor in dashboard/app.py), so this must
    # be mocked too -- otherwise it fires a real, un-mocked live Hopsworks call
    # from a background thread while the test only expects the API to fail.
    monkeypatch.setattr(
        "dashboard.app._load_current_conditions",
        lambda: {"event_time": "2026-08-27T23:00:00+00:00", "pm2_5": 101.0},
    )

    with pytest.raises(OpenMeteoClientError, match="API down"):
        load_dashboard_payload.clear()
        load_dashboard_payload("http://localhost:8000")


def test_alert_rank_orders_severity() -> None:
    from dashboard.app import _alert_rank

    assert _alert_rank("none") < _alert_rank("warning") < _alert_rank("critical") < _alert_rank("hazardous")
