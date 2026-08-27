"""Tests for Day 1 Open-Meteo data ingestion."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.data.open_meteo import (
    AIR_QUALITY_HOURLY_FIELDS,
    WEATHER_HOURLY_FIELDS,
    OpenMeteoClientError,
    fetch_air_quality,
    fetch_weather,
    load_city_config,
    merge_air_quality_and_weather,
)


def _mock_response(payload: dict, status_code: int = 200) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.ok = status_code < 400
    response.json.return_value = payload
    response.text = "failure"
    return response


def _hourly_payload(columns: list[str]) -> dict:
    return {
        "hourly": {
            "time": ["2026-08-20T00:00", "2026-08-20T01:00"],
            **{column: [1.0, 2.0] for column in columns},
        }
    }


def test_load_city_config_reads_repo_default() -> None:
    city = load_city_config()
    assert city.city_id == "lahore"
    assert city.timezone == "Asia/Karachi"


@patch("src.data.open_meteo.requests.get")
def test_fetch_air_quality_expected_columns(mock_get: Mock) -> None:
    mock_get.return_value = _mock_response(_hourly_payload(AIR_QUALITY_HOURLY_FIELDS))
    frame = fetch_air_quality("2026-08-20", "2026-08-21")

    assert list(frame.columns) == ["city_id", "latitude", "longitude", "event_time", *AIR_QUALITY_HOURLY_FIELDS]
    assert frame["event_time"].is_monotonic_increasing
    assert frame["event_time"].is_unique
    assert frame["us_aqi"].notna().any()


@patch("src.data.open_meteo.requests.get")
def test_fetch_weather_expected_columns(mock_get: Mock) -> None:
    mock_get.return_value = _mock_response(_hourly_payload(WEATHER_HOURLY_FIELDS))
    frame = fetch_weather("2026-08-20", "2026-08-21")

    assert list(frame.columns) == ["city_id", "latitude", "longitude", "event_time", *WEATHER_HOURLY_FIELDS]
    assert frame["event_time"].is_monotonic_increasing
    assert frame["event_time"].is_unique


@patch("src.data.open_meteo.requests.get")
def test_weather_join_produces_no_row_explosion(mock_get: Mock) -> None:
    mock_get.side_effect = [
        _mock_response(_hourly_payload(AIR_QUALITY_HOURLY_FIELDS)),
        _mock_response(_hourly_payload(WEATHER_HOURLY_FIELDS)),
    ]

    air_quality = fetch_air_quality("2026-08-20", "2026-08-21")
    weather = fetch_weather("2026-08-20", "2026-08-21")
    merged = merge_air_quality_and_weather(air_quality, weather)

    assert len(merged) == len(air_quality)
    assert merged["event_time"].is_unique


@patch("src.data.open_meteo.requests.get")
def test_bad_request_raises_clean_error(mock_get: Mock) -> None:
    mock_get.return_value = _mock_response({}, status_code=400)

    with pytest.raises(OpenMeteoClientError, match="returned HTTP 400"):
        fetch_air_quality("2026-08-20", "2026-08-21")


@patch("src.data.open_meteo.requests.get")
def test_bad_request_is_not_retried(mock_get: Mock) -> None:
    """4xx errors are our bug (bad params) — retrying them is wasteful and wrong."""
    mock_get.return_value = _mock_response({}, status_code=400)

    with pytest.raises(OpenMeteoClientError):
        fetch_air_quality("2026-08-20", "2026-08-21")

    assert mock_get.call_count == 1


@patch("src.data.open_meteo.requests.get")
def test_missing_columns_raise_clean_error(mock_get: Mock) -> None:
    payload = {"hourly": {"time": ["2026-08-20T00:00"]}}
    mock_get.return_value = _mock_response(payload)

    with pytest.raises(OpenMeteoClientError, match="missing columns"):
        fetch_air_quality("2026-08-20", "2026-08-21")


@patch("src.data.open_meteo.requests.get")
def test_duplicate_timestamps_raise_clean_error(mock_get: Mock) -> None:
    payload = {
        "hourly": {
            "time": ["2026-08-20T00:00", "2026-08-20T00:00"],
            **{col: [1.0, 2.0] for col in AIR_QUALITY_HOURLY_FIELDS},
        }
    }
    mock_get.return_value = _mock_response(payload)

    with pytest.raises(OpenMeteoClientError, match="duplicate event_time"):
        fetch_air_quality("2026-08-20", "2026-08-21")


@patch("src.data.open_meteo.requests.get")
def test_merge_raises_when_frames_share_no_timestamps(mock_get: Mock) -> None:
    air_payload = {
        "hourly": {
            "time": ["2026-08-20T00:00"],
            **{col: [1.0] for col in AIR_QUALITY_HOURLY_FIELDS},
        }
    }
    weather_payload = {
        "hourly": {
            "time": ["2026-08-21T00:00"],
            **{col: [1.0] for col in WEATHER_HOURLY_FIELDS},
        }
    }
    mock_get.side_effect = [
        _mock_response(air_payload),
        _mock_response(weather_payload),
    ]

    air = fetch_air_quality("2026-08-20", "2026-08-20")
    weather = fetch_weather("2026-08-21", "2026-08-21")

    with pytest.raises(OpenMeteoClientError, match="empty"):
        merge_air_quality_and_weather(air, weather)
