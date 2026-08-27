"""Open-Meteo data access helpers for AQI and weather ingestion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from requests import Response
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from pandas import DatetimeTZDtype

LOGGER = logging.getLogger(__name__)

DEFAULT_AIR_QUALITY_BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
DEFAULT_WEATHER_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_SECONDS = 30

AIR_QUALITY_HOURLY_FIELDS = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]

WEATHER_HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]


class OpenMeteoClientError(RuntimeError):
    """Raised when an Open-Meteo request or response validation fails."""


class _RetriableError(OpenMeteoClientError):
    """Raised only for transient failures that should be retried (network, 5xx)."""


@dataclass(frozen=True)
class CityConfig:
    """Coordinates and timezone for the configured city."""

    city_id: str
    name: str
    latitude: float
    longitude: float
    timezone: str


def load_city_config(path: str | Path = "config/config.yaml") -> CityConfig:
    """Load the configured city from the repository config file."""

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as fh:
            payload = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise OpenMeteoClientError(f"City config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise OpenMeteoClientError(f"City config file is not valid YAML: {config_path}") from exc

    city = payload.get("city", {})
    required_keys = {"id", "name", "latitude", "longitude", "timezone"}
    missing_keys = sorted(required_keys.difference(city))
    if missing_keys:
        raise OpenMeteoClientError(f"City config missing required keys: {missing_keys}")

    return CityConfig(
        city_id=str(city["id"]),
        name=str(city["name"]),
        latitude=float(city["latitude"]),
        longitude=float(city["longitude"]),
        timezone=str(city["timezone"]),
    )


def _normalize_date(value: date | datetime | str) -> str:
    """Convert input dates to ISO date strings for Open-Meteo queries."""

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RetriableError),
)
def _get_json(url: str, params: dict[str, Any], timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Fetch JSON from Open-Meteo with retry on transient failures only."""

    LOGGER.info("Requesting Open-Meteo data", extra={"url": url, "params": params})
    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise _RetriableError(f"Open-Meteo request failed: {exc}") from exc

    _raise_for_status(response)
    payload = response.json()
    if "hourly" not in payload:
        raise OpenMeteoClientError("Open-Meteo response did not include an 'hourly' payload")
    return payload


def _raise_for_status(response: Response) -> None:
    """Raise a retriable error for 5xx, non-retriable for 4xx."""

    if response.ok:
        return
    detail = response.text.strip()
    msg = f"Open-Meteo returned HTTP {response.status_code}: {detail or 'no response body'}"
    if response.status_code >= 500:
        raise _RetriableError(msg)
    raise OpenMeteoClientError(msg)


def _hourly_payload_to_frame(
    payload: dict[str, Any],
    expected_columns: list[str],
    city: CityConfig,
) -> pd.DataFrame:
    """Convert an Open-Meteo hourly payload to a validated DataFrame."""

    hourly = payload["hourly"]
    required_columns = ["time", *expected_columns]
    missing_columns = [column for column in required_columns if column not in hourly]
    if missing_columns:
        raise OpenMeteoClientError(f"Open-Meteo hourly payload missing columns: {missing_columns}")

    frame = pd.DataFrame({column: hourly[column] for column in required_columns})
    frame["event_time"] = pd.to_datetime(frame.pop("time"), utc=True)
    frame["city_id"] = city.city_id
    frame["latitude"] = city.latitude
    frame["longitude"] = city.longitude
    frame = frame[["city_id", "latitude", "longitude", "event_time", *expected_columns]]
    frame = frame.sort_values("event_time").reset_index(drop=True)

    if frame["event_time"].duplicated().any():
        raise OpenMeteoClientError("Open-Meteo returned duplicate event_time values")
    if not isinstance(frame["event_time"].dtype, DatetimeTZDtype):
        raise OpenMeteoClientError("event_time must be timezone-aware UTC")

    return frame


def fetch_air_quality(
    start: date | datetime | str,
    end: date | datetime | str,
    city: CityConfig | None = None,
    base_url: str = DEFAULT_AIR_QUALITY_BASE_URL,
) -> pd.DataFrame:
    """Fetch historical air-quality data for the configured city."""

    city = city or load_city_config()
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": _normalize_date(start),
        "end_date": _normalize_date(end),
        "timezone": "UTC",
        "hourly": ",".join(AIR_QUALITY_HOURLY_FIELDS),
    }
    payload = _get_json(base_url, params)
    return _hourly_payload_to_frame(payload, AIR_QUALITY_HOURLY_FIELDS, city)


def fetch_weather(
    start: date | datetime | str,
    end: date | datetime | str,
    city: CityConfig | None = None,
    base_url: str = DEFAULT_WEATHER_BASE_URL,
) -> pd.DataFrame:
    """Fetch historical weather data for the configured city."""

    city = city or load_city_config()
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": _normalize_date(start),
        "end_date": _normalize_date(end),
        "timezone": "UTC",
        "hourly": ",".join(WEATHER_HOURLY_FIELDS),
    }
    payload = _get_json(base_url, params)
    return _hourly_payload_to_frame(payload, WEATHER_HOURLY_FIELDS, city)


def fetch_air_quality_recent(days: int = 7, city: CityConfig | None = None) -> pd.DataFrame:
    """Fetch recent air-quality history using the forecast-compatible endpoint."""

    end = datetime.now(timezone.utc).date()
    start = end - pd.Timedelta(days=days - 1)
    return fetch_air_quality(start=start, end=end, city=city)


def fetch_weather_recent(days: int = 7, city: CityConfig | None = None) -> pd.DataFrame:
    """Fetch recent weather history from the forecast endpoint."""

    end = datetime.now(timezone.utc).date()
    start = end - pd.Timedelta(days=days - 1)
    return fetch_weather(start=start, end=end, city=city)


def merge_air_quality_and_weather(air_quality: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge AQI and weather frames on the Day 1 primary key."""

    merged = air_quality.merge(
        weather.drop(columns=["latitude", "longitude"]),
        on=["city_id", "event_time"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values("event_time").reset_index(drop=True)
    if merged.empty:
        raise OpenMeteoClientError("Merged AQI and weather frame is empty")
    return merged
