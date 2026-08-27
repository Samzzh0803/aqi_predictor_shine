"""Shared AQI category and alert helpers for inference and API responses."""

from __future__ import annotations

from src.data.open_meteo import OpenMeteoClientError


def aqi_category(value: float) -> str:
    """Return the US EPA AQI category for a clipped AQI value."""

    if value < 0 or value > 500:
        raise OpenMeteoClientError("AQI category expects a value in the inclusive range [0, 500]")
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    if value <= 150:
        return "Unhealthy for Sensitive Groups"
    if value <= 200:
        return "Unhealthy"
    if value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def aqi_alert_level(value: float) -> str:
    """Return the project alert level for a clipped AQI value."""

    if value < 0 or value > 500:
        raise OpenMeteoClientError("AQI alert expects a value in the inclusive range [0, 500]")
    if value >= 301:
        return "hazardous"
    if value >= 201:
        return "critical"
    if value >= 151:
        return "warning"
    return "none"
