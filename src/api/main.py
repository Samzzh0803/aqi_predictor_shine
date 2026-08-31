"""FastAPI endpoints for health, forecast, model info, and recent history."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.config import get_config
from src.data.open_meteo import OpenMeteoClientError
from src.feature_store import load_features
from src.inference.predictor import PredictionArtifacts, predict_next_3_days
from src.models.registry import get_champion

app = FastAPI(title="Pearls AQI Predictor API")


class HealthResponse(BaseModel):
    status: str


class ForecastPointResponse(BaseModel):
    horizon: str
    aqi: float
    category: str
    alert: str


class ForecastResponse(BaseModel):
    city: str
    generated_at: str
    model_version: int
    model_type: str
    current_aqi: float
    forecast: list[ForecastPointResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    version: int
    model_type: str
    trained_at: str
    metrics: dict[str, float]
    feature_count: int
    data_start: str
    data_end: str
    backend: str


class HistoryPointResponse(BaseModel):
    event_time: str
    us_aqi: float


class HistoryResponse(BaseModel):
    city: str
    days: int
    points: list[HistoryPointResponse]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/forecast", response_model=ForecastResponse)
def forecast() -> ForecastResponse:
    try:
        payload = predict_next_3_days()
    except OpenMeteoClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert backend dependency failures into a useful API response
        raise HTTPException(status_code=503, detail=f"Forecast backend failed: {exc}") from exc
    return _forecast_response(payload)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        champion = get_champion()
    except OpenMeteoClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert backend dependency failures into a useful API response
        raise HTTPException(status_code=503, detail=f"Model registry backend failed: {exc}") from exc
    return ModelInfoResponse(
        model_name=champion.model_name,
        version=champion.version,
        model_type=champion.model_type,
        trained_at=champion.trained_at,
        metrics=champion.metrics,
        feature_count=len(champion.feature_list),
        data_start=champion.data_start,
        data_end=champion.data_end,
        backend=champion.backend,
    )


@app.get("/history", response_model=HistoryResponse)
def history(days: int = Query(default=7, ge=1, le=30)) -> HistoryResponse:
    try:
        features = load_features().sort_values(["city_id", "event_time"]).reset_index(drop=True)
    except OpenMeteoClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert backend dependency failures into a useful API response
        raise HTTPException(status_code=503, detail=f"Feature-store backend failed: {exc}") from exc
    if features.empty:
        raise HTTPException(status_code=503, detail="Feature store is empty")

    cutoff = features["event_time"].max() - pd.Timedelta(days=days)
    history_frame = features.loc[features["event_time"] >= cutoff, ["event_time", "us_aqi"]].copy()
    city_name = str(get_config().get("city", {}).get("name", "Unknown"))
    points = [
        HistoryPointResponse(event_time=row.event_time.isoformat(), us_aqi=float(row.us_aqi))
        for row in history_frame.itertuples(index=False)
    ]
    return HistoryResponse(city=city_name, days=days, points=points)


def _forecast_response(payload: PredictionArtifacts) -> ForecastResponse:
    return ForecastResponse(
        city=payload.city,
        generated_at=payload.generated_at,
        model_version=payload.model_version,
        model_type=payload.model_type,
        current_aqi=payload.current_aqi,
        forecast=[
            ForecastPointResponse(
                horizon=point.horizon,
                aqi=point.aqi,
                category=point.category,
                alert=point.alert,
            )
            for point in payload.forecast
        ],
    )
