"""Single-page Streamlit dashboard for the AQI forecast system."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.data.open_meteo import OpenMeteoClientError
from src.feature_store import load_features
from src.inference.aqi import aqi_category

DEFAULT_API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEFAULT_TIMEOUT_SECONDS = 15
SHAP_DIR = Path("data") / "metrics" / "shap"
DAY5_SUMMARY_PATH = Path("data") / "metrics" / "day5_summary.json"


@dataclass(frozen=True)
class DashboardPayload:
    """Combined API and local artifact payload for the dashboard."""

    forecast: dict[str, Any]
    model_info: dict[str, Any]
    history: dict[str, Any]
    day5_summary: dict[str, Any]
    shap_importance: pd.DataFrame
    current_conditions: dict[str, float | str]


def main() -> None:
    """Render the Day 7 dashboard."""

    st.set_page_config(
        page_title="Pearls AQI Predictor",
        page_icon="AQI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()

    st.title("Pearls AQI Predictor")
    st.caption("Forecast, not measurement. CAMS via Open-Meteo with model-specific SHAP explanations.")

    try:
        payload = load_dashboard_payload()
    except OpenMeteoClientError as exc:
        _render_error_state(str(exc))
        return

    forecast = payload.forecast
    model_info = payload.model_info
    history = payload.history
    city = forecast["city"]

    st.markdown(
        f"""
        <div class="hero-card">
          <div>
            <div class="eyebrow">Configured city</div>
            <h2>{city}</h2>
            <p>Last model training: {model_info["trained_at"]}</p>
            <p>Last data update: {history["points"][-1]["event_time"]}</p>
          </div>
          <div class="hero-pill">Champion v{forecast["model_version"]} · {forecast["model_type"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_alert_banner(forecast["forecast"])
    _render_kpi_row(forecast)
    _render_forecast_chart(history["points"], forecast["forecast"])
    _render_current_conditions_row(payload.current_conditions)
    _render_driver_row(payload.shap_importance)
    _render_quality_and_shap(payload)
    _render_footer()


@st.cache_data(ttl=900)
def load_dashboard_payload(api_base_url: str = DEFAULT_API_BASE_URL) -> DashboardPayload:
    """Load all dashboard data with caching for a smoother UI."""

    forecast = _get_json(f"{api_base_url}/forecast")
    model_info = _get_json(f"{api_base_url}/model-info")
    history = _get_json(f"{api_base_url}/history", params={"days": 7})
    day5_summary = _load_day5_summary()
    shap_importance = _load_shap_importance(model_info["model_type"])
    current_conditions = _load_current_conditions()
    return DashboardPayload(
        forecast=forecast,
        model_info=model_info,
        history=history,
        day5_summary=day5_summary,
        shap_importance=shap_importance,
        current_conditions=current_conditions,
    )


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise OpenMeteoClientError(f"Dashboard could not reach the API at {url}: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise OpenMeteoClientError(
            f"Dashboard request failed for {url} with HTTP {response.status_code}: {detail or 'no response body'}"
        )
    return response.json()


def _load_day5_summary(path: Path = DAY5_SUMMARY_PATH) -> dict[str, Any]:
    if not path.exists():
        raise OpenMeteoClientError(f"Expected Day 5 summary artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_shap_importance(model_type: str, base_dir: Path = SHAP_DIR) -> pd.DataFrame:
    shap_path = base_dir / f"{model_type}_target_aqi_day1_importance.csv"
    if not shap_path.exists():
        raise OpenMeteoClientError(f"Expected SHAP importance artifact is missing: {shap_path}")
    return pd.read_csv(shap_path).head(5)


def _load_current_conditions() -> dict[str, float | str]:
    features = load_features().sort_values(["city_id", "event_time"]).reset_index(drop=True)
    if features.empty:
        raise OpenMeteoClientError("Feature store is empty; current pollutant and weather conditions are unavailable")

    latest = features.iloc[-1]
    return {
        "event_time": latest["event_time"].isoformat(),
        "pm2_5": float(latest["pm2_5"]),
        "pm10": float(latest["pm10"]),
        "ozone": float(latest["ozone"]),
        "nitrogen_dioxide": float(latest["nitrogen_dioxide"]),
        "relative_humidity_2m": float(latest["relative_humidity_2m"]),
        "wind_speed_10m": float(latest["wind_speed_10m"]),
    }


def _render_error_state(message: str) -> None:
    st.markdown(
        f"""
        <div class="error-card">
          <div class="eyebrow">Dashboard unavailable</div>
          <h3>We could not load the latest forecast.</h3>
          <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_alert_banner(points: list[dict[str, Any]]) -> None:
    highest_alert = max((point["alert"] for point in points), key=_alert_rank)
    if highest_alert == "none":
        return
    highest_value = max(point["aqi"] for point in points)
    st.markdown(
        f"""
        <div class="alert-banner">
          Alert: forecast reaches {highest_alert.upper()} levels with AQI up to {highest_value:.0f}.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_row(forecast: dict[str, Any]) -> None:
    cards = [
        ("Current AQI", forecast["current_aqi"], aqi_category(float(forecast["current_aqi"]))),
        ("Day +1", forecast["forecast"][0]["aqi"], forecast["forecast"][0]["category"]),
        ("Day +2", forecast["forecast"][1]["aqi"], forecast["forecast"][1]["category"]),
        ("Day +3", forecast["forecast"][2]["aqi"], forecast["forecast"][2]["category"]),
    ]
    columns = st.columns(4)
    for column, (label, value, category) in zip(columns, cards, strict=True):
        color = _category_color(category)
        column.markdown(
            f"""
            <div class="kpi-card" style="border-top: 6px solid {color};">
              <div class="eyebrow">{label}</div>
              <div class="kpi-value">{float(value):.0f}</div>
              <div class="kpi-category">{category}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_forecast_chart(history_points: list[dict[str, Any]], forecast_points: list[dict[str, Any]]) -> None:
    history_frame = pd.DataFrame(history_points)
    history_frame["event_time"] = pd.to_datetime(history_frame["event_time"], utc=True)

    forecast_origin = history_frame["event_time"].max()
    forecast_frame = pd.DataFrame(
        {
            "event_time": [forecast_origin + pd.Timedelta(days=index) for index in range(1, 4)],
            "us_aqi": [point["aqi"] for point in forecast_points],
        }
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history_frame["event_time"],
            y=history_frame["us_aqi"],
            mode="lines",
            name="Past 7 days",
            line={"color": "#0b6e4f", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_frame["event_time"],
            y=forecast_frame["us_aqi"],
            mode="lines+markers",
            name="3-day forecast",
            line={"color": "#d94841", "width": 3, "dash": "dash"},
            marker={"size": 10},
        )
    )
    figure.add_vline(x=forecast_origin, line_dash="dot", line_color="#2f4858")
    figure.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8f4ec",
        title="AQI history and forecast",
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        xaxis_title="Time (UTC)",
        yaxis_title="US AQI",
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_current_conditions_row(current_conditions: dict[str, float | str]) -> None:
    st.markdown("### Current pollutant and weather drivers")
    st.caption(f'Latest feature snapshot: {current_conditions["event_time"]}')

    cards = [
        ("PM2.5", current_conditions["pm2_5"], "ug/m3"),
        ("PM10", current_conditions["pm10"], "ug/m3"),
        ("O3", current_conditions["ozone"], "ug/m3"),
        ("NO2", current_conditions["nitrogen_dioxide"], "ug/m3"),
        ("Humidity", current_conditions["relative_humidity_2m"], "%"),
        ("Wind", current_conditions["wind_speed_10m"], "km/h"),
    ]
    columns = st.columns(6)
    for column, (label, value, unit) in zip(columns, cards, strict=True):
        column.markdown(
            f"""
            <div class="kpi-card">
              <div class="eyebrow">{label}</div>
              <div class="kpi-value" style="font-size: 1.7rem;">{float(value):.1f}</div>
              <div class="kpi-category">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_driver_row(shap_importance: pd.DataFrame) -> None:
    left, right = st.columns([1.2, 1.0])
    left.markdown("### SHAP drivers")
    left.dataframe(shap_importance, hide_index=True, use_container_width=True)

    right.markdown("### Plain-language explanation")
    for row in shap_importance.itertuples(index=False):
        feature_name = str(row.feature).replace("_", " ")
        right.markdown(f"- `{feature_name}` is a top contributor right now.")


def _render_quality_and_shap(payload: DashboardPayload) -> None:
    quality, shap_col = st.columns([0.95, 1.05])

    champion_row = next(
        row for row in payload.day5_summary["comparison_rows"] if row["model"] == payload.day5_summary["champion_name"]
    )
    quality.markdown("### Model quality")
    quality.markdown(
        f"""
        <div class="quality-card">
          <p><strong>Champion:</strong> {payload.day5_summary["champion_name"]}</p>
          <p><strong>Validation mean MAE:</strong> {champion_row["selection_mae_mean"]:.2f}</p>
          <p><strong>Final test mean MAE:</strong> {champion_row["mae_mean"]:.2f}</p>
          <p><strong>Final test mean RMSE:</strong> {champion_row["rmse_mean"]:.2f}</p>
          <p><strong>Final test mean R²:</strong> {champion_row["r2_mean"]:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    shap_col.markdown("### SHAP view")
    shap_image = SHAP_DIR / f'{payload.day5_summary["champion_name"]}_target_aqi_day1_top10.png'
    if shap_image.exists():
        shap_col.image(str(shap_image), caption="Top 10 Day +1 SHAP importance")
    else:
        shap_col.info("SHAP image is missing, but the importance table is available.")


def _render_footer() -> None:
    st.markdown(
        """
        <div class="footer-note">
          Built for one configurable city. Data source: CAMS + ERA5 via Open-Meteo. Forecasts support planning; they are not direct measurements.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _category_color(category: str) -> str:
    mapping = {
        "Good": "#2a9d8f",
        "Moderate": "#e9c46a",
        "Unhealthy for Sensitive Groups": "#f4a261",
        "Unhealthy": "#e76f51",
        "Very Unhealthy": "#9b2226",
        "Hazardous": "#5f0f40",
    }
    return mapping.get(category, "#2f4858")


def _alert_rank(alert: str) -> int:
    mapping = {"none": 0, "warning": 1, "critical": 2, "hazardous": 3}
    return mapping.get(alert, -1)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(244, 162, 97, 0.18), transparent 35%),
            radial-gradient(circle at top right, rgba(42, 157, 143, 0.18), transparent 30%),
            linear-gradient(180deg, #f8f4ec 0%, #efe6d7 100%);
        }
        .hero-card, .kpi-card, .quality-card, .error-card {
          background: rgba(255, 250, 241, 0.92);
          border: 1px solid rgba(47, 72, 88, 0.12);
          border-radius: 20px;
          padding: 1rem 1.2rem;
          box-shadow: 0 18px 45px rgba(47, 72, 88, 0.08);
        }
        .hero-card {
          display: flex;
          justify-content: space-between;
          align-items: end;
          margin-bottom: 1rem;
        }
        .hero-pill, .alert-banner {
          background: #2f4858;
          color: #f8f4ec;
          padding: 0.65rem 1rem;
          border-radius: 999px;
          font-weight: 600;
        }
        .alert-banner {
          background: #9b2226;
          margin: 0.5rem 0 1rem 0;
          text-align: center;
        }
        .eyebrow {
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-size: 0.75rem;
          color: #6b705c;
          margin-bottom: 0.35rem;
        }
        .kpi-value {
          font-size: 2.4rem;
          font-weight: 700;
          color: #1f2933;
        }
        .kpi-category {
          color: #52606d;
        }
        .footer-note {
          margin-top: 2rem;
          color: #52606d;
          font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
