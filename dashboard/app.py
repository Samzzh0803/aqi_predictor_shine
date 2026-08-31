"""Redesigned Streamlit dashboard for the AQI forecast system."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.open_meteo import OpenMeteoClientError
from src.feature_store import load_features
from src.inference.aqi import aqi_alert_level, aqi_category

DEFAULT_API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEFAULT_TIMEOUT_SECONDS = 180
SHAP_DIR = Path("data") / "metrics" / "shap"
DAY5_SUMMARY_PATH = Path("data") / "metrics" / "day5_summary.json"

AQI_BANDS: list[tuple[int, int, str, str]] = [
    (0, 50, "Good", "#2A9D8F"),
    (51, 100, "Moderate", "#E9C46A"),
    (101, 150, "USG", "#F4A261"),
    (151, 200, "Unhealthy", "#E76F51"),
    (201, 300, "Very Unhealthy", "#9B2226"),
    (301, 500, "Hazardous", "#5F0F40"),
]

HEALTH_GUIDANCE: list[dict[str, str]] = [
    {
        "level": "Good",
        "threshold": "0-50",
        "action": "Normal outdoor activity is fine; keep monitoring the next 3-day trend.",
    },
    {
        "level": "Moderate",
        "threshold": "51-100",
        "action": "Sensitive groups should shorten heavy outdoor exposure if the trend is rising.",
    },
    {
        "level": "Unhealthy for Sensitive Groups",
        "threshold": "101-150",
        "action": "Shift strenuous outdoor work earlier and warn sensitive staff or family members.",
    },
    {
        "level": "Unhealthy",
        "threshold": "151-200",
        "action": "Reduce outdoor duration, use masks for essential exposure, and postpone discretionary outdoor tasks.",
    },
    {
        "level": "Very Unhealthy",
        "threshold": "201-300",
        "action": "Move activities indoors where possible and treat outdoor exposure as an operational risk.",
    },
    {
        "level": "Hazardous",
        "threshold": "301-500",
        "action": "Avoid outdoor activity unless critical, activate alert plans, and communicate protective guidance broadly.",
    },
]


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
    """Render the redesigned dashboard."""

    st.set_page_config(
        page_title="Pearls AQI Predictor",
        page_icon="AQI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()

    st.markdown('<div class="app-kicker">Pearls AQI Predictor</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">Operational AQI Forecast Center</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Live city conditions, 3-day AQI outlook, model evidence, and practical guidance in one place.</p>',
        unsafe_allow_html=True,
    )

    try:
        payload = load_dashboard_payload()
    except OpenMeteoClientError as exc:
        _render_error_state(str(exc))
        _render_footer()
        return

    _render_hero(payload)
    _render_alert_banner(payload.forecast["forecast"])

    live_tab, metrics_tab, insights_tab, scenario_tab, health_tab = st.tabs(
        [
            "Live & 3-Day Forecast",
            "Model Metrics & SHAP",
            "City Data Insights",
            "Custom Scenario Simulator",
            "Health Guidelines",
        ]
    )

    with live_tab:
        _render_live_forecast_tab(payload)
    with metrics_tab:
        _render_metrics_tab(payload)
    with insights_tab:
        _render_city_insights_tab(payload)
    with scenario_tab:
        _render_scenario_tab(payload)
    with health_tab:
        _render_health_tab(payload)

    _render_footer()


@st.cache_data(ttl=900)
def load_dashboard_payload(api_base_url: str = DEFAULT_API_BASE_URL) -> DashboardPayload:
    """Load all dashboard data with caching for a smoother UI.

    The three API calls and the direct Hopsworks read are independent of each
    other, so they run concurrently rather than one after another. On the
    free-tier stack this matters a lot: each one alone can take anywhere from
    30s to a couple of minutes (Render cold start, Hopsworks query-service
    latency), and running them sequentially could push total load time past
    several minutes. Concurrently, total time is close to the slowest single
    call instead of the sum of all of them.
    """

    with ThreadPoolExecutor(max_workers=4) as pool:
        forecast_future = pool.submit(_get_json, f"{api_base_url}/forecast")
        model_info_future = pool.submit(_get_json, f"{api_base_url}/model-info")
        history_future = pool.submit(_get_json, f"{api_base_url}/history", params={"days": 7})
        current_conditions_future = pool.submit(_load_current_conditions)

        forecast = forecast_future.result()
        model_info = model_info_future.result()
        history = history_future.result()
        current_conditions = current_conditions_future.result()

    day5_summary = _load_day5_summary()
    shap_importance = _load_shap_importance(model_info["model_type"])
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
        "sulphur_dioxide": float(latest["sulphur_dioxide"]),
        "carbon_monoxide": float(latest["carbon_monoxide"]),
        "relative_humidity_2m": float(latest["relative_humidity_2m"]),
        "wind_speed_10m": float(latest["wind_speed_10m"]),
        "temperature_2m": float(latest.get("temperature_2m", 0.0)),
    }


def _render_hero(payload: DashboardPayload) -> None:
    forecast = payload.forecast
    model_info = payload.model_info
    history_points = payload.history["points"]
    last_data_update = history_points[-1]["event_time"] if history_points else "Unavailable"
    current_category = aqi_category(float(forecast["current_aqi"]))

    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-copy">
            <div class="section-kicker">Configured City</div>
            <h2>{forecast["city"]}</h2>
            <p>Champion v{forecast["model_version"]} · {forecast["model_type"]}</p>
            <p>Last model training: {model_info["trained_at"]}</p>
            <p>Last data update: {last_data_update}</p>
          </div>
          <div class="hero-status">
            <div class="status-chip">Current band: {current_category}</div>
            <div class="hero-links">
              <a class="link-chip" href="https://github.com/Samzzh0803/aqi_predictor_shine" target="_blank" rel="noopener">GitHub ↗</a>
              <a class="link-chip" href="https://open-meteo.com/" target="_blank" rel="noopener">Data: Open-Meteo ↗</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_error_state(message: str) -> None:
    st.markdown(
        f"""
        <div class="error-card">
          <div class="section-kicker">Dashboard unavailable</div>
          <h3>We could not load the latest forecast.</h3>
          <p>{message}</p>
          <p>Check the API deployment, `API_BASE_URL`, and Hopsworks secrets before retrying.</p>
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
          Forecast alert: {highest_alert.upper()} conditions are expected, with AQI reaching {highest_value:.0f}.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_live_forecast_tab(payload: DashboardPayload) -> None:
    st.markdown('<div class="section-title">Current Air Quality</div>', unsafe_allow_html=True)
    gauge_col, cards_col = st.columns([1.15, 1.35], gap="large")

    with gauge_col:
        current_aqi = float(payload.forecast["current_aqi"])
        st.plotly_chart(
            _build_gauge_figure(current_aqi),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with cards_col:
        _render_forecast_kpis(payload.forecast)

    st.markdown('<div class="section-title">History & Forecast</div>', unsafe_allow_html=True)
    _render_forecast_chart(payload.history["points"], payload.forecast["forecast"])


def _render_metrics_tab(payload: DashboardPayload) -> None:
    st.markdown('<div class="section-title">Model Metrics</div>', unsafe_allow_html=True)
    quality_col, summary_col = st.columns([0.95, 1.05], gap="large")

    with quality_col:
        _render_quality_card(payload)
    with summary_col:
        _render_shap_gallery(payload.day5_summary["champion_name"])

    st.markdown('<div class="section-title">Top Day +1 Drivers</div>', unsafe_allow_html=True)
    driver_col, language_col = st.columns([1.1, 0.9], gap="large")

    with driver_col:
        styled = payload.shap_importance.copy()
        styled["mean_abs_shap"] = styled["mean_abs_shap"].map(lambda value: round(float(value), 3))
        st.dataframe(styled, hide_index=True, use_container_width=True)

    with language_col:
        st.markdown('<div class="info-panel"><h4>Plain-language explanation</h4>', unsafe_allow_html=True)
        for row in payload.shap_importance.itertuples(index=False):
            feature_name = str(row.feature).replace("_", " ").upper()
            st.markdown(
                f'<p class="explanation-line"><span class="feature-tag">{feature_name}</span> is a strong driver of the next-day forecast.</p>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_city_insights_tab(payload: DashboardPayload) -> None:
    st.markdown('<div class="section-title">Live City Snapshot</div>', unsafe_allow_html=True)
    _render_current_conditions_grid(payload.current_conditions)


SCENARIO_INPUTS: list[tuple[str, str, str, float, float, float]] = [
    ("pm2_5", "PM2.5", "ug/m3", 0.0, 500.0, 1.0),
    ("pm10", "PM10", "ug/m3", 0.0, 600.0, 1.0),
    ("carbon_monoxide", "Carbon Monoxide", "ug/m3", 0.0, 30000.0, 100.0),
    ("nitrogen_dioxide", "Nitrogen Dioxide", "ug/m3", 0.0, 400.0, 1.0),
    ("sulphur_dioxide", "Sulphur Dioxide", "ug/m3", 0.0, 500.0, 1.0),
    ("ozone", "Ozone", "ug/m3", 0.0, 400.0, 1.0),
    ("temperature_2m", "Temperature", "C", -10.0, 50.0, 0.5),
    ("relative_humidity_2m", "Humidity", "%", 0.0, 100.0, 1.0),
]


def _render_scenario_tab(payload: DashboardPayload) -> None:
    st.markdown('<div class="section-title">Custom Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Adjust this hour\'s pollutant and weather readings and see how the 3-day '
        "forecast responds. Everything else — lags, rolling trends, calendar effects — is recomputed from real "
        "recent history through the same feature pipeline the live model runs on, not a separate approximation."
        "</p>",
        unsafe_allow_html=True,
    )

    defaults = payload.current_conditions
    slider_values: dict[str, float] = {}
    columns = st.columns(4, gap="medium")
    for index, (column, label, unit, low, high, step) in enumerate(SCENARIO_INPUTS):
        default = min(max(float(defaults.get(column, (low + high) / 2)), low), high)
        with columns[index % 4]:
            slider_values[column] = st.slider(
                f"{label} ({unit})", min_value=low, max_value=high, value=default, step=step, key=f"scenario_{column}"
            )

    if st.button("Run Scenario Prediction", type="primary"):
        try:
            response = requests.post(
                f"{DEFAULT_API_BASE_URL}/predict-scenario",
                json=slider_values,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Scenario prediction failed: {exc}")
            return

        st.markdown('<div class="section-title">Scenario Forecast</div>', unsafe_allow_html=True)
        cards = st.columns(3, gap="medium")
        for card, point in zip(cards, result["forecast"], strict=True):
            color = _category_color(point["category"])
            card.markdown(
                f"""
                <div class="forecast-card" style="border-left: 8px solid {color};">
                  <div class="section-kicker">{point["horizon"].replace("_", " ").title()}</div>
                  <div class="forecast-value">{point["aqi"]:.0f}</div>
                  <div class="forecast-category">{point["category"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_health_tab(payload: DashboardPayload) -> None:
    st.markdown('<div class="section-title">Health Guidance</div>', unsafe_allow_html=True)
    max_forecast = max(float(point["aqi"]) for point in payload.forecast["forecast"])
    max_category = aqi_category(max_forecast)
    max_alert = aqi_alert_level(max_forecast)

    st.markdown(
        f"""
        <div class="info-panel">
          <h4>3-day planning view</h4>
          <p>Highest forecast in the next 3 days: <strong>{max_forecast:.0f}</strong> AQI.</p>
          <p>Expected category: <strong>{max_category}</strong>. Alert level: <strong>{max_alert}</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for item in HEALTH_GUIDANCE:
        st.markdown(
            f"""
            <div class="guideline-card">
              <div class="guideline-range">{item["threshold"]}</div>
              <div class="guideline-copy">
                <h4>{item["level"]}</h4>
                <p>{item["action"]}</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_forecast_kpis(forecast: dict[str, Any]) -> None:
    cards = [
        ("Current AQI", forecast["current_aqi"], aqi_category(float(forecast["current_aqi"]))),
        ("Day +1", forecast["forecast"][0]["aqi"], forecast["forecast"][0]["category"]),
        ("Day +2", forecast["forecast"][1]["aqi"], forecast["forecast"][1]["category"]),
        ("Day +3", forecast["forecast"][2]["aqi"], forecast["forecast"][2]["category"]),
    ]
    columns = st.columns(2, gap="medium")
    for column, (label, value, category) in zip(columns * 2, cards, strict=True):
        color = _category_color(category)
        column.markdown(
            f"""
            <div class="forecast-card" style="border-left: 8px solid {color};">
              <div class="section-kicker">{label}</div>
              <div class="forecast-value">{float(value):.0f}</div>
              <div class="forecast-category">{category}</div>
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
            name="Past 7 Days",
            line={"color": "#113F67", "width": 4},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_frame["event_time"],
            y=forecast_frame["us_aqi"],
            mode="lines+markers",
            name="3-Day Forecast",
            line={"color": "#E76F51", "width": 4, "dash": "dash"},
            marker={"size": 11, "color": "#E76F51"},
        )
    )
    figure.add_vline(x=forecast_origin, line_dash="dot", line_color="#1D3557")
    figure.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFF8EE",
        title={"text": "AQI history and forecast", "font": {"size": 24, "color": "#102A43"}},
        legend={"orientation": "h", "y": 1.1, "x": 0.0, "font": {"color": "#102A43"}},
        xaxis={"title": "Time (UTC)", "title_font": {"color": "#243B53"}, "tickfont": {"color": "#243B53"}},
        yaxis={"title": "US AQI", "title_font": {"color": "#243B53"}, "tickfont": {"color": "#243B53"}},
        font={"color": "#102A43"},
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_current_conditions_grid(current_conditions: dict[str, float | str]) -> None:
    st.caption(f'Latest feature snapshot: {current_conditions["event_time"]}')
    cards = [
        ("PM2.5", current_conditions["pm2_5"], "ug/m3"),
        ("PM10", current_conditions["pm10"], "ug/m3"),
        ("Ozone", current_conditions["ozone"], "ug/m3"),
        ("NO2", current_conditions["nitrogen_dioxide"], "ug/m3"),
        ("Humidity", current_conditions["relative_humidity_2m"], "%"),
        ("Wind", current_conditions["wind_speed_10m"], "km/h"),
    ]
    columns = st.columns(3, gap="medium")
    for column, (label, value, unit) in zip(columns * 2, cards, strict=True):
        column.markdown(
            f"""
            <div class="mini-card">
              <div class="section-kicker">{label}</div>
              <div class="mini-value">{float(value):.1f}</div>
              <div class="mini-unit">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_quality_card(payload: DashboardPayload) -> None:
    champion_row = next(
        row for row in payload.day5_summary["comparison_rows"] if row["model"] == payload.day5_summary["champion_name"]
    )
    st.markdown(
        f"""
        <div class="info-panel metrics-panel">
          <h4>Champion model</h4>
          <p><strong>Name:</strong> {payload.day5_summary["champion_name"]}</p>
          <p><strong>Validation mean MAE:</strong> {champion_row["selection_mae_mean"]:.2f}</p>
          <p><strong>Final test mean MAE:</strong> {champion_row["mae_mean"]:.2f}</p>
          <p><strong>Final test mean RMSE:</strong> {champion_row["rmse_mean"]:.2f}</p>
          <p><strong>Final test mean R²:</strong> {champion_row["r2_mean"]:.2f}</p>
          <p><strong>Trained at:</strong> {payload.model_info["trained_at"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_shap_gallery(champion_name: str) -> None:
    st.markdown('<div class="info-panel"><h4>SHAP visual summaries</h4></div>', unsafe_allow_html=True)
    view = st.radio(
        "SHAP view",
        ("Top 10 bar chart", "Beeswarm summary"),
        horizontal=True,
        label_visibility="collapsed",
    )
    suffix = "top10" if view == "Top 10 bar chart" else "summary"
    shap_image = SHAP_DIR / f"{champion_name}_target_aqi_day1_{suffix}.png"
    if shap_image.exists():
        st.image(str(shap_image), caption=f"Day +1 {view.lower()}")
    else:
        st.info("Requested SHAP image is missing, but the feature-importance table is still available.")


def _build_gauge_figure(current_aqi: float) -> go.Figure:
    steps = [
        {"range": [low, high], "color": color}
        for low, high, _label, color in AQI_BANDS
    ]
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            number={"font": {"size": 52, "color": "#102A43"}},
            title={"text": "CURRENT AQI", "font": {"size": 20, "color": "#102A43"}},
            gauge={
                "axis": {"range": [0, 500], "tickcolor": "#102A43", "tickfont": {"color": "#102A43"}},
                "bar": {"color": "#0B2545", "thickness": 0.32},
                "bgcolor": "#FFF8EE",
                "borderwidth": 0,
                "steps": steps,
                "threshold": {"line": {"color": "#102A43", "width": 5}, "value": current_aqi},
            },
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#102A43"},
    )
    return figure


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
        "Good": "#2A9D8F",
        "Moderate": "#E9C46A",
        "Unhealthy for Sensitive Groups": "#F4A261",
        "Unhealthy": "#E76F51",
        "Very Unhealthy": "#9B2226",
        "Hazardous": "#5F0F40",
        "USG": "#F4A261",
    }
    return mapping.get(category, "#1D3557")


def _alert_rank(alert: str) -> int:
    mapping = {"none": 0, "warning": 1, "critical": 2, "hazardous": 3}
    return mapping.get(alert, -1)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --cream: #F8F1E5;
          --cream-strong: #FFF8EE;
          --navy: #0B2545;
          --navy-soft: #1D4E89;
          --ink: #102A43;
          --muted: #486581;
          --border: rgba(16, 42, 67, 0.12);
          --shadow: 0 20px 50px rgba(11, 37, 69, 0.10);
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(231, 111, 81, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(42, 157, 143, 0.16), transparent 30%),
            linear-gradient(180deg, var(--cream) 0%, #F3E6D1 100%);
          color: var(--ink);
        }
        .block-container {
          padding-top: 2rem;
          padding-bottom: 3rem;
        }
        .app-kicker, .section-kicker {
          text-transform: uppercase;
          letter-spacing: 0.16em;
          font-weight: 800;
          color: var(--navy-soft);
          font-size: 0.78rem;
        }
        .app-title {
          color: var(--ink);
          font-size: 3.1rem;
          line-height: 1.0;
          text-transform: uppercase;
          letter-spacing: 0.03em;
          margin-bottom: 0.4rem;
        }
        .app-subtitle {
          color: var(--muted);
          font-size: 1.05rem;
          max-width: 760px;
          margin-bottom: 1.25rem;
        }
        .section-title {
          color: var(--ink);
          font-size: 1.45rem;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin: 0.6rem 0 0.8rem 0;
        }
        .hero-card, .forecast-card, .mini-card, .info-panel, .error-card, .guideline-card {
          background: var(--cream-strong);
          color: var(--ink);
          border: 1px solid var(--border);
          border-radius: 22px;
          box-shadow: var(--shadow);
        }
        .hero-card {
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          padding: 1.4rem 1.5rem;
          margin-bottom: 1rem;
        }
        .hero-card h2, .hero-card p, .error-card h3, .error-card p, .info-panel h4, .info-panel p, .guideline-card h4, .guideline-card p {
          color: var(--ink);
          margin: 0.2rem 0;
        }
        .hero-status {
          display: flex;
          flex-direction: column;
          gap: 0.7rem;
          justify-content: flex-start;
          align-items: flex-end;
        }
        .status-chip {
          background: var(--navy);
          color: #FDF7EA;
          padding: 0.7rem 1rem;
          border-radius: 999px;
          font-weight: 700;
          text-align: center;
        }
        .hero-links {
          display: flex;
          gap: 0.5rem;
        }
        .link-chip {
          background: transparent;
          color: var(--navy-soft);
          border: 1.5px solid var(--navy-soft);
          padding: 0.4rem 0.85rem;
          border-radius: 999px;
          font-weight: 700;
          font-size: 0.85rem;
          text-decoration: none;
          transition: background 0.15s ease, color 0.15s ease;
        }
        .link-chip:hover {
          background: var(--navy-soft);
          color: #FDF7EA;
        }
        .alert-banner {
          background: #9B2226;
          color: #FFF8EE;
          padding: 0.9rem 1.1rem;
          border-radius: 18px;
          font-weight: 800;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          text-align: center;
          margin: 0.4rem 0 1.2rem 0;
          box-shadow: var(--shadow);
        }
        .forecast-card, .mini-card, .info-panel {
          padding: 1rem 1.1rem;
        }
        .forecast-value {
          color: var(--ink);
          font-size: 2.35rem;
          font-weight: 900;
          line-height: 1.0;
          margin-top: 0.2rem;
        }
        .forecast-category, .mini-unit, .footer-note, .explanation-line {
          color: var(--muted);
        }
        .mini-value {
          color: var(--ink);
          font-size: 1.65rem;
          font-weight: 800;
        }
        .metrics-panel p {
          font-size: 1rem;
        }
        .feature-tag {
          display: inline-block;
          background: var(--navy);
          color: #FFF8EE;
          padding: 0.18rem 0.55rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 700;
          margin-right: 0.35rem;
        }
        .guideline-card {
          display: flex;
          gap: 1rem;
          padding: 1rem 1.15rem;
          margin-bottom: 0.8rem;
        }
        .guideline-range {
          min-width: 88px;
          background: var(--navy);
          color: #FFF8EE;
          border-radius: 16px;
          padding: 0.8rem 0.6rem;
          font-weight: 800;
          text-align: center;
          align-self: center;
        }
        .insight-card {
          margin-bottom: 0.75rem;
        }
        .error-card {
          padding: 1.35rem 1.4rem;
          border-left: 10px solid #9B2226;
        }
        .footer-note {
          margin-top: 2rem;
          font-size: 0.95rem;
        }
        div[data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--muted);
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
          color: var(--ink);
        }
        div[role="radiogroup"] label p, .stCaption, [data-testid="stCaptionContainer"] {
          color: var(--muted) !important;
        }
        @media (max-width: 900px) {
          .hero-card {
            flex-direction: column;
          }
          .hero-status {
            align-items: flex-start;
          }
          .app-title {
            font-size: 2.3rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
