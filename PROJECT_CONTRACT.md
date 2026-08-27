# PROJECT_CONTRACT.md

**Status: FROZEN.** This file is the law of the project. No agent (Claude Code, Codex, or any other) may change the objective, stack, feature set, or target definition. If a change is genuinely required, the human owner edits this file and records an ADR in `DECISIONS.md`. Until then, this document wins over any agent's opinion about "a better architecture."

---

## 1. The objective (verbatim, non-negotiable)

> Build a reproducible, serverless AQI forecasting system for one configurable city that ingests hourly weather and air-quality data, generates leakage-free time-series features, stores them in Hopsworks Feature Store, forecasts average US AQI for the next 24, 48 and 72 hours using multiple ML approaches, registers the best model in Hopsworks Model Registry, exposes predictions through FastAPI, visualizes them through Streamlit with SHAP explanations and AQI alerts, and automatically updates features hourly and retrains daily using GitHub Actions.

Deadline: **10 working days**, ~6 focused hours per day.

## 2. Locked stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.11 | Pinned. Do not use 3.12+ without checking TensorFlow wheels. |
| AQI + pollutants | Open-Meteo Air Quality API | Free, no key for non-commercial use |
| Weather (history) | Open-Meteo Historical Weather API | Same provider, consistent schema |
| Weather (forecast) | Open-Meteo Forecast API | Used at inference only unless ADR-006 is accepted |
| Validation source | OpenAQ v3 | **Optional**, Day 9 only |
| Processing | pandas, NumPy | |
| Feature Store | **Hopsworks Serverless (Free tier)** | 1 project, Feature Store + Model Registry, no credit card |
| Model Registry | **Hopsworks Model Registry** | Same platform |
| Model serving | **FastAPI (self-hosted)** | Hopsworks Model Serving is NOT on the free tier. Do not attempt it. |
| Classical models | scikit-learn: Ridge, RandomForest, HistGradientBoosting | |
| Deep model | TensorFlow/Keras tabular MLP | LSTM is stretch only |
| Explainability | SHAP | Model-specific SHAP explainer chosen by champion model type |
| Orchestration | **GitHub Actions** | Not Airflow. See ADR-003. |
| Frontend | Streamlit + Plotly | |
| Hosting (primary) | Streamlit Community Cloud (dashboard) + Hugging Face Spaces (API) | No credit card required |
| Hosting (stretch) | Google Cloud Run | Requires billing account; Day 8 stretch only |
| Testing | pytest | |
| Lint | ruff | |
| VCS | Git / GitHub | |

**Forbidden without an ADR:** Vertex AI Feature Store, Airflow, Kubernetes, Docker Compose orchestration, a database, user authentication, multi-city support before Day 9, any model not listed above.

## 3. The forecasting problem (exact definition)

This is **time-series forecasting from a forecast origin `t`**, not tabular regression on shuffled rows.

Given all information available **at or before hour `t`**, predict three scalars:

| Target | Definition |
|---|---|
| `target_aqi_day1` | mean `us_aqi` over `t+1 … t+24` hours |
| `target_aqi_day2` | mean `us_aqi` over `t+25 … t+48` hours |
| `target_aqi_day3` | mean `us_aqi` over `t+49 … t+72` hours |

Model output shape: `[day1, day2, day3]`.

**Hard rules:**
- No `train_test_split(shuffle=True)`. Ever. Chronological split only.
- Validation uses `TimeSeriesSplit` or rolling-origin. Test set is the **most recent** contiguous block (last ~15% of time).
- Every feature for row `t` must be computable from data timestamped `<= t`. Targets may look forward; features may not.
- Rows near the end of the series where targets are incomplete must be dropped, not imputed.

### Known property of the target (do not "fix" this)
US AQI for PM2.5 is computed from a **24-hour rolling average** of concentrations, and CO from an 8-hour average. Our targets are then 24-hour means of that index. The target is therefore heavily smoothed and **highly autocorrelated**. Consequences that are expected and acceptable:
- Persistence baselines will be strong.
- R² will look high; this is not evidence of a good model on its own.
- A realistic win is **20–40% MAE reduction vs persistence**, not 80%.
- Report improvement *relative to persistence*, never raw MAE alone.

## 4. Locked feature set

Agents may not add features without recording them in `DECISIONS.md` with a stated reason.

**Raw air quality:** `us_aqi`, `pm2_5`, `pm10`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `dust`

**Raw weather:** `temperature_2m`, `relative_humidity_2m`, `precipitation`, `surface_pressure`, `cloud_cover`, `wind_speed_10m`, `wind_direction_10m`

**Metadata / keys:** `city_id`, `latitude`, `longitude`, `event_time` (UTC, tz-aware)

**Calendar (cyclical, computed in *local* city time):** `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `is_weekend`

**Wind direction is circular** — encode as `wind_dir_sin`, `wind_dir_cos`, never as raw degrees.

**Lags:** `aqi_lag_{3,6,12,24,48,72}h`, `pm25_lag_{6,24}h`
> Note: source data for the global CAMS domain is natively **3-hourly**. A 1-hour lag is near-duplicate of the current value. `aqi_lag_1h` and `pm25_lag_1h` are **dropped** from the original draft for this reason.

**Rolling:** `aqi_mean_{3,6,12,24,72}h`, `aqi_std_{24,72}h`, `pm25_mean_24h`, `temperature_mean_24h`, `humidity_mean_24h`, `wind_mean_24h`

**Change rate (explicitly required by the brief):** `aqi_change_6h`, `aqi_change_24h`, `pm25_change_24h`

All rolling/lag operations use **closed-left windows** (`.shift(1)` before `.rolling()`) so the current hour never leaks into its own aggregate.

## 5. Metrics and champion rule

Report per horizon and averaged: **MAE, RMSE, R²**.

- **Primary selection metric:** `selection_mae_mean` = mean of MAE across day1/day2/day3 on time-aware validation data.
- **Tie-breaker:** `rmse_mean`.
- **Diagnostic only:** R².
- **Mandatory comparison row:** persistence baseline. A model that does not beat persistence is reported as not beating persistence.

**Champion selection rule**
- The champion is the candidate model with the lowest mean MAE across the three forecast horizons on time-aware validation data.
- The final chronological test set is not used for model selection. It is evaluated once after champion selection for unbiased reporting.
- Explainability is model-specific and must not constrain champion selection.
- Newest does **not** automatically win. The daily training job registers a candidate and only promotes it if it beats the incumbent.

## 6. AQI categories and alerts (US EPA bands)

```
0–50      Good
51–100    Moderate
101–150   Unhealthy for Sensitive Groups
151–200   Unhealthy
201–300   Very Unhealthy
301–500   Hazardous
```

Alert thresholds on any predicted horizon:
```
warning   >= 151
critical  >= 201
hazardous >= 301
```
Predictions are clipped to `[0, 500]` before display.

## 7. MVP vs stretch

**Must be complete and working by end of Day 8:**
historical backfill · EDA · feature engineering · Hopsworks Feature Store · Ridge · Random Forest · HistGradientBoosting · TensorFlow MLP · persistence baseline · MAE/RMSE/R² table · Model Registry · 3-day prediction · FastAPI · Streamlit · SHAP · alerts · hourly GitHub Action · daily GitHub Action · deployed dashboard.

**Only after every item above works:**
LSTM · OpenAQ cross-validation · multiple cities · uncertainty intervals · weather-forecast features (ADR-006) · Cloud Run · email/SMS alerts · map view · drift detection.

## 8. Non-goals

Not building: user accounts, auth, a database, a mobile app, real-time streaming, a custom AQI formula, model monitoring dashboards, or anything requiring a credit card.
