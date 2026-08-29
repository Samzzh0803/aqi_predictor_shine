# Pearls AQI Predictor

A reproducible, serverless AQI forecasting system for one configurable city. Ingests hourly weather and air-quality data, builds leakage-free time-series features, stores them in a Hopsworks-backed feature store, forecasts average US AQI for the next 24, 48, and 72 hours across four modelling approaches, registers the validation-selected champion in the Hopsworks Model Registry, serves predictions through FastAPI, and visualizes them in a Streamlit dashboard with SHAP explanations and alerts.

Configured city: **Lahore, Pakistan** (31.5497, 74.3436).

## What it does

- Fetches ~4 years of hourly air-quality (CAMS via Open-Meteo) and weather (ERA5) history for the configured city.
- Builds a locked, leakage-safe feature set (calendar cyclics, lags, rolling windows, change rates) with one canonical function shared by every pipeline.
- Trains four candidate models per forecast horizon (Ridge, Random Forest, HistGradientBoosting, a TensorFlow MLP) and selects the champion by **time-aware rolling-origin validation MAE**, never by final-test performance.
- Explains the champion with model-specific SHAP (`TreeExplainer` for trees, `LinearExplainer` for Ridge, `GradientExplainer` for the MLP).
- Registers the champion — model files, ordered feature list, metrics, SHAP artifacts, training date range — in the **Hopsworks Model Registry**.
- Serves `/health`, `/forecast`, `/model-info`, `/history` via FastAPI, clipping predictions to `[0, 500]` and attaching US EPA AQI categories and alert levels.
- Renders a one-page Streamlit dashboard: KPI cards, a 7-day-history-plus-3-day-forecast chart, current pollutant/weather readings, SHAP drivers, model-quality card, and an alert banner.
- Translates the forecast into operational decisions (workforce exposure, construction, logistics, travel, events, occupational health) — see [`docs/operational_decision_support.md`](docs/operational_decision_support.md).

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system diagram and [`docs/REPORT.md`](docs/REPORT.md) §4 for the as-built version with current status per component.

```
Open-Meteo (CAMS + ERA5)
        |
src/features/build_features.py, build_targets.py   <- one function, every pipeline
        |
Hopsworks Feature Store   (aqi_features_v1, aqi_targets_v1, aqi_fv_v1)
        |
src/pipelines/train.py   <- Ridge, RandomForest, HistGradientBoosting, TensorFlow MLP
        |                     champion selected by rolling-origin validation MAE
Hopsworks Model Registry  (pearls_aqi_forecaster)
        |
src/inference/predictor.py  ->  src/api/main.py (FastAPI)  ->  dashboard/app.py (Streamlit)
```

## Locked stack

Python 3.11 · pandas/NumPy · scikit-learn (Ridge, RandomForest, HistGradientBoosting) · TensorFlow/Keras · SHAP · Hopsworks (Feature Store + Model Registry) · FastAPI · Streamlit + Plotly · GitHub Actions (CI) · pytest · ruff.

See [`PROJECT_CONTRACT.md`](PROJECT_CONTRACT.md) for the frozen objective, feature list, and metrics definitions, and [`DECISIONS.md`](DECISIONS.md) for why each of these was chosen over the alternatives.

## Install

```bash
python -m venv .venv          # Python 3.11
# or use the project's conda env if you have one
pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` and fill in real values. Never commit `.env`.

```
HOPSWORKS_API_KEY=       # from Hopsworks: profile -> Settings -> API Keys
                          # needs FEATURESTORE, DATASET_CREATE, DATASET_VIEW,
                          # PROJECT, MODELREGISTRY scopes (not SERVING - free
                          # tier doesn't grant it, and this project doesn't need it)
HOPSWORKS_PROJECT=       # your Hopsworks project name
AQI_API_BASE=            # optional; leave blank to use Open-Meteo defaults
```

`dashboard/app.py` also reads `API_BASE_URL` (defaults to `http://localhost:8000`) — set it to a deployed API's URL when hosting the dashboard separately from the API.

## Run backfill

Fetches full history and populates the Hopsworks Feature Store (`src/feature_store/store.py`; `aqi_features_v1`, `aqi_targets_v1`, `aqi_fv_v1`):

```bash
python -m src.pipelines.backfill
```

## Run training

Runs the full Day 4/5 pipeline: baselines, four candidate models, rolling-origin validation, champion selection, SHAP generation, and registration into the live Hopsworks Model Registry.

```bash
python -m src.pipelines.train
```

Outputs land in `data/metrics/` (comparison tables, rolling-validation summary, SHAP artifacts) and the champion is registered at your Hopsworks project's Model Registry page.

## Run the API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/forecast
```

`/forecast` returns `503` if the latest feature row is more than 48 hours old (run backfill again to refresh) or if no model is registered yet.

## Run the dashboard

With the API running (see above):

```bash
streamlit run dashboard/app.py
```

## Run tests

```bash
pytest tests/ -q
ruff check src tests dashboard
```

The suite (68 tests) mocks all external services (Open-Meteo HTTP calls, Hopsworks via in-memory fake clients for both the Feature Store and the Model Registry) — no credentials or network access are required to run it. CI (`.github/workflows/ci.yml`) runs this on every push/PR to `main`.

## Deployment

Not yet deployed publicly. Per `ADR-007`, the intended path is Streamlit Community Cloud (dashboard) + Hugging Face Spaces, Docker SDK (API), both free and requiring no credit card. This was deferred alongside hourly/daily automation — see [Known limitations](#known-limitations).

## Known limitations

- **No hourly/daily automation yet.** `hourly_features.yml` and `daily_training.yml` (referenced in `ARCHITECTURE.md`) don't exist yet.
- **Not publicly deployed.** See Deployment above.
- **CAMS air-quality coverage for this region is 3-hourly**, interpolated to hourly by Open-Meteo; see `ADR-002`.
- **Weather-forecast features are intentionally excluded from v1** to avoid training/serving skew — see `ADR-006`.

## Data attribution

Air quality: CAMS global reanalysis via [Open-Meteo](https://open-meteo.com/). Weather: ERA5 via Open-Meteo. Free tier, non-commercial use.

**Forecasts support planning; they are not direct measurements.**
