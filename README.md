# Pearls AQI Predictor

A reproducible, serverless AQI forecasting system for one configurable city. It ingests hourly weather and air-quality data, builds leakage-free time-series features, stores them in a Hopsworks-backed feature store, forecasts average US AQI for the next 24, 48, and 72 hours across four modeling approaches, registers the validation-selected champion in the Hopsworks Model Registry, serves predictions through FastAPI, and visualizes them in a Streamlit dashboard with SHAP explanations and alerts.

Configured city: **Karachi, Pakistan** (24.8608, 67.0104).

## What it does

- Fetches about 4 years of hourly air-quality (CAMS via Open-Meteo) and weather (ERA5) history for the configured city.
- Builds a locked, leakage-safe feature set with one canonical feature function shared by backfill, hourly refresh, training, and inference.
- Trains four candidate models per forecast horizon: Ridge, Random Forest, HistGradientBoosting, and a TensorFlow MLP.
- Selects the champion by time-aware rolling-origin validation MAE, never by final-test performance.
- Registers the champion model, ordered feature list, metrics, and SHAP artifacts in the Hopsworks Model Registry.
- Serves `/health`, `/forecast`, `/model-info`, `/history`, and `/predict-scenario` via FastAPI, clipping predictions to `[0, 500]` and attaching AQI categories and alert levels.
- Renders a tabbed Streamlit dashboard: live 3-day forecast with a gauge and history chart, model metrics with browsable SHAP visuals, current city conditions, an interactive scenario simulator (adjust pollutant/weather inputs and get a real recomputed forecast), and health guidance.
- Translates the forecast into operational decisions (workforce exposure, construction, logistics, travel, events, occupational health) — see [`docs/operational_decision_support.md`](docs/operational_decision_support.md).

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system diagram and [`docs/REPORT.md`](docs/REPORT.md) for the as-built report.

```text
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

Python 3.11 · pandas/NumPy · scikit-learn (Ridge, RandomForest, HistGradientBoosting) · TensorFlow/Keras · SHAP · Hopsworks (Feature Store + Model Registry) · FastAPI · Streamlit + Plotly · GitHub Actions (CI + automation) · pytest · ruff.

## Install

```bash
python -m venv .venv
pip install -r requirements.txt
```

Use Python 3.11.

## Environment variables

Copy `.env.example` to `.env` and fill in real values. Never commit `.env`.

```env
HOPSWORKS_API_KEY=
HOPSWORKS_PROJECT=
AQI_API_BASE=
```

`HOPSWORKS_API_KEY` needs `FEATURESTORE`, `DATASET_CREATE`, `DATASET_VIEW`, `PROJECT`, and `MODELREGISTRY` scopes. `dashboard/app.py` also reads `API_BASE_URL` and defaults to `http://localhost:8000`.

## Run backfill

```bash
python -m src.pipelines.backfill
```

## Run hourly refresh

Appends newly available feature rows and backfills targets only for rows now at least 72 hours old:

```bash
python -m src.pipelines.hourly_features
```

## Run training

Runs the full model-comparison pipeline and registers the champion immediately:

```bash
python -m src.pipelines.train
```

For the automated daily job path used by GitHub Actions:

```bash
python -c "from src.pipelines.train import run_daily_training_job; run_daily_training_job()"
```

That command trains a fresh candidate and only registers it if its validation selection metrics beat the incumbent champion.

## Run the API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/forecast
```

`/forecast` returns `503` if the latest feature row is more than 48 hours old or if no model is registered yet.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Run tests

```bash
pytest tests/ -q
ruff check src tests dashboard
```

The suite mocks all external services, including Open-Meteo HTTP calls and Hopsworks via in-memory fake clients for both the Feature Store and the Model Registry. No credentials or network access are required locally. CI in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request to `main`.

## Automation

- [`.github/workflows/hourly_features.yml`](.github/workflows/hourly_features.yml) runs hourly at minute `17` and supports `workflow_dispatch`.
- [`.github/workflows/daily_training.yml`](.github/workflows/daily_training.yml) runs daily at `03:37` UTC and supports `workflow_dispatch`.
- [`.github/workflows/keep_dashboard_awake.yml`](.github/workflows/keep_dashboard_awake.yml) pings the live dashboard every 6 hours so it never hits Streamlit Community Cloud's 12-hour inactivity sleep. No secrets required.
- The two pipeline workflows require `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` GitHub repository secrets.
- Both pipeline workflows are verified live, not just offline-tested: a real dispatch has grown the Feature Store and registered a new Model Registry version against the live project.

## Deployment

Publicly deployed and live:

- API: https://aqi-predictor-shine-4au0.onrender.com
- Dashboard: https://aqipredictor-samzzh.streamlit.app

The API is hosted on [Render](https://render.com) (free tier, no card required), building from [docker/Dockerfile.api](docker/Dockerfile.api). The dashboard is on Streamlit Community Cloud.

### Render API

1. Create a Web Service from this GitHub repo.
2. Runtime: **Docker**. Dockerfile Path: `docker/Dockerfile.api`. Root Directory: blank (build context is the repo root).
3. Instance Type: **Free**.
4. Environment Variables: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`.
5. Deploy. Render auto-redeploys on every push to `main`.

Local verification (requires the `docker` CLI, not available in every dev environment):

```bash
docker build -f docker/Dockerfile.api -t pearls-aqi-api .
docker run --rm -p 7860:7860 -e HOPSWORKS_API_KEY=... -e HOPSWORKS_PROJECT=... pearls-aqi-api
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/forecast
```

### Streamlit Community Cloud Dashboard

1. Create a new app from this GitHub repository, entrypoint `dashboard/app.py`.
2. Advanced settings: Python `3.11`.
3. Secrets: `API_BASE_URL=<render-api-url>` plus `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT`.
4. Streamlit Community Cloud also auto-redeploys on every push to `main`.

## Known limitations

- **Render's free tier spins down after 15 minutes of inactivity.** The first request after idle can take 50+ seconds to wake the instance; the dashboard's API request timeout is set accordingly (60s).
- **Hopsworks' free-tier Feature Query Service is occasionally slow independent of data volume.** Reads are filtered server-side by `city_id` (not transferred in full and discarded client-side), which cut a typical `aqi_features_v1` read from ~134s to ~30-60s after the Karachi backfill roughly doubled the feature group's row count — but an occasional slow Hopsworks response can still exceed Render's own gateway timeout, surfacing as an intermittent `502` on `/history` or `/forecast`. `src/feature_store/store.py::_read_with_retry` retries transient failures within the request; a hung request that Render's gateway kills before a retry completes will still show as a one-off failure to the client. Retrying the request from the browser resolves it.
- CAMS air-quality coverage for this region is 3-hourly and interpolated to hourly by Open-Meteo.
- The hourly refresh currently only upserts rows newer than the latest stored `event_time`; if Open-Meteo revises already-seen historical timestamps later, this pipeline will not replay those timestamps automatically.
- Weather-forecast features are intentionally excluded to avoid training/serving skew.

## Data attribution

Air quality: CAMS global reanalysis via [Open-Meteo](https://open-meteo.com/). Weather: ERA5 via Open-Meteo. Free tier, non-commercial use.

**Forecasts support planning; they are not direct measurements.**
