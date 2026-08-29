# Pearls AQI Predictor

A reproducible, serverless AQI forecasting system for one configurable city. It ingests hourly weather and air-quality data, builds leakage-free time-series features, stores them in a Hopsworks-backed feature store, forecasts average US AQI for the next 24, 48, and 72 hours across four modeling approaches, registers the validation-selected champion in the Hopsworks Model Registry, serves predictions through FastAPI, and visualizes them in a Streamlit dashboard with SHAP explanations and alerts.

Configured city: **Lahore, Pakistan** (31.5497, 74.3436).

## What it does

- Fetches about 4 years of hourly air-quality (CAMS via Open-Meteo) and weather (ERA5) history for the configured city.
- Builds a locked, leakage-safe feature set with one canonical feature function shared by backfill, hourly refresh, training, and inference.
- Trains four candidate models per forecast horizon: Ridge, Random Forest, HistGradientBoosting, and a TensorFlow MLP.
- Selects the champion by time-aware rolling-origin validation MAE, never by final-test performance.
- Registers the champion model, ordered feature list, metrics, and SHAP artifacts in the Hopsworks Model Registry.
- Serves `/health`, `/forecast`, `/model-info`, and `/history` via FastAPI, clipping predictions to `[0, 500]` and attaching AQI categories and alert levels.
- Renders a one-page Streamlit dashboard with KPI cards, a history-plus-forecast chart, current pollutant/weather readings, SHAP drivers, a model-quality card, and alerts.
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

See [`PROJECT_CONTRACT.md`](PROJECT_CONTRACT.md) for the frozen objective, feature list, and metrics definitions, and [`DECISIONS.md`](DECISIONS.md) for why each of these was chosen over the alternatives.

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

Runs the full Day 4/5 pipeline and registers the champion immediately:

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
- Both workflows require `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` GitHub repository secrets.

## Deployment

Not yet deployed publicly. Per `ADR-007`, the intended path is Streamlit Community Cloud for the dashboard and Hugging Face Spaces for the API.

## Known limitations

- Not publicly deployed yet.
- CAMS air-quality coverage for this region is 3-hourly and interpolated to hourly by Open-Meteo; see `ADR-002`.
- The hourly refresh currently only upserts rows newer than the latest stored `event_time`; if Open-Meteo revises already-seen historical timestamps later, this pipeline will not replay those timestamps automatically.
- Weather-forecast features are intentionally excluded from v1 to avoid training/serving skew; see `ADR-006`.

## Data attribution

Air quality: CAMS global reanalysis via [Open-Meteo](https://open-meteo.com/). Weather: ERA5 via Open-Meteo. Free tier, non-commercial use.

**Forecasts support planning; they are not direct measurements.**
