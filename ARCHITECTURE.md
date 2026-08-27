# ARCHITECTURE.md

Companion to `PROJECT_CONTRACT.md`. Describes how the system actually fits together. Agents read this before writing any module.

---

## 1. System diagram

```
                    ┌──────────────────────────────┐
                    │        Open-Meteo APIs       │
                    │  /v1/air-quality  (CAMS)     │
                    │  /v1/archive      (ERA5)     │
                    │  /v1/forecast     (weather)  │
                    └──────────────┬───────────────┘
                                   │ raw JSON
                                   ▼
                    ┌──────────────────────────────┐
                    │   src/data/open_meteo.py     │
                    │   retries · timeouts · tz    │
                    │   schema validation          │
                    └──────────────┬───────────────┘
                                   │ tidy DataFrame
                                   ▼
                    ┌──────────────────────────────┐
                    │  src/features/               │
                    │    build_features(df)        │  ← ONE function,
                    │    build_targets(df)         │    used by BOTH
                    └──────────────┬───────────────┘    backfill and hourly
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Hopsworks Feature Store    │
                    │   aqi_features_v1            │
                    │   aqi_targets_v1             │
                    │   → Feature View: aqi_fv_v1  │
                    └───────┬──────────────┬───────┘
                            │              │
              training data │              │ latest row
                            ▼              │
              ┌───────────────────────┐    │
              │ src/pipelines/train.py│    │
              │ baseline · ridge · rf │    │
              │ hgb · tf-mlp          │    │
              │ TimeSeriesSplit eval  │    │
              └───────────┬───────────┘    │
                          │ champion       │
                          ▼                │
              ┌───────────────────────┐    │
              │ Hopsworks Model Reg.  │    │
              │ pearls_aqi_forecaster │    │
              └───────────┬───────────┘    │
                          │                │
                          └────────┬───────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │   src/inference/predictor.py │
                    │   load champion + latest fx  │
                    │   → [day1, day2, day3]       │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │      src/api/main.py         │
                    │  /health /forecast           │
                    │  /model-info /history        │
                    │  /explanation                │
                    └──────────────┬───────────────┘
                                   │ JSON over HTTPS
                                   ▼
                    ┌──────────────────────────────┐
                    │     dashboard/app.py         │
                    │  KPIs · chart · SHAP · alert │
                    └──────────────────────────────┘

GitHub Actions
  ├── ci.yml              on push/PR      → ruff + pytest
  ├── hourly_features.yml cron "17 * * * *" → append latest features
  └── daily_training.yml  cron "37 3 * * *" → retrain, evaluate, maybe promote
```

## 2. Three pipelines, one feature function

The single most important architectural constraint: **`build_features()` is called identically by the backfill job and the hourly job.** There is no second implementation. This is what prevents training/serving skew, and it is the thing a reviewer should check first.

| Pipeline | Entry point | Trigger | Writes |
|---|---|---|---|
| Backfill | `src/pipelines/backfill.py` | manual, once (Day 3) | full history → both feature groups |
| Hourly features | `src/pipelines/hourly_features.py` | cron, hourly | last N hours → both feature groups (upsert) |
| Training | `src/pipelines/train.py` | cron, daily | Model Registry |

## 3. Feature groups

**`aqi_features_v1`**
- primary key: `city_id`
- event time: `event_time`
- payload: all raw + engineered predictor columns

**`aqi_targets_v1`**
- primary key: `city_id`
- event time: `event_time`
- payload: `target_aqi_day1`, `target_aqi_day2`, `target_aqi_day3`

Kept separate because targets are only knowable 72h after the fact. The hourly job writes features immediately and **backfills targets for rows that are now ≥72h old**. This is the one place where a subtle bug is likely — write a test for it.

**Feature View `aqi_fv_v1`** joins the two on (`city_id`, `event_time`) and is the only read path used by training.

## 4. Data source facts that constrain the design

- **Air quality history:** the `/v1/air-quality` endpoint's `past_days` parameter is capped at 92. Backfill must use `start_date` / `end_date`.
- **Coverage:** for Pakistan only the **CAMS global** domain applies — 0.4° (~45 km), natively 3-hourly, available from **August 2022**. The European 11 km / hourly reanalysis does not cover this region. Hourly values are interpolated. Say this in the report's Limitations section.
- **Weather history** comes from the separate `/v1/archive` endpoint (ERA5), which is hourly and goes back decades — it is the *air quality* side that bounds the dataset to ~4 years.
- **Timezone:** request with `timezone=UTC` and store `event_time` as UTC. Convert to `Asia/Karachi` **only** for calendar features and for display. Mixing these up is the classic silent bug here.
- **Licence:** free tier is non-commercial use. Attribution to CAMS + Open-Meteo goes in the README and the dashboard footer.

## 5. Modelling shape

Train **one model per horizon** (three fitted estimators) rather than a single `MultiOutputRegressor`.

Reason: SHAP's `TreeExplainer` does not accept a `MultiOutputRegressor` wrapper cleanly, and per-horizon models let you report honest per-horizon feature importance ("wind matters more for day 1, season matters more for day 3"), which is a far better slide than one blended chart. Cost is 3× fit time, which is negligible at this data size.

The TensorFlow MLP is the exception: it has a `Dense(3)` output head and is trained once.

## 6. Deployment

Primary (no credit card):
- Dashboard → Streamlit Community Cloud, pointed at the GitHub repo, secrets via its UI.
- API → Hugging Face Spaces (Docker SDK) running uvicorn.

Fallback if the API host misbehaves: Streamlit imports `predictor.py` directly and calls it in-process. The FastAPI service still exists in the repo and is still documented — you lose a deployment, not a deliverable.

Stretch: Cloud Run for both services. Requires an enabled billing account. Do not start this before Day 8 evening.

## 7. Secrets

```
HOPSWORKS_API_KEY
HOPSWORKS_PROJECT
AQI_API_BASE          (optional override)
```
Stored in GitHub repo secrets and in the host's secrets UI. `.env` is gitignored. `.env.example` is committed with empty values. No secret ever appears in a log line, a notebook output, or a screenshot in the report.
