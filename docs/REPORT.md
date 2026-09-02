# Pearls AQI Predictor — Final Report

**City:** Karachi, Pakistan (24.8608, 67.0104) · **Status:** Complete and publicly deployed (§18).

---

## 1. Executive summary

This project builds a reproducible, serverless air-quality forecasting system for one configurable city. It ingests roughly four years of hourly air-quality and weather history, engineers a locked, leakage-tested feature set, trains four candidate models per forecast horizon, selects a production champion using time-aware validation rather than a single lucky test split, explains that champion with model-specific SHAP, registers it in a live Hopsworks Model Registry, and serves it through a FastAPI backend and a Streamlit dashboard — including an interactive scenario simulator that recomputes real predictions from user-adjusted pollutant and weather inputs.

The registered champion — a HistGradientBoosting model — reduces mean absolute error by **20.6% relative to a persistence baseline** (7.81 vs. 9.84 AQI points, averaged across the three forecast horizons), a result consistent with the target's known heavy autocorrelation and squarely inside the 20–40% realistic range this project targeted from the outset, rather than an implausibly large "80% improvement" that would suggest leakage.

Every major component — Feature Store, Model Registry, hourly refresh, daily retraining, and both public endpoints — is genuinely live, not just green on offline tests: verified with real dispatches against production infrastructure, not mocks.

## 2. Business problem

Air quality changes hour to hour, but the decisions it should inform — whether to schedule outdoor labor, whether to run a construction task, whether to proceed with an outdoor event — are made in advance, not in the moment. A single current reading tells you nothing about tomorrow. The system's job is to turn a noisy, autocorrelated pollution signal into a defensible 3-day forecast, with an honest sense of its own reliability, that a non-technical operator can read in under five seconds and act on.

## 3. Requirements

Forecast average US AQI at +24h, +48h, +72h for one configurable city, using only features computable from data at or before the forecast origin (no future information may leak into a feature). Multiple modelling approaches were required, including a deep-learning baseline evaluated fairly rather than manufactured into a winner. The system needed a managed feature store and model registry, served predictions with categories and alerts, an explainability layer, automated hourly feature refresh and daily retraining, and a dashboard usable by a non-technical operator.

## 4. Architecture

```
Open-Meteo (CAMS air quality + ERA5 weather)
        |
src/features/build_features.py, build_targets.py    <- one function, every pipeline
        |
Hopsworks Feature Store   (aqi_features_v1, aqi_targets_v1, aqi_fv_v1)
        |
src/pipelines/train.py
  Ridge · RandomForest · HistGradientBoosting · TensorFlow MLP
  rolling-origin validation -> champion selection -> single final-test evaluation
        |
Hopsworks Model Registry ("pearls_aqi_forecaster")
        |
src/inference/predictor.py -> src/api/main.py (FastAPI) -> dashboard/app.py (Streamlit)
```

| Component | Status |
|---|---|
| Data ingestion (`src/data/open_meteo.py`) | Live, ~4 years of real history |
| Feature engineering (`src/features/`) | Live, leakage-tested |
| Feature Store | Live on Hopsworks |
| Model training (`src/pipelines/train.py`) | Live, 4 candidates, rolling-origin validation |
| Model Registry | Live on Hopsworks |
| Inference (`src/inference/predictor.py`) | Live, reads the registry directly |
| API (`src/api/main.py`) | Live, publicly deployed |
| Dashboard (`dashboard/app.py`) | Live, publicly deployed |
| Hourly refresh / daily retraining | Live, scheduled GitHub Actions |
| CI (`.github/workflows/ci.yml`) | Live and green on every push |

## 5. Data sources

- **Air quality:** Open-Meteo Air Quality API, CAMS global reanalysis domain (`us_aqi`, PM2.5, PM10, CO, NO₂, SO₂, ozone, dust). For this region only the CAMS global domain applies — 0.4° (~45 km) resolution, natively 3-hourly and interpolated to hourly by the API, available from August 2022.
- **Weather:** Open-Meteo Historical Weather API (ERA5) — temperature, humidity, precipitation, pressure, cloud cover, wind speed and direction. Hourly, available for decades; the air-quality side is what bounds the usable history.
- **License:** free tier, non-commercial use; CAMS and Open-Meteo attribution is carried in the README and the dashboard footer.

## 6. Data quality

The Feature Store holds continuously growing hourly history for Karachi from 2022-08-08 onward (over 35,700 rows as of the last verified check). Missingness is very low overall — pollutant columns show only trace nulls in the first few days of coverage, weather columns are effectively complete. No synthetic filling is used anywhere; rolling and lag features naturally drop the small number of early rows lacking full context, and target-building drops trailing rows whose forward-looking window isn't fully observed yet, rather than imputing them.

## 7. Exploratory data analysis

Full notebook: `notebooks/01_eda.ipynb`. Ten charts, each with a finding, a candidate explanation, and a stated modelling implication. The findings below shaped the feature set; they were computed on the original single-city dataset the modelling pipeline was first built against, and the seasonal/diurnal patterns they describe are expected to hold qualitatively for another dense urban South Asian site, though the exact numbers reflect that original run.

- **AQI is strongly autocorrelated and regime-like**, not white noise — multi-month waves, repeated winter peaks, sustained persistence. Confirms that lagged AQI and rolling summaries are essential, and that any shuffled validation split would leak these regimes.
- **Strong seasonality**, highest in winter, lowest in spring — supports cyclical month encoding over a raw integer.
- **Clear intra-day pattern**: flat overnight, rising sharply through late afternoon/evening — supports cyclical hour encoding.
- **Day-of-week effects exist but are small** relative to month and hour — included as a low-cost feature, not relied upon.
- **PM2.5 is the most direct physical driver** of AQI in this dataset — justifies PM2.5 lags, rolling means, and change features.
- **Wind speed is inversely, noisily related to AQI**; humidity/temperature have weaker secondary relationships — all three retained as smoothed 24h features rather than single raw points.
- **24-hour autocorrelation is strong and pollutant associations are consistent with a persistent atmospheric regime** — this is exactly why a persistence baseline is expected to be strong, and why beating it by 20–40% (not 80%) is the realistic, credible target (§11).

## 8. Feature engineering

One canonical function, `build_features()`, called identically by every pipeline — backfill, training, inference, and the dashboard's scenario simulator — which is the single most important anti-skew guarantee in the system. All lag and rolling operations use closed-left windows (`.shift(1)` before `.rolling()`), so the current hour never leaks into its own aggregate.

**Locked feature set (46 columns fed to the model):** raw air quality (`us_aqi`, PM2.5, PM10, CO, NO₂, SO₂, ozone, dust), raw weather (temperature, humidity, precipitation, pressure, cloud cover, wind speed/direction), cyclical calendar features (hour, day-of-week, month, is-weekend), circular wind direction (sin/cos, never raw degrees), AQI/PM2.5 lags at {3,6,12,24,48,72}h / {6,24}h, rolling means and stds, and change-rate features. A 1-hour lag is deliberately excluded across the board: the underlying source data is natively 3-hourly, so a 1-hour lag would be a near-duplicate of the current value rather than genuinely new information.

An early draft still had that 1-hour lag present in a few columns despite the intent to exclude it. This was caught during review, removed, and the pipeline retrained from a clean state — the fix changed final-test MAE by less than 0.1%, confirming those columns were redundant, not load-bearing.

Leakage tests (`tests/test_features.py`, `tests/test_targets.py`) recompute features from truncated data and assert equality with the full-history computation, assert no feature correlates ≥0.999 with any target, and assert target columns never appear in the feature matrix.

## 9. Forecasting methodology

This is time-series forecasting from a forecast origin `t`, not tabular regression on shuffled rows. Given data at or before `t`, the model predicts three scalars: mean `us_aqi` over `t+1..t+24` (day 1), `t+25..t+48` (day 2), `t+49..t+72` (day 3). No `train_test_split(shuffle=True)` is used anywhere. The chronological split reserves the most recent ~15% of the timeline as a **final test set that is never touched during model selection** — only evaluated once, after the champion is chosen, for unbiased reporting.

Champion selection uses **4-fold rolling-origin (`TimeSeriesSplit`) validation on the pre-test timeline only**, not the final-test score. This distinction mattered in practice (§11): the TensorFlow MLP's final-test error looked competitive with the other candidates, but its rolling-origin validation error carried roughly five times the variance of any other candidate — an instability the final-test view alone would have hidden. Selecting by validation rather than a single held-out score is what catches this.

## 10. Model experiments

Four candidates, one model per forecast horizon (three fitted estimators each) except the MLP, which uses a single `Dense(3)` head. Per-horizon models were chosen because SHAP's `TreeExplainer` doesn't cleanly accept a multi-output wrapper, and because separate models yield honest per-horizon feature importance.

- **Ridge** — `StandardScaler` → `Ridge`, small alpha grid, selected by validation MAE.
- **Random Forest** — 200 estimators, small `RandomizedSearchCV` with `TimeSeriesSplit`, capped search time.
- **HistGradientBoosting** — small fixed candidate grid, selected by validation MAE.
- **TensorFlow MLP** — `Dense(128, relu) → Dropout(0.2) → Dense(64, relu) → Dense(32, relu) → Dense(3)`, MAE loss, `EarlyStopping` + `ReduceLROnPlateau`, no extended training.

Two baselines are mandatory comparison rows: **persistence** (last observed AQI carried to all three horizons) and **seasonal-naive** (trailing 24h mean carried forward).

Deep learning was included as a fairly-evaluated experiment, not manufactured into a winner. It did not win on the validation view that actually selects the champion — a legitimate, expected result given ~35k rows of smoothed, strongly autocorrelated tabular data (§7, §9), where tree- and linear-model inductive biases suit the signal better than a neural network needs to.

## 11. Model evaluation

**Final-test metrics** (held out, never used for selection):

| Model | D+1 MAE | D+2 MAE | D+3 MAE | Mean MAE | RMSE | R² |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 5.92 | 10.97 | 12.62 | 9.84 | 13.75 | 0.050 |
| Seasonal naive | 8.01 | 11.08 | 12.39 | 10.49 | 14.31 | 0.008 |
| Random Forest | 3.91 | 9.26 | 10.58 | 7.92 | 11.25 | 0.326 |
| Ridge | 3.93 | 9.13 | 10.59 | 7.88 | 10.71 | 0.393 |
| **HistGradientBoosting (champion)** | 3.74 | 9.20 | 10.50 | **7.81** | 11.19 | 0.330 |
| TensorFlow MLP | 4.45 | 8.72 | 9.92 | 7.70 | 10.89 | 0.391 |

**Validation metrics** (rolling-origin, 4 folds, pre-test timeline — this is what actually selected the champion):

| Model | Selection mean MAE | Selection std |
|---|---:|---:|
| **HistGradientBoosting (champion)** | **11.89** | 1.88 |
| Ridge | 12.40 | 2.76 |
| Random Forest | 12.43 | 2.14 |
| TensorFlow MLP | 15.30 | 9.32 |

The TensorFlow MLP shows exactly why final-test alone is the wrong way to pick a champion: on the single final-test split it has the lowest mean MAE of all four candidates (7.70), which would make it look like the best model. Its validation std (9.32) is roughly 3–5x every other candidate's, meaning that final-test result was one favorable split away from a much worse one. HistGradientBoosting's validation score is both the lowest mean and among the lowest variance — the more trustworthy basis for a production choice, and the one this project's champion rule actually uses.

**Champion improvement vs. persistence: (9.84 − 7.81) / 9.84 = 20.6%.** This sits inside the realistic 20–40% range this project targeted from the outset — evidence against leakage, since a leaked model would show an implausibly large improvement, not a modest, physically sensible one.

## 12. Explainability

SHAP is dispatched by champion model type — `TreeExplainer` for Random Forest/HistGradientBoosting, `LinearExplainer` for Ridge, `GradientExplainer` for the MLP — so explainability never constrains which model can win. For the current champion (HistGradientBoosting), the top day-1 drivers by mean |SHAP value| are: `pm25_mean_24h`, `pm2_5`, `us_aqi`, `pm25_change_24h`, `pm25_lag_6h`, `sulphur_dioxide`, `wind_speed_10m`, `wind_mean_24h`, `aqi_std_72h`, `wind_dir_sin` — consistent with the EDA's finding that PM2.5 and recent AQI history are the dominant drivers (§7). Per-horizon SHAP artifacts (summary plot, top-10 bar, importance CSV) are generated separately for day 1/2/3 and attached to the registered model version (§14), and are also browsable live in the dashboard's Model Metrics tab.

## 13. Feature store

**Status: live on Hopsworks, verified with a real unmocked backfill.** `src/feature_store/store.py` reads and writes the real Hopsworks Feature Store behind stable function signatures shared by `backfill.py`, `train.py`, `predictor.py`, and `api/main.py`. Reads are filtered server-side by `city_id` — the Feature Store holds Karachi's history in the same feature groups as an earlier city's, and pushing the filter into the query rather than discarding rows client-side keeps reads fast as that shared history grows.

The training frame is built from a verified exact `(city_id, event_time)` inner merge of the features and targets feature groups, rather than trusting Hopsworks' own default Feature View join — that default is point-in-time-correct (matching each row to the *most recently available* value at or before its event_time), which would silently reuse stale target values for rows that shouldn't have known targets yet. A real leakage risk, not a cosmetic mismatch, and one worth naming plainly rather than glossing over.

## 14. Model registry

**Status: live on Hopsworks, verified with a real registered version**, not just a login test. The current live champion is `hist_gradient_boosting`, with the ordered 46-feature list, both final-test and validation metrics, training timestamp, data date range, and SHAP artifacts attached as files alongside the per-horizon model files. Champion selection reads the registry directly and is scoped to the configured city, so a version trained for a different city can never be silently served — this matters concretely once more than one city's models exist in the same registry, as they do here. The daily retraining job registers a fresh candidate and only promotes it over the current champion if it beats it on validation MAE; a newer version never wins by default.

## 15. Automated pipelines

**Status: live, verified with real dispatches, not just green offline tests.** `.github/workflows/ci.yml` runs `ruff` plus the full test suite (mocking all external services, including in-memory fake Hopsworks clients for both the Feature Store and the Model Registry — no credentials needed) on every push and PR to `main`. `hourly_features.yml` (cron `17 * * * *`) and `daily_training.yml` (cron `37 3 * * *`), both with manual-dispatch support, have each had real successful runs against the live project: the hourly job grows the Feature Store with each newly available hour, and the daily job evaluates a fresh candidate against the live incumbent and promotes it only when it's genuinely better.

Live dispatch surfaced real integration issues that no offline test caught, because offline fakes don't reproduce what a live Hopsworks feature-group schema or a live Open-Meteo response actually look like: a missing transitive dependency, and a feature-group schema mismatch that could break in either direction depending on which columns a given live fetch happened to contain fractional values for that hour. The fix conforms every insert to the feature group's actual live-registered schema rather than guessing a dtype at the ingestion layer — both a real bug and a reminder that offline-green is necessary but not sufficient for a system this dependent on a third-party service's real behavior.

## 16. FastAPI

`src/api/main.py` exposes `GET /health`, `GET /forecast`, `GET /model-info`, `GET /history?days=N`, and `POST /predict-scenario`. `/forecast` loads the live champion from Hopsworks, asserts the inference feature order matches the registered feature list, clips predictions to `[0, 500]`, and attaches AQI category and alert level. `/predict-scenario` accepts overrides for up to eight raw pollutant/weather readings, splices them into real recent history, and recomputes a forecast through the exact same `build_features()` function training and normal inference use — never a separate, hand-rolled approximation. Every endpoint returns a clean `503` with a real error message (never a stack trace) when a backend dependency fails, verified with `TestClient` and against the live deployed API.

## 17. Streamlit dashboard

A tabbed single-page dashboard: a hero card with the configured city, champion version, training/data-currency timestamps, and a data-driven one-line summary generated from the live forecast (trend direction, top SHAP driver, alert status — never a fabricated line). Five tabs: **Live & 3-Day Forecast** (gauge, KPI cards, history-plus-forecast chart with a visual break at the forecast origin), **Model Metrics & SHAP** (champion metrics card, browsable SHAP visualizations), **City Data Insights** (current pollutant/weather readings read live from the Feature Store), **Custom Scenario Simulator** (adjust pollutant/weather sliders and get a real recomputed 3-day forecast via `/predict-scenario`), and **Health Guidelines** (category-by-category guidance plus an alert banner when any horizon crosses AQI 151). Verified live against the deployed FastAPI backend, not just mocked tests.

## 18. Deployment

**Publicly deployed and live:**

- API: https://aqi-predictor-shine-4au0.onrender.com
- Dashboard: https://aqipredictor-samzzh.streamlit.app

The API runs on [Render](https://render.com) (free tier, no card required), building from `docker/Dockerfile.api`. The dashboard runs on Streamlit Community Cloud. Both platforms auto-redeploy on every push to `main`. Both the Model Registry (§14) and the Feature Store (§13) are live and remotely reachable; the API and dashboard read from them directly at request time, not from any local cache.

Hosting artifacts:

- `docker/Dockerfile.api` runs `uvicorn src.api.main:app --host 0.0.0.0 --port 7860`.
- `.dockerignore` excludes local-only content (`data/`, `notebooks/`, `.venv/`, `tests/`, `docs/`, `.git/`) so the API image does not ship raw caches or the local development environment.
- `dashboard/app.py` reads `API_BASE_URL` from the environment.

## 19. Limitations

- **Render's free tier spins down after 15 minutes of inactivity.** The first request after idle can take 50+ seconds to wake the instance; the dashboard's API request timeout accounts for this.
- **Hopsworks' free-tier Feature Query Service is occasionally slow independent of data volume.** Reads are filtered server-side by `city_id` rather than transferred in full and discarded client-side, which meaningfully cut typical read latency — but an occasional slow Hopsworks response can still exceed Render's own gateway timeout, surfacing as an intermittent error on a request. Retrying resolves it; this is a free-tier characteristic, not a code defect. Login itself, not just reads, has the same retry logic for the same reason.
- **The hourly refresh only inserts rows newer than the latest stored `event_time`.** If Open-Meteo later revises an already-seen historical reading, this pipeline won't replay and re-upsert it automatically.
- **CAMS coverage for this region is 3-hourly, interpolated to hourly** by the data provider — a real resolution limit, not an artifact of this project's processing.
- **Weather-forecast features are excluded by design** to avoid training/serving skew that would otherwise inflate validation numbers without a genuine accuracy gain.
- **The Feature Store and Model Registry hold more than one city's data in the same shared feature groups and model name.** Reads and champion selection are correctly scoped by city, so this is a storage detail rather than a correctness risk — but a genuine multi-city *product* (serving more than one city simultaneously through the API/dashboard) is still out of scope.
- **The Hopsworks free-tier API key cannot use Hopsworks Model Serving** — self-hosted FastAPI inference is the deliberate substitute, not a workaround to apologize for.

## 20. Cost and scalability

Every component used is on a free tier: Open-Meteo (non-commercial), Hopsworks Serverless Free (Feature Store + Model Registry, no card), GitHub Actions (free minutes on a repo of this size), Streamlit Community Cloud (free, no card), and Render (free, no card). The real scaling constraints are Hopsworks' Model Registry free-tier storage and the free-tier absence of Model Serving (both acceptable at this project's scale), plus Render's free-tier spin-down and Hopsworks' free-tier query-service latency (§19).

## 21. Business applications

The forecast is designed to feed operational decisions via a `DATA → FORECAST → RISK → ACTION` framing across six categories: outdoor workforce exposure and shift scheduling, construction and site planning, logistics and last-mile routing, employee travel policy, outdoor event go/no-go decisions, and occupational health planning. Full decision matrices and a worked example (a Day+2 forecast of AQI 165 triggering specific actions) are in [`docs/operational_decision_support.md`](operational_decision_support.md).

## 22. Future improvements

In priority order: (1) archived historical weather-forecast features via Open-Meteo's Historical Forecast API, done in a way that keeps training and serving features identical; (2) a real multi-city *product* (serving more than one city simultaneously through the API/dashboard, not just correctly isolated in storage); (3) prediction uncertainty intervals; (4) drift detection comparing live prediction error against the registered validation metrics over time; (5) OpenAQ ground-sensor cross-validation as an independent data-quality check; (6) revision-aware hourly refresh (§19) so a later Open-Meteo correction to an already-seen reading gets replayed, not permanently missed.

## 23. Conclusion

The core data-science claim of this project is real and defensible: a HistGradientBoosting champion, selected by disciplined time-aware validation rather than a single lucky test split, reduces forecast error by 20.6% relative to a persistence baseline — a result that sits exactly where the project's own methodology predicted a genuine, non-leaked model should land. Getting that result required real engineering discipline along the way: catching a redundant near-duplicate feature after the fact and re-verifying the fix changed almost nothing (§8), catching that a single-split evaluation would have made an unstable neural network look like the best candidate (§9, §11), getting a genuinely live Hopsworks Model Registry and Feature Store working against real, unmocked infrastructure rather than settling for green offline tests (§13, §14), and getting the hourly/daily automation to survive a real live dispatch, not just an offline-green checkmark (§15). The system is fully built, fully live, and publicly deployed, with an interactive scenario simulator on top that lets anyone explore how changing pollutant conditions would move the forecast — built on the same feature pipeline the live model actually runs on, not a separate approximation.
