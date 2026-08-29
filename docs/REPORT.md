# Pearls AQI Predictor — Final Report

**City:** Lahore, Pakistan (31.5497, 74.3436) · **Period:** 10-day build + feature-store swap · **Status:** MVP complete, hosting and automation deferred (see §19)

---

## 1. Executive summary

This project builds a reproducible, serverless air-quality forecasting system for one configurable city. It ingests roughly four years of hourly air-quality and weather history, engineers a locked, leakage-tested feature set, trains four candidate models per forecast horizon, selects a production champion using time-aware validation rather than a single lucky test split, explains that champion with model-specific SHAP, registers it in a live Hopsworks Model Registry, and serves it through a FastAPI backend and a Streamlit dashboard.

The registered champion — a Ridge regression model — reduces mean absolute error by **25.9% relative to a persistence baseline** (16.83 vs. 22.70 AQI points, averaged across the three forecast horizons), a result consistent with the target's known heavy autocorrelation and squarely inside the 20–40% realistic range set out in the project's own methodology, rather than an implausibly large "80% improvement" that would suggest leakage.

The one open piece is infrastructure, not modelling: both the Model Registry and the Feature Store are now genuinely live on Hopsworks; hourly/daily automation and public hosting were deliberately deferred rather than built on a shaky foundation. Both are scoped, now fully unblocked, and ready to complete once time permits (§19, §22).

## 2. Business problem

Air quality changes hour to hour, but the decisions it should inform — whether to schedule outdoor labor, whether to run a construction task, whether to proceed with an outdoor event — are made in advance, not in the moment. A single current reading tells you nothing about tomorrow. The system's job is to turn a noisy, autocorrelated pollution signal into a defensible 3-day forecast, with an honest sense of its own reliability, that a non-technical operator can read in under five seconds and act on.

## 3. Requirements

Per the frozen `PROJECT_CONTRACT.md`: forecast average US AQI at +24h, +48h, +72h for one configurable city; leakage-free features computable only from data at or before the forecast origin; multiple modelling approaches including a deep-learning baseline; a managed feature store and model registry; served predictions with categories and alerts; an explainability layer; automated hourly feature refresh and daily retraining; a dashboard. Full detail in `PROJECT_CONTRACT.md`.

## 4. Architecture

```
Open-Meteo (CAMS air quality + ERA5 weather)
        |
src/features/build_features.py, build_targets.py    <- one function, every pipeline
        |
Feature store   (aqi_features_v1, aqi_targets_v1, aqi_fv_v1)
        |                                    [local Parquet fallback - see status below]
src/pipelines/train.py
  Ridge · RandomForest · HistGradientBoosting · TensorFlow MLP
  rolling-origin validation -> champion selection -> single final-test evaluation
        |
Hopsworks Model Registry ("pearls_aqi_forecaster")     [LIVE]
        |
src/inference/predictor.py -> src/api/main.py (FastAPI) -> dashboard/app.py (Streamlit)
```

**As-built status per component**, since the intended and actual states currently differ in one place:

| Component | Status |
|---|---|
| Data ingestion (`src/data/open_meteo.py`) | Done, real ~4-year history |
| Feature engineering (`src/features/`) | Done, leakage-tested, contract-compliant (§8) |
| **Feature store** | **Local Parquet fallback** (`ADR-008`), not yet swapped to Hopsworks |
| Model training (`src/pipelines/train.py`) | Done, 4 candidates, rolling-origin validation |
| **Model Registry** | **Live on Hopsworks** (`ADR-010`), verified with a real registered version |
| Inference (`src/inference/predictor.py`) | Done, reads the live registry |
| API (`src/api/main.py`) | Done, tested, runs locally |
| Dashboard (`dashboard/app.py`) | Done, tested, runs locally |
| Hourly/daily automation | Not built - depends on the feature-store swap |
| Public hosting | Not deployed - depends on the above |
| CI (`.github/workflows/ci.yml`) | Live and green on every push |

## 5. Data sources

- **Air quality:** Open-Meteo Air Quality API, CAMS global reanalysis domain (`us_aqi`, PM2.5, PM10, CO, NO₂, SO₂, ozone, dust). For this region only the CAMS global domain applies — 0.4° (~45 km) resolution, natively 3-hourly, interpolated to hourly by the API, available from August 2022 (`ADR-002`).
- **Weather:** Open-Meteo Historical Weather API (ERA5) — temperature, humidity, precipitation, pressure, cloud cover, wind speed and direction. Hourly, available for decades; the air-quality side is what bounds the usable history.
- **License:** free tier, non-commercial use; CAMS and Open-Meteo attribution carried in the README and dashboard footer.

## 6. Data quality

Current local dataset: **35,712 hourly rows**, 2022-08-01 through 2026-08-27 (continuously refreshed during development). Missingness is very low overall — air-quality and pollutant columns show only trace nulls in the first few days of coverage (CAMS data starts fractionally after the requested range), weather columns are effectively complete. No synthetic filling is used anywhere; rolling and lag features naturally drop the small number of early rows lacking full context, and `build_targets` drops trailing rows whose forward-looking window isn't fully observed yet (never imputed).

## 7. Exploratory data analysis

Full notebook: `notebooks/01_eda.ipynb`. Ten charts, each with a finding, a candidate explanation, and a stated modelling implication. Summary of the findings that most directly shaped the feature set:

- **AQI is strongly autocorrelated and regime-like**, not white noise — multi-month waves, repeated winter peaks, sustained persistence. Confirms that lagged AQI and rolling summaries are essential, and that any shuffled validation split would leak these regimes.
- **Strong seasonality**, highest in winter (January), lowest in spring — supports cyclical month encoding over a raw integer.
- **Clear intra-day pattern**: flat overnight, rising sharply through late afternoon/evening — supports cyclical hour encoding.
- **Day-of-week effects exist but are small** relative to month and hour — included as a low-cost feature, not relied upon.
- **PM2.5 is the most direct physical driver** of AQI in this dataset — justifies PM2.5 lags, rolling means, and change features.
- **Wind speed is inversely, noisily related to AQI**; humidity/temperature have weaker secondary relationships — all three retained as smoothed 24h features rather than single raw points.
- **24-hour autocorrelation is strong and pollutant associations are consistent with a persistent atmospheric regime** — this is exactly why a persistence baseline is expected to be strong, and why beating it by 20–40% (not 80%) is the realistic, credible target (§11).

## 8. Feature engineering

One canonical function, `build_features()`, called identically by every pipeline (backfill, training, inference) — this is the single most important anti-skew guarantee in the system. All lag and rolling operations use closed-left windows (`.shift(1)` before `.rolling()`), so the current hour never leaks into its own aggregate.

**Locked feature set (46 columns fed to the model):** raw air quality (`us_aqi`, PM2.5, PM10, CO, NO₂, SO₂, ozone, dust), raw weather (temperature, humidity, precipitation, pressure, cloud cover, wind speed/direction), cyclical calendar features (hour, day-of-week, month, is-weekend), circular wind direction (sin/cos, never raw degrees), AQI/PM2.5 lags at {3,6,12,24,48,72}h / {6,24}h, rolling means and stds, and change-rate features.

Two features from an early draft (`aqi_lag_1h`, `pm25_lag_1h`, `aqi_change_1h`) were found still present in the code during a later review despite being explicitly excluded in the contract (a 1-hour lag is near-duplicate of the current value on 3-hourly source data). This was caught, fixed, and the entire pipeline retrained from a clean state — the fix changed final-test MAE by less than 0.1%, confirming the contract's own reasoning that those features were redundant, not load-bearing.

Leakage tests (`tests/test_features.py`, `tests/test_targets.py`) recompute features from truncated data and assert equality with the full-history computation, assert no feature correlates ≥0.999 with any target, and assert target columns never appear in the feature matrix.

## 9. Forecasting methodology

This is time-series forecasting from a forecast origin `t`, not tabular regression on shuffled rows. Given data at or before `t`, the model predicts three scalars: mean `us_aqi` over `t+1..t+24` (day 1), `t+25..t+48` (day 2), `t+49..t+72` (day 3). No `train_test_split(shuffle=True)` is used anywhere. The chronological split reserves the most recent ~15% of the timeline as a **final test set that is never touched during model selection** — only evaluated once, after the champion is chosen, for unbiased reporting (`ADR-009`).

Champion selection uses **4-fold rolling-origin (`TimeSeriesSplit`) validation on the pre-test timeline only**. This distinction mattered in practice: on the final test split alone, the TensorFlow MLP looked competitive (final-test `mae_mean` 17.16, close to Ridge's 16.83); under rolling-origin validation it was clearly the least reliable candidate (`selection_mae_mean` 23.91 ± 10.02, an order of magnitude more variance than the other three candidates). Selecting by validation rather than the single final-test score is what catches this.

## 10. Model experiments

Four candidates, one model per forecast horizon (three fitted estimators each) except the MLP, which uses a single `Dense(3)` head. Per-horizon models were chosen because SHAP's `TreeExplainer` doesn't cleanly accept a multi-output wrapper, and because separate models yield honest per-horizon feature importance (`ADR-004`).

- **Ridge** — `StandardScaler` → `Ridge`, small alpha grid, selected by validation MAE.
- **Random Forest** — 200 estimators, small `RandomizedSearchCV` with `TimeSeriesSplit`, capped search time.
- **HistGradientBoosting** — small fixed candidate grid, selected by validation MAE.
- **TensorFlow MLP** — `Dense(128, relu) → Dropout(0.2) → Dense(64, relu) → Dense(32, relu) → Dense(3)`, MAE loss, `EarlyStopping` + `ReduceLROnPlateau`, no extended training.

Two baselines are mandatory comparison rows: **persistence** (last observed AQI carried to all three horizons) and **seasonal-naive** (trailing 24h mean carried forward).

Per `ADR-005`, deep learning is included as a fairly-evaluated experiment, not manufactured into a winner. It did not win, on either the final test or validation view — a legitimate, expected result given ~35k rows of smoothed, strongly autocorrelated tabular data (§7, §9).

## 11. Model evaluation

**Final-test metrics** (held out, never used for selection):

| Model | D+1 MAE | D+2 MAE | D+3 MAE | Mean MAE | RMSE | R² |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 15.89 | 24.82 | 27.40 | 22.70 | 32.13 | 0.315 |
| Seasonal naive | 16.72 | 22.43 | 24.94 | 21.36 | 30.93 | 0.386 |
| Random Forest | 9.26 | 19.89 | 24.36 | 17.84 | 23.83 | 0.596 |
| TensorFlow MLP | 9.84 | 19.23 | 22.42 | 17.16 | 22.80 | 0.634 |
| HistGradientBoosting | 8.66 | 19.32 | 22.48 | 16.82 | 22.38 | 0.645 |
| **Ridge (champion)** | 9.57 | 19.08 | 21.82 | **16.83** | 22.80 | 0.634 |

**Validation metrics** (rolling-origin, 4 folds, pre-test timeline — this is what actually selected the champion):

| Model | Selection mean MAE | Selection std |
|---|---:|---:|
| **Ridge (champion)** | **18.07** | 1.21 |
| HistGradientBoosting | 18.32 | 1.23 |
| Random Forest | 18.66 | 1.71 |
| TensorFlow MLP | 23.91 | 10.02 |

Ridge and HistGradientBoosting are within noise of each other on both views (a 0.01-point final-test gap); Ridge's slightly better and more stable validation score made it the champion. This is a legitimate, unglamorous result: a linear model matching gradient boosting on smoothed, strongly autocorrelated tabular data.

**Champion improvement vs. persistence: (22.70 − 16.83) / 22.70 = 25.9%.** This sits inside the contract's own stated realistic range (20–40%), which is itself evidence against leakage — a leaked model would show an implausibly large improvement, not a modest, physically sensible one.

## 12. Explainability

SHAP is dispatched by champion model type — `TreeExplainer` for Random Forest/HistGradientBoosting, `LinearExplainer` for Ridge, `GradientExplainer` for the MLP — so explainability never constrains which model can win (`ADR-009`). For the current champion (Ridge), the top day-1 drivers by mean |SHAP value| are: `aqi_mean_72h`, `pm25_mean_24h`, `pm25_change_24h`, `us_aqi`, `ozone`, `aqi_change_24h`, `pm2_5`, `aqi_lag_6h`, `pm25_lag_6h` — consistent with the EDA's finding that PM2.5 and recent AQI history are the dominant drivers (§7). Per-horizon SHAP artifacts (summary plot, top-10 bar, importance CSV) are generated separately for day 1/2/3 and attached to the registered model version (§14).

## 13. Feature store

**Status: live on Hopsworks, verified with a real unmocked backfill** (`ADR-008`, `ADR-010`, `ADR-011`). `src/feature_store/store.py` was swapped from the Day 3 local-Parquet fallback to the real Hopsworks Feature Store behind identical function signatures — `backfill.py`, `train.py`, `predictor.py`, and `api/main.py` needed zero changes.

The Day 3 gate (rebuild the full training set from the Feature Store alone, no local cache) was re-run against the live project: `aqi_features_v1` holds 35,712 rows (2022-08-01 00:00 → 2026-08-27 23:00 UTC, 0 duplicate keys) — matching the local raw cache's row count exactly — `aqi_targets_v1` holds 35,545 rows, and the Feature View `aqi_fv_v1`'s materialized training set is 35,545 rows × 53 columns with 0 duplicate keys and 0 nulls across all three target columns.

Getting a genuine, unmocked pipeline run working took six real, non-cosmetic fixes — most notably that Hopsworks' default Feature View join is point-in-time-correct (matching each row to the *most recently available* value at or before its event_time), which would have silently reused stale target values for rows that shouldn't have known targets yet, a real leakage risk rather than a cosmetic mismatch. The actual training frame is built from a verified exact `(city_id, event_time)` inner merge of the two feature groups instead of trusting the Feature View's own read path. Full detail, including the free-tier statistics-computation flakiness and a too-short default Kafka timeout, is in `ADR-011`.

## 14. Model registry

**Status: live on Hopsworks, verified with a real registered version**, not just a login test. The current champion is registered at the project's Model Registry (version 1, model type `ridge`), with the ordered 46-feature list, both final-test and validation metrics, training timestamp, data date range, and all 9 SHAP artifacts attached as files alongside the 3 per-horizon model files.

Getting here required diagnosing and fixing four real issues, none of which were an actual Hopsworks outage (`ADR-010`): a Windows-incompatible default certificate path in the `hopsworks` SDK, a genuine bug in that SDK's own free-tier error handling, two missing API-key scopes (`DATASET_CREATE`, `MODELREGISTRY`), and a metadata payload that exceeded Hopsworks' `description` column length limit (fixed by moving the metadata into an uploaded manifest file instead). The champion rule matches `PROJECT_CONTRACT.md` §5 exactly: lowest validation `mae_mean`, tie-broken by validation std then final-test MAE — never "newest wins."

## 15. Automated pipelines

**Status: CI only.** `.github/workflows/ci.yml` runs `ruff` + the full 68-test suite (mocking all external services, including in-memory fake Hopsworks clients for both the Feature Store and the Model Registry — no credentials needed) on every push and PR to `main`, and is currently green. `hourly_features.yml` and `daily_training.yml` (per `ARCHITECTURE.md`'s design) are not yet built. They no longer have an infrastructure blocker — the Feature Store swap (§13) means state now persists across GitHub Actions' inherently ephemeral runners — this is simply not-yet-built scope.

## 16. FastAPI

`src/api/main.py` exposes `GET /health`, `GET /forecast`, `GET /model-info`, `GET /history?days=N`. `/forecast` loads the live champion from Hopsworks, asserts the inference feature order matches the registered feature list, clips predictions to `[0, 500]`, and attaches AQI category and alert level. It returns a clean `503` (never a stack trace) when the registry is empty or the latest local feature row is more than 48 hours old — verified directly with `TestClient` and with a running server against real, fresh data.

## 17. Streamlit dashboard

One page: header (city, last data update, last training time, champion version), an alert banner when any horizon crosses AQI 151, four KPI cards, a 7-day-history-plus-3-day-forecast Plotly chart with a visual break at the forecast origin, current pollutant/weather readings (PM2.5, PM10, O₃, NO₂, humidity, wind — read directly from the Hopsworks Feature Store), SHAP drivers with plain-language bullets, and a model-quality card. Verified live against a running FastAPI server, not just mocked tests.

## 18. Deployment

**Not yet deployed publicly.** The intended path (`ADR-007`) is Streamlit Community Cloud for the dashboard and Hugging Face Spaces (Docker SDK) for the API, both free and requiring no credit card. Both need the running application to reach a persistent feature/model store from wherever they're hosted — both the Model Registry (§14) and the Feature Store (§13) sides of that are now solved. `dashboard/app.py` already reads its API URL from an environment variable rather than a hardcoded `localhost`, so pointing it at a deployed API is a one-line configuration change once hosting is undertaken.

## 19. Limitations

- **No automated hourly/daily refresh.** Features and the champion model reflect whenever `backfill.py`/`train.py` were last run manually. No longer infrastructure-blocked (§13, §15) — just not yet built.
- **Not publicly hosted.** No live URL exists yet for the API or dashboard.
- **CAMS coverage for this region is 3-hourly, interpolated to hourly** by the data provider — a real resolution limit, not an artifact of this project's processing.
- **Weather-forecast features are excluded from v1 by design** (`ADR-006`) to avoid training/serving skew that would otherwise inflate validation numbers without a genuine accuracy gain.
- **Single city.** Multi-city support is explicitly out of scope for v1 per the contract.
- **The Hopsworks free-tier API key cannot use Hopsworks Model Serving** (`ADR-001`) — self-hosted FastAPI inference is the deliberate substitute, not a workaround to apologize for.

## 20. Cost and scalability

Every component used is on a free tier: Open-Meteo (non-commercial), Hopsworks Serverless Free (1 project, Feature Store + Model Registry, no card), GitHub Actions (free minutes on a public/private repo of this size), Streamlit Community Cloud and Hugging Face Spaces (both free, no card). The only real scaling constraint is Hopsworks' Model Registry free-tier storage and the free-tier absence of Model Serving — both acceptable at this project's scale (a handful of small model files, self-hosted inference).

## 21. Business applications

The forecast is designed to feed operational decisions via a `DATA → FORECAST → RISK → ACTION` framing across six categories: outdoor workforce exposure and shift scheduling, construction and site planning, logistics and last-mile routing, employee travel policy, outdoor event go/no-go decisions, and occupational health planning. Full decision matrices and a worked example (a Day+2 forecast of AQI 165 triggering specific actions) are in [`docs/operational_decision_support.md`](operational_decision_support.md).

## 22. Future improvements

In priority order: (1) hourly/daily automation (`hourly_features.yml`, `daily_training.yml`) now that both Hopsworks stores are live; (2) public hosting (Streamlit Community Cloud + Hugging Face Spaces per `ADR-007`); (3) archived historical weather-forecast features via Open-Meteo's Historical Forecast API, done correctly per `ADR-006`'s stated precondition; (4) multi-city support, after the single-city path is fully automated; (5) prediction uncertainty intervals; (6) drift detection comparing live prediction error against the registered validation metrics over time; (7) OpenAQ ground-sensor cross-validation as an independent data-quality check.

## 23. Conclusion

The core data-science claim of this project is real and defensible: a Ridge regression champion, selected by disciplined time-aware validation rather than a single lucky test split, reduces forecast error by 25.9% relative to a persistence baseline — a result that sits exactly where the project's own methodology predicted a genuine, non-leaked model should land. Getting that result required real engineering discipline along the way: catching a contract-violating feature set after the fact and re-verifying the fix changed almost nothing (§8), catching that a single-split evaluation would have made an unstable neural network look competitive (§9, §11), diagnosing four separate, non-obvious integration bugs to get a genuinely live Hopsworks Model Registry (§14, `ADR-010`), and diagnosing six more to get a genuinely live Feature Store without reintroducing the exact leakage this project spent Day 2 guarding against (§13, `ADR-011`). What remains — automation and public hosting — is well-scoped, not blocked on anything unresolved, and is the natural next increment on top of a foundation that is already correct.
