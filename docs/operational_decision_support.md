# Operational Decision Support

## Purpose

This section translates the AQI forecast into practical operational decisions for a single configured city. The goal is not just to predict air quality, but to support safer and more defensible planning for outdoor work, logistics, travel, and events.

Decision framing:

`DATA -> FORECAST -> RISK -> ACTION`

- `DATA`: hourly AQI, pollutant, and weather features from Open-Meteo
- `FORECAST`: predicted average US AQI for Day +1, Day +2, and Day +3
- `RISK`: health and operational exposure implied by the AQI band
- `ACTION`: concrete operational response

## AQI bands used by the system

| AQI range | Category | Operational meaning |
|---|---|---|
| 0-50 | Good | No AQI-driven restrictions needed |
| 51-100 | Moderate | Low operational disruption; monitor sensitive staff |
| 101-150 | Unhealthy for Sensitive Groups | Outdoor exposure should be managed for at-risk people |
| 151-200 | Unhealthy | Broad outdoor exposure controls should begin |
| 201-300 | Very Unhealthy | Strong restrictions and schedule changes required |
| 301-500 | Hazardous | Outdoor operations should be minimized or suspended |

## Decision matrix

### 1. Outdoor workforce exposure and shift scheduling

| Forecast band | Risk | Action |
|---|---|---|
| 0-100 | Low | Normal shift planning; continue routine monitoring |
| 101-150 | Moderate | Limit prolonged exposure for sensitive workers; add hydration and mask guidance |
| 151-200 | High | Shift heavy outdoor work to lower-AQI hours; shorten exposure blocks; increase breaks |
| 201-300 | Very high | Reschedule non-essential outdoor tasks; require protective measures for essential crews |
| 301-500 | Severe | Suspend non-critical outdoor work until conditions improve |

### 2. Construction and site planning

| Forecast band | Risk | Action |
|---|---|---|
| 0-100 | Low | Normal site activity |
| 101-150 | Moderate | Review dust-generating work and limit discretionary exposure |
| 151-200 | High | Delay dust-intensive tasks where possible; tighten PPE enforcement |
| 201-300 | Very high | Pause non-essential earthmoving, cutting, and demolition activity |
| 301-500 | Severe | Stop outdoor construction except emergency work |

### 3. Logistics and last-mile routing

| Forecast band | Risk | Action |
|---|---|---|
| 0-100 | Low | Normal dispatch |
| 101-150 | Moderate | Prefer shorter outdoor loading windows |
| 151-200 | High | Reduce rider or loader exposure time; consolidate stops where feasible |
| 201-300 | Very high | Shift delivery windows; prioritize enclosed-vehicle work over exposed last-mile tasks |
| 301-500 | Severe | Restrict non-essential dispatch and defer discretionary deliveries |

### 4. Employee travel policy

| Forecast band | Risk | Action |
|---|---|---|
| 0-100 | Low | Normal travel |
| 101-150 | Moderate | Advise sensitive employees to reduce unnecessary outdoor time |
| 151-200 | High | Reduce non-essential local travel and site visits |
| 201-300 | Very high | Move meetings remote unless operationally necessary |
| 301-500 | Severe | Cancel non-essential travel |

### 5. Outdoor event go / no-go

| Forecast band | Risk | Action |
|---|---|---|
| 0-100 | Low | Proceed normally |
| 101-150 | Moderate | Proceed with advisories, water access, and sensitive-group warnings |
| 151-200 | High | Reassess attendance profile, duration, and medical support |
| 201-300 | Very high | Postpone or relocate if possible |
| 301-500 | Severe | Cancel outdoor event |

### 6. Occupational health planning

| Forecast band | Risk | Action |
|---|---|---|
| 0-100 | Low | Routine communications only |
| 101-150 | Moderate | Notify teams with respiratory sensitivity |
| 151-200 | High | Issue formal health advisory and exposure controls |
| 201-300 | Very high | Activate elevated health precautions and supervisor check-ins |
| 301-500 | Severe | Trigger emergency continuity posture for outdoor operations |

## Worked example

Example policy statement:

If the system predicts **Day +2 AQI = 165**, that falls in the **Unhealthy** band.

Using the project decision model:

- `DATA`: recent AQI, PM2.5, PM10, NO2, O3, humidity, and wind trends indicate persistent poor dispersion conditions
- `FORECAST`: Day +2 AQI of 165
- `RISK`: broad outdoor exposure risk, not limited to highly sensitive individuals
- `ACTION`:
  - move labor-intensive outdoor work to the lowest-AQI hours available
  - defer non-essential site visits
  - tighten PPE and break requirements for essential crews
  - issue a same-day advisory to field managers
  - reassess any outdoor event or delivery surge planned for that day

## How this fits the dashboard and API

- The API provides the forecast horizons and AQI categories.
- The dashboard surfaces current AQI, the 3-day forecast, current pollutant and weather drivers, and SHAP context for why the model expects risk to rise or fall.
- The operational layer converts those outputs into decisions that a manager can act on quickly.

## Current limitation

This decision layer is based on AQI categories and operational judgement rules, not medical diagnosis. It is designed for planning support. Organizations should still apply their own health and safety policies, legal obligations, and local guidance.
