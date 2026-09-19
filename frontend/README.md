# 🌾 StubbleAI

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-frontend-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-prototype-orange)]()
[![SDG](https://img.shields.io/badge/Primary%20SDG-13%20Climate%20Action-2C4A3E)]()

## AI-Based Crop Residue Burning Risk Prediction & Sustainable Management Assistant

StubbleAI is an AI-powered environmental **decision-support and early-warning prototype** for predicting **next-day elevated satellite-derived active-fire activity at district level** across Punjab and Haryana during the October–November crop-residue-burning season.

The project combines satellite fire observations, weather information, temporal feature engineering, and a Random Forest classifier, then exposes the resulting risk through a FastAPI backend and React + Leaflet dashboard.

> **Primary SDG:** SDG 13 — Climate Action  
> **Secondary alignment:** SDG 3 — Good Health and Well-being; SDG 11 — Sustainable Cities and Communities

Developed as part of the **1M1B AI for Sustainability Virtual Internship**, in collaboration with **IBM SkillsBuild & AICTE**.

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [SDG Alignment](#-sdg-alignment)
- [Target Users](#-target-users)
- [AI Solution](#-ai-solution)
- [Machine Learning](#-machine-learning)
- [Data Sources](#-data-sources)
- [Feature Engineering](#️-feature-engineering)
- [Experimental Design](#-experimental-design)
- [Model Evaluation](#-model-evaluation)
- [Persistence Baseline](#-persistence-baseline)
- [Explainability](#-explainability)
- [Web Dashboard](#️-web-dashboard)
- [Screenshots](#-screenshots)
- [System Architecture](#️-system-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Historical Evaluation vs 2026 Live Mode](#-historical-evaluation-vs-2026-live-mode)
- [Cold-Start Handling](#-cold-start-handling)
- [Security](#-security)
- [Local Setup](#-local-setup)
- [API](#-api)
- [Reproducibility](#-reproducibility)
- [Responsible AI](#️-responsible-ai)
- [Expected Impact](#-expected-impact)
- [Limitations](#️-limitations)
- [Future Scope](#-future-scope)
- [Internship Learning & Project Journey](#-internship-learning--project-journey)
- [License](#-license)
- [Author](#-author)

---


##  Problem Statement

> **How might we use AI to identify elevated crop-residue burning risk so that agriculture and environmental stakeholders can act earlier toward sustainable residue management?**

Crop-residue burning is a recurring seasonal problem in northern India. Satellite systems provide valuable observations of active fires, but a predictive district-level signal can help stakeholders identify potentially elevated activity before the following day.

StubbleAI therefore focuses on a practical one-day-ahead prediction task rather than attempting to solve the entire crop-residue-burning problem.

---

##  SDG Alignment

### Primary — SDG 13: Climate Action

The project supports environmental monitoring, climate-awareness, and preventive decision support around seasonal crop-residue burning activity.

### Secondary alignment

- **SDG 3 — Good Health and Well-being:** seasonal biomass-burning activity can contribute to degraded air quality and associated health concerns.
- **SDG 11 — Sustainable Cities and Communities:** regional fire activity can affect downwind urban areas, including Delhi-NCR.

The prototype does **not** claim a quantified reduction in pollution or emissions; measuring that would require a separate intervention and impact study.

---

##  Target Users

### Primary
- District Agriculture Officers
- Farmer-support organizations
- Local environmental / pollution-control authorities

### Secondary
- Environmental researchers and analysts
- Urban and regional planners

### Responsible positioning

StubbleAI does not tell farmers when to burn. It is intended as an additional district-level decision-support signal for awareness, preparedness, and sustainable residue-management outreach.

---

##  AI Solution

```text
NASA FIRMS observations
        +
Weather data / forecast
        ↓
District-level aggregation
        ↓
Temporal + weather feature engineering
        ↓
Random Forest classifier
        ↓
Risk probability
        ↓
Normal / Elevated
        ↓
FastAPI backend
        ↓
React + Leaflet dashboard
```

### Why AI?

The system uses machine learning to identify patterns across recent fire activity, weather, seasonality, and location that can produce a probability-based next-day risk signal at district level.

---

##  Machine Learning

### Final evaluated model

**Random Forest Classifier**

Configuration:

- 400 trees
- `class_weight="balanced"`
- `min_samples_leaf=2`
- `random_state=42`
- Decision probability threshold: **0.40**

The 0.40 decision threshold was selected on the **2024 validation set using maximum F1**, then frozen before the 2025 final test.

---

## Target Definition

The target is binary:

```text
Normal:
next_day_fire_count <= 2

Elevated:
next_day_fire_count > 2
```

The target represents elevated **satellite-derived active-fire activity**, not a confirmed count of stubble-burning incidents.

---

##  Data Sources

### NASA FIRMS

Historical fire observations were processed using VIIRS satellite products.

Historical pipeline:

```text
149,461 raw detections
        ↓
148,196 Oct–Nov detections
        ↓
104,259 matched to Punjab/Haryana districts
```

The evaluated historical period covers:

- 2023
- 2024
- 2025
- October 1 – November 30
- 45 districts

### Important limitation

FIRMS active-fire detections are a **proxy for fire activity**. A satellite detection is not automatically a confirmed crop-residue-burning event.

### Historical weather

NASA POWER daily meteorological data:

- `T2M` — temperature
- `RH2M` — relative humidity
- `WS2M` — wind speed
- `PRECTOTCORR` — precipitation

### Live 2026 weather

The operational 2026 pipeline uses **Open-Meteo forecast data** for next-day weather inputs.

The historical evaluation and live operational pipeline are intentionally documented separately because their weather sources differ.

---

##  Feature Engineering

### Fire-history features

- `fire_lag_1d`
- `fire_lag_3d`
- `fire_lag_7d`
- `fire_mean_3d`
- `fire_mean_7d`

### Weather

- `T2M`
- `RH2M`
- `WS2M`
- `PRECTOTCORR`

### Seasonality

- `season_sin`
- `season_cos`

### Location

- District
- State

Lag and rolling features use past observations so that future target information is not leaked into the predictors.

---

##  Experimental Design

A strict temporal split was used instead of a random shuffle:

```text
2023
TRAINING
2,430 rows
       ↓
2024
VALIDATION
2,745 rows
       ↓
2025
FINAL TEST
2,700 rows
```

Final ML dataset:

**7,875 rows**

The 2025 test season remained unseen during model development and threshold selection.

---

##  Model Evaluation

Models / baselines evaluated:

- Random Forest
- Logistic Regression
- XGBoost
- Persistence baseline

The persistence baseline predicts tomorrow's class as today's class.

### Final Random Forest — 2025 Test

| Metric | Result |
|---|---:|
| Accuracy | **81.48%** |
| Balanced Accuracy | **79.35%** |
| Precision | **66.78%** |
| Recall | **74.15%** |
| F1 Score | **70.27%** |
| ROC-AUC | **0.8779** |
| PR-AUC | **0.7926** |

### Confusion matrix

```text
                    Predicted
                 Normal  Elevated

Actual Normal      1609     294
Actual Elevated     206     591
```

The model correctly identified **591 of 797 actual elevated-activity cases** in the 2025 test season.

---

##  Persistence Baseline

The persistence baseline performed strongly because fire activity exhibits substantial temporal persistence.

| Metric | Random Forest | Persistence |
|---|---:|---:|
| Accuracy | 81.48% | 83.89% |
| Balanced Accuracy | 79.35% | 80.59% |
| F1 | 70.27% | 72.66% |

The Random Forest did **not** outperform persistence on aggregate accuracy or F1.

However, the Random Forest correctly identified **81 elevated cases that persistence missed**, providing complementary probabilistic information for some rising-activity cases.

This comparison is intentionally reported rather than hidden.

---

##  Explainability
Important model features included:

1. `fire_mean_7d`
2. `fire_mean_3d`
3. `fire_lag_1d`
4. `fire_lag_3d`
5. `season_cos`
6. `fire_lag_7d`
7. `T2M`

Recent fire-history features dominate feature importance.

> Feature importance describes model association; it does not establish causation for an individual prediction.

---

## Web Dashboard

The working prototype provides:

- Live prediction date
- Total monitored districts
- Normal / Elevated counts
- Highest-risk district
- District search
- Risk filters
- Risk probability
- District analysis
- Interactive district risk map
- Map-based district selection
- Model performance information
- 2026 prediction tracker

### Map

The Leaflet map visualizes model-estimated probability using:

```text
<20%       Low
20–40%     Moderate
40–60%     Elevated
>60%       High
```

The map is for situational awareness and does not represent confirmed burning events.

---

## 📸Screenshots

**Risk Overview — district monitoring, live-history mode**

![StubbleAI Risk Overview](docs/screenshots/dashboard-overview.png)

Total districts, Normal/Elevated counts, highest-risk district, and a searchable, sortable risk table across all 45 districts.

**District Risk Map — geospatial view**

![StubbleAI District Risk Map](docs/screenshots/district-risk-map.png)

Live crop-residue fire risk plotted across monitored districts, with a click-through to district-level analysis.

> Place the corresponding PNG files in `docs/screenshots/` before publishing so these render on GitHub.

---

##  System Architecture

```text
NASA FIRMS ──────────────┐
                         │
                         ↓
                 Data Processing
                         ↑
Weather ────────────────┘
                         ↓
                Feature Engineering
                         ↓
                  Random Forest
                         ↓
                 Risk Probability
                         ↓
                    FastAPI
                         ↓
               React Web Dashboard
                    ↙        ↘
             District Map   Analysis
                         ↓
                 Prediction Tracker
```

---

## 💻 Technology Stack

### Data / ML
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Random Forest
- XGBoost
- Logistic Regression

### Backend
- FastAPI
- Uvicorn
- Python prediction pipeline

### Frontend
- React
- Vite
- Axios
- Leaflet
- React Leaflet
- CSS

### Data / APIs
- NASA FIRMS
- NASA POWER
- Open-Meteo
- District boundary GeoJSON

### Development
- Git
- GitHub
- Visual Studio Code

---

## Project Structure

```text
StubbleAI/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── public/
│   │   └── districts_punjab_haryana.geojson
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── ...
│   ├── package.json
│   └── ...
│
├── model/
│   ├── stubbleai_final_model.pkl
│   ├── stubbleai_deployment_model.pkl
│   ├── stubbleai_final_results.json
│   └── stubbleai_threshold.json
│
├── process_fire_data.py
├── download_weather.py
├── build_ml_dataset.py
├── prepare_ml.py
├── predict_2026.py
├── live_firms.py
├── update_live_firms_2026.py
├── live_district_fire_2026.py
├── build_live_2026_features.py
├── live_weather_2026.py
├── prediction_tracker.py
│
├── ml_dataset_base.csv
├── live_district_fire_2026.csv
├── live_weather_2026.csv
├── live_2026_fire_features.csv
├── stubbleai_2026_predictions.csv
├── prediction_tracker.csv
│
├── .env.example
├── .gitignore
└── README.md
```

> File names can vary slightly depending on the final repository cleanup. Keep the README synchronized with the actual repository before publishing.

---

##  Historical Evaluation vs 2026 Live Mode

These are intentionally separate.

### Historical research evaluation

```text
2023 → Training
2024 → Validation / threshold selection
2025 → Final unseen test
```

The reported metrics belong only to this experiment.

### 2026 operational mode

```text
FIRMS NRT observations
        +
Open-Meteo forecast
        ↓
Live feature generation
        ↓
Deployment model
        ↓
Next-day district risk
```

No 2026 accuracy is claimed because future ground-truth observations are not available at prediction time.

### FIRMS processing distinction

The historical evaluation uses processed historical FIRMS data, while the live 2026 pipeline uses near-real-time FIRMS observations. NASA's NRT stream can later differ from the finalized historical product after processing/reprocessing.

---

## 🧊 Cold-Start Handling

The model depends on recent fire-history features.

When sufficient live history is unavailable during the supported operating season, the implemented pipeline can use its historical seasonal fallback rather than treating missing observations as real zeros.

This prevents unsupported predictions from being presented as measured live history.

---

##  Security

### Environment variables

Private API credentials belong in `.env`:

```env
FIRMS_MAP_KEY=your_private_key
```

Never commit the real `.env`.

Use `.env.example` as the public template.

### Important

- Never put `FIRMS_MAP_KEY` in frontend React/Vite code.
- Never use a `VITE_*` variable for a private API key.
- Store production secrets in the hosting provider's secret manager.
- Rotate a key immediately if it is accidentally exposed.
- Do not log credentials or full environment-variable contents.

### Model artifact security

The Random Forest is stored as a trusted Joblib/Pickle artifact.

Pickle-based model files can execute code during deserialization, so:

- Only load model files produced by this project or another trusted source.
- Do not load arbitrary `.pkl` files from users or the internet.
- Keep the model artifact under controlled repository/deployment access.
- The API should return prediction values, not raw model objects.
- For production, consider artifact checksums/signing and a safer serialized model format where practical.

See [`SECURITY.md`](SECURITY.md) for the repository security checklist.

---

## Local Setup

### 1. Clone

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd StubbleAI
```

### 2. Python environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is blocked:

```powershell
.\.venv\Scripts\python.exe
```

### 3. Install backend dependencies

```bash
pip install pandas numpy scikit-learn joblib requests fastapi uvicorn geopandas shapely pyogrio
```

If a final `requirements.txt` is included, prefer:

```bash
pip install -r requirements.txt
```

### 4. Configure secrets

Copy:

```text
.env.example
```

to:

```text
.env
```

Add the private NASA FIRMS key.

### 5. Start backend

```bash
cd backend
uvicorn main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

### 6. Start frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

##  API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API status |
| `/api/health` | GET | Health check |
| `/api/predictions` | GET | All district predictions |
| `/api/predictions/{district}` | GET | District prediction |
| `/api/summary` | GET | Dashboard summary |
| `/api/run-prediction` | POST | Run live prediction pipeline |

Example:

```text
GET /api/predictions/Amritsar
```

---

##  Reproducibility

The research evaluation is deliberately separated from deployment.

```text
2023 → TRAIN
2024 → VALIDATE
2025 → TEST
```

The reported 2025 test metrics are frozen after evaluation.

For future operational inference, a separate deployment model may use the available historical data after the research evaluation has been finalized. This prevents the reported test result from being contaminated by future operational data.

---

## 🛡️ Responsible AI

### Transparency
The dashboard exposes risk probability, decision threshold, supporting information, and evaluation metrics.

### Fairness
District aggregation can hide sub-district differences. The system must not be used to assign blame or penalties to individual farmers or communities.

### Privacy
The prototype uses district-level environmental data and does not require personally identifiable farmer information.

### Ethics
The system is intended for awareness, preparedness, and decision support—not autonomous enforcement.

### Data limitations
FIRMS detections are satellite-derived active-fire proxies and may include sources other than crop-residue burning.

### Human oversight
Predictions should be considered alongside local knowledge and other official information.

---

## Expected Impact

StubbleAI aims to support:

1. Earlier awareness of potentially elevated fire-activity days.
2. Better targeting of district-level outreach.
3. Sustainable residue-management efforts.
4. Environmental preparedness and monitoring.
5. Reusable AI-based sustainability decision-support workflows.

No specific pollution-reduction percentage is claimed because this prototype has not conducted an intervention impact study.

---

##  Limitations

- FIRMS observations are proxies, not confirmed stubble-burning events.
- Geographic scope is currently Punjab + Haryana.
- Seasonal scope is October–November.
- Binary classification reduces risk granularity.
- Historical and live data streams have different processing characteristics.
- Historical weather and live forecast weather come from different sources.
- Random Forest does not outperform persistence on aggregate 2025 accuracy or F1.
- District-level aggregation can mask local variation.
- 2026 predictive accuracy cannot be established until future observations are available.

---

##  Future Scope

- Expand to additional crop-residue-burning states.
- Incorporate satellite imagery as a complementary signal.
- Use a consistent historical forecast-data source for stronger operational evaluation.
- Add authorized SMS/WhatsApp alerts.
- Improve spatial and temporal resolution.
- Evaluate advanced spatiotemporal models.
- Conduct intervention studies to quantify environmental impact.
- Continue validating 2026 predictions against subsequent observations.

---

##  Internship Learning & Project Journey

This project demonstrates the internship learning journey:

### Problem framing
Converted a sustainability challenge into a measurable AI prediction problem.

### Data and AI
Applied data preparation, feature engineering, temporal validation, classification, and evaluation.

### Responsible AI
Considered transparency, fairness, privacy, environmental-data limitations, and human oversight.

### Deployment
Connected the ML pipeline to a FastAPI backend and React dashboard.

### Sustainability
Applied AI as a decision-support tool for an environmental problem rather than building technology without a defined impact objective.

---

##  License

This repository is an educational/research prototype developed for the 1M1B AI for Sustainability Virtual Internship.

Before selecting an open-source license, verify the licensing and attribution requirements of external datasets, boundary files, APIs, and other third-party resources used by the project.

---

##  Author

**Vansh Jain**

B.Tech — Computer Science & Engineering (AI & ML)

**StubbleAI — AI-Based Crop Residue Burning Risk Prediction & Sustainable Management Assistant**

1M1B AI for Sustainability Virtual Internship  
In collaboration with IBM SkillsBuild & AICTE

---

##  Disclaimer

StubbleAI is an educational and research-oriented decision-support prototype.

Its predictions represent model-estimated probabilities of elevated **satellite-derived active-fire activity**, not confirmed crop-residue-burning events.

The system should not be used as the sole basis for enforcement, penalties, or decisions affecting individuals or communities.

---

##  Project Philosophy

> **Observe the problem → understand the data → apply AI responsibly → evaluate honestly → build for sustainability.**