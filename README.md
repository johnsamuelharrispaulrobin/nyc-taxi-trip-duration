# NYC Taxi Trip Duration Prediction

[![CI](https://github.com/johnsamuelharrispaulrobin/nyc-taxi-trip-duration/actions/workflows/ci.yml/badge.svg)](https://github.com/johnsamuelharrispaulrobin/nyc-taxi-trip-duration/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9-blue)
![Dataset](https://img.shields.io/badge/dataset-3.2M%20rows-orange)
![R2](https://img.shields.io/badge/R%CA%B2-0.80-green)

Predicting trip duration for NYC Yellow Taxi trips using XGBoost — a production-grade ML pipeline from raw data to Streamlit app.

---

## 1. Project Overview

**Problem**: Predict how long a Yellow Taxi trip will take at booking time.

**End Users**: Dispatch systems, ride-hailing apps, logistics planners.

**Input**: Pickup zone, dropoff zone, trip distance, pickup datetime.

**Output**: Predicted duration in minutes with context (rush hour, airport, route type).

**Key Constraint**: No data leakage. Fare, tip, and dropoff datetime are dropped at cleaning — they are unavailable at booking time.

| Property | Value |
|----------|-------|
| Dataset | NYC Yellow Taxi Trip Records |
| Rows | 3,201,131 |
| Format | Parquet |
| Raw Features | 8 columns (post-cleaning) |
| Engineered Features | 28 created, 25 selected |
| Target | `trip_duration_seconds` (log scale) |

---

## 2. Architecture

```
Raw Parquet Files
       │
       ▼
┌──────────────┐
│  loader.py   │  Load + concatenate yellow + green tripdata
└──────────────┘
       │
       ▼
┌──────────────┐
│ quality.py   │  Check schema, detect outliers, validate dtypes
└──────────────┘
       │
       ▼
┌──────────────┐
│ cleaner.py   │  Drop leaks, compute trip_duration_seconds
└──────────────┘
       │
       ▼
┌─────────────────┐
│ engineering.py  │  Temporal, distance, location, interaction features
└─────────────────┘
       │
       ▼
   cleaned.parquet ──► features.parquet
       │                    │
       ▼                    ▼
┌──────────────┐    ┌──────────────┐
│ baseline.py  │    │train_xgboost │    tuning.py
│ (Lin. Reg.)  │    │ (XGB default)│──►(Optuna 30 trials)
└──────────────┘    └──────────────┘    │
       │                   │            │
       └───────────────────┴────────────┘
                     │
                     ▼
           production_model.pkl
                     │
                     ▼
           streamlit_app.py ──► User
```

---

## 3. Results

| Model | RMSE (s) | R² | MAE (s) | Notes |
|-------|----------|-----|---------|-------|
| Linear Regression | 857 | -0.04 | — | Failed — non-linear relationships |
| XGBoost Default | 387 | 0.79 | — | Strong baseline |
| **XGBoost Tuned** | **373** | **0.80** | **232** | Production model |

**Key metric**: MAE 232 seconds (~3.9 minutes) on median trip of 853 seconds (14 min).

---

## 4. Key Technical Decisions

### Why log transform on target?
Trip duration is right-skewed (mean 1070s > median 853s). Log-transform normalizes the distribution, improves model convergence, and reduces the influence of extreme outliers.

### Why MAE over RMSE as primary metric?
For ETA prediction, we care equally about all errors. RMSE penalizes large errors more heavily, which can mislead stakeholders about average user experience. MAE is more interpretable: "off by 4 minutes on average."

### Why drop fare/tip columns?
**Data leakage**. Fare and tip are only known after the trip completes. Including them would train the model on future information, producing optimistic but useless predictions at booking time.

### Why XGBoost over Linear Regression?
Trip duration has non-linear relationships: congestion effects, airport access patterns, and borough-specific traffic. XGBoost automatically captures feature interactions. Linear Regression failed with R² = -0.04 — it couldn't model these patterns.

### Why Optuna over GridSearchCV?
GridSearchCV scales exponentially with hyperparameter count. Optuna uses Bayesian optimization (TPE sampler) to find better parameters in fewer trials. 30 Optuna trials outperformed what would require hundreds of grid searches.

---

## 5. Feature Engineering

| Feature | Category | Rationale |
|---------|----------|----------|
| `log_trip_distance` | Distance | Log-transform handles skewed distribution; primary duration driver |
| `is_rush_hour` | Temporal | Rush hours (7-9am, 4-7pm) add 15-20% to median duration |
| `is_airport_trip` | Location | JFK/LaGuardia trips have distinct traffic patterns; 7% of data |
| `hour_sin` / `hour_cos` | Temporal | Cyclical encoding preserves continuity between hour 23 and 0 |
| `distance_rush_interaction` | Interaction | Long trips during rush hour compound delays |
| `pu_borough` / `do_borough` | Location | Inter-borough trips cross congestion boundaries |
| `is_weekend` | Temporal | Weekend traffic patterns differ from weekday |
| `same_borough` | Location | 85% of trips stay within same borough; shorter, more predictable |

---

## 6. Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9 | Core language |
| pandas | latest | Data loading and manipulation |
| numpy | latest | Numerical operations |
| scikit-learn | latest | Linear regression, train/test split |
| XGBoost | latest | Gradient boosted regression |
| Optuna | latest | Hyperparameter optimization |
| MLflow | latest | Experiment tracking |
| Streamlit | latest | Web app UI |
| pytest | ≥7.0.0 | Unit testing |
| ruff | latest | Linting |
| Docker | latest | Containerization |
| GitHub Actions | latest | CI/CD |

---

## 7. Setup & Installation

```bash
# Clone the repository
git clone https://github.com/johnsamuelharrispaulrobin/nyc-taxi-trip-duration.git
cd nyc-taxi-trip-duration

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download raw data (NYC Taxi & Limousine Commission)
# Place yellow_tripdata.parquet and green_tripdata.parquet in data/raw/
```

---

## 8. How to Run

### Full Training Pipeline

```bash
# 1. Load and clean data
python -m src.data.loader
python -m src.data.quality
python -m src.data.cleaner

# 2. Engineer features
python -m src.features.engineering

# 3. Train models
python -m src.models.baseline
python -m src.models.train_xgboost
python -m src.models.tuning  # Optuna 30 trials (~10 min)

# 4. Run full MLflow pipeline (all 3 models)
python -m src.models.run_training
```

### Streamlit App

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

### Docker

```bash
# Build image
docker build -t nyc-taxi-predictor .

# Run container
docker run -p 8501:8501 nyc-taxi-predictor

# Or use docker-compose
docker-compose up --build
```

### Run Tests

```bash
pytest tests/ -v
```

### Lint Code

```bash
ruff check src/ app/
```

---

## 10. File Structure

```
nyc-taxi-trip-duration/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── app/
│   └── streamlit_app.py              # 4-page Streamlit app
├── data/
│   ├── raw/
│   │   ├── yellow_tripdata.parquet   # Raw trip data
│   │   ├── green_tripdata.parquet    # Raw trip data
│   │   └── taxi_zone_lookup.csv      # Zone ID → Borough/Zone
│   └── processed/
│       ├── cleaned.parquet           # After quality + cleaning
│       ├── features.parquet          # After feature engineering
│       └── target_log.pkl            # Target variable (log scale)
├── models/
│   ├── baseline.pkl                 # Linear Regression
│   ├── xgboost_default.pkl           # XGBoost default params
│   ├── tuned_xgboost.pkl             # XGBoost tuned (production)
│   ├── production_model.pkl          # Alias for tuned model
│   ├── best_params.json              # Optuna best hyperparameters
│   ├── feature_columns.pkl           # Ordered feature list
│   └── category_mappings.pkl          # Categorical encoders
├── notebooks/
│   ├── eda.ipynb                     # Exploratory data analysis
│   ├── images/                       # EDA visualizations
│   └── run_eda.py                    # EDA script
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Load raw parquet files
│   │   ├── quality.py               # Data quality checks
│   │   └── cleaner.py               # Clean + remove leaks
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py           # Feature creation + encoding
│   └── models/
│       ├── __init__.py
│       ├── baseline.py              # Linear Regression baseline
│       ├── train_xgboost.py          # XGBoost default
│       ├── tuning.py                 # Optuna hyperparameter tuning
│       └── run_training.py          # MLflow experiment tracking
└── tests/
    ├── __init__.py
    └── test_features.py              # pytest suite (9 tests)
```

---

## 11. Lessons Learned

### What Worked
- **Separating target from features** early prevented leakage throughout the pipeline.
- **Feature engineering** drove the biggest R² improvement — going from 0.79 to 0.80 with tuning was marginal; better features would yield more.
- **`inference_mode` parameter** in `create_features()` cleanly separated training (needs target) from prediction (no target).

### What Didn't Work
- **Linear Regression as baseline** failed completely (R² = -0.04). In hindsight, starting with XGBoost default would have saved time — but the failure clearly demonstrates why non-linear models are necessary here.
- **Full 3-fold CV during Optuna tuning** was too slow on 2.5M rows (~60s/trial). Switching to train/val split brought trial time down to ~30s and completed 30 trials in under 7 minutes.

### What I'd Do Differently
- Add **real-time traffic data** or **weather features** — the biggest remaining error comes from unpredictable congestion not captured by time-of-day alone.
- Use **HAVERSine distance** instead of raw trip_distance for better route-based predictions.
- Consider **LightGBM** as a faster alternative for larger hyperparameter search spaces.

---

## License

MIT License