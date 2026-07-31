# 🌫️ VayuVision — Indian City AQI Forecasting & Data Quality Dashboard

An end-to-end data science project that analyzes, forecasts, and monitors air quality across major Indian cities using historical pollution data, machine learning, and an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

---

## 📖 Overview

VayuVision analyzes historical air quality data (2015–2020) for Delhi, Mumbai, and Bengaluru, builds a machine learning model to forecast next-day AQI, tracks data quality issues in the underlying dataset, and presents everything through an interactive Streamlit dashboard.

## ❓ Problem Statement

Air pollution is a major public health concern in Indian cities, but raw pollution data is often incomplete, inconsistent, and hard to interpret at a glance. This project builds a pipeline that cleans messy real-world air quality data, quantifies its reliability, and uses it to forecast near-term air quality — giving a clearer, more actionable picture than raw sensor readings alone.

## 📊 Dataset

- **Source:** [Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) — Kaggle
- **File used:** `city_day.csv`
- **Cities analyzed:** Delhi, Mumbai, Bengaluru
- **Columns:** City, Date, pollutant concentrations (PM2.5, PM10, NO2, SO2, CO, O3, etc.), AQI, AQI_Bucket

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data processing | Python, pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Database | PostgreSQL, SQLAlchemy |
| Dashboard | Streamlit |
| Version control | Git, GitHub |
| Development | Jupyter Notebook, VS Code |

## 🏗️ Project Architecture

```
Raw CSV (Kaggle)
      │
      ▼
Data Cleaning & Quality Scoring (src/data_cleaning.py)
      │
      ▼
Feature Engineering — lag features, rolling stats (src/features.py)
      │
      ▼
Model Training — Baseline, Linear Regression, Random Forest (src/train_model.py)
      │
      ▼
Streamlit Dashboard (dashboard/app.py)
```

## 🧹 Data Cleaning Approach

- Converted `Date` to datetime and sorted by city and date
- Removed duplicate rows
- Filtered to three cities: Delhi, Mumbai, Bengaluru
- Filled missing pollutant values using per-city median
- Dropped rows with missing AQI (the model's target variable)
- Flagged outliers in AQI and PM2.5 using the IQR method
- Engineered a **data quality score** (0–100) per row, penalizing missing values and flagged outliers


## 🔧 Feature Engineering

- Calendar features: year, month, day of week, weekend flag
- Time-series features computed **per city**: `aqi_lag_1`, `aqi_lag_7`, `aqi_rolling_mean_3`, `aqi_rolling_mean_7`, `pm25_lag_1`, `pm10_lag_1`
- Target variable: `next_day_aqi` (strictly using only past/present data — no future leakage)

## 🤖 Machine Learning

**Models compared:**
- Naive Baseline (predict tomorrow's AQI = today's AQI)
- Linear Regression
- Random Forest Regressor

**Evaluation strategy:** Chronological time-based split — Train: 2015–2018, Validation: 2019, Test: 2020 — to avoid data leakage inherent in random splits for time-series data.

**Results (2020 test set, held out):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive Baseline | [20.309] | [34.098] | [0.878] |
| Linear Regression | [17.356] | [27.657] | [0.92] |
| Random Forest | [17.197] | [27.76] | [0.919] |
| **Final Model (Test)** | **[14.27]** | **[21.43]** | **[0.915]** |

**Final model chosen:** Linear Regression — performed on par with Random Forest on validation, but with better interpretability and no meaningful loss in accuracy, since AQI shows strong linear relationships with its own recent lag/rolling-average values.


## 📊 Dashboard

The Streamlit dashboard has four pages:

1. **Overview** — key city-level metrics, AQI trend, and current AQI category
2. **City Analysis** — city selector, date-range filtering, monthly trends, pollutant comparison
3. **AQI Forecast** — live next-day AQI prediction, historical prediction explorer, model metrics, feature importance
4. **Data Quality** — missing value breakdown, outlier detection, data quality score distribution

## 🚀 How to Run Locally

### Prerequisites
- Python 3.11+

### Setup

```bash
# Clone the repository
git clone https://github.com/[your-username]/vayuvision-aqi-dashboard.git
cd vayuvision-aqi-dashboard

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place it at:
# data/raw/city_day.csv

# Run the data pipeline
python src/data_cleaning.py
python src/features.py
python src/train_model.py

# Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`.

## ⚠️ Limitations

- The model predicts only 1 day ahead; multi-day forecasting would require a different modeling approach
- A linear model cannot capture sudden pollution spikes from one-off events (fireworks, stubble burning, dust storms) not reflected in recent trends
- Currently limited to three cities (Delhi, Mumbai, Bengaluru); the underlying pipeline is designed to scale to more

## 🔮 Future Improvements

- Add multi-day (3-day, 7-day) AQI forecasting
- Expand to additional Indian cities
- Deploy the dashboard publicly (Streamlit Community Cloud)
- Add automated data refresh via scheduled pipeline runs

## 👤 Author

**[Shraddha Sharma]**
---

*This project was built as a portfolio piece demonstrating an end-to-end data science workflow — from raw data to a deployed, interactive application.*