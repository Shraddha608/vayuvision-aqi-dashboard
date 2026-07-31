"""
train_model.py
Trains and evaluates AQI forecasting models (Baseline, Linear Regression, Random Forest)
and saves the best model + metrics to disk.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Paths (relative to project root, so run this from src/ or via `python -m src.train_model`) ----
DATA_PATH = Path("data/processed/model_ready_data.csv")
MODEL_OUTPUT_PATH = Path("models/best_aqi_forecaster.pkl")
METRICS_OUTPUT_PATH = Path("reports/model_metrics.csv")

TARGET_COLUMN = "next_day_aqi"
NUMERIC_FEATURES = [
    "AQI", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3",
    "month", "day_of_week", "is_weekend",
    "aqi_lag_1", "aqi_lag_7", "aqi_rolling_mean_3", "aqi_rolling_mean_7",
    "pm25_lag_1", "pm10_lag_1", "data_quality_score",
]
CATEGORICAL_FEATURES = ["City"]
def load_and_split_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    train_df = df[df["Date"].dt.year.between(2015, 2018)].copy()
    validation_df = df[df["Date"].dt.year == 2019].copy()
    test_df = df[df["Date"].dt.year == 2020].copy()

    return train_df, validation_df, test_df
def build_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("city", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return preprocessor
def calculate_metrics(actual, predicted, model_name):
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }
def train_and_select_best_model(train_df, validation_df):
    preprocessor = build_preprocessor()
    model_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    results = []

    # Baseline
    baseline_preds = validation_df["AQI"]
    results.append(calculate_metrics(validation_df[TARGET_COLUMN], baseline_preds, "Naive Baseline"))

    # Linear Regression
    linear_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])
    linear_model.fit(train_df[model_features], train_df[TARGET_COLUMN])
    lr_preds = linear_model.predict(validation_df[model_features])
    results.append(calculate_metrics(validation_df[TARGET_COLUMN], lr_preds, "Linear Regression"))

    # Random Forest
    from sklearn.base import clone
    rf_model = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    rf_model.fit(train_df[model_features], train_df[TARGET_COLUMN])
    rf_preds = rf_model.predict(validation_df[model_features])
    results.append(calculate_metrics(validation_df[TARGET_COLUMN], rf_preds, "Random Forest"))

    validation_comparison = pd.DataFrame(results).round(3)

    # pick best by lowest MAE among LR and RF (skip baseline)
    candidates = validation_comparison[validation_comparison["Model"] != "Naive Baseline"]
    best_model_name = candidates.sort_values("MAE").iloc[0]["Model"]
    best_model = linear_model if best_model_name == "Linear Regression" else rf_model

    return best_model, best_model_name, validation_comparison
def retrain_and_evaluate_on_test(best_model, train_df, validation_df, test_df):
    model_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    development_df = pd.concat([train_df, validation_df], ignore_index=True).sort_values("Date")

    best_model.fit(development_df[model_features], development_df[TARGET_COLUMN])
    test_preds = best_model.predict(test_df[model_features])

    test_metrics = calculate_metrics(test_df[TARGET_COLUMN], test_preds, "Final Model (Test)")

    test_results = test_df[["City", "Date", TARGET_COLUMN]].copy()
    test_results["predicted_aqi"] = test_preds
    test_results = test_results.rename(columns={TARGET_COLUMN: "actual_aqi"})

    per_city_metrics = []
    for city in test_results["City"].unique():
        subset = test_results[test_results["City"] == city]
        per_city_metrics.append(calculate_metrics(subset["actual_aqi"], subset["predicted_aqi"], city))

    return best_model, test_metrics, pd.DataFrame(per_city_metrics), test_results
def save_artifacts(model, validation_comparison, test_metrics, per_city_df):
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_OUTPUT_PATH)

    final_report = pd.concat([
        validation_comparison,
        pd.DataFrame([test_metrics]),
        per_city_df,
    ], ignore_index=True)
    final_report.to_csv(METRICS_OUTPUT_PATH, index=False)

    print(f"Model saved to {MODEL_OUTPUT_PATH}")
    print(f"Metrics saved to {METRICS_OUTPUT_PATH}")

def main():
    train_df, validation_df, test_df = load_and_split_data()
    best_model, best_model_name, validation_comparison = train_and_select_best_model(train_df, validation_df)
    print(f"Best model on validation: {best_model_name}")
    print(validation_comparison)

    final_model, test_metrics, per_city_df, test_results = retrain_and_evaluate_on_test(
        best_model, train_df, validation_df, test_df
    )
    print("\nFinal test metrics:", test_metrics)
    print("\nPer-city test metrics:\n", per_city_df)

    save_artifacts(final_model, validation_comparison, test_metrics, per_city_df)


if __name__ == "__main__":
    main()