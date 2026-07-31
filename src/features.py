from pathlib import Path
import pandas as pd
import numpy as np


def build_model_ready_dataset(cleaned_df):
    """Create time-safe AQI forecasting features and next-day target."""

    data = cleaned_df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = (
        data
        .dropna(subset=["Date"])
        .sort_values(["City", "Date"])
        .reset_index(drop=True)
    )

    # Calendar features
    data["year"] = data["Date"].dt.year
    data["month"] = data["Date"].dt.month
    data["day_of_week"] = data["Date"].dt.dayofweek
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Time-series features, calculated separately for each city.
    city_groups = data.groupby("City")

    data["aqi_lag_1"] = city_groups["AQI"].shift(1)
    data["aqi_lag_7"] = city_groups["AQI"].shift(7)

    data["aqi_rolling_mean_3"] = city_groups["AQI"].transform(
        lambda values: values.rolling(
            window=3,
            min_periods=3
        ).mean()
    )

    data["aqi_rolling_mean_7"] = city_groups["AQI"].transform(
        lambda values: values.rolling(
            window=7,
            min_periods=7
        ).mean()
    )

    data["pm25_lag_1"] = city_groups["PM2.5"].shift(1)
    data["pm10_lag_1"] = city_groups["PM10"].shift(1)

    # Target: AQI on the next actual calendar day.
    next_observation_date = city_groups["Date"].shift(-1)
    data["next_day_aqi"] = city_groups["AQI"].shift(-1)

    is_actual_next_day = (
        next_observation_date
        == data["Date"] + pd.Timedelta(days=1)
    )

    data.loc[
        ~is_actual_next_day,
        "next_day_aqi"
    ] = np.nan

    # Keep rows with complete features and a valid target.
    required_model_columns = [
        "AQI",
        "aqi_lag_1",
        "aqi_lag_7",
        "aqi_rolling_mean_3",
        "aqi_rolling_mean_7",
        "pm25_lag_1",
        "pm10_lag_1",
        "next_day_aqi"
    ]

    return (
        data
        .dropna(subset=required_model_columns)
        .reset_index(drop=True)
    )


def main():
    project_root = Path(__file__).resolve().parents[1]

    input_path = (
        project_root / "data" / "processed" / "cleaned_city_day.csv"
    )

    output_path = (
        project_root / "data" / "processed" / "model_ready_data.csv"
    )

    cleaned_df = pd.read_csv(input_path)

    model_ready_df = build_model_ready_dataset(cleaned_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_ready_df.to_csv(output_path, index=False)

    print(f"Saved model-ready data to: {output_path}")
    print(f"Model-ready dataset shape: {model_ready_df.shape}")


if __name__ == "__main__":
    main()