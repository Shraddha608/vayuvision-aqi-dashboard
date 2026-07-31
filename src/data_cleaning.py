from pathlib import Path
import pandas as pd


TARGET_CITIES = ["Delhi", "Mumbai", "Bengaluru"]

POLLUTANT_COLUMNS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]


def get_iqr_outlier_flags(dataframe, column):
    """Flag city-specific IQR outliers without removing them."""
    q1 = dataframe.groupby("City")[column].transform(
        lambda values: values.quantile(0.25)
    )
    q3 = dataframe.groupby("City")[column].transform(
        lambda values: values.quantile(0.75)
    )

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (
        (dataframe[column] < lower_bound)
        | (dataframe[column] > upper_bound)
    )


def clean_air_quality_data(raw_df):
    """Clean raw AQI data and return a model-ready cleaned dataframe."""

    data = raw_df[raw_df["City"].isin(TARGET_CITIES)].copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = (
        data
        .dropna(subset=["Date"])
        .sort_values(["City", "Date"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    available_pollutants = [
        column for column in POLLUTANT_COLUMNS
        if column in data.columns
    ]

    # Preserve original missingness before filling pollutant values.
    data["missing_value_count"] = (
        data[available_pollutants]
        .isna()
        .sum(axis=1)
    )

    # Fill each pollutant with its city-level median where possible.
    for column in available_pollutants:
        city_median = data.groupby("City")[column].transform("median")
        data[column] = data[column].fillna(city_median)

    # AQI is the target, so rows without it cannot train a model.
    data = data.dropna(subset=["AQI"]).copy()

    data["is_aqi_outlier"] = get_iqr_outlier_flags(data, "AQI")
    data["is_pm25_outlier"] = get_iqr_outlier_flags(data, "PM2.5")

    data["data_quality_score"] = (
        100
        - (data["missing_value_count"] * 5)
        - (data["is_aqi_outlier"].astype(int) * 15)
    ).clip(0, 100).astype(int)

    return data


def main():
    project_root = Path(__file__).resolve().parents[1]

    raw_path = project_root / "data" / "raw" / "city_day.csv"
    output_path = project_root / "data" / "processed" / "cleaned_city_day.csv"

    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_air_quality_data(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    print(f"Saved cleaned data to: {output_path}")
    print(f"Final dataset shape: {cleaned_df.shape}")


if __name__ == "__main__":
    main()