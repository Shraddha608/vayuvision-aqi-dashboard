"""
page_forecast.py
AQI Forecast page — live next-day prediction, historical exploration,
model metrics, and feature importance.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from theme import apply_chart_style, CITY_COLORS, ACCENT_COLOR


def render(model_ready_df, model, metrics_df, cities):
    apply_chart_style()
    st.title("AQI Forecast")

    selected_city = st.selectbox("Select City", cities, key="forecast_city_selector")
    city_color = CITY_COLORS.get(selected_city, ACCENT_COLOR)

    city_df = model_ready_df[model_ready_df["City"] == selected_city].sort_values("Date")
    feature_cols = [
    "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "month", "day_of_week", "is_weekend",
    "aqi_lag_1", "aqi_lag_7", "aqi_rolling_mean_3", "aqi_rolling_mean_7",
    "pm25_lag_1", "pm10_lag_1", "data_quality_score",
    "City",
    ]

    # ---------- Live next-day prediction ----------
    latest_row = city_df.dropna(subset=feature_cols).iloc[-1:]

    if not latest_row.empty:
        predicted = model.predict(latest_row[feature_cols])[0]
        based_on_date = latest_row["Date"].iloc[0].strftime("%B %d, %Y")

        with st.container(border=True):
            st.subheader(f"Next-Day AQI Prediction — {selected_city}")
            st.caption(f"Based on data through {based_on_date}")
            st.metric("Predicted AQI", f"{predicted:.1f}")
    else:
        st.warning("Not enough recent data to generate a live prediction for this city.")

    st.divider()

    # ---------- Historical exploration ----------
    st.subheader("Explore a Historical Prediction")
    st.caption("See how the model would have predicted next-day AQI on any date in the 2020 test period.")

    city_df_2020 = city_df[city_df["Date"].dt.year == 2020].dropna(subset=["next_day_aqi"] + feature_cols)
    if not city_df_2020.empty:
        selected_date = st.date_input(
            "Select a date",
            value=city_df_2020["Date"].max(),
            min_value=city_df_2020["Date"].min(),
            max_value=city_df_2020["Date"].max(),
        )

        row = city_df_2020[city_df_2020["Date"] == pd.Timestamp(selected_date)]

        if not row.empty:
            hist_predicted = model.predict(row[feature_cols])[0]
            actual = row["next_day_aqi"].iloc[0]

            col1, col2 = st.columns(2)
            col1.metric("Predicted Next-Day AQI", f"{hist_predicted:.1f}")
            col2.metric(
                "Actual Next-Day AQI",
                f"{actual:.1f}",
                delta=f"{hist_predicted - actual:+.1f}",
                delta_color="inverse",
            )
        else:
            st.info("No data available for this exact date.")
    else:
        st.info("No 2020 test data available for this city.")

    st.divider()

    # ---------- Actual vs Predicted chart over full test period ----------
    if not city_df_2020.empty:
        st.subheader(f"Actual vs Predicted AQI — {selected_city} (2020 Test Set)")

        preds_2020 = model.predict(city_df_2020[feature_cols])

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(city_df_2020["Date"], city_df_2020["next_day_aqi"],
                label="Actual", color=city_color, linewidth=2)
        ax.plot(city_df_2020["Date"], preds_2020,
                label="Predicted", color="#FFCA3A", linewidth=2, linestyle="--")
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        st.pyplot(fig)

    st.divider()

    # ---------- Model metrics ----------
    st.subheader("Model Performance Metrics")
    st.dataframe(metrics_df, use_container_width=True)

    st.divider()

    # ---------- Feature importance ----------
    st.subheader("Feature Importance")
    try:
        model_step = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]
        numeric_features = [c for c in feature_cols if c != "City"]
        ohe_cols = preprocessor.named_transformers_["city"].get_feature_names_out(["City"])
        all_names = numeric_features + list(ohe_cols)

        if hasattr(model_step, "coef_"):
            importances = model_step.coef_
        elif hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
        else:
            importances = None

        if importances is not None:
            importance_series = pd.Series(importances, index=all_names).sort_values()

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            colors = ["#9C89B8" if v < 0 else ACCENT_COLOR for v in importance_series.values]
            importance_series.plot(kind="barh", ax=ax2, color=colors)
            ax2.axvline(x=0, color="#9CA3AF", linewidth=0.8)
            ax2.grid(True, alpha=0.2, axis="x")
            st.pyplot(fig2)
        else:
            st.info("Feature importance not available for this model type.")
    except Exception:
        st.info("Feature importance could not be computed for this model.")