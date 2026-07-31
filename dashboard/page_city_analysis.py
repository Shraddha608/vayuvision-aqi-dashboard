from theme import apply_chart_style, CITY_COLORS, ACCENT_COLOR
apply_chart_style()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render(cleaned_df, cities):
    apply_chart_style()
    st.title("City Analysis")

    selected_city = st.selectbox("Select City", cities, key="city_analysis_selector")
    date_range = st.date_input(
        "Date Range",
        value=(cleaned_df["Date"].min(), cleaned_df["Date"].max())
    )

    city_df = cleaned_df[cleaned_df["City"] == selected_city].copy()

    if len(date_range) == 2:
        start, end = date_range
        city_df = city_df[(city_df["Date"] >= pd.Timestamp(start)) & (city_df["Date"] <= pd.Timestamp(end))]

    st.subheader("AQI Trend")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(city_df["Date"], city_df["AQI"])
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Month-wise Average AQI")
    monthly_avg = city_df.groupby(city_df["Date"].dt.month)["AQI"].mean()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    monthly_avg.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Average AQI")
    st.pyplot(fig2)

    st.subheader("Pollutant Comparison")
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    city_df[pollutants].mean().plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

    st.subheader("AQI Distribution")
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    sns.histplot(city_df["AQI"], bins=30, ax=ax4)
    st.pyplot(fig4)