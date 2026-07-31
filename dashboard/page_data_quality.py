from theme import apply_chart_style, CITY_COLORS, ACCENT_COLOR
apply_chart_style()
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def render(cleaned_df, cities):
    apply_chart_style()
    st.title("Data Quality")

    st.subheader("Missing Values by City")
    missing_by_city = cleaned_df.groupby("City")["missing_value_count"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    missing_by_city.plot(kind="bar", ax=ax,color="#F4A261")
    st.pyplot(fig)

    st.subheader("AQI Outliers by City")
    outliers_by_city = cleaned_df.groupby("City")["is_aqi_outlier"].sum()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    outliers_by_city.plot(kind="bar", ax=ax2,color="#E9C46A")
    st.pyplot(fig2)

    st.subheader("Data Quality Score Distribution")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.histplot(cleaned_df["data_quality_score"], bins=20, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Lowest-Quality Records")
    lowest_quality = cleaned_df.sort_values("data_quality_score").head(20)
    st.dataframe(lowest_quality[["City", "Date", "AQI", "missing_value_count", "is_aqi_outlier", "data_quality_score"]])