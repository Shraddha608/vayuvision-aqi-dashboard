from theme import apply_chart_style, CITY_COLORS, ACCENT_COLOR
apply_chart_style()
import streamlit as st
import matplotlib.pyplot as plt
from theme import aqi_badge_html

def render(cleaned_df, cities):
    apply_chart_style()
    st.title("Overview")

    selected_city = st.selectbox("Select City", cities)
    city_df = cleaned_df[cleaned_df["City"] == selected_city].sort_values("Date")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average AQI", f"{city_df['AQI'].mean():.1f}")
    col2.metric("Highest AQI", f"{city_df['AQI'].max():.0f}")
    col3.metric("Latest AQI", f"{city_df['AQI'].iloc[-1]:.0f}")
    col4.metric("Avg Data Quality Score", f"{city_df['data_quality_score'].mean():.1f}")
    latest_bucket = city_df["AQI_Bucket"].iloc[-1]
    st.markdown(f"**Current status:** {aqi_badge_html(latest_bucket)}", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader(f"AQI Trend — {selected_city}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(city_df["Date"], city_df["AQI"], color=CITY_COLORS.get(selected_city, ACCENT_COLOR), linewidth=2)
        ax.fill_between(city_df["Date"], city_df["AQI"], alpha=0.15, color=CITY_COLORS.get(selected_city, ACCENT_COLOR))
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    st.subheader("All Cities — AQI Comparison")
    fig_all, ax_all = plt.subplots(figsize=(12, 4))
    for city in cities:
        subset = cleaned_df[cleaned_df["City"] == city].sort_values("Date")
        ax_all.plot(subset["Date"], subset["AQI"], label=city, color=CITY_COLORS.get(city), linewidth=1.5, alpha=0.85)
    ax_all.legend()
    ax_all.grid(True, alpha=0.2)
    st.pyplot(fig_all)