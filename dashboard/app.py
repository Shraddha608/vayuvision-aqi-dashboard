"""
app.py
VayuVision — Indian City AQI Forecasting & Data Quality Dashboard
"""

import streamlit as st
from data_loader import load_cleaned_data, load_model_ready_data, load_model, load_metrics

st.set_page_config(page_title="VayuVision AQI Dashboard", layout="wide")
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #1C2128;
        border: 1px solid #31333F;
        border-radius: 10px;
        padding: 15px 20px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #9CA3AF;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    h1 {
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 2px solid #2E86AB;
    }
    h3 {
        color: #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align:center; padding: 15px 0;">
    <div style="font-size:42px; line-height:1;">🌥️</div>
    <h2 style="color:#2E86AB; margin:5px 0 0 0;">VayuVision</h2>
    <p style="color:#9CA3AF; font-size:12px; margin-top:2px;">AQI Forecasting & Data Quality</p>
</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Overview", "City Analysis", "AQI Forecast", "Data Quality"])

cleaned_df = load_cleaned_data()
model_ready_df = load_model_ready_data()
model = load_model()
metrics_df = load_metrics()

cities = sorted(cleaned_df["City"].unique())

if page == "Overview":
    from page_overview import render
    render(cleaned_df, cities)

elif page == "City Analysis":
    from page_city_analysis import render
    render(cleaned_df, cities)

elif page == "AQI Forecast":
    from page_forecast import render
    render(model_ready_df, model, metrics_df, cities)

elif page == "Data Quality":
    from page_data_quality import render
    render(cleaned_df, cities)