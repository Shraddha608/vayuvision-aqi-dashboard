"""
data_loader.py
Cached data loading functions for the Streamlit dashboard.
"""

from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

CLEANED_DATA_PATH = Path("data/processed/cleaned_city_day.csv")
MODEL_READY_PATH = Path("data/processed/model_ready_data.csv")
MODEL_PATH = Path("models/best_aqi_forecaster.pkl")
METRICS_PATH = Path("reports/model_metrics.csv")


@st.cache_data
def load_cleaned_data():
    df = pd.read_csv(CLEANED_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_model_ready_data():
    df = pd.read_csv(MODEL_READY_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)