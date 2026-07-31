"""
theme.py
Shared color palette and chart styling for consistency across all dashboard pages.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Consistent city colors — use these everywhere a city is plotted
# City colors — distinct, no city coded as "red" to avoid implying poor performance
CITY_COLORS = {
    "Delhi": "#F4A261",       # warm orange — distinct, not alarming
    "Mumbai": "#2E86AB",      # blue
    "Bengaluru": "#06A77D",   # green
}

ACCENT_COLOR = "#2E86AB"
BACKGROUND_COLOR = "#0E1117"
GRID_COLOR = "#31333F"

def apply_chart_style():
    """Call once at the top of each page to style all matplotlib/seaborn charts consistently."""
    plt.style.use("dark_background")
    sns.set_palette(list(CITY_COLORS.values()))
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND_COLOR,
        "axes.facecolor": BACKGROUND_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": "#FAFAFA",
        "xtick.color": "#FAFAFA",
        "ytick.color": "#FAFAFA",
        "text.color": "#FAFAFA",
        "grid.color": GRID_COLOR,
        "font.size": 11,
    })
AQI_BUCKET_COLORS = {
    "Good": "#06A77D",
    "Satisfactory": "#8AC926",
    "Moderate": "#FFCA3A",
    "Poor": "#FF7B00",
    "Very Poor": "#E63946",
    "Severe": "#8B0000",
}

def aqi_badge_html(bucket):
    color = AQI_BUCKET_COLORS.get(bucket, "#9CA3AF")
    return f"""
    <span style="background-color:{color}; color:white; padding:4px 12px;
    border-radius:12px; font-size:13px; font-weight:600;">{bucket}</span>
    """