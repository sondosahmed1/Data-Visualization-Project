import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

csv_path = BASE_DIR / "preprocessing" / "owid_energy_visualization_cleaned.csv"
# =========================================================
# Load + Prepare Data
# =========================================================
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = ["population", "gdp", "electricity_generation", "year"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)

    # === AGGREGATE FILTER (fixed regex warning) ===
    AGGREGATE_KEYWORDS = [
        r"(Ember)", r"(Ei)", r"(Eia)", r"(Shift)", "World", "Income Countries",
        "OECD", "Oecd", "G20", "G7", "Opec", "Europe", "Asia", "Africa",
        "America", "Oceania", "Eurasia", "Middle East", "Cis", "Persian Gulf",
        "Rest Of World", "Non-Oecd", "Non-Opec", "Pacific Islands",
        "Territories", "Wake Island",
    ]
    pattern = "|".join(AGGREGATE_KEYWORDS)
    mask = ~df["country"].str.contains(pattern, case=False, na=False, regex=False)
    df = df[mask]

    # === DROP PLACEHOLDER VALUES ===
    df = df[~np.isclose(df["electricity_generation"], 21.86, atol=0.1)]

    # Positive values only
    df = df[
        (df["gdp"] > 0) &
        (df["population"] > 0) &
        (df["electricity_generation"] > 0)
    ]

    # Log-scale
    df["log_gdp"] = np.log10(df["gdp"])
    df["log_population"] = np.log10(df["population"])
    df["log_electricity_generation"] = np.log10(df["electricity_generation"])

    # Safe size column for bubbles (prevents negative sizes)
    df["size_electricity"] = df["log_electricity_generation"] - df["log_electricity_generation"].min() + 1
    df["size_electricity"] = df["size_electricity"].fillna(1).clip(lower=1)

    if "renewables_share_energy" in df.columns:
        df["renewables_share_energy"] = df["renewables_share_energy"].clip(lower=0)

    return df


# =========================================================
# =========================================================
# Detect Outliers + Outlier ID (using Z-score)
# =========================================================
def detect_outliers(df, column="log_electricity_generation", z_threshold=1.5):
    """Detect outliers using Z-score method for better robustness"""
    df = df.copy()
    
    # Calculate Z-scores
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    
    # Mark as outlier if |Z| > threshold
    df["is_outlier"] = z_scores > z_threshold
    df["Point Type"] = df["is_outlier"].map({False: "Normal", True: "Outlier"})
    
    df["outlier_id"] = df.apply(
        lambda row: f"{row['country']}_{int(row['year'])}" if row["is_outlier"] else None, axis=1
    )
    return df


# =========================================================
# Add Outlier Labels on Chart
# =========================================================
def add_outlier_labels(fig, df, x_col, y_col, max_labels=8):
    outliers = df[df["is_outlier"]].copy()
    if len(outliers) == 0:
        return fig

    top_outliers = outliers.nlargest(max_labels, "electricity_generation")
    
    for _, row in top_outliers.iterrows():
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text=row["outlier_id"],
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            ax=35,
            ay=-35,
            font=dict(size=9.5, color="#F28C28"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#F28C28",
            borderwidth=1
        )
    return fig


# =========================================================
# Shared Layout
# =========================================================
_COLOR_MAP = {"Normal": "#4C9BE8", "Outlier": "#F28C28"}

def _apply_layout(fig, x_label, y_label):
    fig.update_traces(
        marker=dict(opacity=0.55, line=dict(width=0.4, color="white")),
        selector=dict(mode="markers"),
    )
    # Make outliers stand out more
    fig.update_traces(
        marker=dict(size=10, line=dict(width=1.8, color="white")),
        selector=dict(mode="markers", name="Outlier")
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=13),
        legend=dict(
            title="Point Type",
            orientation="v",
            x=1.01,
            y=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(r=180),
    )
    return fig


# =========================================================
# Bubble Charts
# =========================================================
def bubble_gdp_vs_electricity(df):
    df = df.copy()
    df["log_gdp"] = np.log10(df["gdp"])
    df["log_electricity_generation"] = np.log10(df["electricity_generation"])
    
    df = df.dropna(subset=["log_gdp", "log_electricity_generation"])
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
        return fig

    min_gen = df["log_electricity_generation"].min()
    df["size_electricity"] = (df["log_electricity_generation"] - min_gen + 1).fillna(1).clip(lower=1)
    
    if "Point Type" not in df.columns:
        df["Point Type"] = "Normal"
        df["is_outlier"] = False

    fig = px.scatter(
        df,
        x="log_gdp",
        y="log_electricity_generation",
        size="size_electricity",
        size_max=35,
        color="Point Type",
        color_discrete_map=_COLOR_MAP,
        trendline="ols",
        hover_name="country",
        hover_data={"year": True},
        title="GDP vs Electricity Generation (Bubble = Generation)",
        category_orders={"Point Type": ["Normal", "Outlier"]},
    )
    return _apply_layout(fig, "GDP (log₁₀ USD)", "Electricity Generation (log₁₀ TWh)")

def bubble_population_vs_gdp(df):
    df = df.copy()
    df["log_population"] = np.log10(df["population"])
    df["log_gdp"] = np.log10(df["gdp"])
    df["log_electricity_generation"] = np.log10(df["electricity_generation"])
    
    df = df.dropna(subset=["log_population", "log_gdp", "log_electricity_generation"])
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for this selection", showarrow=False, font=dict(size=20))
        return fig

    min_gen = df["log_electricity_generation"].min()
    df["size_electricity"] = (df["log_electricity_generation"] - min_gen + 1).fillna(1).clip(lower=1)
    
    if "Point Type" not in df.columns:
        df["Point Type"] = "Normal"

    fig = px.scatter(
        df,
        x="log_population",
        y="log_gdp",
        size="size_electricity",
        size_max=35,
        color="Point Type",
        color_discrete_map=_COLOR_MAP,
        trendline="ols",
        hover_name="country",
        hover_data={"year": True},
        title="Population vs GDP (Bubble = Generation)",
        category_orders={"Point Type": ["Normal", "Outlier"]},
    )
    return _apply_layout(fig, "Population (log₁₀)", "GDP (log₁₀ USD)")

def bubble_renewables_vs_gdp(df):
    if "renewables_share_energy" not in df.columns:
        return None

    df = df.copy()
    df["log_gdp"] = np.log10(df["gdp"])
    df["log_electricity_generation"] = np.log10(df["electricity_generation"])
    
    df = df.dropna(subset=["renewables_share_energy", "log_gdp", "log_electricity_generation"])
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
        return fig

    min_gen = df["log_electricity_generation"].min()
    df["size_electricity"] = (df["log_electricity_generation"] - min_gen + 1).fillna(1).clip(lower=1)
    
    if "Point Type" not in df.columns:
        df["Point Type"] = "Normal"

    fig = px.scatter(
        df,
        x="renewables_share_energy",
        y="log_gdp",
        size="size_electricity",
        size_max=35,
        color="Point Type",
        color_discrete_map=_COLOR_MAP,
        trendline="ols",
        hover_name="country",
        hover_data={"year": True},
        title="Renewables Share vs GDP (Bubble = Generation)",
        category_orders={"Point Type": ["Normal", "Outlier"]},
    )
    return _apply_layout(fig, "Renewables Share (%)", "GDP (log₁₀ USD)")

def bubble_avg_generation_per_year(df):
    yearly = df.groupby("year", as_index=False)[["electricity_generation", "population"]].mean()
    yearly = yearly.dropna(subset=["electricity_generation", "population"])
    if yearly.empty:
        fig = go.Figure()
        fig.add_annotation(text="No yearly data available", showarrow=False)
        return fig
    
    fig = px.scatter(
        yearly,
        x="year",
        y="electricity_generation",
        size="population",
        size_max=40,
        title="Avg Electricity Generation Per Year",
    )
    fig.update_traces(marker=dict(color="#4C9BE8", opacity=0.8, line=dict(width=0.8, color="white")))
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=13),
        xaxis_title="Year",
        yaxis_title="Avg Generation (TWh)",
    )
    return fig
