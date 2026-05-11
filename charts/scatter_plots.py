import pandas as pd
import numpy as np
import plotly.express as px


# =========================================================
# Load + Prepare Data
# =========================================================
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = [
        "population",
        "gdp",
        "electricity_generation",
        "year"
    ]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)

    return df


# =========================================================
# Detect Outliers
# =========================================================
def detect_outliers(df, column="electricity_generation"):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df.copy()

    df["is_outlier"] = (
        (df[column] < lower) |
        (df[column] > upper)
    )

    return df


# =========================================================
# Scatter Plot
# GDP vs Electricity Generation
# =========================================================
def scatter_gdp_vs_electricity(df):

    plot_df = df.copy()

    fig = px.scatter(
        plot_df,
        x="gdp",
        y="electricity_generation",
        color="is_outlier",
        color_discrete_map={
            False: "lightblue",
            True: "orange"
        },
        hover_data=["year"],
        opacity=0.7,
        title="GDP vs Electricity Generation"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# =========================================================
# Scatter Plot
# Population vs Electricity Generation
# =========================================================
def scatter_population_vs_electricity(df):

    plot_df = df.copy()

    fig = px.scatter(
        plot_df,
        x="population",
        y="electricity_generation",
        color="is_outlier",
        color_discrete_map={
            False: "lightblue",
            True: "orange"
        },
        hover_data=["year"],
        opacity=0.7,
        title="Population vs Electricity Generation"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# =========================================================
# Scatter Plot
# GDP vs Population
# =========================================================
def scatter_gdp_vs_population(df):

    plot_df = df.copy()

    fig = px.scatter(
        plot_df,
        x="population",
        y="gdp",
        color="is_outlier",
        color_discrete_map={
            False: "lightblue",
            True: "orange"
        },
        hover_data=["year"],
        opacity=0.7,
        title="GDP vs Population"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# =========================================================
# Scatter Plot
# Renewables vs Electricity Generation
# =========================================================
def scatter_renewables_vs_generation(df):

    if "renewables_share_energy" not in df.columns:
        return None

    plot_df = df.copy()

    fig = px.scatter(
        plot_df,
        x="renewables_share_energy",
        y="electricity_generation",
        color="is_outlier",
        color_discrete_map={
            False: "lightblue",
            True: "orange"
        },
        hover_data=["year"],
        opacity=0.7,
        title="Renewables Share vs Electricity Generation"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


if __name__ == "__main__":

    csv_path = "owid_energy_visualization_cleaned(2).csv"

    df = load_data(csv_path)
    df = detect_outliers(df)

    fig1 = scatter_gdp_vs_electricity(df)
    fig2 = scatter_population_vs_electricity(df)
    fig3 = scatter_gdp_vs_population(df)

    fig1.show()
    fig2.show()
    fig3.show()
