import pandas as pd
import plotly.express as px


# =========================================================
# Load Data
# =========================================================
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = [
        "population",
        "gdp",
        "electricity_generation",
        "year"
    ]

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
# Bubble Chart
# GDP vs Electricity Generation
# Bubble Size = Population
# =========================================================
def bubble_gdp_vs_electricity(df):

    fig = px.scatter(
        df,
        x="gdp",
        y="electricity_generation",
        size="population",
        color="is_outlier",
        trendline="ols",
        color_discrete_map={
            False: "lightblue",
            True: "orange"
        },
        hover_data=["year"],
        title="GDP vs Electricity Generation"
    )

    fig.update_layout(template="plotly_white")

    return fig


# =========================================================
# Bubble Chart
# Population vs GDP
# Bubble Size = Electricity Generation
# =========================================================
def bubble_population_vs_gdp(df):

    fig = px.scatter(
        df,
        x="population",
        y="gdp",
        size="electricity_generation",
        color="is_outlier",
        trendline="ols",
        color_discrete_map={
            False: "lightblue",
            True: "orange"
        },
        hover_data=["year"],
        title="Population vs GDP"
    )

    fig.update_layout(template="plotly_white")

    return fig


# =========================================================
# Bubble Chart
# Renewables vs GDP
# Bubble Size = Electricity Generation
# =========================================================
def bubble_renewables_vs_gdp(df):

    if "renewables_share_energy" not in df.columns:
        return None

    fig = px.scatter(
        df,
        x="renewables_share_energy",
        y="gdp",
        size="electricity_generation",
        color="is_outlier",
        trendline="ols",
        color_discrete_map={
            False: "lightblue",
            True: "orange"
        },
        hover_data=["year"],
        title="Renewables Share vs GDP"
    )

    fig.update_layout(template="plotly_white")

    return fig


# =========================================================
# Aggregated Bubble Chart
# Average Electricity Generation Per Year
# =========================================================
def bubble_avg_generation_per_year(df):

    yearly = (
        df.groupby("year", as_index=False)
        [["electricity_generation", "population"]]
        .mean()
    )

    fig = px.scatter(
        yearly,
        x="year",
        y="electricity_generation",
        size="population",
        title="Average Electricity Generation Per Year"
    )

    fig.update_traces(marker=dict(color="lightblue"))

    fig.update_layout(template="plotly_white")

    return fig


if __name__ == "__main__":

    csv_path = r"C:\\Uni\\Semester 6\\Data Visiualization\\Data-Visualization-Project\\preprocessing\\owid_energy_visualization_cleaned.csv"

    df = load_data(csv_path)
    df = detect_outliers(df)

    fig1 = bubble_gdp_vs_electricity(df)
    fig2 = bubble_population_vs_gdp(df)
    fig3 = bubble_avg_generation_per_year(df)

    fig1.show()
    fig2.show()
    fig3.show()
