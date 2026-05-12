import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from pathlib import Path

# Import chart functions
from charts.notebook_charts import (
    load_data as load_notebook_data,
    bar_top_countries,
    bar_avg_by_decade,
    stacked_bar_china_us,
    horizontal_stacked_before_after,
    clustered_bar_gdp_elec,
    network_centrality_mock,
    violin_asia_na,
    line_trend_ma,
    area_trend,
    clustered_bar_pop_elec
)
from charts.bubble_plots import (
    bubble_gdp_vs_electricity,
    bubble_population_vs_gdp,
    bubble_renewables_vs_gdp,
    bubble_avg_generation_per_year
)
from charts.scatter_plots import (
    scatter_gdp_vs_electricity,
    scatter_population_vs_electricity,
    scatter_gdp_vs_population,
    scatter_renewables_vs_generation
)
from charts.Histogram_and_BoxPlot import (
    make_histogram,
    make_kde,
    make_boxplot
)

# Initialize Dash App
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
app.title = "Global Energy Dashboard"

# Load Data Once (Global)
df = load_notebook_data()

# Styling Constants for 1920x1080
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2.5rem 1.5rem",
    "background-color": "#FFFFFF",
    "border-right": "1px solid #E9ECEF",
    "box-shadow": "4px 0 10px rgba(0,0,0,0.03)",
    "zIndex": 1000
}

CONTENT_STYLE = {
    "margin-left": "22rem",
    "margin-right": "2rem",
    "padding": "2rem 2.5rem",
    "background-color": "#F8F9FA",
    "min-height": "100vh"
}

# Sidebar Component
sidebar = html.Div(
    [
        html.Div([
            html.Img(src="https://img.icons8.com/fluency/96/lightning-bolt.png", style={"width": "80px"}),
            html.H2("Energy Insights", className="mt-3 mb-4", style={"fontWeight": "800", "letterSpacing": "-1px", "color": "#1A202C"}),
        ], style={"textAlign": "center"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-home me-2"), "Overview"], href="/", active="exact"),
                dbc.NavLink([html.I(className="fas fa-link me-2"), "Relation Charts"], href="/relation", active="exact"),
                dbc.NavLink([html.I(className="fas fa-chart-bar me-2"), "Distribution"], href="/distribution", active="exact"),
                dbc.NavLink([html.I(className="fas fa-chart-line me-2"), "Trend Analysis"], href="/trend", active="exact"),
                dbc.NavLink([html.I(className="fas fa-layer-group me-2"), "Composition"], href="/composition", active="exact"),
            ],
            vertical=True,
            pills=True,
            className="nav-pills-custom mt-4"
        ),
        html.Hr(),
        html.Div([
            html.P("🚀 High-End Analytics Dashboard", className="small fw-bold text-primary mb-1"),
            html.P("Optimized for 1080p Desktop Viewing", className="small text-muted")
        ], style={"position": "absolute", "bottom": "2rem", "left": "1.5rem", "right": "1.5rem"})
    ],
    style=SIDEBAR_STYLE,
)

# Main Layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    html.Div(id="page-content", style=CONTENT_STYLE)
])

# Utility to check if figure has data
def is_fig_empty(fig):
    if not fig or not hasattr(fig, 'data') or len(fig.data) == 0:
        return True
    # Check if scatter/bubble has valid points
    for trace in fig.data:
        if hasattr(trace, 'x') and trace.x is not None and len(trace.x) > 0:
            return False
    return True

# Config for responsive graphs
GRAPH_CONFIG = {
    'responsive': True,
    'displayModeBar': False,
    'scrollZoom': False
}

# Callback for Relation Charts Selection
@app.callback(
    Output("relation-content", "children"),
    [Input("relation-selector", "value")]
)
def update_relation_chart(selected_chart):
    if selected_chart == "gdp_elec":
        return dcc.Graph(figure=scatter_gdp_vs_electricity(df), className="shadow-sm", style={"height": "750px", "width": "100%"}, config=GRAPH_CONFIG)
        
    elif selected_chart == "pop_elec":
        return html.Div([
            dcc.Graph(figure=scatter_population_vs_electricity(df), className="mb-5 shadow-sm", style={"height": "700px", "width": "100%"}, config=GRAPH_CONFIG),
            dcc.Graph(figure=bubble_population_vs_gdp(df), className="shadow-sm", style={"height": "700px", "width": "100%"}, config=GRAPH_CONFIG)
        ])
    elif selected_chart == "pop_gdp":
        return dcc.Graph(figure=scatter_gdp_vs_population(df), className="shadow-sm", style={"height": "800px", "width": "100%"}, config=GRAPH_CONFIG)
    elif selected_chart == "network":
        return dcc.Graph(figure=network_centrality_mock(), className="shadow-sm", style={"height": "850px", "width": "100%"}, config=GRAPH_CONFIG)
    return html.Div("Please select a valid chart category.")

# Page Render Callback
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return dbc.Container([
            html.H1("🌏 Global Energy Overview", className="mb-4 display-5 fw-bold"),
            html.P("Comprehensive analysis of global energy production trends and economic correlations.", className="lead text-muted mb-5"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.H6("Total Countries", className="text-uppercase small text-muted mb-2"), html.H2(f"{df['country'].nunique()}", className="mb-0")])], className="metric-card"), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H6("Timeline Range", className="text-uppercase small text-muted mb-2"), html.H2(f"{df['year'].min()}-{df['year'].max()}", className="mb-0")])], className="metric-card"), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H6("Total Generation", className="text-uppercase small text-muted mb-2"), html.H2(f"{df['electricity_generation'].sum()/1e3:.1f}k TWh", className="mb-0")])], className="metric-card"), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H6("World Population", className="text-uppercase small text-muted mb-2"), html.H2(f"{df['population'].max()/1e9:.1f}B", className="mb-0")])], className="metric-card"), width=3),
            ], className="mb-5 g-4"),
            dbc.Card([
                dbc.CardHeader(html.H5("Top 10 Global Energy Producers", className="m-0 py-2")),
                dbc.CardBody(dcc.Graph(figure=bar_top_countries(df), style={"height": "800px", "width": "100%"}, config=GRAPH_CONFIG))
            ], className="border-0 shadow-sm rounded-4 overflow-hidden")
        ], fluid=True)
    
    elif pathname == "/relation":
        return dbc.Container([
            html.H1("🔗 Relation Analysis", className="mb-4 display-5 fw-bold"),
            html.P("Exploring the nexus between economic growth and energy demand.", className="text-muted mb-5"),
            dbc.RadioItems(
                id="relation-selector",
                className="btn-group-toggle d-flex flex-wrap mb-5",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary px-4 py-2 me-3 rounded-pill",
                labelCheckedClassName="active",
                options=[
                    {"label": "GDP vs Electricity", "value": "gdp_elec"},
                    {"label": "Population vs Electricity", "value": "pop_elec"},
                    {"label": "Population vs GDP", "value": "pop_gdp"},
                    {"label": "Network Centrality", "value": "network"},
                ],
                value="gdp_elec",
            ),
            dcc.Loading(id="loading-relation", type="default", children=html.Div(id="relation-content", className="animate-fade-in"))
        ], fluid=True)

    elif pathname == "/distribution":
        return dbc.Container([
            html.H1("📊 Distribution Analysis", className="mb-4 display-5 fw-bold"),
            dbc.Tabs([
                dbc.Tab(label="Global Histogram", children=[
                    dbc.Card(dbc.CardBody(dcc.Graph(figure=make_histogram(df), style={"height": "800px", "width": "100%"}, config=GRAPH_CONFIG)), className="mt-4 border-0 shadow-sm rounded-4")
                ], tab_id="hist"),
                dbc.Tab(label="Decade Variance (KDE)", children=[
                    dbc.Card(dbc.CardBody([
                        dcc.Graph(figure=make_kde(df), className="mb-5", style={"height": "600px", "width": "100%"}, config=GRAPH_CONFIG),
                        dcc.Graph(figure=make_boxplot(df), style={"height": "600px", "width": "100%"}, config=GRAPH_CONFIG)
                    ]), className="mt-4 border-0 shadow-sm rounded-4")
                ], tab_id="kde"),
                dbc.Tab(label="Regional Comparison", children=[
                    dbc.Card(dbc.CardBody(dcc.Graph(figure=violin_asia_na(df), style={"height": "800px", "width": "100%"}, config=GRAPH_CONFIG)), className="mt-4 border-0 shadow-sm rounded-4")
                ], tab_id="violin")
            ], active_tab="hist", className="nav-justified custom-tabs")
        ], fluid=True)

    elif pathname == "/trend":
        return dbc.Container([
            html.H1("📈 Trend Analysis", className="mb-4 display-5 fw-bold"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=line_trend_ma(df), style={"height": "700px", "width": "100%"}, config=GRAPH_CONFIG)), className="mb-5 shadow-sm border-0 rounded-4"), width=12),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=area_trend(df), style={"height": "700px", "width": "100%"}, config=GRAPH_CONFIG)), className="mb-5 shadow-sm border-0 rounded-4"), width=12),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=bubble_avg_generation_per_year(df), style={"height": "700px", "width": "100%"}, config=GRAPH_CONFIG)), className="shadow-sm border-0 rounded-4"), width=12),
            ], className="g-4")
        ], fluid=True)

    elif pathname == "/composition":
        return dbc.Container([
            html.H1("🧱 Energy Composition", className="mb-4 display-5 fw-bold"),
            html.P("Each chart displayed in full width for maximum clarity.", className="text-muted mb-5"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=bar_avg_by_decade(df), style={"height": "700px", "width": "100%"}, config=GRAPH_CONFIG)), className="mb-5 shadow-sm border-0 rounded-4"), width=12),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=clustered_bar_gdp_elec(df), style={"height": "700px", "width": "100%"}, config=GRAPH_CONFIG)), className="mb-5 shadow-sm border-0 rounded-4"), width=12),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=clustered_bar_pop_elec(df), style={"height": "700px", "width": "100%"}, config=GRAPH_CONFIG)), className="mb-5 shadow-sm border-0 rounded-4"), width=12),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=stacked_bar_china_us(df), style={"height": "750px", "width": "100%"}, config=GRAPH_CONFIG)), className="mb-5 shadow-sm border-0 rounded-4"), width=12),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=horizontal_stacked_before_after(df), style={"height": "750px", "width": "100%"}, config=GRAPH_CONFIG)), className="mb-5 shadow-sm border-0 rounded-4"), width=12),
            ], className="g-4")
        ], fluid=True)
    
    return dbc.Container([html.H1("404: Page Not Found", className="text-danger mt-5")], className="text-center")

if __name__ == "__main__":
    app.run(debug=True, port=8050)
