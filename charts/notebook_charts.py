import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

def load_data():
    csv_path = Path("preprocessing") / "owid_energy_visualization_cleaned.csv"
    if not csv_path.exists():
        csv_path = Path("Global Energy Production & Consumption Dashboard") / "Data" / "owid_energy_visualization_cleaned.csv"
    
    df = pd.read_csv(csv_path)
    if "decade" in df.columns:
        df["decade_str"] = df["decade"].astype(str)
    return df

def bar_top_countries(df):
    latest_year = df["year"].max()
    latest_df = df[df["year"] == latest_year].copy()
    exclude_names = ["World", "G20 (Ember)", "Asia (Ember)", "Oecd (Ember)", "OECD (Ember)", "G7 (Ember)", "North America (Ember)", "Europe (Ember)", "Europe"]
    latest_df = latest_df[~latest_df["country"].isin(exclude_names)]
    top10 = latest_df.groupby("country", as_index=False)["electricity_generation"].sum().sort_values("electricity_generation", ascending=False).head(10)
    top10_plot = top10.sort_values("electricity_generation", ascending=True)
    max_value = top10["electricity_generation"].max()
    top_country = top10.iloc[0]["country"]
    MAX_COLOR, NORMAL_COLOR = "#ECFF16", "#21A098"
    colors = [MAX_COLOR if country == top_country else NORMAL_COLOR for country in top10_plot["country"]]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top10_plot["electricity_generation"], y=top10_plot["country"], orientation="h",
        marker=dict(color=colors, line=dict(color="black", width=0.6)),
        text=[f"{v/1000:.0f}K" if v >= 1000 else f"{v:.0f}" for v in top10_plot["electricity_generation"]],
        textposition="outside", textfont=dict(color="black", size=11),
        hovertemplate="<b>%{y}</b><br>Electricity Generation: %{x:,.0f} TWh<extra></extra>",
        showlegend=False
    ))
    # Legend traces
    fig.add_trace(go.Bar(x=[None], y=[None], name="Maximum Generation", marker=dict(color=MAX_COLOR, line=dict(color="black", width=0.4))))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Standard Countries", marker=dict(color=NORMAL_COLOR, line=dict(color="black", width=0.4))))

    fig.update_layout(
        title=dict(text=f"Top 10 Countries by Electricity Generation ({latest_year})", x=0.5, font=dict(size=16, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=600, margin=dict(l=200, r=50, t=90, b=80), font=dict(color="black", family="Arial"),
        legend=dict(x=0.985, y=0.97, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,1)", bordercolor="lightgray", borderwidth=1, font=dict(size=9, color="black"), itemwidth=30, traceorder="normal"),
        xaxis=dict(title=dict(text="Electricity Generation (TWh)", font=dict(size=15, color="black", family="Arial Black")), range=[0, max_value * 1.33], zeroline=True, zerolinecolor="black", zerolinewidth=1, showgrid=True, gridcolor="lightgray", griddash="dot", tickfont=dict(size=10, color="black"), showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(title=dict(text="Country Name", font=dict(size=15, color="black", family="Arial Black")), tickfont=dict(size=10, color="black"), showgrid=False, showline=True, linecolor="black", linewidth=1, mirror=True),
        bargap=0.22
    )
    return fig

def bar_avg_by_decade(df):
    decade_df = df.groupby("decade", as_index=False)["electricity_generation"].mean().sort_values("electricity_generation", ascending=False)
    max_value, top_decade = decade_df["electricity_generation"].max(), decade_df.iloc[0]["decade"]
    MAX_COLOR, NORMAL_COLOR = "#E9FF00", "#2A9D8F"
    colors = [MAX_COLOR if d == top_decade else NORMAL_COLOR for d in decade_df["decade"]]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=decade_df["decade"].astype(str), y=decade_df["electricity_generation"],
        marker=dict(color=colors, line=dict(color="black", width=0.8)),
        text=[f"{v:,.0f}" for v in decade_df["electricity_generation"]],
        textposition="outside", textfont=dict(color="black", size=11),
        hovertemplate="<b>Decade:</b> %{x}<br><b>Average Electricity Generation:</b> %{y:,.0f} TWh<extra></extra>",
        showlegend=False
    ))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Highest Decade", marker=dict(color=MAX_COLOR, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Other Decades", marker=dict(color=NORMAL_COLOR, line=dict(color="black", width=0.5))))

    fig.update_layout(
        title=dict(text="Average Electricity Generation by Decade", x=0.5, font=dict(size=18, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=600, margin=dict(l=90, r=90, t=100, b=90), font=dict(color="black", family="Arial"),
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,1)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="black")),
        xaxis=dict(title=dict(text="Decade", font=dict(size=15, color="black", family="Arial Black")), tickangle=0, tickfont=dict(size=11, color="black"), showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(title=dict(text="Average Electricity Generation (TWh)", font=dict(size=15, color="black", family="Arial Black")), range=[0, max_value * 1.18], zeroline=True, zerolinecolor="black", zerolinewidth=1, showgrid=True, gridcolor="lightgray", griddash="dot", tickfont=dict(size=11, color="black"), showline=True, linecolor="black", linewidth=1, mirror=True),
        bargap=0.35
    )
    return fig

def stacked_bar_china_us(df):
    selected_countries = ["China", "United States"]
    top2_df = df[df["country"].isin(selected_countries)].copy()
    stacked_df = top2_df.groupby(["decade", "country"], as_index=False)["electricity_generation"].sum()
    pivot_df = stacked_df.pivot(index="decade", columns="country", values="electricity_generation").fillna(0).sort_index(ascending=False)
    max_total = pivot_df[selected_countries].sum(axis=1).max()
    peak_decade = pivot_df[selected_countries].sum(axis=1).idxmax()
    
    USA_STANDARD_COLOR, CHINA_STANDARD_COLOR = "#A9D6E5", "#2C7DA0"
    USA_PEAK_COLOR, CHINA_PEAK_COLOR = "#FFDCA8", "#F4A261"
    usa_colors = [USA_PEAK_COLOR if d == peak_decade else USA_STANDARD_COLOR for d in pivot_df.index]
    china_colors = [CHINA_PEAK_COLOR if d == peak_decade else CHINA_STANDARD_COLOR for d in pivot_df.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pivot_df.index.astype(str), y=pivot_df["United States"], name="United States",
        marker=dict(color=usa_colors, line=dict(color="black", width=0.7)),
        text=[f"{v/1000:.1f}K" if v >= 1000 else f"{v:.0f}" for v in pivot_df["United States"]],
        textposition="inside", textfont=dict(color="black", size=10, family="Arial Black"),
        customdata=pivot_df["United States"], hovertemplate="<b>Country:</b> United States<br><b>Decade:</b> %{x}<br><b>Real Electricity Generation:</b> %{customdata:,.0f} TWh<extra></extra>", showlegend=False
    ))
    fig.add_trace(go.Bar(
        x=pivot_df.index.astype(str), y=pivot_df["China"], name="China",
        marker=dict(color=china_colors, line=dict(color="black", width=0.7)),
        text=[f"{v/1000:.1f}K" if v >= 1000 else f"{v:.0f}" for v in pivot_df["China"]],
        textposition="inside", textfont=dict(color="black", size=10, family="Arial Black"),
        customdata=pivot_df["China"], hovertemplate="<b>Country:</b> China<br><b>Decade:</b> %{x}<br><b>Real Electricity Generation:</b> %{customdata:,.0f} TWh<extra></extra>", showlegend=False
    ))
    # Legend
    fig.add_trace(go.Bar(x=[None], y=[None], name="China (Peak Decade)", marker=dict(color=CHINA_PEAK_COLOR, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], name="United States (Peak Decade)", marker=dict(color=USA_PEAK_COLOR, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], name="China (Standard Decades)", marker=dict(color=CHINA_STANDARD_COLOR, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], name="United States (Standard Decades)", marker=dict(color=USA_STANDARD_COLOR, line=dict(color="black", width=0.5))))

    fig.update_layout(
        barmode="stack", title=dict(text="China and United States Electricity Generation by Decade", x=0.5, font=dict(size=20, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=700, margin=dict(l=100, r=90, t=100, b=90), font=dict(color="black", family="Arial"),
        legend=dict(title="Electricity Generation Groups", x=0.98, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,1)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="black")),
        xaxis=dict(title=dict(text="Decade", font=dict(size=14, color="black", family="Arial Black")), tickangle=0, tickfont=dict(size=11, color="black", family="Arial Black"), showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(title=dict(text="Electricity Generation (TWh)", font=dict(size=14, color="black", family="Arial Black")), range=[0, max_total * 1.18], zeroline=True, zerolinecolor="black", zerolinewidth=1, showgrid=True, gridcolor="lightgray", griddash="dot", tickfont=dict(size=11, color="black"), showline=True, linecolor="black", linewidth=1, mirror=True),
        bargap=0.22
    )
    return fig

def horizontal_stacked_before_after(df):
    df["country_clean"] = df["country"].astype(str).str.strip()
    exclude = ["World", "G20", "Ember", "Ei", "OECD", "Oecd", "G7", "European Union", "Europe", "High-Income", "Upper-Middle-Income", "Lower-Middle-Income", "Asia Pacific", "North America", "Middle East", "Latin America", "Asean"]
    pattern = "|".join(exclude)
    df_clean = df[~df["country_clean"].str.contains(pattern, case=False, na=False)].copy()
    before = df_clean[df_clean["year"] < 2000].groupby("country_clean")["electricity_generation"].sum()
    after = df_clean[df_clean["year"] >= 2000].groupby("country_clean")["electricity_generation"].sum()
    stacked = pd.DataFrame({"Before 2000": before, "After 2000": after}).fillna(0)
    stacked = stacked[(stacked["Before 2000"] > 0) & (stacked["After 2000"] > 0)].copy()
    stacked["Total"] = stacked["Before 2000"] + stacked["After 2000"]
    stacked = stacked.sort_values("Total", ascending=False).head(7)
    peak_country, max_total = stacked.index[0], stacked["Total"].max()
    BEFORE_STANDARD, AFTER_STANDARD = "#A9D6E5", "#2C7DA0"
    BEFORE_PEAK, AFTER_PEAK = "#FFDCA8", "#F4A261"
    before_colors = [BEFORE_PEAK if c == peak_country else BEFORE_STANDARD for c in stacked.index]
    after_colors = [AFTER_PEAK if c == peak_country else AFTER_STANDARD for c in stacked.index]
    fmt_k = lambda v: f"{v/1000:.1f}K" if v >= 1000 else f"{v:.0f}"

    fig = go.Figure()
    fig.add_trace(go.Bar(y=stacked.index, x=stacked["Before 2000"], orientation="h", marker=dict(color=before_colors, line=dict(color="black", width=0.7)), text=[fmt_k(v) for v in stacked["Before 2000"]], textposition="inside", textfont=dict(color="black", size=11, family="Arial Black"), showlegend=False))
    fig.add_trace(go.Bar(y=stacked.index, x=stacked["After 2000"], orientation="h", marker=dict(color=after_colors, line=dict(color="black", width=0.7)), text=[fmt_k(v) for v in stacked["After 2000"]], textposition="inside", textfont=dict(color="black", size=11, family="Arial Black"), showlegend=False))
    # Legend
    fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="After 2000 (Peak Country)", marker=dict(color=AFTER_PEAK, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Before 2000 (Peak Country)", marker=dict(color=BEFORE_PEAK, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="After 2000 (Standard)", marker=dict(color=AFTER_STANDARD, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Before 2000 (Standard)", marker=dict(color=BEFORE_STANDARD, line=dict(color="black", width=0.5))))

    fig.update_layout(
        barmode="stack", title=dict(text="Electricity Generation Before and After 2000 by Country", x=0.5, font=dict(size=22, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=780, margin=dict(l=220, r=120, t=110, b=100), font=dict(color="black", family="Arial"),
        legend=dict(title="Electricity Generation Groups", x=0.98, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.75)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="black")),
        xaxis=dict(title=dict(text="Electricity Generation (TWh)", font=dict(size=15, color="black", family="Arial Black")), range=[0, max_total * 1.25], showgrid=True, gridcolor="lightgray", griddash="dot", zeroline=True, zerolinecolor="black", zerolinewidth=1, tickfont=dict(size=11, color="black"), showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(title=dict(text="Country Name", font=dict(size=15, color="black", family="Arial Black")), tickfont=dict(size=12, color="black", family="Arial Black"), autorange="reversed", showline=True, linecolor="black", linewidth=1, mirror=True),
        bargap=0.25
    )
    return fig

def clustered_bar_gdp_elec(df):
    df["country_clean"] = df["country"].astype(str).str.strip()
    exclude = ["World", "G20", "Ember", "Ei", "OECD", "Oecd", "G7", "European Union", "Europe", "High-Income", "Upper-Middle-Income", "Lower-Middle-Income", "Asia Pacific", "North America", "Middle East", "Latin America", "Asean"]
    pattern = "|".join(exclude)
    df_clean = df[~df["country_clean"].str.contains(pattern, case=False, na=False)].dropna(subset=["gdp", "electricity_generation", "decade"]).copy()
    decade_df = df_clean.groupby("decade", as_index=False).agg({"gdp": "mean", "electricity_generation": "mean"}).sort_values("decade", ascending=False)
    decade_df["GDP_Index"] = (decade_df["gdp"] / decade_df["gdp"].max()) * 100
    decade_df["Elec_Index"] = (decade_df["electricity_generation"] / decade_df["electricity_generation"].max()) * 100
    peak_decade, max_y = decade_df.loc[decade_df["GDP_Index"].idxmax(), "decade"], max(decade_df["GDP_Index"].max(), decade_df["Elec_Index"].max())
    x_pos, bar_width = np.arange(len(decade_df)), 0.36
    GDP_STANDARD, ELEC_STANDARD, GDP_PEAK, ELEC_PEAK = "#A9D6E5", "#5DADE2", "#FFDCA8", "#F6C36A"
    gdp_colors = [GDP_PEAK if d == peak_decade else GDP_STANDARD for d in decade_df["decade"]]
    elec_colors = [ELEC_PEAK if d == peak_decade else ELEC_STANDARD for d in decade_df["decade"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_pos - bar_width/2, y=decade_df["GDP_Index"], width=bar_width, name="GDP Index", marker=dict(color=gdp_colors, line=dict(color="black", width=0.5)), text=[f"{v:.0f}" for v in decade_df["GDP_Index"]], textposition="outside", textfont=dict(color="black", size=11, family="Arial Black"), showlegend=False))
    fig.add_trace(go.Bar(x=x_pos + bar_width/2, y=decade_df["Elec_Index"], width=bar_width, name="Elec Index", marker=dict(color=elec_colors, line=dict(color="black", width=0.5)), text=[f"{v:.0f}" for v in decade_df["Elec_Index"]], textposition="outside", textfont=dict(color="black", size=11, family="Arial Black"), showlegend=False))
    # Legend
    fig.add_trace(go.Bar(x=[None], y=[None], name="GDP (Highest Peak)", marker=dict(color=GDP_PEAK, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Electricity (Peak)", marker=dict(color=ELEC_PEAK, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], name="GDP (Standard)", marker=dict(color=GDP_STANDARD, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Electricity (Standard)", marker=dict(color=ELEC_STANDARD, line=dict(color="black", width=0.5))))

    fig.update_layout(
        title=dict(text="Average GDP and Electricity Generation by Decade", x=0.5, font=dict(size=22, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=720, margin=dict(l=100, r=120, t=110, b=100), font=dict(color="black", family="Arial"),
        legend=dict(title="Indicators Breakdown", x=0.98, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.90)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="black")),
        xaxis=dict(title=dict(text="Decade", font=dict(size=15, color="black", family="Arial Black")), tickmode="array", tickvals=x_pos, ticktext=decade_df["decade"].astype(str), tickangle=0, tickfont=dict(size=12, color="black", family="Arial Black"), showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(title=dict(text="Normalized Index", font=dict(size=15, color="black", family="Arial Black")), range=[0, max_y * 1.20], zeroline=True, zerolinecolor="black", zerolinewidth=1, showgrid=True, gridcolor="lightgray", griddash="dot", tickfont=dict(size=11, color="black"), showline=True, linecolor="black", linewidth=1, mirror=True)
    )
    return fig

def clustered_bar_pop_elec(df):
    df["country_clean"] = df["country"].astype(str).str.strip()
    exclude = ["World", "G20", "Ember", "Ei", "OECD", "Oecd", "G7", "European Union", "Europe", "High-Income", "Upper-Middle-Income", "Lower-Middle-Income", "Asia Pacific", "North America", "Middle East", "Latin America", "Asean"]
    pattern = "|".join(exclude)
    df_clean = df[~df["country_clean"].str.contains(pattern, case=False, na=False)].dropna(subset=["population", "electricity_generation"]).copy()
    country_df = df_clean.groupby("country_clean", as_index=False).agg({"population": "mean", "electricity_generation": "mean"}).sort_values("electricity_generation", ascending=False).head(7)
    country_df["Pop_Index"] = (country_df["population"] / country_df["population"].max()) * 100
    country_df["Elec_Index"] = (country_df["electricity_generation"] / country_df["electricity_generation"].max()) * 100
    peak_country, max_x = country_df.iloc[0]["country_clean"], max(country_df["Pop_Index"].max(), country_df["Elec_Index"].max())
    y_pos, bar_height = np.arange(len(country_df)), 0.36
    POP_STANDARD, ELEC_STANDARD, POP_PEAK, ELEC_PEAK = "#A9D6E5", "#5DADE2", "#FFDCA8", "#F6C36A"
    pop_colors = [POP_PEAK if c == peak_country else POP_STANDARD for c in country_df["country_clean"]]
    elec_colors = [ELEC_PEAK if c == peak_country else ELEC_STANDARD for c in country_df["country_clean"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=y_pos - bar_height/2, x=country_df["Pop_Index"], orientation="h", width=bar_height, name="Population Index", marker=dict(color=pop_colors, line=dict(color="black", width=0.5)), text=[f"{v:.0f}" for v in country_df["Pop_Index"]], textposition="outside", textfont=dict(color="black", size=11, family="Arial Black"), showlegend=False))
    fig.add_trace(go.Bar(y=y_pos + bar_height/2, x=country_df["Elec_Index"], orientation="h", width=bar_height, name="Elec Index", marker=dict(color=elec_colors, line=dict(color="black", width=0.5)), text=[f"{v:.0f}" for v in country_df["Elec_Index"]], textposition="outside", textfont=dict(color="black", size=11, family="Arial Black"), showlegend=False))
    # Legend
    fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Pop (Highest Combined)", marker=dict(color=POP_PEAK, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Elec (Highest Combined)", marker=dict(color=ELEC_PEAK, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Pop (Standard)", marker=dict(color=POP_STANDARD, line=dict(color="black", width=0.5))))
    fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Elec (Standard)", marker=dict(color=ELEC_STANDARD, line=dict(color="black", width=0.5))))

    fig.update_layout(
        title=dict(text="Population and Electricity Generation Comparison by Country", x=0.5, font=dict(size=22, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=720, margin=dict(l=260, r=120, t=110, b=100), font=dict(color="black", family="Arial"),
        legend=dict(title="Indicators Breakdown", x=0.98, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.70)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="black")),
        xaxis=dict(title=dict(text="Normalized Index", font=dict(size=15, color="black", family="Arial Black")), range=[0, max_x * 1.20], zeroline=True, zerolinecolor="black", zerolinewidth=1, showgrid=True, gridcolor="lightgray", griddash="dot", tickfont=dict(size=11, color="black"), showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(title=dict(text="Country Name", font=dict(size=15, color="black", family="Arial Black")), tickmode="array", tickvals=y_pos, ticktext=country_df["country_clean"], tickfont=dict(size=12, color="black", family="Arial Black"), autorange="reversed", showline=True, linecolor="black", linewidth=1, mirror=True)
    )
    return fig

def network_centrality_mock():
    nodes = {"United States": {"x": 0.10, "y": 0.62}, "China": {"x": 0.48, "y": 0.55}, "India": {"x": 0.82, "y": 0.72}, "Japan": {"x": 0.91, "y": 0.20}, "Russia": {"x": 0.62, "y": 0.20}, "Germany": {"x": 0.22, "y": 0.88}, "Brazil": {"x": 0.35, "y": 0.30}}
    path_nodes, central_node = ["Germany", "China", "India"], "China"
    edges = [("United States", "Germany", "Energy Trade", "normal"), ("United States", "China", "Economic Link", "normal"), ("Germany", "China", "Industrial Path", "path"), ("China", "India", "Growth Path", "path"), ("China", "Russia", "Energy Supply", "normal"), ("China", "Brazil", "Emerging Demand", "normal"), ("India", "Japan", "Regional Link", "normal"), ("Russia", "Japan", "Power Route", "normal")]
    STANDARD_NODE_COLOR, PATH_NODE_COLOR, CENTRAL_NODE_COLOR = "cyan", "lightpink", "lightpink"
    fig = go.Figure()
    for source, target, label, edge_type in edges:
        x0, y0, x1, y1 = nodes[source]["x"], nodes[source]["y"], nodes[target]["x"], nodes[target]["y"]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color="black", width=3 if edge_type == "path" else 2, dash="dash" if edge_type == "path" else "solid"), hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=label, showarrow=False, font=dict(size=12, color="black"), bgcolor="rgba(255,255,255,0.75)")
    # Nodes
    for node, pos in nodes.items():
        color = CENTRAL_NODE_COLOR if node == central_node else (PATH_NODE_COLOR if node in path_nodes else STANDARD_NODE_COLOR)
        size = 95 if node == central_node else 85
        fig.add_trace(go.Scatter(x=[pos["x"]], y=[pos["y"]], mode="markers+text", text=[node], textposition="middle center", marker=dict(size=size, color=color, line=dict(color="black", width=3 if node not in path_nodes else 0)), textfont=dict(size=12, color="black", family="Arial Black"), showlegend=False))
    
    fig.update_layout(
        title=dict(text="Network Centrality: Global Electricity & Economic Influence", x=0.5, font=dict(size=22, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=760, margin=dict(l=60, r=60, t=90, b=60),
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, showline=True, linecolor="black", linewidth=2, mirror=True),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, showline=True, linecolor="black", linewidth=2, mirror=True)
    )
    return fig

def violin_asia_na(df):
    region_map = {"China": "Asia", "India": "Asia", "Japan": "Asia", "Russia": "Asia", "United States": "North America", "Canada": "North America", "Mexico": "North America"}
    df["region"] = df["country"].map(region_map)
    df_reg = df.dropna(subset=["region"]).copy()
    df_reg["log_gen"] = np.log10(df_reg["electricity_generation"] + 1)
    
    fig = go.Figure()
    for r in ["Asia", "North America"]:
        subset = df_reg[df_reg["region"] == r]
        median_log, median_real = subset["log_gen"].median(), subset["electricity_generation"].median()
        fig.add_trace(go.Violin(x=subset["region"], y=subset["log_gen"], name=r, box_visible=True, line_color="black", fillcolor="#A9D6E5" if r=="Asia" else "#B7E4C7", opacity=0.75))
        fig.add_annotation(x=r, y=median_log, text=f"{median_real:,.0f} TWh", showarrow=False, xshift=55, font=dict(size=12, color="black", family="Arial Black"), bgcolor="rgba(255,255,255,0.80)")
    
    fig.update_layout(
        title=dict(text="Electricity Generation Distribution: Asia vs North America", x=0.5, font=dict(size=22, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=720, font=dict(color="black", family="Arial"),
        xaxis=dict(title=dict(text="Region", font=dict(size=15, color="black", family="Arial Black")), showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(title=dict(text="Log Electricity Generation", font=dict(size=15, color="black", family="Arial Black")), showline=True, linecolor="black", linewidth=1, mirror=True)
    )
    return fig

def line_trend_ma(df):
    line_df = df[df["year"] >= 1980].groupby("year", as_index=False)["electricity_generation"].mean().sort_values("year")
    line_df["MA"] = line_df["electricity_generation"].rolling(window=5, center=True).mean()
    RAW_COLOR, TREND_COLOR = "#30E114", "#2C7DA0"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=line_df["year"], y=line_df["electricity_generation"], name="Yearly Average", line=dict(color=RAW_COLOR, width=2), opacity=0.25))
    fig.add_trace(go.Scatter(x=line_df["year"], y=line_df["MA"], name="5-Year Moving Average", line=dict(color=TREND_COLOR, width=4)))
    
    fig.update_layout(
        title=dict(text="Average Electricity Generation Trend Over Time", x=0.5, font=dict(size=22, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=700, font=dict(color="black", family="Arial"),
        xaxis=dict(title=dict(text="Year", font=dict(size=15, color="black", family="Arial Black")), showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(title=dict(text="Average Electricity Generation (TWh)", font=dict(size=15, color="black", family="Arial Black")), showline=True, linecolor="black", linewidth=1, mirror=True)
    )
    return fig

def area_trend(df):
    area_df = df[df["year"] >= 1980].groupby("year", as_index=False)["electricity_generation"].mean().sort_values("year")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=area_df["year"], y=area_df["electricity_generation"], fill="tozeroy", name="Average Electricity Generation", line_color="#2C7DA0", fillcolor="rgba(169, 214, 229, 0.65)"))
    
    fig.update_layout(
        title=dict(text="Average Electricity Generation Area Trend Over Time", x=0.5, font=dict(size=22, color="black", family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="white", height=700, font=dict(color="black", family="Arial"),
        xaxis=dict(title=dict(text="Year", font=dict(size=15, color="black", family="Arial Black")), showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(title=dict(text="Average Electricity Generation (TWh)", font=dict(size=15, color="black", family="Arial Black")), showline=True, linecolor="black", linewidth=1, mirror=True)
    )
    return fig
