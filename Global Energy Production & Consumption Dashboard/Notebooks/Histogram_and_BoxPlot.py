"""

    from Histogram_and_BoxPlot import make_histogram, make_kde, make_boxplot

    fig1 = make_histogram(df_countries)
    fig2 = make_kde(df_modern)
    fig3 = make_boxplot(df_modern)

    fig1.show()
"""

import pandas as pd

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# ── Shared style constants 
df = pd.read_csv("preprocessing/owid_energy_visualization_cleaned.csv")
TEMPLATE   = 'plotly_white'
HIST_COLOR = '#1f77b4'
BOX_COLORS = px.colors.qualitative.Bold

LEGEND_STYLE = dict(
    orientation='v',
    x=1.02, y=1,
    xanchor='left',
    bgcolor='rgba(255,255,255,0.8)',
    bordercolor='#cccccc',
    borderwidth=1,
)


# ── 1. Side-by-side Histogram (Raw vs Log) 
def make_histogram(df_countries: "pd.DataFrame") -> go.Figure:
    """
    Side-by-side histogram comparing raw vs log-transformed
    electricity generation.
 
    Drops the repeated placeholder value (~21.86 TWh) that accounts for
    65% of rows and collapses both histograms into a single spike.
    The remaining 6,590 rows represent countries with real variance.
 
    
    """
    # Drop the 21.86 placeholder (65.5% of rows) and zeros — keep real variance only
    mask = (
        df_countries['electricity_generation'] > 0
    ) & ~(
        df_countries['electricity_generation'].between(21.85, 21.87)
    )
    clean = df_countries[mask].copy()
 
    elec_raw = clean['electricity_generation'].dropna()
    elec_log = clean['log_elec'].dropna()
 
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            ' Raw scale  (nearly unusable)',
            ' Log scale  (near-normal)',
        )
    )
 
    # Raw histogram
    fig.add_trace(
        go.Histogram(
            x=elec_raw, nbinsx=60,
            marker_color=HIST_COLOR, opacity=0.80,
            name='Raw scale', legendgroup='raw',
        ),
        row=1, col=1,
    )
 
    # Log histogram
    fig.add_trace(
        go.Histogram(
            x=elec_log, nbinsx=50,
            marker_color='#2ca02c', opacity=0.80,
            name='Log scale', legendgroup='log',
        ),
        row=1, col=2,
    )
 
    # Dummy trace so "Median" appears in legend
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='crimson', dash='dash', width=1.5),
            name='Median',
        ),
        row=1, col=1,
    )
 
    # Actual median vlines
    for col_idx, (series, label) in enumerate(
        [
            (elec_raw, f'Median: {elec_raw.median():.0f} TWh'),
            (elec_log, f'Median (log): {elec_log.median():.2f}'),
        ],
        start=1,
    ):
        fig.add_vline(
            x=series.median(), line_dash='dash', line_color='crimson',
            annotation_text=label, annotation_position='top right',
            annotation_font_color='crimson', row=1, col=col_idx,
        )
 
    fig.update_layout(
        title_text=f'Distribution of Electricity Generation',
        title_font_size=16,
        template=TEMPLATE,
        height=480,
        showlegend=True,
        legend=dict(title='Series', **LEGEND_STYLE),
        font=dict(size=12),
    )
    fig.update_xaxes(title_text='Electricity Generation (TWh)', col=1)
    fig.update_xaxes(title_text='log₁₊ₓ (Electricity Generation)', col=2)
    fig.update_yaxes(title_text='Number of Observations', col=1)
 
    return fig
 


# ── 2. KDE by Decade 
def make_kde(df_modern: "pd.DataFrame") -> go.Figure:

# Histogram + per-decade KDE curves on the log-transformed scale.
        
    
    decade_labels = sorted(df_modern['decade_label'].unique())
    decade_colors = dict(zip(decade_labels, BOX_COLORS[:len(decade_labels)]))

    fig = go.Figure()

    # Background histogram
    fig.add_trace(go.Histogram(
        x=df_modern['log_elec'].dropna(),
        nbinsx=50,
        histnorm='probability density',
        marker_color='lightgrey', opacity=0.5,
        name='All (1980–2025)',
    ))

    # KDE curve per decade
    for dlabel in decade_labels:
        vals = df_modern.loc[
            df_modern['decade_label'] == dlabel, 'log_elec'
        ].dropna().values
        if len(vals) < 10:
            continue
        kde     = gaussian_kde(vals, bw_method=0.25)
        x_range = np.linspace(vals.min(), vals.max(), 300)
        fig.add_trace(go.Scatter(
            x=x_range, y=kde(x_range),
            mode='lines', name=dlabel,
            line=dict(color=decade_colors[dlabel], width=2.2),
        ))

    fig.update_layout(
        title='Distribution of Electricity Generation by Decade — KDE on log Scale',
        yaxis_title='Density',
        template=TEMPLATE,
        height=520,
        font=dict(size=12),
        title_font_size=16,
        legend=dict(title='Decade', **LEGEND_STYLE),
    )
    fig.update_xaxes(
        tickvals=[np.log1p(v) for v in [1, 10, 50, 200, 500, 2000, 5000]],
        ticktext=['1', '10', '50', '200', '500', '2,000', '5,000'],
        title_text='Electricity Generation (TWh) — log scale',
    )

    return fig


# ── 3. Box Plot by Decade
def make_boxplot(df_modern: "pd.DataFrame") -> go.Figure:
    
    #Box plot of log-transformed electricity generation, one box per decade.

    decade_labels = sorted(df_modern['decade_label'].unique())

    twh_ticks   = [1, 5, 20, 50, 200, 500, 2000, 5000, 10000]
    log_ticks   = [np.log1p(v) for v in twh_ticks]
    tick_labels = ['1', '5', '20', '50', '200', '500', '2k', '5k', '10k']

    fig = go.Figure()

    for i, dlabel in enumerate(decade_labels):
        subset = df_modern[df_modern['decade_label'] == dlabel]['log_elec'].dropna()
        fig.add_trace(go.Box(
            y=subset,
            name=dlabel,
            marker_color=BOX_COLORS[i % len(BOX_COLORS)],
            line=dict(width=1.5),
            marker=dict(size=3, opacity=0.3),
            boxpoints='outliers',
            showlegend=True,
        ))

    fig.update_layout(
        title='Distribution of Electricity Generation by Decade — Log Scale',
        yaxis=dict(
            title='Electricity Generation (TWh) — log scale',
            tickvals=log_ticks,
            ticktext=tick_labels,
            gridcolor='#e8e8e8',
        ),
        xaxis_title='Decade',
        template=TEMPLATE,
        height=560,
        font=dict(size=12),
        title_font_size=16,
        showlegend=True,
        legend=dict(title='Decade', **LEGEND_STYLE),
    )

    return fig



if __name__ == '__main__':

    
    AGGREGATE_PATTERN = r'\(Ember\)|\(Ei\)|World|G20|Asia$|Africa$|Europe$|America|Income|Oecd|Opec|Asean|Non-'
    df_countries = df[~df['country'].str.contains(AGGREGATE_PATTERN, case=False, regex=True)].copy()
    df_countries['log_elec'] = np.log1p(df_countries['electricity_generation'])
    df_modern = df_countries[df_countries['year'] >= 1980].copy()
    df_modern['decade_label'] = df_modern['decade'].astype(str) + 's'


    make_histogram(df_countries).show()
    make_kde(df_modern).show()
    make_boxplot(df_modern).show()