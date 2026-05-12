import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# ── Shared style constants 
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
def make_histogram(df):
    df = df.copy()
    if 'log_elec' not in df.columns:
        df['log_elec'] = np.log1p(df['electricity_generation'])
    
    mask = (df['electricity_generation'] > 0) & ~(df['electricity_generation'].between(21.85, 21.87))
    clean = df[mask].copy()
    elec_raw = clean['electricity_generation'].dropna()
    elec_log = clean['log_elec'].dropna()

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Raw scale', 'Log scale'))
    fig.add_trace(go.Histogram(x=elec_raw, nbinsx=60, marker_color=HIST_COLOR, opacity=0.8, name='Raw scale'), row=1, col=1)
    fig.add_trace(go.Histogram(x=elec_log, nbinsx=50, marker_color='#2ca02c', opacity=0.8, name='Log scale'), row=1, col=2)

    for col_idx, (series, label) in enumerate([(elec_raw, f'Median: {elec_raw.median():.0f}'), (elec_log, f'Median (log): {elec_log.median():.2f}')], start=1):
        fig.add_vline(x=series.median(), line_dash='dash', line_color='crimson', annotation_text=label, row=1, col=col_idx)

    fig.update_layout(title='Distribution of Electricity Generation', template=TEMPLATE, height=450)
    return fig

# ── 2. KDE by Decade 
def make_kde(df):
    df = df[df['year'] >= 1980].copy()
    if 'log_elec' not in df.columns:
        df['log_elec'] = np.log1p(df['electricity_generation'])
    df['decade_label'] = df['decade'].astype(str) + 's'
    
    decade_labels = sorted(df['decade_label'].unique())
    decade_colors = dict(zip(decade_labels, BOX_COLORS[:len(decade_labels)]))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['log_elec'].dropna(), nbinsx=50, histnorm='probability density', marker_color='lightgrey', opacity=0.5, name='All'))

    for dlabel in decade_labels:
        vals = df.loc[df['decade_label'] == dlabel, 'log_elec'].dropna().values
        if len(vals) < 10: continue
        kde = gaussian_kde(vals, bw_method=0.25)
        x_range = np.linspace(vals.min(), vals.max(), 300)
        fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', name=dlabel, line=dict(color=decade_colors.get(dlabel, 'blue'), width=2)))

    fig.update_layout(title='Electricity Generation KDE by Decade (Log Scale)', template=TEMPLATE, height=500)
    return fig

# ── 3. Box Plot by Decade
def make_boxplot(df):
    df = df[df['year'] >= 1980].copy()
    if 'log_elec' not in df.columns:
        df['log_elec'] = np.log1p(df['electricity_generation'])
    df['decade_label'] = df['decade'].astype(str) + 's'
    
    decade_labels = sorted(df['decade_label'].unique())
    fig = go.Figure()

    for i, dlabel in enumerate(decade_labels):
        subset = df[df['decade_label'] == dlabel]['log_elec'].dropna()
        fig.add_trace(go.Box(y=subset, name=dlabel, marker_color=BOX_COLORS[i % len(BOX_COLORS)], boxpoints='outliers'))

    fig.update_layout(title='Electricity Generation Box Plot by Decade', template=TEMPLATE, height=500)
    return fig