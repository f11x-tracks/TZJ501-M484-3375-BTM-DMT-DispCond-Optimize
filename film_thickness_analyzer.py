import dash
from dash import dcc, html, Input, Output, ALL, callback_context, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
import glob
import xml.etree.ElementTree as ET

# Load and process data
def load_data():
    """Load film thickness data and condition table"""
    # Load condition table
    conditions_df = pd.read_csv('BTM/condition-table.txt', sep='\t')
    
    # Load all CSV measurement files in the BTM folder (excluding the condition table)
    csv_files = [
        f for f in os.listdir('BTM')
        if f.endswith('.csv') and f != 'condition-table.txt'
    ]
    
    if not csv_files:
        raise FileNotFoundError("No CSV measurement files found in BTM/")
    
    frames = []
    for csv_file in csv_files:
        part = pd.read_csv(os.path.join('BTM', csv_file))
        part.columns = part.columns.str.strip()
        # Strip whitespace from all string columns
        str_cols = part.select_dtypes(include=['object', 'str']).columns
        part[str_cols] = part[str_cols].apply(lambda c: c.str.strip())
        frames.append(part)
    
    thickness_df = pd.concat(frames, ignore_index=True)
    
    # Strip condition table string columns too
    str_cols = conditions_df.select_dtypes(include=['object', 'str']).columns
    conditions_df[str_cols] = conditions_df[str_cols].apply(lambda c: c.str.strip())
    
    # Merge with conditions
    merged_df = thickness_df.merge(conditions_df, on='WaferID', how='left')
    
    # Calculate radial distance from center
    merged_df['Radius'] = np.sqrt(merged_df['X[mm]']**2 + merged_df['Y[mm]']**2)
    
    # Define zones
    def get_zone(radius):
        if radius <= 70:
            return 'Center'
        elif radius <= 140:
            return 'Mid'
        else:
            return 'Edge'
    
    merged_df['Zone'] = merged_df['Radius'].apply(get_zone)
    
    return merged_df

# Create condition identifier for each unique combination
def create_condition_id(df):
    """Create a unique condition identifier"""
    condition_cols = ['DispT', 'PumpT', 'DispSS', 'RlxT', 'RlxSS', 'Cast', 'Other']
    df['Condition_ID'] = df[condition_cols].apply(
        lambda x: f"DispT:{x['DispT']}, PumpT:{x['PumpT']}, DispSS:{x['DispSS']}, RlxT:{x['RlxT']}, RlxSS:{x['RlxSS']}, Cast:{x['Cast']}, Other:{x['Other']}", axis=1
    )
    return df

# Load data
df = load_data()
df = create_condition_id(df)

CONDITION_COLS = ['DispT', 'PumpT', 'DispSS', 'RlxT', 'RlxSS', 'Cast', 'Other']


def load_dmt_data():
    """Load Layer 1 Thickness data from DMT XML files and merge with condition table"""
    conditions_df = pd.read_csv('DMT/condition-table.txt', sep='\t')
    str_cols = conditions_df.select_dtypes(include=['object', 'str']).columns
    conditions_df[str_cols] = conditions_df[str_cols].apply(lambda c: c.str.strip())

    xml_files = glob.glob(os.path.join('DMT', '*.xml'))
    if not xml_files:
        raise FileNotFoundError("No XML files found in DMT/")

    records = []
    for filepath in xml_files:
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            for dr in root.findall('.//DataRecord'):
                label = dr.findtext('Label')
                if label != 'Layer 1 Thickness':
                    continue
                datum = dr.findtext('Datum')
                wafer_id = dr.findtext('WaferID')
                x_str = dr.findtext('XWaferLoc')
                y_str = dr.findtext('YWaferLoc')
                try:
                    datum_val = float(datum)
                    x_val = float(x_str)
                    y_val = float(y_str)
                except (TypeError, ValueError):
                    continue
                records.append({
                    'WaferID': wafer_id,
                    'Film Thickness': round(datum_val, 1),
                    'X[mm]': x_val,
                    'Y[mm]': y_val,
                })
        except Exception as e:
            print(f"DMT: error parsing {filepath}: {e}")

    if not records:
        raise ValueError("No Layer 1 Thickness records found in DMT XML files")

    dmt_df = pd.DataFrame(records)
    dmt_df['WaferID'] = dmt_df['WaferID'].str.strip()

    dmt_df = dmt_df.merge(conditions_df, on='WaferID', how='left')

    dmt_df['Radius'] = np.sqrt(dmt_df['X[mm]']**2 + dmt_df['Y[mm]']**2)

    def get_zone(r):
        if r <= 70:
            return 'Center'
        elif r <= 140:
            return 'Mid'
        return 'Edge'

    dmt_df['Zone'] = dmt_df['Radius'].apply(get_zone)
    return dmt_df


dmt_df = load_dmt_data()
dmt_df = create_condition_id(dmt_df)

# Initialize Dash app
app = dash.Dash(__name__)


# ── DMT plot helpers ──────────────────────────────────────────────────────────

def create_dmt_contour_plots(filtered_df, selected_conditions, plot_size=600):
    """Contour plots for each DMT condition using XWaferLoc / YWaferLoc"""
    plots = []
    for condition in selected_conditions:
        cond_data = filtered_df[filtered_df['Condition_ID'] == condition]
        if cond_data.empty:
            continue

        xi = np.linspace(-150, 150, 100)
        yi = np.linspace(-150, 150, 100)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        mask = np.sqrt(xi_grid**2 + yi_grid**2) <= 150

        points = cond_data[['X[mm]', 'Y[mm]']].values
        values = cond_data['Film Thickness'].values
        zi = griddata(points, values, (xi_grid, yi_grid), method='cubic')
        zi[~mask] = np.nan

        fig = go.Figure(data=go.Contour(
            z=zi, x=xi, y=yi,
            colorscale='Viridis',
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
            hovertemplate='X: %{x} mm<br>Y: %{y} mm<br>Thickness: %{z:.1f}<extra></extra>'
        ))

        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=150 * np.cos(theta), y=150 * np.sin(theta),
            mode='lines', line=dict(color='black', width=2),
            name='Wafer Edge', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=cond_data['X[mm]'], y=cond_data['Y[mm]'],
            mode='markers', marker=dict(color='red', size=4),
            name='Measurement Points',
            hovertemplate='X: %{x:.1f} mm<br>Y: %{y:.1f} mm<extra></extra>'
        ))

        fig.update_layout(
            title=f'DMT Thickness Contour - {condition}',
            xaxis_title='X [mm]', yaxis_title='Y [mm]',
            xaxis=dict(range=[-150, 150], constrain='domain'),
            yaxis=dict(range=[-150, 150], scaleanchor='x', scaleratio=1),
            width=plot_size, height=plot_size, showlegend=True
        )
        plots.append(html.Div(dcc.Graph(figure=fig), style={'flex': '0 0 auto', 'margin': '5px'}))

    return html.Div(plots, style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-start'})


def create_dmt_radial_plots(filtered_df, selected_conditions):
    """Radial profile plot for DMT data"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for i, condition in enumerate(selected_conditions):
        cond_data = filtered_df[filtered_df['Condition_ID'] == condition].copy()
        if cond_data.empty:
            continue

        cond_data = cond_data.sort_values('Radius')
        color = colors[i % len(colors)]

        # LOWESS trend on raw data
        if len(cond_data) >= 5:
            try:
                smoothed = lowess(
                    cond_data['Film Thickness'],
                    cond_data['Radius'],
                    frac=0.3, it=3, return_sorted=True
                )
                fig.add_trace(go.Scatter(
                    x=smoothed[:, 0], y=smoothed[:, 1],
                    mode='lines', name=f'{condition} (LOWESS)',
                    line=dict(color=color, width=3)
                ))
            except Exception:
                pass

        fig.add_trace(go.Scatter(
            x=cond_data['Radius'], y=cond_data['Film Thickness'],
            mode='markers', name=f'{condition} (Points)',
            marker=dict(color=color, size=6, symbol='circle-open'),
            hovertemplate='Radius: %{x:.1f} mm<br>Thickness: %{y:.1f}<br>WaferID: %{text}<extra></extra>',
            text=cond_data['WaferID']
        ))

    fig.add_vline(x=70, line_dash='dash', line_color='gray',
                  annotation_text='Center/Mid', annotation_position='top')
    fig.add_vline(x=140, line_dash='dash', line_color='gray',
                  annotation_text='Mid/Edge', annotation_position='top')

    fig.update_layout(
        title='DMT Radial Film Thickness Profile (Center to Edge)',
        xaxis_title='Radius [mm]', yaxis_title='Film Thickness',
        width=1200, height=600, showlegend=True, hovermode='x unified'
    )
    return dcc.Graph(figure=fig)


def create_dmt_summary_table(filtered_df, selected_conditions):
    """Summary table for DMT conditions - one row per wafer"""
    summary_data = []
    for condition in selected_conditions:
        cond_data = filtered_df[filtered_df['Condition_ID'] == condition]
        if cond_data.empty:
            continue
        
        # Group by WaferID to get individual wafer statistics
        for wafer_id, wafer_data in cond_data.groupby('WaferID'):
            params = wafer_data[CONDITION_COLS].iloc[0]
            overall_mean = wafer_data['Film Thickness'].mean()
            overall_std = wafer_data['Film Thickness'].std()
            zone_stats = wafer_data.groupby('Zone')['Film Thickness'].agg(['mean', 'std', 'min', 'max']).round(2)

            def zone_val(zone, col, fmt):
                if zone in zone_stats.index and not np.isnan(zone_stats.loc[zone, col]):
                    return fmt.format(zone_stats.loc[zone, col])
                return 'N/A'

            row = {'WaferID': wafer_id}  # Add WaferID as first column
            row.update({col: params[col] for col in CONDITION_COLS})
            row.update({
                'Overall Mean': f"{overall_mean:.1f}",
                'Overall Std': f"{overall_std:.2f}",
                'Center Std': zone_val('Center', 'std', '{:.2f}'),
                'Mid Std': zone_val('Mid', 'std', '{:.2f}'),
                'Edge Mean': zone_val('Edge', 'mean', '{:.1f}'),
                'Edge Std': zone_val('Edge', 'std', '{:.2f}'),
                'Total Points': len(wafer_data)
            })
            summary_data.append(row)

    if not summary_data:
        return html.Div("No data available.")

    stat_cols = ['Overall Mean', 'Overall Std',
                 'Center Std', 'Mid Std', 'Edge Mean', 'Edge Std',
                 'Total Points']
    all_cols = ['WaferID'] + CONDITION_COLS + stat_cols

    return html.Div([
        dash_table.DataTable(
            data=summary_data,
            columns=[{'name': c, 'id': c} for c in all_cols],
            style_cell={'textAlign': 'center', 'padding': '8px', 'minWidth': '80px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data={'backgroundColor': 'rgb(248, 248, 248)'},
            style_data_conditional=[
                {'if': {'column_id': col},
                 'backgroundColor': 'rgb(220, 230, 245)', 'fontWeight': 'bold'}
                for col in CONDITION_COLS
            ]
        )
    ])

app.layout = html.Div([
    html.H1("Film Thickness Analysis Dashboard",
            style={'textAlign': 'center', 'marginBottom': 30}),

    dcc.Tabs([

        # ── BTM Tab ───────────────────────────────────────────────────────────
        dcc.Tab(label='BTM', children=[
            # One dropdown per condition parameter
            html.Div(
                [
                    html.Div([
                        html.Label(col, style={'fontWeight': 'bold', 'marginBottom': 4}),
                        dcc.Dropdown(
                            id={'type': 'param-drop', 'index': col},
                            options=[{'label': str(v), 'value': v}
                                     for v in sorted(df[col].dropna().unique())],
                            multi=True,
                            clearable=True,
                            placeholder='All',
                        )
                    ], style={'flex': '1', 'minWidth': '130px', 'margin': '5px'})
                    for col in CONDITION_COLS
                ],
                style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '20px'}
            ),

            # Contour size selector
            html.Div([
                html.Label("Contour Plot Size:", style={'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Dropdown(
                    id='contour-size-drop',
                    options=[{'label': f'{s}px', 'value': s} for s in [400, 500, 600, 700, 800]],
                    value=600,
                    clearable=False,
                    style={'width': '120px', 'display': 'inline-block'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'margin': '0 20px 20px 20px'}),

            dcc.Tabs([
                dcc.Tab(label='Maps & Profile', children=[
                    html.Div(id='page-content', style={'margin': '20px'})
                ]),
                dcc.Tab(label='Charts', children=[
                    html.Div(id='btm-charts-content', style={'margin': '20px'})
                ]),
            ], style={'marginTop': '10px'})
        ]),

        # ── DMT Tab ───────────────────────────────────────────────────────────
        dcc.Tab(label='DMT', children=[
            # One dropdown per condition parameter (DMT)
            html.Div(
                [
                    html.Div([
                        html.Label(col, style={'fontWeight': 'bold', 'marginBottom': 4}),
                        dcc.Dropdown(
                            id={'type': 'dmt-param-drop', 'index': col},
                            options=[{'label': str(v), 'value': v}
                                     for v in sorted(dmt_df[col].dropna().unique())],
                            multi=True,
                            clearable=True,
                            placeholder='All',
                        )
                    ], style={'flex': '1', 'minWidth': '130px', 'margin': '5px'})
                    for col in CONDITION_COLS
                ],
                style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '20px'}
            ),

            # Contour size selector (DMT)
            html.Div([
                html.Label("Contour Plot Size:", style={'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Dropdown(
                    id='dmt-contour-size-drop',
                    options=[{'label': f'{s}px', 'value': s} for s in [400, 500, 600, 700, 800]],
                    value=600,
                    clearable=False,
                    style={'width': '120px', 'display': 'inline-block'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'margin': '0 20px 20px 20px'}),

            dcc.Tabs([
                dcc.Tab(label='Maps & Profile', children=[
                    html.Div(id='dmt-page-content', style={'margin': '20px'})
                ]),
                dcc.Tab(label='Charts', children=[
                    html.Div(id='dmt-charts-content', style={'margin': '20px'})
                ]),
            ], style={'marginTop': '10px'})
        ]),

    ])
])

@app.callback(
    Output('page-content', 'children'),
    Input({'type': 'param-drop', 'index': ALL}, 'value'),
    Input('contour-size-drop', 'value')
)
def render_content(param_values, contour_size):
    # Filter df by each parameter dropdown; empty/None means "all values"
    filtered_df = df.copy()
    for col, vals in zip(CONDITION_COLS, param_values):
        if vals:
            filtered_df = filtered_df[filtered_df[col].isin(vals)]

    selected_conditions = filtered_df['Condition_ID'].unique().tolist()
    if not selected_conditions:
        return html.Div("No conditions match the selected filters.")

    return html.Div([
        html.H2("Contour Plots", style={'marginTop': 20}),
        create_contour_plots(filtered_df, selected_conditions, contour_size),
        html.Hr(),
        html.H2("Radial Profile", style={'marginTop': 20}),
        create_radial_plots(filtered_df, selected_conditions),
        html.Hr(),
        html.H2("Summary Statistics", style={'marginTop': 20}),
        create_summary_table(filtered_df, selected_conditions),
    ])


@app.callback(
    Output('dmt-page-content', 'children'),
    Input({'type': 'dmt-param-drop', 'index': ALL}, 'value'),
    Input('dmt-contour-size-drop', 'value')
)
def render_dmt_content(param_values, contour_size):
    filtered_df = dmt_df.copy()
    for col, vals in zip(CONDITION_COLS, param_values):
        if vals:
            filtered_df = filtered_df[filtered_df[col].isin(vals)]

    selected_conditions = filtered_df['Condition_ID'].unique().tolist()
    if not selected_conditions:
        return html.Div("No conditions match the selected filters.")

    return html.Div([
        html.H2("Contour Plots", style={'marginTop': 20}),
        create_dmt_contour_plots(filtered_df, selected_conditions, contour_size),
        html.Hr(),
        html.H2("Radial Profile", style={'marginTop': 20}),
        create_dmt_radial_plots(filtered_df, selected_conditions),
        html.Hr(),
        html.H2("Summary Statistics", style={'marginTop': 20}),
        create_dmt_summary_table(filtered_df, selected_conditions),
    ])


@app.callback(
    Output('btm-charts-content', 'children'),
    Input({'type': 'param-drop', 'index': ALL}, 'value'),
)
def render_btm_charts(param_values):
    filtered_df = df.copy()
    for col, vals in zip(CONDITION_COLS, param_values):
        if vals:
            filtered_df = filtered_df[filtered_df[col].isin(vals)]
    selected_conditions = filtered_df['Condition_ID'].unique().tolist()
    if not selected_conditions:
        return html.Div("No conditions match the selected filters.")
    return html.Div([
        html.H2("Mean & Std Dev by Condition", style={'marginTop': 20}),
        create_condition_stats_plot(filtered_df, selected_conditions),
    ])


@app.callback(
    Output('dmt-charts-content', 'children'),
    Input({'type': 'dmt-param-drop', 'index': ALL}, 'value'),
)
def render_dmt_charts(param_values):
    filtered_df = dmt_df.copy()
    for col, vals in zip(CONDITION_COLS, param_values):
        if vals:
            filtered_df = filtered_df[filtered_df[col].isin(vals)]
    selected_conditions = filtered_df['Condition_ID'].unique().tolist()
    if not selected_conditions:
        return html.Div("No conditions match the selected filters.")
    return html.Div([
        html.H2("Mean & Std Dev by Condition", style={'marginTop': 20}),
        create_condition_stats_plot(filtered_df, selected_conditions),
    ])

def create_condition_stats_plot(filtered_df, selected_conditions):
    """Line chart of mean and std dev per selected condition"""
    rows = []
    for condition in selected_conditions:
        vals = filtered_df[filtered_df['Condition_ID'] == condition]['Film Thickness']
        if vals.empty:
            continue
        rows.append({
            'Condition': condition,
            'Mean': round(vals.mean(), 2),
            'Std Dev': round(vals.std(), 2),
        })

    if not rows:
        return html.Div("No data available.")

    stats_df = pd.DataFrame(rows)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stats_df['Condition'],
        y=stats_df['Mean'],
        mode='lines+markers+text',
        name='Mean',
        yaxis='y1',
        marker=dict(size=8),
        line=dict(width=2),
        text=stats_df['Mean'].map(lambda v: f'{v:.1f}'),
        textposition='top center',
        hovertemplate='Condition: %{x}<br>Mean: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=stats_df['Condition'],
        y=stats_df['Std Dev'],
        mode='lines+markers+text',
        name='Std Dev',
        yaxis='y2',
        marker=dict(size=8, symbol='diamond'),
        line=dict(width=2, dash='dash'),
        text=stats_df['Std Dev'].map(lambda v: f'{v:.2f}'),
        textposition='bottom center',
        hovertemplate='Condition: %{x}<br>Std Dev: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Mean & Std Dev by Condition',
        xaxis=dict(title='Condition', tickangle=-45),
        yaxis=dict(title='Mean Thickness', side='left'),
        yaxis2=dict(title='Std Dev', side='right', overlaying='y', showgrid=False),
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=60, t=60, b=120),
        hovermode='x unified'
    )

    return dcc.Graph(figure=fig)


def create_contour_plots(filtered_df, selected_conditions, plot_size=600):
    """Create contour plots for each selected condition"""
    plots = []
    
    for condition in selected_conditions:
        condition_data = filtered_df[filtered_df['Condition_ID'] == condition]
        
        if condition_data.empty:
            continue
            
        # Create grid for interpolation
        xi = np.linspace(-150, 150, 100)
        yi = np.linspace(-150, 150, 100)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate thickness data
        points = condition_data[['X[mm]', 'Y[mm]']].values
        values = condition_data['Film Thickness'].values
        
        # Only interpolate within the wafer boundary (150mm radius)
        mask = np.sqrt(xi_grid**2 + yi_grid**2) <= 150
        
        zi = griddata(points, values, (xi_grid, yi_grid), method='cubic')
        zi[~mask] = np.nan
        
        # Create contour plot
        fig = go.Figure(data=go.Contour(
            z=zi,
            x=xi,
            y=yi,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            hovertemplate='X: %{x} mm<br>Y: %{y} mm<br>Thickness: %{z:.0f}<extra></extra>'
        ))
        
        # Add wafer boundary circle
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=150*np.cos(theta),
            y=150*np.sin(theta),
            mode='lines',
            line=dict(color='black', width=2),
            name='Wafer Edge',
            hoverinfo='skip'
        ))
        
        # Add measurement points
        fig.add_trace(go.Scatter(
            x=condition_data['X[mm]'],
            y=condition_data['Y[mm]'],
            mode='markers',
            marker=dict(color='red', size=4),
            name='Measurement Points',
            hovertemplate='X: %{x} mm<br>Y: %{y} mm<br>Point: %{text}<extra></extra>',
            text=condition_data['Point No']
        ))
        
        fig.update_layout(
            title=f'Film Thickness Contour - {condition}',
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            xaxis=dict(range=[-150, 150], constrain='domain'),
            yaxis=dict(range=[-150, 150], scaleanchor='x', scaleratio=1),
            width=plot_size,
            height=plot_size,
            showlegend=True
        )

        plots.append(html.Div(dcc.Graph(figure=fig), style={'flex': '0 0 auto', 'margin': '5px'}))

    return html.Div(plots, style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-start'})

def create_radial_plots(filtered_df, selected_conditions):
    """Create radial profile plots (center to edge)"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, condition in enumerate(selected_conditions):
        condition_data = filtered_df[filtered_df['Condition_ID'] == condition].copy()
        
        if condition_data.empty:
            continue
        
        condition_data = condition_data.sort_values('Radius')
        color = colors[i % len(colors)]

        # LOWESS trend on raw data
        if len(condition_data) >= 5:
            try:
                smoothed = lowess(
                    condition_data['Film Thickness'],
                    condition_data['Radius'],
                    frac=0.3, it=3, return_sorted=True
                )
                fig.add_trace(go.Scatter(
                    x=smoothed[:, 0],
                    y=smoothed[:, 1],
                    mode='lines',
                    name=f'{condition} (LOWESS)',
                    line=dict(color=color, width=3)
                ))
            except Exception:
                pass
        
        # Add actual measurement points
        fig.add_trace(go.Scatter(
            x=condition_data['Radius'],
            y=condition_data['Film Thickness'],
            mode='markers',
            name=f'{condition} (Points)',
            marker=dict(color=color, size=6, symbol='circle-open'),
            hovertemplate='Radius: %{x:.1f} mm<br>Thickness: %{y:.0f}<br>WaferID: %{text}<extra></extra>',
            text=condition_data['WaferID']
        ))
    
    # Add zone boundaries
    fig.add_vline(x=70, line_dash="dash", line_color="gray", 
                  annotation_text="Center/Mid", annotation_position="top")
    fig.add_vline(x=140, line_dash="dash", line_color="gray", 
                  annotation_text="Mid/Edge", annotation_position="top")
    
    fig.update_layout(
        title='Radial Film Thickness Profile (Center to Edge)',
        xaxis_title='Radius [mm]',
        yaxis_title='Film Thickness',
        width=1200,
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return dcc.Graph(figure=fig)

def create_summary_table(filtered_df, selected_conditions):
    """Create summary table: one row per wafer, parameter columns + stats columns"""
    summary_data = []

    for condition in selected_conditions:
        condition_data = filtered_df[filtered_df['Condition_ID'] == condition]
        if condition_data.empty:
            continue
        
        # Group by WaferID to get individual wafer statistics
        for wafer_id, wafer_data in condition_data.groupby('WaferID'):
            # Individual parameter values (from first row of this wafer group)
            params = wafer_data[CONDITION_COLS].iloc[0]

            overall_mean = wafer_data['Film Thickness'].mean()
            overall_std = wafer_data['Film Thickness'].std()
            zone_stats = wafer_data.groupby('Zone')['Film Thickness'].agg(['mean', 'std', 'min', 'max']).round(2)

            def zone_val(zone, col, fmt):
                if zone in zone_stats.index and not np.isnan(zone_stats.loc[zone, col]):
                    return fmt.format(zone_stats.loc[zone, col])
                return 'N/A'

            row = {'WaferID': wafer_id}  # Add WaferID as first column
            row.update({col: params[col] for col in CONDITION_COLS})
            row.update({
                'Overall Mean': f"{overall_mean:.1f}",
                'Overall Std': f"{overall_std:.2f}",
                'Center Std': zone_val('Center', 'std', '{:.2f}'),
                'Mid Std': zone_val('Mid', 'std', '{:.2f}'),
                'Edge Mean': zone_val('Edge', 'mean', '{:.1f}'),
                'Edge Std': zone_val('Edge', 'std', '{:.2f}'),
                'Total Points': len(wafer_data)
            })
            summary_data.append(row)

    if not summary_data:
        return html.Div("No data available.")

    stat_cols = ['Overall Mean', 'Overall Std',
                 'Center Std', 'Mid Std', 'Edge Mean', 'Edge Std',
                 'Total Points']
    all_cols = ['WaferID'] + CONDITION_COLS + stat_cols

    return html.Div([
        dash_table.DataTable(
            data=summary_data,
            columns=[{'name': c, 'id': c} for c in all_cols],
            style_cell={'textAlign': 'center', 'padding': '8px', 'minWidth': '80px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data={'backgroundColor': 'rgb(248, 248, 248)'},
            style_data_conditional=[
                {'if': {'column_id': col},
                 'backgroundColor': 'rgb(220, 230, 245)', 'fontWeight': 'bold'}
                for col in CONDITION_COLS
            ]
        )
    ])

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)