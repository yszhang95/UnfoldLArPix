#!/usr/bin/env python3
"""
Interactive Dash app for event display visualization from deconvolved voxel data.
Filters and displays voxels with charge > threshold from deconv_q array.
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path
import os

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Get available npz files from examples directory (including subdirectories)
EXAMPLES_DIR = Path(__file__).parent
NPZ_FILES = sorted([str(f.relative_to(EXAMPLES_DIR)) for f in EXAMPLES_DIR.rglob("*.npz")])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Event Display - Deconvolved Voxel Viewer", className="mb-4")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select NPZ File:"),
            dcc.Dropdown(
                id='file-selector',
                options=[{'label': f, 'value': f} for f in NPZ_FILES],
                value=NPZ_FILES[0] if NPZ_FILES else None,
                clearable=False
            ),
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Charge Threshold:"),
            dcc.Slider(
                id='threshold-slider',
                min=0,
                max=15,
                step=0.1,
                value=1,
                marks={i: str(i) for i in range(0, 16)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], width=12, lg=6),
        dbc.Col([
            html.Label("Color Scale:"),
            dcc.RadioItems(
                id='color-scale',
                options=[
                    {'label': ' Linear', 'value': 'linear'},
                    {'label': ' Log', 'value': 'log'},
                ],
                value='linear',
                inline=True
            ),
        ], width=12, lg=6),
    ], className="mt-3"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading",
                type="default",
                children=[
                    dcc.Graph(id='event-display', style={'height': '800px'})
                ]
            )
        ], width=12)
    ], className="mt-4"),

    dbc.Row([
        dbc.Col([
            html.Div(id='stats-display', className="mt-3")
        ], width=12)
    ]),

    dcc.Store(id='loaded-data-store'),
], fluid=True, className="p-4")


# Global cache for loaded npz data (to avoid JSON serialization of large arrays)
_loaded_npz_cache = {}


@app.callback(
    Output('loaded-data-store', 'data'),
    Input('file-selector', 'value'),
)
def load_npz_file(filename):
    """Load NPZ file and store metadata (data stays in memory)."""
    if not filename:
        return {}

    filepath = EXAMPLES_DIR / filename
    if not filepath.exists():
        return {'error': f'File not found: {filename}'}

    try:
        data = np.load(filepath, allow_pickle=True)

        # Cache the full data in memory
        _loaded_npz_cache[filename] = data

        # Extract metadata only for the store
        result = {
            'filename': filename,
            'has_deconv_q': 'deconv_q' in data,
            'deconv_q_shape': list(data['deconv_q'].shape) if 'deconv_q' in data else None,
            'loaded': True,
        }

        print(f"Loaded {filename}, deconv_q shape: {result['deconv_q_shape']}")
        return result
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return {'error': str(e)}


@app.callback(
    Output('event-display', 'figure'),
    Output('stats-display', 'children'),
    Input('loaded-data-store', 'data'),
    Input('threshold-slider', 'value'),
    Input('color-scale', 'value'),
)
def update_display(loaded_data, threshold, color_scale):
    """Update event display based on threshold and loaded data."""

    if not loaded_data or not loaded_data.get('loaded'):
        return go.Figure().add_annotation(text="No data loaded"), html.Div("No data to display")

    if loaded_data.get('error'):
        return go.Figure().add_annotation(text=f"Error: {loaded_data['error']}"), html.Div(f"Error: {loaded_data['error']}")

    # Get data from cache
    filename = loaded_data.get('filename')
    if filename not in _loaded_npz_cache:
        return go.Figure().add_annotation(text="Data not in cache"), html.Div("Data not in cache")

    npz_data = _loaded_npz_cache[filename]
    if 'deconv_q' not in npz_data:
        return go.Figure().add_annotation(text="No deconv_q in file"), html.Div("No deconv_q in file")

    deconv_q = np.array(npz_data['deconv_q'])

    # Get voxel coordinates and charges
    mask = deconv_q > threshold
    coords = np.where(mask)
    charges = deconv_q[mask]

    if len(charges) == 0:
        fig = go.Figure().add_annotation(
            text=f"No voxels above threshold {threshold}"
        )
        stats = html.Div([
            html.P(f"Total voxels: {deconv_q.size}"),
            html.P(f"Voxels above threshold {threshold}: 0"),
        ])
        return fig, stats

    # Prepare plot data
    x, y, z = coords

    # Handle color scale
    if color_scale == 'log':
        color_vals = np.log10(charges + 1)
        colorbar_title = "log10(Q+1)"
    else:
        color_vals = charges
        colorbar_title = "Charge"

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=color_vals,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=colorbar_title),
            line=dict(width=0),
        ),
        text=[f"({xi}, {yi}, {zi})<br>Q={q:.2f}"
              for xi, yi, zi, q in zip(x, y, z, charges)],
        hovertemplate='%{text}<extra></extra>',
    )])

    fig.update_layout(
        title=f"Event Display - {filename} (threshold > {threshold})",
        scene=dict(
            xaxis_title="X (pixel)",
            yaxis_title="Y (pixel)",
            zaxis_title="Time",
            aspectmode='data',
        ),
        height=800,
        hovermode='closest',
    )

    # Statistics
    stats = dbc.Row([
        dbc.Col([
            html.H5("Statistics"),
            html.P(f"Grid shape: {deconv_q.shape}"),
            html.P(f"Total voxels: {deconv_q.size}"),
            html.P(f"Voxels above threshold {threshold}: {len(charges)}"),
            html.P(f"Min charge: {deconv_q.min():.3f}"),
            html.P(f"Max charge: {deconv_q.max():.3f}"),
            html.P(f"Mean charge (all): {deconv_q.mean():.3f}"),
            html.P(f"Mean charge (>threshold): {charges.mean():.3f}"),
            html.P(f"Total charge (>threshold): {charges.sum():.3f}"),
        ], width=12, md=4),
        dbc.Col([
            html.H5("Threshold Info"),
            html.P(f"Current threshold: {threshold}"),
            html.P(f"Percentage of voxels above threshold: {100*len(charges)/deconv_q.size:.2f}%"),
        ], width=12, md=4),
    ])

    return fig, stats


if __name__ == '__main__':
    print(f"Found {len(NPZ_FILES)} NPZ files in {EXAMPLES_DIR}")
    print("Starting Dash app at http://127.0.0.1:8050/")
    app.run(debug=True, host='127.0.0.1', port=8050)
