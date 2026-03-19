#!/usr/bin/env python3
"""
Interactive Dash app for event display visualization from deconvolved voxel data.
Filters and displays voxels with charge > threshold from deconv_q array.
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from pathlib import Path
import os

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Get available npz files from examples directory (including subdirectories)
EXAMPLES_DIR = Path(__file__).parent

def get_npz_files():
    """Scan for available NPZ files."""
    return sorted([str(f.relative_to(EXAMPLES_DIR)) for f in EXAMPLES_DIR.rglob("*.npz")])

NPZ_FILES = get_npz_files()

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
        ], width=10),
        dbc.Col([
            html.Label(""),
            dbc.Button(
                "🔄 Refresh Files",
                id='refresh-files-btn',
                color="info",
                className="w-100 mt-2",
                size="sm"
            ),
        ], width=2),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Charge Threshold:"),
            dcc.Slider(
                id='threshold-slider',
                min=0,
                max=15,
                step=0.1,
                value=0.5,
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

    dbc.Row([
        dbc.Col([
            html.H3("Waveform View", className="mt-4"),
            html.P("Click a voxel in the 3D plot above, or enter global pixel coordinates below:"),
        ], width=12)
    ], className="mt-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Global Pixel X:"),
            dcc.Input(
                id='input-pxl-x',
                type='number',
                placeholder='Enter pxl_x',
                className="form-control",
            ),
        ], width=12, md=3),
        dbc.Col([
            html.Label("Global Pixel Y:"),
            dcc.Input(
                id='input-pxl-y',
                type='number',
                placeholder='Enter pxl_y',
                className="form-control",
            ),
        ], width=12, md=3),
        dbc.Col([
            html.Label(""),
            dbc.Button(
                "Plot Waveform",
                id='plot-waveform-btn',
                color="primary",
                className="w-100 mt-2",
                size="sm"
            ),
        ], width=12, md=2),
        dbc.Col([
            html.Label(""),
            dbc.Button(
                "Clear",
                id='clear-coords-btn',
                color="secondary",
                className="w-100 mt-2",
                size="sm"
            ),
        ], width=12, md=2),
        dbc.Col([
            html.Label(""),
            dbc.Button(
                "Shift Truth: OFF",
                id='shift-truth-btn',
                color="secondary",
                className="w-100 mt-2",
                size="sm"
            ),
        ], width=12, md=2),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="waveform-loading",
                type="default",
                children=[
                    dcc.Graph(id='waveform-display', style={'height': '600px'})
                ]
            )
        ], width=12)
    ], className="mt-3"),

    dcc.Store(id='loaded-data-store'),
    dcc.Store(id='selected-coords-store', data={}),
    dcc.Store(id='truth-shift-store', data=0),
], fluid=True, className="p-4")


# Global cache for loaded npz data (to avoid JSON serialization of large arrays)
_loaded_npz_cache = {}


@app.callback(
    Output('file-selector', 'options'),
    Input('refresh-files-btn', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_file_list(n_clicks):
    """Refresh the list of available NPZ files."""
    npz_files = get_npz_files()
    print(f"Refreshed file list: found {len(npz_files)} NPZ files")
    return [{'label': f, 'value': f} for f in npz_files]


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

        # Convert NPZ contents into a plain dict of numpy arrays and store in cache.
        # Storing a plain dict ensures we are not holding onto a file handle or a lazily-loaded object.
        cache_entry = {}
        for key in data.files:
            try:
                cache_entry[key] = np.array(data[key])
            except Exception:
                # As a fallback, store the raw object
                cache_entry[key] = data[key]

        # Close the NpzFile explicitly (safe even if it's already closed)
        try:
            data.close()
        except Exception:
            pass

        _loaded_npz_cache[filename] = cache_entry

        # Extract metadata only for the store
        result = {
            'filename': filename,
            'has_deconv_q': 'deconv_q' in cache_entry,
            'deconv_q_shape': list(cache_entry['deconv_q'].shape) if 'deconv_q' in cache_entry else None,
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


@app.callback(
    Output('selected-coords-store', 'data'),
    Input('event-display', 'clickData'),
    Input('plot-waveform-btn', 'n_clicks'),
    State('input-pxl-x', 'value'),
    State('input-pxl-y', 'value'),
    State('loaded-data-store', 'data'),
    prevent_initial_call=False
)
def update_selected_coords(clickData, plot_nclicks, pxl_x, pxl_y, loaded_data):
    """
    Unified callback to update selected coordinates from either:
      - a click in the 3D event-display (clickData), or
      - the Plot Waveform button using manual pxl_x/pxl_y inputs.

    Uses callback_context to determine which input triggered the callback.
    """

    trig = callback_context.triggered
    if not trig:
        # No trigger (initial call) — do nothing.
        return {}

    prop_id = trig[0].get('prop_id', '')

    # Click from the 3D plot
    if prop_id.startswith('event-display'):
        if not clickData or not loaded_data or not loaded_data.get('loaded'):
            return {}

        filename = loaded_data.get('filename')
        if filename not in _loaded_npz_cache:
            return {}

        npz_data = _loaded_npz_cache[filename]
        if 'boffset' not in npz_data:
            return {}

        try:
            # Extract clicked point coordinates (local to deconv_q)
            point = clickData['points'][0]
            x_local = int(point['x'])
            y_local = int(point['y'])

            boffset = np.array(npz_data['boffset'])
            # Convert local coordinates to global pixel coordinates
            pxl_x_global = int(boffset[0]) + x_local
            pxl_y_global = int(boffset[1]) + y_local

            return {'pxl_x': pxl_x_global, 'pxl_y': pxl_y_global, 'source': 'click'}
        except Exception as e:
            print(f"Error extracting coordinates from click: {e}")
            return {}

    # Manual input via button
    if prop_id.startswith('plot-waveform-btn'):
        if pxl_x is None or pxl_y is None:
            return {}

        try:
            return {'pxl_x': int(pxl_x), 'pxl_y': int(pxl_y), 'source': 'input'}
        except Exception as e:
            print(f"Error parsing input coordinates: {e}")
            return {}

    # Unhandled trigger
    return {}


@app.callback(
    Output('input-pxl-x', 'value'),
    Output('input-pxl-y', 'value'),
    Input('clear-coords-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_coordinates(n_clicks):
    """Clear coordinate inputs."""
    return None, None


@app.callback(
    Output('truth-shift-store', 'data'),
    Output('shift-truth-btn', 'children'),
    Output('shift-truth-btn', 'color'),
    Input('shift-truth-btn', 'n_clicks'),
    State('truth-shift-store', 'data'),
    prevent_initial_call=True
)
def toggle_truth_shift(n_clicks, current_shift):
    """Toggle the truth shift between 0 and 1, and update button appearance."""
    if current_shift == 0:
        return 1, "Shift Truth: ON", "success"
    else:
        return 0, "Shift Truth: OFF", "secondary"


@app.callback(
    Output('waveform-display', 'figure'),
    Input('selected-coords-store', 'data'),
    Input('truth-shift-store', 'data'),
    Input('threshold-slider', 'value'),
    State('loaded-data-store', 'data'),
)
def display_waveform(selected_coords, truth_shift, threshold, loaded_data):
    """Display waveform for selected voxel with aligned smeared_true data."""

    if not selected_coords or not loaded_data or not loaded_data.get('loaded'):
        return go.Figure().add_annotation(text="Click a voxel or enter coordinates to view its waveform")

    filename = loaded_data.get('filename')
    if filename not in _loaded_npz_cache:
        return go.Figure().add_annotation(text="Data not in cache")

    npz_data = _loaded_npz_cache[filename]
    required_keys = ['deconv_q', 'boffset', 'adc_downsample_factor', 'smeared_true', 'smear_offset']
    if not all(k in npz_data for k in required_keys):
        return go.Figure().add_annotation(text=f"Missing required data: {required_keys}")

    try:
        # Extract global coordinates
        pxl_x = selected_coords.get('pxl_x')
        pxl_y = selected_coords.get('pxl_y')

        if pxl_x is None or pxl_y is None:
            return go.Figure().add_annotation(text="Invalid coordinates")

        deconv_q = np.array(npz_data['deconv_q'])
        smeared_true = np.array(npz_data['smeared_true'])
        boffset = np.array(npz_data['boffset'])
        smear_offset = np.array(npz_data['smear_offset'])
        dt_deconv = float(npz_data['adc_downsample_factor'])

        # Convert global coordinates to local coordinates
        x_local = pxl_x - int(boffset[0])
        y_local = pxl_y - int(boffset[1])

        # Check bounds
        if not (0 <= x_local < deconv_q.shape[0] and 0 <= y_local < deconv_q.shape[1]):
            return go.Figure().add_annotation(
                text=f"Global coordinates ({pxl_x}, {pxl_y}) out of bounds for deconv_q"
            )

        # Extract deconv_q waveform
        deconv_waveform = deconv_q[x_local, y_local, :]
        t0_deconv = float(boffset[2])
        times_deconv = t0_deconv + np.arange(len(deconv_waveform)) * dt_deconv

        # Try to get aligned smeared_true waveform
        fig = go.Figure()

        # Add deconv_q trace
        fig.add_trace(go.Scatter(
            x=times_deconv,
            # y=deconv_waveform / dt_deconv,  # Normalize by downsample factor
            y=deconv_waveform,  # per adc_hold_delay
            mode='lines+markers',
            name=f'deconv_q (global: {pxl_x}, {pxl_y}, local: {x_local}, {y_local})',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
        ))

        fig.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text=f"threshold: {threshold:.1f}",
            annotation_position="top left",
            line_color="green"
        )

        # Try to add smeared_true at aligned global position
        try:
            x_smear = pxl_x - int(smear_offset[0])
            y_smear = pxl_y - int(smear_offset[1])

            # Check bounds
            if 0 <= x_smear < smeared_true.shape[0] and 0 <= y_smear < smeared_true.shape[1]:
                smear_waveform = smeared_true[x_smear, y_smear, :]
                t0_smear = float(smear_offset[2])

                # Apply shift if enabled
                time_shift = truth_shift * dt_deconv if truth_shift else 0
                times_smear = t0_smear + np.arange(len(smear_waveform)) * 1 + time_shift  # dt=1 for smeared_true

                shift_label = f" (shifted +{dt_deconv:.1f} ticks)" if truth_shift else ""
                fig.add_trace(go.Scatter(
                    x=times_smear,
                    # y=smear_waveform,
                    y=smear_waveform * dt_deconv, # Scale to ADC_HOLD_DELAY
                    mode='lines+markers',
                    name=f'smeared_true (global: {pxl_x}, {pxl_y}){shift_label} x {dt_deconv}',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                ))

                # Add binned smeared_true trace
                # We want to sum smeared_true in bins of dt_deconv, aligned with deconv_q time bins.
                # The deconv_q time bins are [t0_deconv + i*dt_deconv, t0_deconv + (i+1)*dt_deconv)
                # smeared_true samples are at t0_smear + j*1
                
                # Calculate relative offset in fine ticks
                # offset_fine = t0_deconv - t0_smear
                
                # To align precisely, we find which fine samples fall into each coarse bin.
                # j-th fine sample is at t_j = t0_smear + j.
                # It falls into i-th coarse bin if t0_deconv + i*dt_deconv <= t0_smear + j < t0_deconv + (i+1)*dt_deconv
                
                binned_smear = []
                binned_times = []
                
                # Use the same range as deconv_waveform
                for i in range(len(deconv_waveform)):
                    t_start = t0_deconv + i * dt_deconv
                    t_end = t0_deconv + (i + 1) * dt_deconv
                    
                    # Indices in smear_waveform that fall into this interval
                    idx_start = int(np.ceil(t_start - t0_smear))
                    idx_end = int(np.ceil(t_end - t0_smear))
                    
                    # Clamp indices
                    idx_start = max(0, min(len(smear_waveform), idx_start))
                    idx_end = max(0, min(len(smear_waveform), idx_end))
                    
                    if idx_start < idx_end:
                        val = np.sum(smear_waveform[idx_start:idx_end])
                        binned_smear.append(val)
                        binned_times.append(t_start + dt_deconv) # Plot at upper bound
                    else:
                        # No overlapping fine samples for this coarse bin
                        pass

                if binned_smear:
                    fig.add_trace(go.Scatter(
                        x=binned_times,
                        y=binned_smear,
                        mode='lines+markers',
                        name=f'smeared_true_binned (sum over {int(dt_deconv)} ticks)',
                        line=dict(color='orange', width=3),
                        marker=dict(size=6, symbol='square'),
                    ))
            else:
                # Smeared position out of bounds
                fig.add_annotation(
                    text=f"smeared_true out of bounds: local ({x_smear}, {y_smear})",
                    showarrow=False,
                    y=0.95,
                )
        except Exception as e:
            print(f"Could not plot smeared_true: {e}")

        fig.update_layout(
            title=f"Waveforms: Global pixel ({pxl_x}, {pxl_y})",
            xaxis_title="Time (ticks, 50ns)",
            yaxis_title=f"Charge per {dt_deconv} x 50ns",
            height=600,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99),
        )

        return fig

    except Exception as e:
        print(f"Error displaying waveform: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().add_annotation(text=f"Error: {str(e)}")


if __name__ == '__main__':
    print(f"Found {len(NPZ_FILES)} NPZ files in {EXAMPLES_DIR}")
    print("Starting Dash app at http://127.0.0.1:8050/")
    app.run(debug=True, host='127.0.0.1', port=8050)
