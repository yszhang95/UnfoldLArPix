#!/usr/bin/env python3
"""
Dash app to compare per-channel waveforms from deconvolution NPZ files.

The app shows:
1. merged burst sequence per channel from ``hwf_block``
2. per-channel ``deconv_q`` and ``smeared_true`` waveforms

Usage
-----
python dash_compare_hwf.py
python dash_compare_hwf.py file1.npz file2.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html


EXAMPLES_DIR = Path(__file__).parent
FILE_COLORS = ("#2563eb", "#dc2626")
COMPARABLE_KEYS = {"hwf_block", "deconv_q", "smeared_true"}
PANEL_LABELS = {
    "merged": "Merged Burst",
    "deconv": "deconv_q / Truth",
    "hits": "Hits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare merged burst, deconvolved charge, and smeared truth "
        "from two NPZ files."
    )
    parser.add_argument(
        "npz_files",
        nargs="*",
        help="Optional default NPZ files to preselect in the UI.",
    )
    parser.add_argument(
        "--search-root",
        default=str(EXAMPLES_DIR),
        help="Directory to scan for comparable NPZ files.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Dash host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8051,
        help="Dash port.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode.",
    )
    args = parser.parse_args()
    if len(args.npz_files) > 2:
        parser.error("At most two default NPZ files may be provided.")
    return args


def resolve_npz_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (EXAMPLES_DIR / candidate).resolve()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.array(data[key]) for key in data.files}


def is_comparable_npz(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            return any(key in data.files for key in COMPARABLE_KEYS)
    except Exception:
        return False


def format_display_path(path: Path, search_root: Path) -> str:
    try:
        return str(path.relative_to(search_root))
    except ValueError:
        return str(path)


def discover_npz_files(search_root: Path, extra_paths: list[Path]) -> list[Path]:
    discovered: dict[Path, None] = {}
    if search_root.exists():
        for path in search_root.rglob("*.npz"):
            if path.is_file() and is_comparable_npz(path):
                discovered[path.resolve()] = None

    for path in extra_paths:
        if path.exists():
            discovered[path.resolve()] = None

    return sorted(
        discovered.keys(),
        key=lambda path: format_display_path(path, search_root).lower(),
    )


def build_file_options(paths: list[Path], search_root: Path) -> list[dict[str, str]]:
    return [
        {
            "label": format_display_path(path, search_root),
            "value": str(path),
        }
        for path in paths
    ]


def pick_default_values(paths: list[Path], requested: list[Path]) -> tuple[str | None, str | None]:
    requested_existing = [str(path.resolve()) for path in requested if path.exists()]
    available = [str(path) for path in paths]

    if not available:
        return None, None

    if len(requested_existing) >= 2:
        return requested_existing[0], requested_existing[1]

    if len(requested_existing) == 1:
        first = requested_existing[0]
        second = next((value for value in available if value != first), first)
        return first, second

    first = available[0]
    second = available[1] if len(available) > 1 else available[0]
    return first, second


def normalize_selected_value(value: str | None, available_values: set[str]) -> str | None:
    if not value:
        return None

    try:
        candidate = str(Path(value).resolve())
    except Exception:
        return None

    return candidate if candidate in available_values else None


def get_scalar(data: dict[str, np.ndarray], key: str, default: float) -> float:
    if key not in data:
        return float(default)
    return float(np.asarray(data[key]).squeeze())


def get_offset(
    data: dict[str, np.ndarray],
    *keys: str,
) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key], dtype=int)
    return None


def get_coarse_dt(data: dict[str, np.ndarray]) -> float:
    if "adc_downsample_factor" in data:
        return get_scalar(data, "adc_downsample_factor", 1.0)
    return get_scalar(data, "adc_hold_delay", 1.0)


def get_display_offset(
    data: dict[str, np.ndarray],
    array_key: str,
    offset: np.ndarray,
) -> np.ndarray:
    corrected = np.asarray(offset, dtype=int).copy()
    # Backward compatibility: older deconvolution outputs wrote hwf_block_offset
    # after the manual boffset time shift. When that happens, undo the shift for
    # merged-waveform display so trigger timestamps line up with the HWF axis.
    if array_key == "hwf_block":
        boffset = get_offset(data, "boffset")
        if boffset is not None and np.array_equal(corrected, boffset):
            corrected[-1] += int(round(get_coarse_dt(data)))
    return corrected


def has_saved_merged_sequence(data: dict[str, np.ndarray]) -> bool:
    return "hwf_block" in data and get_offset(data, "hwf_block_offset", "boffset") is not None


def extract_coarse_waveform(
    data: dict[str, np.ndarray],
    pxl_x: int,
    pxl_y: int,
    array_key: str,
    *offset_keys: str,
) -> dict[str, Any]:
    if array_key not in data:
        return {"status": "missing", "array_key": array_key}

    offset = get_offset(data, *offset_keys)
    if offset is None:
        return {"status": "missing_offset", "array_key": array_key}
    offset = get_display_offset(data, array_key, offset)

    array = np.asarray(data[array_key])
    if array.ndim != 3:
        return {"status": "invalid_shape", "array_key": array_key, "shape": array.shape}

    local_x = int(pxl_x - offset[0])
    local_y = int(pxl_y - offset[1])
    dt = get_coarse_dt(data)
    info: dict[str, Any] = {
        "status": "out_of_bounds",
        "array_key": array_key,
        "shape": tuple(int(dim) for dim in array.shape),
        "offset": offset,
        "dt": dt,
        "local_x": local_x,
        "local_y": local_y,
    }
    if not (0 <= local_x < array.shape[0] and 0 <= local_y < array.shape[1]):
        return info

    waveform = np.asarray(array[local_x, local_y, :], dtype=float)
    times = float(offset[2]) + np.arange(waveform.shape[0]) * dt
    info.update(
        {
            "status": "ok",
            "waveform": waveform,
            "times": times,
        }
    )
    return info


def extract_truth_waveform(
    data: dict[str, np.ndarray],
    pxl_x: int,
    pxl_y: int,
) -> dict[str, Any]:
    if "smeared_true" not in data or "smear_offset" not in data:
        return {"status": "missing", "array_key": "smeared_true"}

    array = np.asarray(data["smeared_true"])
    offset = np.asarray(data["smear_offset"], dtype=int)
    if array.ndim != 3:
        return {"status": "invalid_shape", "array_key": "smeared_true", "shape": array.shape}

    local_x = int(pxl_x - offset[0])
    local_y = int(pxl_y - offset[1])
    info: dict[str, Any] = {
        "status": "out_of_bounds",
        "array_key": "smeared_true",
        "shape": tuple(int(dim) for dim in array.shape),
        "offset": offset,
        "dt": 1.0,
        "local_x": local_x,
        "local_y": local_y,
    }
    if not (0 <= local_x < array.shape[0] and 0 <= local_y < array.shape[1]):
        return info

    waveform = np.asarray(array[local_x, local_y, :], dtype=float)
    times = float(offset[2]) + np.arange(waveform.shape[0], dtype=float)
    info.update(
        {
            "status": "ok",
            "waveform": waveform,
            "times": times,
        }
    )
    return info


def rebin_truth_to_coarse(
    truth_info: dict[str, Any],
    coarse_info: dict[str, Any],
) -> dict[str, Any]:
    if truth_info.get("status") != "ok" or coarse_info.get("status") != "ok":
        return {"status": "unavailable"}

    fine_waveform = np.asarray(truth_info["waveform"], dtype=float)
    t0_fine = float(truth_info["offset"][2])
    coarse_times = np.asarray(coarse_info["times"], dtype=float)
    t0_coarse = float(coarse_info["offset"][2])
    dt_coarse = float(coarse_info["dt"])

    rebinned = np.zeros_like(coarse_times, dtype=float)
    for idx in range(coarse_times.shape[0]):
        bin_start = t0_coarse + idx * dt_coarse
        bin_end = bin_start + dt_coarse
        fine_start = int(np.ceil(bin_start - t0_fine))
        fine_end = int(np.ceil(bin_end - t0_fine))
        fine_start = max(0, min(fine_waveform.shape[0], fine_start))
        fine_end = max(0, min(fine_waveform.shape[0], fine_end))
        if fine_start < fine_end:
            rebinned[idx] = np.sum(fine_waveform[fine_start:fine_end])

    return {
        "status": "ok",
        "times": coarse_times,
        "waveform": rebinned,
    }


def extract_hits_waveforms(
    data: dict[str, np.ndarray],
    pxl_x: int,
    pxl_y: int,
) -> dict[str, Any]:
    if "hits_location" not in data or "hits_data" not in data:
        return {"status": "missing", "array_key": "hits"}

    hits_location = np.asarray(data["hits_location"])
    hits_data = np.asarray(data["hits_data"], dtype=float)
    if hits_location.ndim != 2 or hits_location.shape[1] < 3:
        return {"status": "invalid_shape", "array_key": "hits_location", "shape": hits_location.shape}
    if hits_data.ndim != 2 or hits_data.shape[1] <= 3:
        return {"status": "invalid_shape", "array_key": "hits_data", "shape": hits_data.shape}

    mask = (hits_location[:, 0] == int(pxl_x)) & (hits_location[:, 1] == int(pxl_y))
    matching_indices = np.nonzero(mask)[0]
    if matching_indices.size == 0:
        return {
            "status": "missing_channel",
            "array_key": "hits",
            "count": 0,
        }

    dt = get_coarse_dt(data)
    waveforms: list[dict[str, Any]] = []
    for hit_index in matching_indices:
        trigger_time_idx = int(hits_location[hit_index, 2])
        charges = hits_data[hit_index, 3:]
        times = trigger_time_idx + dt + np.arange(charges.shape[0]) * dt
        waveforms.append(
            {
                "hit_index": int(hit_index),
                "trigger_time_idx": trigger_time_idx,
                "times": times,
                "waveform": np.asarray(charges, dtype=float),
            }
        )

    return {
        "status": "ok",
        "count": int(matching_indices.size),
        "waveforms": waveforms,
        "dt": dt,
    }


def extract_template_comp_triggers(
    data: dict[str, np.ndarray],
    pxl_x: int,
    pxl_y: int,
) -> dict[str, Any]:
    required = (
        "template_comp_peak_locations",
        "template_comp_trigger_timestamp",
        "template_comp_transit_fraction",
    )
    if any(key not in data for key in required):
        return {"status": "missing", "array_key": "template_comp"}

    peak_locations = np.asarray(data["template_comp_peak_locations"])
    trigger_timestamps = np.asarray(data["template_comp_trigger_timestamp"], dtype=float)
    transit_fractions = np.asarray(data["template_comp_transit_fraction"], dtype=float)
    if peak_locations.ndim != 2 or peak_locations.shape[1] < 2:
        return {
            "status": "invalid_shape",
            "array_key": "template_comp_peak_locations",
            "shape": peak_locations.shape,
        }
    if trigger_timestamps.ndim != 1 or transit_fractions.ndim != 1:
        return {"status": "invalid_shape", "array_key": "template_comp"}
    if not (
        peak_locations.shape[0] == trigger_timestamps.shape[0] == transit_fractions.shape[0]
    ):
        return {"status": "invalid_shape", "array_key": "template_comp"}

    mask = (peak_locations[:, 0] == int(pxl_x)) & (peak_locations[:, 1] == int(pxl_y))
    matching_indices = np.nonzero(mask)[0]
    if matching_indices.size == 0:
        return {"status": "missing_channel", "count": 0}

    entries = [
        {
            "trigger_timestamp": float(trigger_timestamps[idx]),
            "transit_fraction": float(transit_fractions[idx]),
        }
        for idx in matching_indices
    ]
    return {
        "status": "ok",
        "count": int(matching_indices.size),
        "entries": entries,
    }


def extract_template_comp_entries(data: dict[str, np.ndarray]) -> dict[str, Any]:
    required = (
        "template_comp_peak_locations",
        "template_comp_trigger_timestamp",
        "template_comp_transit_fraction",
        "template_comp_transit_threshold_idx",
    )
    if any(key not in data for key in required):
        return {"status": "missing", "array_key": "template_comp"}

    peak_locations = np.asarray(data["template_comp_peak_locations"], dtype=float)
    trigger_timestamps = np.asarray(data["template_comp_trigger_timestamp"], dtype=float)
    transit_fractions = np.asarray(data["template_comp_transit_fraction"], dtype=float)
    threshold_indices = np.asarray(data["template_comp_transit_threshold_idx"], dtype=int)
    if peak_locations.ndim != 2 or peak_locations.shape[1] < 3:
        return {
            "status": "invalid_shape",
            "array_key": "template_comp_peak_locations",
            "shape": peak_locations.shape,
        }
    if (
        trigger_timestamps.ndim != 1
        or transit_fractions.ndim != 1
        or threshold_indices.ndim != 1
    ):
        return {"status": "invalid_shape", "array_key": "template_comp"}
    if not (
        peak_locations.shape[0]
        == trigger_timestamps.shape[0]
        == transit_fractions.shape[0]
        == threshold_indices.shape[0]
    ):
        return {"status": "invalid_shape", "array_key": "template_comp"}

    entries: list[dict[str, Any]] = []
    for idx in range(peak_locations.shape[0]):
        pixel_x = int(np.rint(peak_locations[idx, 0]))
        pixel_y = int(np.rint(peak_locations[idx, 1]))
        peak_time = float(peak_locations[idx, 2])
        trigger_timestamp = float(trigger_timestamps[idx])
        entries.append(
            {
                "match_key": (
                    pixel_x,
                    pixel_y,
                    int(np.rint(peak_time)),
                    int(np.rint(trigger_timestamp)),
                ),
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "peak_time": peak_time,
                "trigger_timestamp": trigger_timestamp,
                "transit_fraction": float(transit_fractions[idx]),
                "threshold_idx": int(threshold_indices[idx]),
            }
        )

    return {
        "status": "ok",
        "count": int(len(entries)),
        "entries": entries,
    }


def compare_template_comp_entries(
    file1_info: dict[str, Any],
    file2_info: dict[str, Any],
) -> dict[str, Any]:
    if file1_info.get("status") != "ok":
        return {
            "status": "unavailable",
            "message": f"File 1 template-comp diagnostics unavailable ({file1_info.get('status')}).",
        }
    if file2_info.get("status") != "ok":
        return {
            "status": "unavailable",
            "message": f"File 2 template-comp diagnostics unavailable ({file2_info.get('status')}).",
        }

    grouped_1: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    grouped_2: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for entry in file1_info["entries"]:
        grouped_1.setdefault(entry["match_key"], []).append(entry)
    for entry in file2_info["entries"]:
        grouped_2.setdefault(entry["match_key"], []).append(entry)

    matched_pairs: list[dict[str, Any]] = []
    unmatched_file1 = 0
    unmatched_file2 = 0
    for match_key in sorted(set(grouped_1) | set(grouped_2)):
        entries_1 = grouped_1.get(match_key, [])
        entries_2 = grouped_2.get(match_key, [])
        n_match = min(len(entries_1), len(entries_2))
        unmatched_file1 += len(entries_1) - n_match
        unmatched_file2 += len(entries_2) - n_match
        for idx in range(n_match):
            matched_pairs.append(
                {
                    "file1": entries_1[idx],
                    "file2": entries_2[idx],
                }
            )

    if not matched_pairs:
        return {
            "status": "no_overlap",
            "message": "No overlapping template-compensation anchors were found between the two files.",
            "file1_count": int(file1_info["count"]),
            "file2_count": int(file2_info["count"]),
        }

    return {
        "status": "ok",
        "matched_pairs": matched_pairs,
        "matched_count": int(len(matched_pairs)),
        "unmatched_file1": int(unmatched_file1),
        "unmatched_file2": int(unmatched_file2),
        "file1_count": int(file1_info["count"]),
        "file2_count": int(file2_info["count"]),
    }


def add_trace(
    fig: go.Figure,
    info: dict[str, Any],
    *,
    name: str,
    color: str,
    scale: float = 1.0,
    dash_style: str = "solid",
    opacity: float = 1.0,
) -> None:
    if info.get("status") != "ok":
        return

    fig.add_trace(
        go.Scatter(
            x=info["times"],
            y=np.asarray(info["waveform"], dtype=float) * scale,
            mode="lines+markers",
            name=name,
            line=dict(color=color, dash=dash_style, width=2),
            marker=dict(size=4),
            opacity=opacity,
        )
    )


def integrated_waveform_charge(info: dict[str, Any]) -> float | None:
    if info.get("status") != "ok":
        return None
    waveform = np.asarray(info.get("waveform", []), dtype=float)
    if waveform.size == 0:
        return None
    return float(np.sum(waveform))


def empty_figure(message: str, height: int) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    fig.update_layout(height=height, template="plotly_white")
    return fig


def describe_info(label: str, info: dict[str, Any]) -> str:
    status = info.get("status")
    if status == "ok":
        return (
            f"{label}: local=({info['local_x']}, {info['local_y']}), "
            f"shape={info['shape']}, offset={tuple(int(v) for v in info['offset'])}"
        )
    if status == "out_of_bounds":
        return (
            f"{label}: global channel is out of bounds "
            f"(local=({info['local_x']}, {info['local_y']}), shape={info['shape']})"
        )
    if status == "missing_offset":
        return f"{label}: missing offset metadata"
    if status == "invalid_shape":
        return f"{label}: invalid array shape {info.get('shape')}"
    if status == "missing_channel":
        return f"{label}: no hits found in this global channel"
    return f"{label}: not available"


def make_status_card(
    title: str,
    path: Path,
    scales: dict[str, float],
    merged_info: dict[str, Any],
    deconv_info: dict[str, Any],
    truth_info: dict[str, Any],
    truth_binned_info: dict[str, Any],
    hits_info: dict[str, Any],
) -> dbc.Card:
    true_charge = integrated_waveform_charge(truth_binned_info)
    if true_charge is None:
        true_charge = integrated_waveform_charge(truth_info)
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, className="card-title"),
                html.P(path.name, className="text-break mb-1"),
                html.P(str(path), className="text-break small text-muted mb-2"),
                html.P(
                    "Scales: "
                    f"merged x{scales['merged']:.3g}, "
                    f"deconv x{scales['deconv']:.3g}, "
                    f"hits x{scales['hits']:.3g}",
                    className="mb-1",
                ),
                html.P(describe_info("Merged burst", merged_info), className="mb-1"),
                html.P(describe_info("deconv_q", deconv_info), className="mb-1"),
                html.P(describe_info("smeared_true", truth_info), className="mb-1"),
                html.P(
                    (
                        f"Integrated true charge (rebinned): {true_charge:.6g} ke-"
                        if true_charge is not None
                        else "Integrated true charge (rebinned): not available"
                    ),
                    className="mb-1",
                ),
                html.P(
                    f"Hits: {hits_info.get('count', 0)} waveform(s) in this channel"
                    if hits_info.get("status") == "ok"
                    else describe_info("Hits", hits_info),
                    className="mb-1",
                ),
            ]
        ),
        className="h-100",
    )


def build_missing_merged_notification(file_specs: list[dict[str, Any]]) -> html.Div:
    warnings: list[str] = []
    for spec in file_specs:
        data = spec["data"]
        if "hwf_block" not in data:
            warnings.append(
                f"{spec['label']} ({spec['path'].name}) does not contain a saved "
                "merged sequence ('hwf_block')."
            )
        elif not has_saved_merged_sequence(data):
            warnings.append(
                f"{spec['label']} ({spec['path'].name}) has 'hwf_block' but is "
                "missing merged-sequence offset metadata."
            )

    if not warnings:
        return html.Div()

    return html.Div(
        [
            dbc.Alert(
                [
                    html.Div("Merged sequence notification", className="fw-bold"),
                    html.Ul([html.Li(message) for message in warnings], className="mb-0"),
                ],
                color="warning",
                className="mb-3",
            )
        ]
    )


def build_merged_figure(
    pxl_x: int,
    pxl_y: int,
    file_specs: list[dict[str, Any]],
) -> go.Figure:
    fig = go.Figure()
    for spec in file_specs:
        add_trace(
            fig,
            spec["merged_info"],
            name=f"{spec['label']} merged burst x{spec['merged_scale']:.3g}",
            color=spec["color"],
            scale=spec["merged_scale"],
        )
        trigger_info = spec["template_comp_trigger_info"]
        if trigger_info.get("status") == "ok":
            trigger_times = np.asarray(
                [entry["trigger_timestamp"] for entry in trigger_info["entries"]],
                dtype=float,
            )
            transit_fractions = np.asarray(
                [entry["transit_fraction"] for entry in trigger_info["entries"]],
                dtype=float,
            )
            if trigger_times.size > 0:
                merged_waveform = np.asarray(spec["merged_info"].get("waveform", []), dtype=float)
                marker_height = (
                    float(np.max(merged_waveform)) * spec["merged_scale"]
                    if merged_waveform.size > 0
                    else 0.0
                )
                marker_height = marker_height if np.isfinite(marker_height) and marker_height > 0 else 1.0
                x_coords: list[float | None] = []
                y_coords: list[float | None] = []
                customdata: list[list[float] | None] = []
                for trigger_time, transit_fraction in zip(trigger_times, transit_fractions, strict=False):
                    x_coords.extend([float(trigger_time), float(trigger_time), None])
                    y_coords.extend([0.0, marker_height, None])
                    customdata.extend(
                        [
                            [float(trigger_time), float(transit_fraction)],
                            [float(trigger_time), float(transit_fraction)],
                            None,
                        ]
                    )
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode="lines",
                        name=f"{spec['label']} trigger stamps",
                        line=dict(color=spec["color"], width=1.5, dash="dot"),
                        customdata=customdata,
                        hovertemplate="(%{customdata[0]:.0f}, %{customdata[1]:.6g})<extra></extra>",
                    )
                )

    if not fig.data:
        return empty_figure(
            f"No merged burst waveform found at global channel ({pxl_x}, {pxl_y})",
            430,
        )

    fig.update_layout(
        title=f"Merged Burst Sequence at Global Channel ({pxl_x}, {pxl_y})",
        xaxis_title="Time tick",
        yaxis_title="Merged burst charge per coarse bin",
        hovermode="x unified",
        template="plotly_white",
        height=430,
    )
    return fig


def build_deconv_truth_figure(
    pxl_x: int,
    pxl_y: int,
    file_specs: list[dict[str, Any]],
) -> go.Figure:
    fig = go.Figure()
    for spec in file_specs:
        add_trace(
            fig,
            spec["deconv_info"],
            name=f"{spec['label']} deconv_q x{spec['deconv_scale']:.3g}",
            color=spec["color"],
            scale=spec["deconv_scale"],
        )
        add_trace(
            fig,
            spec["truth_binned_info"],
            name=f"{spec['label']} smeared_true rebinned x{spec['deconv_scale']:.3g}",
            color=spec["color"],
            scale=spec["deconv_scale"],
            dash_style="dash",
        )
        add_trace(
            fig,
            spec["truth_info"],
            name=f"{spec['label']} smeared_true fine",
            color=spec["color"],
            dash_style="dot",
            opacity=0.45,
        )

    if not fig.data:
        return empty_figure(
            f"No deconv_q or smeared_true waveform found at global channel ({pxl_x}, {pxl_y})",
            620,
        )

    fig.update_layout(
        title=f"deconv_q and smeared_true at Global Channel ({pxl_x}, {pxl_y})",
        xaxis_title="Time tick",
        yaxis_title="Charge",
        hovermode="x unified",
        template="plotly_white",
        height=620,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def build_hits_figure(
    pxl_x: int,
    pxl_y: int,
    file_specs: list[dict[str, Any]],
) -> go.Figure:
    fig = go.Figure()
    any_hits = False
    for spec in file_specs:
        hits_info = spec["hits_info"]
        if hits_info.get("status") != "ok":
            continue
        any_hits = True
        for idx, hit in enumerate(hits_info["waveforms"], start=1):
            fig.add_trace(
                go.Scatter(
                    x=hit["times"],
                    y=np.asarray(hit["waveform"], dtype=float) * spec["hits_scale"],
                    mode="lines+markers",
                    name=(
                        f"{spec['label']} hit {idx}/{hits_info['count']} "
                        f"(idx {hit['hit_index']}) x{spec['hits_scale']:.3g}"
                    ),
                    line=dict(color=spec["color"], width=2),
                    marker=dict(size=4),
                    opacity=0.9,
                )
            )

    if not any_hits:
        return empty_figure(
            f"No hits waveform found at global channel ({pxl_x}, {pxl_y})",
            480,
        )

    fig.update_layout(
        title=f"Hits per Channel at Global Channel ({pxl_x}, {pxl_y})",
        xaxis_title="Time tick",
        yaxis_title="Hit charge",
        hovermode="x unified",
        template="plotly_white",
        height=480,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def build_transit_fraction_compare_figure(file_specs: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    any_entries = False
    for spec in file_specs:
        template_comp_entries = spec["template_comp_entries"]
        if template_comp_entries.get("status") != "ok":
            continue
        any_entries = True
        entries = template_comp_entries["entries"]
        anchor_index = np.arange(len(entries), dtype=int)
        values = np.asarray([entry["transit_fraction"] for entry in entries], dtype=float)
        customdata = np.asarray(
            [
                [
                    entry["pixel_x"],
                    entry["pixel_y"],
                    entry["peak_time"],
                    entry["trigger_timestamp"],
                    entry["threshold_idx"],
                ]
                for entry in entries
            ],
            dtype=float,
        )
        fig.add_trace(
            go.Scatter(
                x=anchor_index,
                y=values,
                mode="lines+markers",
                marker=dict(color=spec["color"], size=5, opacity=0.8),
                line=dict(color=spec["color"], width=2),
                name=f"{spec['label']} transit fraction",
                customdata=customdata,
                hovertemplate=(
                    "anchor=%{x}<br>"
                    "transit=%{y:.6g}<br>"
                    "pixel=(%{customdata[0]:.0f}, %{customdata[1]:.0f})<br>"
                    "peak_time=%{customdata[2]:.0f}<br>"
                    "trigger=%{customdata[3]:.0f}<br>"
                    "threshold_idx=%{customdata[4]:.0f}"
                    "<extra></extra>"
                ),
            )
        )

    if not any_entries:
        return empty_figure("Transit-fraction comparison unavailable.", 420)

    fig.update_layout(
        title="Transit Fraction by Compensation Index",
        xaxis_title="Compensation index in saved order",
        yaxis_title="Transit fraction",
        template="plotly_white",
        height=420,
    )
    return fig


def build_threshold_idx_compare_figure(file_specs: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    any_entries = False
    global_min: int | None = None
    global_max: int | None = None
    for spec in file_specs:
        template_comp_entries = spec["template_comp_entries"]
        if template_comp_entries.get("status") != "ok":
            continue
        any_entries = True
        entries = template_comp_entries["entries"]
        values = np.asarray([entry["threshold_idx"] for entry in entries], dtype=float)
        values_int = values.astype(int)
        current_min = int(np.min(values_int))
        current_max = int(np.max(values_int))
        global_min = current_min if global_min is None else min(global_min, current_min)
        global_max = current_max if global_max is None else max(global_max, current_max)
        fig.add_trace(
            go.Histogram(
                x=values_int,
                name=f"{spec['label']} threshold idx",
                marker=dict(color=spec["color"]),
                opacity=0.65,
                bingroup="threshold-idx",
                hovertemplate="threshold_idx=%{x}<br>count=%{y}<extra></extra>",
            )
        )

    if not any_entries:
        return empty_figure("Threshold-idx comparison unavailable.", 420)

    fig.update_layout(
        title="Template Threshold Idx Histogram",
        xaxis_title="Compensation threshold idx",
        yaxis_title="Count",
        template="plotly_white",
        height=420,
        barmode="overlay",
    )
    if global_min is not None and global_max is not None:
        fig.update_traces(
            xbins=dict(
                start=global_min - 0.5,
                end=global_max + 0.5,
                size=1,
            ),
            selector=dict(type="histogram"),
        )
    return fig


def build_delta_threshold_idx_histogram(compare_info: dict[str, Any]) -> go.Figure:
    if compare_info.get("status") != "ok":
        return empty_figure(
            compare_info.get("message", "Delta threshold-idx histogram unavailable."),
            380,
        )

    deltas = np.asarray(
        [
            pair["file2"]["threshold_idx"] - pair["file1"]["threshold_idx"]
            for pair in compare_info["matched_pairs"]
        ],
        dtype=int,
    )
    unique_deltas, counts = np.unique(deltas, return_counts=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=unique_deltas,
            y=counts,
            marker_color="#ea580c",
            name="Matched anchors",
            hovertemplate="delta=%{x}<br>count=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Histogram of Delta Template Threshold Idx",
        xaxis_title="File 2 - File 1 compensation threshold idx",
        yaxis_title="Matched anchor count",
        template="plotly_white",
        height=380,
    )
    return fig


def describe_template_comp_comparison(compare_info: dict[str, Any]) -> str:
    if compare_info.get("status") != "ok":
        return compare_info.get("message", "Template-compensation comparison unavailable.")

    return (
        "Template-compensation comparison: "
        f"matched {compare_info['matched_count']} anchor(s), "
        f"File 1 only {compare_info['unmatched_file1']}, "
        f"File 2 only {compare_info['unmatched_file2']}. "
        "The transit-fraction plot shows the full saved compensation sequence for each file. "
        "The threshold-idx plot shows the saved compensation-threshold distribution as a histogram. "
        "The delta-threshold histogram uses explicitly matched anchors and the saved "
        "compensation threshold idx (`template_comp_transit_threshold_idx`)."
    )


def create_app(default_paths: list[Path], search_root: Path) -> Dash:
    available_paths = discover_npz_files(search_root, default_paths)
    if not available_paths:
        raise FileNotFoundError(
            f"No comparable NPZ files found under {search_root}"
        )

    file_options = build_file_options(available_paths, search_root)
    default_file1, default_file2 = pick_default_values(available_paths, default_paths)
    cache: dict[str, dict[str, np.ndarray]] = {}

    def load_npz_cached(path_str: str) -> dict[str, np.ndarray]:
        path = Path(path_str).resolve()
        key = str(path)
        if key not in cache:
            cache[key] = load_npz(path)
        return cache[key]

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1(
                                "Channel Comparison: Merged Burst, deconv_q, smeared_true",
                                className="mb-2 text-center",
                            ),
                            html.P(
                                f"Selectable files are scanned from {search_root}",
                                className="text-center text-muted",
                            ),
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("File 1"),
                            dcc.Dropdown(
                                id="file-selector-1",
                                options=file_options,
                                value=default_file1,
                                clearable=False,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("File 2"),
                            dcc.Dropdown(
                                id="file-selector-2",
                                options=file_options,
                                value=default_file2,
                                clearable=False,
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Refresh Files",
                            id="refresh-files-btn",
                            color="secondary",
                            outline=True,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Div(
                            "Rescan the file list after adding, removing, or generating NPZ files.",
                            id="file-refresh-status",
                            className="text-muted small align-self-center",
                        )
                    ),
                ],
                className="mb-4 g-2 align-items-center",
            ),
            *[
                dcc.Store(id=f"scale-factor-{panel}-1", data=1.0)
                for panel in PANEL_LABELS
            ],
            *[
                dcc.Store(id=f"scale-factor-{panel}-2", data=1.0)
                for panel in PANEL_LABELS
            ],
            *[
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(f"{PANEL_LABELS[panel]} Scale: File 1", className="mb-2"),
                                        html.Div(
                                            [
                                                dbc.Button("Down", id=f"scale-down-{panel}-1", color="secondary", outline=True, size="sm"),
                                                dcc.Input(
                                                    id=f"scale-input-{panel}-1",
                                                    type="text",
                                                    value="1.0",
                                                    inputMode="decimal",
                                                    debounce=True,
                                                    className="form-control form-control-sm",
                                                    style={"maxWidth": "8rem"},
                                                ),
                                                html.Span(id=f"scale-display-{panel}-1", className="fw-bold"),
                                                dbc.Button("Up", id=f"scale-up-{panel}-1", color="secondary", outline=True, size="sm"),
                                            ],
                                            className="d-flex align-items-center justify-content-center gap-2",
                                        ),
                                    ]
                                )
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(f"{PANEL_LABELS[panel]} Scale: File 2", className="mb-2"),
                                        html.Div(
                                            [
                                                dbc.Button("Down", id=f"scale-down-{panel}-2", color="secondary", outline=True, size="sm"),
                                                dcc.Input(
                                                    id=f"scale-input-{panel}-2",
                                                    type="text",
                                                    value="1.0",
                                                    inputMode="decimal",
                                                    debounce=True,
                                                    className="form-control form-control-sm",
                                                    style={"maxWidth": "8rem"},
                                                ),
                                                html.Span(id=f"scale-display-{panel}-2", className="fw-bold"),
                                                dbc.Button("Up", id=f"scale-up-{panel}-2", color="secondary", outline=True, size="sm"),
                                            ],
                                            className="d-flex align-items-center justify-content-center gap-2",
                                        ),
                                    ]
                                )
                            ),
                            width=6,
                        ),
                    ],
                    className="mb-3",
                )
                for panel in PANEL_LABELS
            ],
            dbc.Row(
                [
                    dbc.Col(html.Div(id="notification-display"))
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Global Pixel X"),
                            dcc.Input(
                                id="input-pxl-x",
                                type="text",
                                value="0",
                                inputMode="numeric",
                                debounce=True,
                                className="form-control",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Global Pixel Y"),
                            dcc.Input(
                                id="input-pxl-y",
                                type="text",
                                value="0",
                                inputMode="numeric",
                                debounce=True,
                                className="form-control",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Label(""),
                            dbc.Button(
                                "Update Channel",
                                id="update-btn",
                                color="primary",
                                className="w-100 mt-2",
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            type="default",
                            children=[dcc.Graph(id="merged-plot")],
                        )
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            type="default",
                            children=[dcc.Graph(id="deconv-truth-plot")],
                        )
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            type="default",
                            children=[dcc.Graph(id="hits-plot")],
                        )
                    )
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            type="default",
                            children=[dcc.Graph(id="transit-fraction-compare-plot")],
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dcc.Loading(
                            type="default",
                            children=[dcc.Graph(id="threshold-idx-compare-plot")],
                        ),
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            type="default",
                            children=[dcc.Graph(id="delta-threshold-idx-hist-plot")],
                        )
                    )
                ],
                className="mb-3",
            ),
            dbc.Row([dbc.Col(html.Div(id="info-display"))]),
        ],
        fluid=True,
        className="p-4",
    )

    @app.callback(
        *[
            Output(f"scale-factor-{panel}-{file_idx}", "data")
            for panel in PANEL_LABELS
            for file_idx in (1, 2)
        ],
        *[
            Output(f"scale-input-{panel}-{file_idx}", "value")
            for panel in PANEL_LABELS
            for file_idx in (1, 2)
        ],
        *[
            Output(f"scale-display-{panel}-{file_idx}", "children")
            for panel in PANEL_LABELS
            for file_idx in (1, 2)
        ],
        *[
            Input(f"scale-down-{panel}-{file_idx}", "n_clicks")
            for panel in PANEL_LABELS
            for file_idx in (1, 2)
        ],
        *[
            Input(f"scale-up-{panel}-{file_idx}", "n_clicks")
            for panel in PANEL_LABELS
            for file_idx in (1, 2)
        ],
        *[
            Input(f"scale-input-{panel}-{file_idx}", "value")
            for panel in PANEL_LABELS
            for file_idx in (1, 2)
        ],
        *[
            State(f"scale-factor-{panel}-{file_idx}", "data")
            for panel in PANEL_LABELS
            for file_idx in (1, 2)
        ],
    )
    def update_scale_factors(*args: Any) -> tuple[Any, ...]:
        panel_keys = [(panel, file_idx) for panel in PANEL_LABELS for file_idx in (1, 2)]
        n_pairs = len(panel_keys)
        inputs_start = 2 * n_pairs
        states_start = 3 * n_pairs
        scale_inputs = args[inputs_start:states_start]
        current_scales = args[states_start:states_start + n_pairs]
        scales = {
            key: float(value or 1.0)
            for key, value in zip(panel_keys, current_scales)
        }
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

        def parse_scale(raw_value: str | None, fallback: float) -> float:
            try:
                value = float(str(raw_value).strip())
            except (TypeError, ValueError):
                return fallback
            if not np.isfinite(value) or value <= 0:
                return fallback
            return min(max(value, 1e-6), 1e6)

        input_map = {
            key: scale_inputs[idx]
            for idx, key in enumerate(panel_keys)
        }
        if trigger:
            for key in panel_keys:
                panel, file_idx = key
                if trigger == f"scale-down-{panel}-{file_idx}":
                    scales[key] = max(scales[key] / 2.0, 1e-6)
                    break
                if trigger == f"scale-up-{panel}-{file_idx}":
                    scales[key] = min(scales[key] * 2.0, 1e6)
                    break
                if trigger == f"scale-input-{panel}-{file_idx}":
                    scales[key] = parse_scale(input_map[key], scales[key])
                    break
        else:
            for key in panel_keys:
                scales[key] = parse_scale(input_map[key], scales[key])

        scale_values = [scales[key] for key in panel_keys]
        input_values = [f"{scales[key]:.6g}" for key in panel_keys]
        display_values = [f"x{scales[key]:.3g}" for key in panel_keys]
        return tuple(scale_values + input_values + display_values)

    @app.callback(
        Output("file-selector-1", "options"),
        Output("file-selector-2", "options"),
        Output("file-selector-1", "value"),
        Output("file-selector-2", "value"),
        Output("file-refresh-status", "children"),
        Input("refresh-files-btn", "n_clicks"),
        State("file-selector-1", "value"),
        State("file-selector-2", "value"),
        prevent_initial_call=True,
    )
    def refresh_file_options(
        _n_clicks: int | None,
        file1: str | None,
        file2: str | None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], str | None, str | None, str]:
        selected_paths = [
            Path(value).expanduser()
            for value in (file1, file2)
            if value
        ]
        available_paths = discover_npz_files(search_root, default_paths + selected_paths)
        file_options = build_file_options(available_paths, search_root)
        available_values = {str(path) for path in available_paths}

        selected1 = normalize_selected_value(file1, available_values)
        selected2 = normalize_selected_value(file2, available_values)
        default1, default2 = pick_default_values(available_paths, default_paths)

        if selected1 is None:
            selected1 = default1
        if selected2 is None:
            selected2 = default2

        if selected1 == selected2 and len(available_paths) > 1:
            selected2 = next(
                (str(path) for path in available_paths if str(path) != selected1),
                selected2,
            )

        return (
            file_options,
            file_options,
            selected1,
            selected2,
            f"Refreshed {len(available_paths)} comparable NPZ file(s) from {search_root}.",
        )

    @app.callback(
        Output("notification-display", "children"),
        Output("merged-plot", "figure"),
        Output("deconv-truth-plot", "figure"),
        Output("hits-plot", "figure"),
        Output("transit-fraction-compare-plot", "figure"),
        Output("threshold-idx-compare-plot", "figure"),
        Output("delta-threshold-idx-hist-plot", "figure"),
        Output("info-display", "children"),
        Input("file-selector-1", "value"),
        Input("file-selector-2", "value"),
        Input("update-btn", "n_clicks"),
        *[
            Input(f"scale-factor-{panel}-{file_idx}", "data")
            for panel in PANEL_LABELS
            for file_idx in (1, 2)
        ],
        State("input-pxl-x", "value"),
        State("input-pxl-y", "value"),
    )
    def update_channel_view(
        file1: str | None,
        file2: str | None,
        _n_clicks: int | None,
        *scale_values_and_coords: Any,
    ) -> tuple[html.Div, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, html.Div]:
        scale_values = scale_values_and_coords[:-2]
        pxl_x_raw = scale_values_and_coords[-2]
        pxl_y_raw = scale_values_and_coords[-1]
        panel_keys = [(panel, file_idx) for panel in PANEL_LABELS for file_idx in (1, 2)]
        scales = {
            key: float(value or 1.0)
            for key, value in zip(panel_keys, scale_values)
        }
        if not file1 or not file2:
            message = "Select two NPZ files."
            return (
                html.Div(),
                empty_figure(message, 430),
                empty_figure(message, 620),
                empty_figure(message, 480),
                empty_figure(message, 420),
                empty_figure(message, 420),
                empty_figure(message, 380),
                html.Div(message),
            )

        try:
            pxl_x = int(str(pxl_x_raw).strip())
            pxl_y = int(str(pxl_y_raw).strip())
        except (TypeError, ValueError):
            message = "Enter both global pixel coordinates."
            return (
                html.Div(),
                empty_figure(message, 430),
                empty_figure(message, 620),
                empty_figure(message, 480),
                empty_figure(message, 420),
                empty_figure(message, 420),
                empty_figure(message, 380),
                html.Div(message),
            )

        selected_specs = [
            {
                "label": "File 1",
                "path": Path(file1).resolve(),
                "color": FILE_COLORS[0],
                "merged_scale": scales[("merged", 1)],
                "deconv_scale": scales[("deconv", 1)],
                "hits_scale": scales[("hits", 1)],
                "data": load_npz_cached(file1),
            },
            {
                "label": "File 2",
                "path": Path(file2).resolve(),
                "color": FILE_COLORS[1],
                "merged_scale": scales[("merged", 2)],
                "deconv_scale": scales[("deconv", 2)],
                "hits_scale": scales[("hits", 2)],
                "data": load_npz_cached(file2),
            },
        ]
        notification = build_missing_merged_notification(selected_specs)
        template_comp_entries_1 = extract_template_comp_entries(selected_specs[0]["data"])
        template_comp_entries_2 = extract_template_comp_entries(selected_specs[1]["data"])
        template_comp_compare_info = compare_template_comp_entries(
            template_comp_entries_1,
            template_comp_entries_2,
        )

        enriched_specs = []
        for spec, template_comp_entries in zip(
            selected_specs,
            (template_comp_entries_1, template_comp_entries_2),
            strict=False,
        ):
            merged_info = extract_coarse_waveform(
                spec["data"],
                int(pxl_x),
                int(pxl_y),
                "hwf_block",
                "hwf_block_offset",
                "boffset",
            )
            deconv_info = extract_coarse_waveform(
                spec["data"],
                int(pxl_x),
                int(pxl_y),
                "deconv_q",
                "boffset",
                "hwf_block_offset",
            )
            truth_info = extract_truth_waveform(
                spec["data"],
                int(pxl_x),
                int(pxl_y),
            )
            hits_info = extract_hits_waveforms(
                spec["data"],
                int(pxl_x),
                int(pxl_y),
            )
            template_comp_trigger_info = extract_template_comp_triggers(
                spec["data"],
                int(pxl_x),
                int(pxl_y),
            )
            enriched_specs.append(
                {
                    **spec,
                    "merged_info": merged_info,
                    "deconv_info": deconv_info,
                    "truth_info": truth_info,
                    "hits_info": hits_info,
                    "template_comp_trigger_info": template_comp_trigger_info,
                    "template_comp_entries": template_comp_entries,
                    "truth_binned_info": rebin_truth_to_coarse(truth_info, deconv_info),
                }
            )

        merged_fig = build_merged_figure(int(pxl_x), int(pxl_y), enriched_specs)
        deconv_truth_fig = build_deconv_truth_figure(
            int(pxl_x),
            int(pxl_y),
            enriched_specs,
        )
        hits_fig = build_hits_figure(
            int(pxl_x),
            int(pxl_y),
            enriched_specs,
        )
        transit_fraction_compare_fig = build_transit_fraction_compare_figure(
            enriched_specs,
        )
        threshold_idx_compare_fig = build_threshold_idx_compare_figure(
            enriched_specs,
        )
        delta_threshold_idx_hist_fig = build_delta_threshold_idx_histogram(
            template_comp_compare_info,
        )

        info_children = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            make_status_card(
                                spec["label"],
                                spec["path"],
                                {
                                    "merged": spec["merged_scale"],
                                    "deconv": spec["deconv_scale"],
                                    "hits": spec["hits_scale"],
                                },
                                spec["merged_info"],
                                spec["deconv_info"],
                                spec["truth_info"],
                                spec["truth_binned_info"],
                                spec["hits_info"],
                            ),
                            width=6,
                        )
                        for spec in enriched_specs
                    ],
                    className="g-3",
                ),
                html.P(
                    describe_template_comp_comparison(template_comp_compare_info),
                    className="mt-3 mb-1 text-muted",
                ),
                html.P(
                    "Dashed traces are smeared_true rebinned onto the coarse deconv grid. "
                    "Dotted traces are the raw fine-tick smeared_true waveforms. "
                    "Each panel has its own File 1 / File 2 scale controls. "
                    "The hits plot overlays every hit waveform found in the selected channel. "
                    "The diagnostic plots compare saved template-compensation quantities "
                    "across the full files.",
                    className="mt-3 text-muted",
                ),
            ]
        )

        return (
            notification,
            merged_fig,
            deconv_truth_fig,
            hits_fig,
            transit_fraction_compare_fig,
            threshold_idx_compare_fig,
            delta_threshold_idx_hist_fig,
            info_children,
        )

    return app


def main() -> None:
    args = parse_args()
    search_root = resolve_npz_path(args.search_root)
    default_paths = [resolve_npz_path(path_str) for path_str in args.npz_files]
    app = create_app(default_paths, search_root)
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
