#!/usr/bin/env python3
"""Compare template-compensation trigger inference against a waveform reference.

For each first sequence per channel in a deconvolution-output NPZ, this script
computes

    recorded_trigger_idx - inferred_trigger_idx

where the inferred trigger index is defined as

    template_comp_peak_locations[:, 2] + template_comp_transit_threshold_idx

This uses the saved template-compensation anchor time together with the saved
compensation threshold index.

The script then loads a waveform-reference TRED NPZ and, for the first sequence
per channel in that reference, uses the recorded trigger index directly. For
matched channels `(x, y)`, it computes

    (recorded_trigger_idx - inferred_trigger_idx) - reference_trigger_idx

and writes:

- a CSV table for the first study sequence per channel
- a histogram PNG of the final matched-channel difference
- a short text summary
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REFERENCE = Path("examples/data/pgun_positron_3gev_tred_nonoises_effq_nt1_wf_5tks.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare template-compensation trigger inference to a waveform reference."
    )
    parser.add_argument(
        "study_npz",
        help="Deconvolution-output NPZ with template_comp_* diagnostics.",
    )
    parser.add_argument(
        "--reference-npz",
        default=str(DEFAULT_REFERENCE),
        help=f"Reference TRED waveform NPZ (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--tpc-id",
        type=int,
        default=0,
        help="TPC ID for the reference file (default: 0)",
    )
    parser.add_argument(
        "--batch-id",
        type=int,
        default=0,
        help="Batch/event ID for the reference file (default: 0)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Number of bins for the matched-difference histogram (default: 120)",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix (default: derived from the study NPZ stem)",
    )
    return parser.parse_args()


def require_keys(data: np.lib.npyio.NpzFile, keys: list[str], *, source: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{source} is missing required arrays: {', '.join(missing)}")


def build_study_first_sequence_rows(study_data: np.lib.npyio.NpzFile) -> list[dict[str, float | int]]:
    require_keys(
        study_data,
        [
            "hits_location",
            "template_comp_peak_locations",
            "template_comp_trigger_time_idx",
            "template_comp_transit_threshold_idx",
            "template_comp_transit_fraction",
        ],
        source="study NPZ",
    )

    hits_location = np.asarray(study_data["hits_location"], dtype=int)
    peak_locations = np.asarray(study_data["template_comp_peak_locations"], dtype=float)
    trigger_time_idx = np.asarray(study_data["template_comp_trigger_time_idx"], dtype=int)
    threshold_idx = np.asarray(study_data["template_comp_transit_threshold_idx"], dtype=int)
    transit_fraction = np.asarray(study_data["template_comp_transit_fraction"], dtype=float)

    if peak_locations.ndim != 2 or peak_locations.shape[1] != 3:
        raise ValueError("template_comp_peak_locations must have shape (N, 3)")
    if (
        trigger_time_idx.ndim != 1
        or threshold_idx.ndim != 1
        or transit_fraction.ndim != 1
        or peak_locations.shape[0] != trigger_time_idx.shape[0]
        or peak_locations.shape[0] != threshold_idx.shape[0]
        or peak_locations.shape[0] != transit_fraction.shape[0]
    ):
        raise ValueError("Study template_comp arrays have inconsistent shapes.")

    comp_by_key: dict[tuple[int, int, int], dict[str, float | int]] = {}
    for idx in range(peak_locations.shape[0]):
        key = (
            int(peak_locations[idx, 0]),
            int(peak_locations[idx, 1]),
            int(trigger_time_idx[idx]),
        )
        if key in comp_by_key:
            raise ValueError(f"Duplicate template compensation entry for key {key}")
        peak_time = float(peak_locations[idx, 2])
        # print(peak_time)
        comp_threshold_idx = int(threshold_idx[idx])
        # inferred_trigger_idx = peak_time + comp_threshold_idx
        inferred_trigger_idx = comp_threshold_idx
        comp_by_key[key] = {
            "peak_time": peak_time,
            "comp_threshold_idx": comp_threshold_idx,
            "transit_fraction": float(transit_fraction[idx]),
            "inferred_trigger_idx": inferred_trigger_idx,
        }

    first_row_by_channel: dict[tuple[int, int], dict[str, float | int]] = {}
    for row in hits_location:
        pixel_x = int(row[0])
        pixel_y = int(row[1])
        recorded_trigger_idx = int(row[2])
        key = (pixel_x, pixel_y, recorded_trigger_idx)
        comp = comp_by_key.get(key)
        if comp is None:
            raise KeyError(f"No template compensation entry found for study hit key {key}")

        channel_key = (pixel_x, pixel_y)
        current = first_row_by_channel.get(channel_key)
        if current is not None and recorded_trigger_idx >= int(current["recorded_trigger_idx"]):
            continue

        inferred_trigger_idx = float(comp["inferred_trigger_idx"])
        first_row_by_channel[channel_key] = {
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "recorded_trigger_idx": recorded_trigger_idx,
            "peak_time": float(comp["peak_time"]),
            "comp_threshold_idx": int(comp["comp_threshold_idx"]),
            "transit_fraction": float(comp["transit_fraction"]),
            "inferred_trigger_idx": inferred_trigger_idx,
            "recorded_minus_inferred": float(recorded_trigger_idx - inferred_trigger_idx),
        }

    return sorted(first_row_by_channel.values(), key=lambda row: (int(row["pixel_x"]), int(row["pixel_y"])))


def build_reference_first_trigger_map(
    reference_data: np.lib.npyio.NpzFile,
    *,
    tpc_id: int,
    batch_id: int,
) -> dict[tuple[int, int], int]:
    key = f"hits_tpc{tpc_id}_batch{batch_id}_location"
    if key not in reference_data:
        raise KeyError(f"Reference NPZ does not contain {key}")

    hits_location = np.asarray(reference_data[key], dtype=int)
    if hits_location.ndim != 2 or hits_location.shape[1] < 3:
        raise ValueError(f"{key} must have shape (N, >=3)")

    first_trigger_by_channel: dict[tuple[int, int], int] = {}
    for row in hits_location:
        channel_key = (int(row[0]), int(row[1]))
        trigger_idx = int(row[2])
        current = first_trigger_by_channel.get(channel_key)
        if current is None or trigger_idx < current:
            first_trigger_by_channel[channel_key] = trigger_idx

    return first_trigger_by_channel


def match_to_reference(
    study_rows: list[dict[str, float | int]],
    reference_first_trigger: dict[tuple[int, int], int],
) -> list[dict[str, float | int]]:
    matched_rows = []
    for row in study_rows:
        channel_key = (int(row["pixel_x"]), int(row["pixel_y"]))
        reference_trigger_idx = reference_first_trigger.get(channel_key)
        if reference_trigger_idx is None:
            continue

        study_delta = float(row["recorded_minus_inferred"])
        matched_rows.append(
            {
                **row,
                "reference_trigger_idx": int(reference_trigger_idx),
                "difference_vs_reference": float(study_delta - reference_trigger_idx),
            }
        )

    return matched_rows


def write_table_csv(rows: list[dict[str, float | int]], output_path: Path) -> None:
    fieldnames = [
        "pixel_x",
        "pixel_y",
        "recorded_trigger_idx",
        "peak_time",
        "comp_threshold_idx",
        "transit_fraction",
        "inferred_trigger_idx",
        "recorded_minus_inferred",
        "reference_trigger_idx",
        "difference_vs_reference",
    ]
    with output_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_histogram(differences: np.ndarray, output_path: Path, *, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(differences, bins=bins, histtype="stepfilled", alpha=0.85, color="tab:blue")
    ax.axvline(np.mean(differences), color="tab:red", linestyle="--", linewidth=1.5, label="mean")
    ax.set_xlabel(r"$(\mathrm{recorded} - \mathrm{inferred}) - \mathrm{reference\ trigger}$")
    ax.set_ylabel("Channels")
    ax.set_title("First-Sequence Trigger Difference vs Reference")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    stats_text = "\n".join(
        [
            f"n = {differences.size}",
            f"mean = {np.mean(differences):.2f}",
            f"std = {np.std(differences):.2f}",
            f"min = {np.min(differences):.2f}",
            f"max = {np.max(differences):.2f}",
        ]
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_summary(
    output_path: Path,
    *,
    study_path: Path,
    reference_path: Path,
    study_row_count: int,
    matched_count: int,
    missing_reference_count: int,
    differences: np.ndarray,
) -> None:
    negative_fraction = float(np.mean(differences < 0)) if differences.size else np.nan
    lines = [
        f"study_npz: {study_path}",
        f"reference_npz: {reference_path}",
        "study_formula: recorded_trigger_idx - comp_threshold_idx",
        f"study_first_sequences: {study_row_count}",
        f"matched_channels: {matched_count}",
        f"missing_reference_channels: {missing_reference_count}",
    ]
    if differences.size:
        lines.extend(
            [
                f"difference_mean: {np.mean(differences):.6f}",
                f"difference_std: {np.std(differences):.6f}",
                f"difference_min: {np.min(differences):.6f}",
                f"difference_max: {np.max(differences):.6f}",
                f"difference_negative_fraction: {negative_fraction:.6f}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    study_path = Path(args.study_npz).expanduser()
    reference_path = Path(args.reference_npz).expanduser()
    prefix = Path(args.output_prefix).expanduser() if args.output_prefix else study_path.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with np.load(study_path) as study_data, np.load(reference_path) as reference_data:
        study_rows = build_study_first_sequence_rows(study_data)
        reference_first_trigger = build_reference_first_trigger_map(
            reference_data,
            tpc_id=args.tpc_id,
            batch_id=args.batch_id,
        )
        matched_rows = match_to_reference(study_rows, reference_first_trigger)

    if not matched_rows:
        raise ValueError("No matched channels found between study and reference.")

    table_path = prefix.parent / f"{prefix.name}_first_sequence_trigger_compare.csv"
    hist_path = prefix.parent / f"{prefix.name}_hist_trigger_compare_vs_reference.png"
    summary_path = prefix.parent / f"{prefix.name}_trigger_compare_summary.txt"

    write_table_csv(matched_rows, table_path)
    differences = np.asarray([row["difference_vs_reference"] for row in matched_rows], dtype=float)
    plot_histogram(differences, hist_path, bins=args.bins)
    write_summary(
        summary_path,
        study_path=study_path,
        reference_path=reference_path,
        study_row_count=len(study_rows),
        matched_count=len(matched_rows),
        missing_reference_count=len(study_rows) - len(matched_rows),
        differences=differences,
    )

    print(f"wrote table: {table_path}")
    print(f"wrote histogram: {hist_path}")
    print(f"wrote summary: {summary_path}")
    print(f"study first sequences: {len(study_rows)}")
    print(f"matched channels: {len(matched_rows)}")
    print(f"difference mean/std: {np.mean(differences):.3f} / {np.std(differences):.3f}")


if __name__ == "__main__":
    main()
