#!/usr/bin/env python3
"""Mask sequence-aligned arrays in a source NPZ to match a reference NPZ count.

This utility was written to compare deconvolution outputs where the source file
contains more triggered / compensated sequences than a reference file. It builds
a one-to-one assignment between source and reference hit rows using
``(pixel_x, pixel_y, trigger_time_idx)`` from ``hits_location[:, :3]`` and keeps
only the matched source rows in the sequence-aligned arrays.

All non-sequence arrays are copied unchanged.

Example
-------
python examples/mask_sequences_to_reference.py \
  --source examples/analysis_20260402/deconv_positron_thres5k_nburst256_fastadc0p5_sp005_spp2_event_0_0.npz \
  --reference examples/analysis_20260402/deconv_positron_thres5k_nburst256_sp005_spp2_event_0_0.npz \
  --output data/deconv_positron_thres5k_nburst256_fastadc0p5_sp005_spp2_event_0_0_masked18_to_match_nburst256.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


SEQUENCE_ALIGNED_KEYS = {
    "hits_location",
    "hits_data",
    "template_comp_peak_locations",
    "template_comp_trigger_time_idx",
    "template_comp_trigger_timestamp",
    "template_comp_peak_indices",
    "template_comp_peak_charges",
    "template_comp_transit_threshold_idx",
    "template_comp_transit_fraction",
    "template_comp_is_bootstrap",
    "template_comp_effq_peak_time",
    "template_comp_effq_peak_value",
    "template_comp_effq_peak_distance",
    "template_comp_effq_peak_distance_abs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only the source sequences that can be matched one-to-one to a "
            "reference file using hits_location[:, :3]."
        )
    )
    parser.add_argument("--source", required=True, help="Source NPZ to be masked.")
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference NPZ that defines how many sequences to keep.",
    )
    parser.add_argument("--output", required=True, help="Output NPZ path.")
    parser.add_argument(
        "--pixel-weight",
        type=float,
        default=100.0,
        help="Quadratic weight applied to pixel-coordinate differences.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.array(data[key]) for key in data.files}


def validate_sequence_lengths(payload: dict[str, np.ndarray]) -> int:
    if "hits_location" not in payload or "hits_data" not in payload:
        raise KeyError("Source payload must contain hits_location and hits_data.")

    n_rows = int(payload["hits_location"].shape[0])
    if int(payload["hits_data"].shape[0]) != n_rows:
        raise ValueError("hits_location and hits_data row counts do not match.")

    for key in SEQUENCE_ALIGNED_KEYS:
        if key not in payload:
            continue
        if int(payload[key].shape[0]) != n_rows:
            raise ValueError(
                f"{key} has {payload[key].shape[0]} rows but expected {n_rows}."
            )
    return n_rows


def build_match_mask(
    source_hits_location: np.ndarray,
    reference_hits_location: np.ndarray,
    *,
    pixel_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_xyz = np.asarray(source_hits_location[:, :3], dtype=float)
    reference_xyz = np.asarray(reference_hits_location[:, :3], dtype=float)
    if source_xyz.shape[0] < reference_xyz.shape[0]:
        raise ValueError(
            "Source has fewer sequences than reference; masking would not make sense."
        )

    cost = pixel_weight * (
        (source_xyz[:, None, 0] - reference_xyz[None, :, 0]) ** 2
        + (source_xyz[:, None, 1] - reference_xyz[None, :, 1]) ** 2
    ) + (source_xyz[:, None, 2] - reference_xyz[None, :, 2]) ** 2
    source_rows, reference_cols = linear_sum_assignment(cost)
    keep_mask = np.zeros(source_xyz.shape[0], dtype=bool)
    keep_mask[source_rows] = True
    return keep_mask, source_rows, reference_cols


def mask_payload(
    payload: dict[str, np.ndarray],
    keep_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    masked: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key in SEQUENCE_ALIGNED_KEYS:
            masked[key] = value[keep_mask]
        else:
            masked[key] = value
    return masked


def main() -> None:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    source = load_npz(source_path)
    reference = load_npz(reference_path)
    n_source = validate_sequence_lengths(source)
    n_reference = validate_sequence_lengths(reference)

    keep_mask, source_rows, reference_cols = build_match_mask(
        source["hits_location"],
        reference["hits_location"],
        pixel_weight=args.pixel_weight,
    )
    masked = mask_payload(source, keep_mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **masked)

    removed_indices = np.flatnonzero(~keep_mask)
    matched_cost = args.pixel_weight * (
        (source["hits_location"][source_rows, 0] - reference["hits_location"][reference_cols, 0]) ** 2
        + (source["hits_location"][source_rows, 1] - reference["hits_location"][reference_cols, 1]) ** 2
    ) + (source["hits_location"][source_rows, 2] - reference["hits_location"][reference_cols, 2]) ** 2

    print(f"source: {source_path}")
    print(f"reference: {reference_path}")
    print(f"output: {output_path}")
    print(f"source rows: {n_source}")
    print(f"reference rows: {n_reference}")
    print(f"kept rows: {int(keep_mask.sum())}")
    print(f"removed rows: {int((~keep_mask).sum())}")
    print(
        "matched cost stats: "
        f"min={float(np.min(matched_cost)):.3f} "
        f"median={float(np.median(matched_cost)):.3f} "
        f"max={float(np.max(matched_cost)):.3f}"
    )
    print(f"removed indices: {removed_indices.tolist()}")


if __name__ == "__main__":
    main()
