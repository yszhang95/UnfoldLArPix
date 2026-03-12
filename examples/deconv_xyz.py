#!/usr/bin/env python3
"""Compute (x, y, z) physical coordinates for each voxel in deconv_q
and dump to a JSON file compatible with wire-cell event display.

Coordinate conventions
----------------------
- x    : drift direction; x = anode_position - drift_direction * drift_time * v * dt
         where drift_time is in ticks, v = 0.16 cm/us, dt = 0.05 us/tick
- y, z : transverse (pixel) directions;
         y = tpc_lower[1] + (boffset[0] + i) * pitch
         z = tpc_lower[2] + (boffset[1] + j) * pitch
         pitch = 0.4434 cm

Usage
-----
    python deconv_xyz.py deconv_event_0_42.npz [--threshold 0.5] [--output-dir output] [--prefix deconv]
"""

import json
import os
import re
import sys
import argparse
import numpy as np


DRIFT_VELOCITY = 0.16    # cm / us
TIME_PER_TICK  = 0.05    # us / tick
PIXEL_PITCH    = 0.4434  # cm


def compute_xyz(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, charge) for all voxels in deconv_q.

    Parameters
    ----------
    npz_path : str
        Path to a .npz file saved by deconv_example3.py.

    Returns
    -------
    pos : np.ndarray, shape (N, 3)
        Physical coordinates [x, y, z] in cm for every voxel.
    charge : np.ndarray, shape (N,)
        Deconvolved charge in ke- for every voxel.
    """
    data = np.load(npz_path)

    deconv_q        = data["deconv_q"]       # (n_px, n_py, n_t)
    boffset         = data["boffset"]        # [pixel_x_min, pixel_y_min, t_min_tick]
    global_tref     = data["global_tref"]    # array; [1] is the reference tick
    anode_position  = float(data["anode_position"])
    drift_direction = int(data["drift_direction"])
    tpc_lower       = data["tpc_lower"]      # 2-vector [y_lower, z_lower]
    adc_hold_delay  = int(data["adc_hold_delay"])
    drtoa           = float(np.squeeze(data["drtoa"]))

    tref_tick = float(np.asarray(global_tref).flat[1])

    n_px, n_py, n_t = deconv_q.shape

    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n_px),
        np.arange(n_py),
        np.arange(n_t),
        indexing="ij",
    )

    tick             = boffset[2] + k_idx * adc_hold_delay
    print(boffset[2], tref_tick, np.max(k_idx) * adc_hold_delay)
    drift_time_ticks = tick - tref_tick
    print(drift_direction)

    x = anode_position - drift_direction * drift_time_ticks * DRIFT_VELOCITY * TIME_PER_TICK
    x += drtoa
    print(tpc_lower, anode_position)
    y = tpc_lower[0] + (boffset[0] + i_idx) * PIXEL_PITCH
    z = tpc_lower[1] + (boffset[1] + j_idx) * PIXEL_PITCH

    pos    = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    charge = deconv_q.reshape(-1)
    return pos, charge


def save_json(npz_path: str, output_dir: str, threshold: float, prefix: str, tpc_id: int = 0, event_id: int = None) -> None:
    """Load a deconv npz, filter by threshold, and write a wire-cell JSON."""
    # Try to parse tpc_id and event_id from filename if not provided
    if event_id is None:
        m = re.search(r"event_(\d+)_(\d+)", os.path.basename(npz_path))
        if m is not None:
            tpc_id = int(m.group(1))
            event_id = int(m.group(2))
        else:
            # Use hash of filename for fallback event_id
            event_id = hash(os.path.basename(npz_path)) % 100000
    eid = event_id

    pos, charge = compute_xyz(npz_path)
    print(pos[:,0].min())
    print(pos[:,0].min())

    # Filter below threshold (charge is in ke-)
    mask   = charge >= threshold
    pos    = pos[mask]
    charge = charge[mask]
    tpcs   = np.full(len(charge), tpc_id, dtype=int)

    chg = charge.copy()
    chg *= 1000  # convert ke- to e-

    evt = {
        "runNo": 0,
        "subRunNo": 0,
        "eventNo": int(eid),
        "x": pos[:, 0].tolist(),
        "y": pos[:, 1].tolist(),
        "z": pos[:, 2].tolist(),
        "q": chg.tolist(),
        "nq": [0] * len(chg),
        "cluster_id": tpcs.tolist(),
        "real_cluster_id": tpcs.tolist(),
        "geom": "2x2",
        "type": "wire-cell",
    }

    edir = os.path.join(output_dir, "data", str(eid))
    os.makedirs(edir, exist_ok=True)
    filename = f"{eid}-{prefix}.json"
    out_path = os.path.join(edir, filename)
    with open(out_path, "w") as f:
        json.dump(evt, f, indent=2)

    print(f"  [{tpc_id}/{eid}] kept {len(charge)} voxels (threshold={threshold} ke-)  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert deconv_event npz to wire-cell JSON.")
    parser.add_argument("npz_files", nargs="+", help="deconv_event_*.npz files")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum charge in ke- to include (default: 0.5)")
    parser.add_argument("--output-dir", default="output",
                        help="Root output directory (default: output)")
    parser.add_argument("--prefix", default="deconv",
                        help="Filename prefix inside event folder (default: deconv)")
    parser.add_argument("--tpc-id", type=int, default=0,
                        help="TPC ID (default: 0, or extract from filename)")
    parser.add_argument("--event-id", type=int, default=None,
                        help="Event ID (default: None, or extract from filename/hash)")
    args = parser.parse_args()

    for npz_path in args.npz_files:
        save_json(npz_path, args.output_dir, args.threshold, args.prefix, args.tpc_id, args.event_id)


if __name__ == "__main__":
    main()
