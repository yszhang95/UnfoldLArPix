"""Demonstrate the semi-reverse of field response averaging.

Forward pipeline:
  raw quarter (25, 25, 2000)
  -> _quadrant_copy -> (50, 50, 2000)
  -> _downsample_by_averaging (npath=10) -> (5, 5, 2000)

Semi-reverse:
  averaged (5, 5, 2000)
  -> broadcast each pixel back to npath×npath sub-pixel block -> (50, 50, 2000)
  -> extract the positive-positive quarter [Q:, Q:] -> (25, 25, 2000)

The mapping for the quarter:
  quarter position (r, c) -> averaged pixel ((r+Q)//npath, (c+Q)//npath)
  e.g. (r=0..4, c=0..4)  -> averaged pixel (2, 2)  [since (0+25)//10=2]
       (r=0..4, c=5..14) -> averaged pixel (2, 3)  [since (5+25)//10=3]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from unfoldlarpix.field_response import FieldResponseProcessor

NPZ_PATH = Path(__file__).parent / "data" / "fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npz"

# ── Load raw data ────────────────────────────────────────────────────────────
data = np.load(NPZ_PATH, allow_pickle=True)
raw_quarter = data["response"]          # (25, 25, 2000)
npath = int(data["npath"])              # 10
Q = raw_quarter.shape[0]               # 25  (== raw_quarter.shape[1])
Nt = raw_quarter.shape[2]              # 2000

print(f"raw quarter shape : {raw_quarter.shape}")
print(f"npath             : {npath}")

# ── Forward: get the averaged response ──────────────────────────────────────
proc = FieldResponseProcessor(NPZ_PATH, normalized=True)
averaged = proc.response                # (5, 5, 2000)
n_px = averaged.shape[0]               # 5
print(f"averaged shape    : {averaged.shape}")

# ── Semi-reverse ─────────────────────────────────────────────────────────────
# Step 1: broadcast each averaged pixel to npath×npath sub-pixel positions.
#   np.repeat over spatial axes restores (n_px*npath, n_px*npath, Nt) = (50, 50, 2000)
full_reconstructed = np.repeat(
    np.repeat(averaged, npath, axis=0), npath, axis=1
)
print(f"full reconstructed shape : {full_reconstructed.shape}")

# Step 2: extract the positive-positive quarter (the original quadrant).
#   In _quadrant_copy the original quarter is placed at [Q:, Q:] of the full array.
quarter_reconstructed = full_reconstructed[Q:, Q:, :]
print(f"quarter reconstructed shape : {quarter_reconstructed.shape}")

# ── Verify the mapping at a few positions ────────────────────────────────────
print("\nMapping check (r, c) -> averaged pixel:")
for r, c in [(0, 0), (0, 5), (0, 14), (4, 4), (4, 5), (10, 10)]:
    ap_r = (r + Q) // npath
    ap_c = (c + Q) // npath
    direct = averaged[ap_r, ap_c, :]
    from_full = quarter_reconstructed[r, c, :]
    match = np.allclose(direct, from_full)
    print(f"  ({r:2d},{c:2d}) -> averaged pixel ({ap_r},{ap_c})  match={match}")

# ── Save expanded array to NPZ ───────────────────────────────────────────────
out_npz = NPZ_PATH.parent / (NPZ_PATH.stem + "_average_expanded.npz")
save_dict = {k: data[k] for k in data if k != "response"}
save_dict["response"] = quarter_reconstructed
np.savez(out_npz, **save_dict)
print(f"\nSaved expanded response {quarter_reconstructed.shape} to {out_npz}")

# ── Plot: compare time waveforms at a few quarter positions ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
sample_positions = [(0, 0), (0, 5), (0, 14), (5, 5), (10, 0), (24, 24)]

for ax, (r, c) in zip(axes.flat, sample_positions):
    ap_r = (r + Q) // npath
    ap_c = (c + Q) // npath
    ax.plot(raw_quarter[r, c, :], label="raw quarter", lw=1)
    ax.plot(quarter_reconstructed[r, c, :], "--", label=f"avg pixel ({ap_r},{ap_c})", lw=1)
    ax.set_title(f"quarter ({r},{c})")
    ax.legend(fontsize=7)
    ax.set_xlabel("time tick")

fig.suptitle("Semi-reverse: raw quarter vs value from averaged pixel", fontsize=11)
fig.tight_layout()
out_path = Path(__file__).parent / "fr_expand_from_average.png"
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
