"""Universal-grid evaluation protocol (reconstruction-independent).

Truth and reco are rebinned INDEPENDENTLY onto bins at global multiples
of adc_hold_delay; reco charges are deposited as Gaussian shapes at
their regressed (sub-bin) centers.  Moved from examples/ so the
protocol is import-safe and testable; the examples CLI is a thin shim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constrained_solver import gaussian_post_smooth  # noqa: F401 (API)

def pool_block(block: np.ndarray, group_pixels: int = 1,
               group_time: int = 1) -> np.ndarray:
    """Sum-pool a (nx, ny, nt) block over non-overlapping voxel groups."""
    gp, gt = int(group_pixels), int(group_time)
    if gp == 1 and gt == 1:
        return block
    nx, ny, nt = block.shape
    px, py, pt = (-nx) % gp, (-ny) % gp, (-nt) % gt
    b = np.pad(block, ((0, px), (0, py), (0, pt)))
    return b.reshape((nx + px) // gp, gp, (ny + py) // gp, gp,
                     (nt + pt) // gt, gt).sum(axis=(1, 3, 5))


def universal_rebin(npz_path: Path, truth_npz: Path | None = None,
                    content_offset_ticks: float = 0.0,
                    deposit_shape: str = "linear",
                    sigma_time: float = 0.005,
                    sigma_pxl: float = 0.2,
                    time_offsets: np.ndarray | None = None,
                    edge_anchor: str = "universal",
                    return_origin: bool = False):
    """Rebin truth and reco INDEPENDENTLY onto the universal grid.

    Universal time bins have edges at global multiples of adc_hold_delay
    (anchored to tick 0 of the common clock); pixels are absolute
    hardware indices.  The smeared truth is summed from its own fine-tick
    grid; the reconstruction's bin contents are deposited from their
    PHYSICAL centers (boffset + k*B under the release-point convention,
    boffset + (k + 1/2)*B for legacy files, plus an optional
    diagnostic ``content_offset_ticks``).  Neither binning depends on the
    other — the protocol required for absolute (cross-config,
    cross-event) statements.

    ``deposit_shape`` controls how reco content is deposited:
    - "linear": charge-conserving linear split of the SMOOTHED coarse
      content (treats each bin as uniform — carries a rebinning cost of
      up to half a bin of artificial spread).
    - "gaussian": each SHARP fitted charge is deposited as a Gaussian
      around its regressed mean with the analysis filter width
      (sigma_time, frequency-domain; time-domain sigma = 1/(2 pi
      sigma_time) ticks), then the spatial analysis Gaussian is applied —
      i.e. the smeared field G (x) q_hat is evaluated DIRECTLY on the
      universal grid at fine-tick precision.  This removes the rebinning
      artifact entirely; the shape width is hypothetical (filter- or
      diffusion-motivated), matching the smeared-truth convention.

    ``time_offsets`` (gaussian mode only): optional array shaped like
    ``deconv_q_sharp`` with a per-voxel shift [fine ticks] added to that
    charge's deposit center — the hook for sub-bin regressed positions.

    ``edge_anchor`` selects the bin EDGES both sides are binned against:
    - "universal": edges at global multiples of adc_hold_delay.  Neither
      binning depends on the reconstruction — the protocol required for
      absolute (cross-config, cross-event) statements.
    - "fit": edges anchored on the reconstruction's own declared block
      origin, i.e. the fit grid.  Everything else (the deposit shape, the
      sub-bin centers, the truth smearing) is unchanged, so this isolates
      the EDGE SET as the only difference.  Reconstruction-dependent by
      construction: each event is then scored on its own grid.
    """
    f = np.load(npz_path, allow_pickle=True)
    t = np.load(truth_npz, allow_pickle=True) if truth_npz is not None else f
    B = int(f["adc_hold_delay"])
    smeared = np.asarray(t["smeared_true"], dtype=np.float64)
    s_off = np.asarray(t["smear_offset"], dtype=np.int64)
    dq = np.asarray(f["deconv_q"], dtype=np.float64)
    b_off = np.asarray(f["boffset"], dtype=np.float64)

    # bin m spans [phi + m*B, phi + (m+1)*B).  phi = 0 is the universal grid;
    # phi = b_off[2] mod B puts the edges on the reconstruction's own bins.
    if edge_anchor == "universal":
        phi = 0.0
    elif edge_anchor == "fit":
        phi = float(b_off[2] - np.floor(b_off[2] / B) * B)
    else:
        raise ValueError(f"unknown edge_anchor: {edge_anchor}")

    # ---- truth: fine ticks -> bins (pad front to a bin edge)
    pre = int(s_off[2] - (np.floor((s_off[2] - phi) / B) * B + phi))
    nt_f = smeared.shape[2] + pre
    post = (-nt_f) % B
    tr_fine = np.pad(smeared, ((0, 0), (0, 0), (pre, post)))
    tr_u = tr_fine.reshape(*tr_fine.shape[:2], -1, B).sum(axis=3)
    tr_t0 = int(np.floor((s_off[2] - phi) / B))     # first bin index
    tr_p0 = (int(s_off[0]), int(s_off[1]))

    # ---- reco: physical bin centers -> bins
    nx, ny, ntq = dq.shape
    # bin k's physical instant.  New files declare the release point
    # (boffset = raw corner, charge at b_off + k*B); legacy files declare
    # half a bin early and are deposited at the bin centre, which lands on
    # the same instant for even B.  The marker distinguishes them.
    conv = f["time_convention"] if "time_convention" in f else None
    conv = str(conv) if conv is not None else "legacy_half_bin"
    half = 0.0 if conv == "release_point" else 0.5
    centers = b_off[2] + (np.arange(ntq) + half) * B + content_offset_ticks
    fpos = (centers - phi) / B - 0.5    # fractional bin position
    i0 = np.floor(fpos).astype(np.int64)
    frac = fpos - i0
    u_min = int(min(i0.min(), tr_t0))
    u_max = int(max(i0.max() + 1, tr_t0 + tr_u.shape[2] - 1))
    ntu = u_max - u_min + 1
    p_min = (min(int(b_off[0]), tr_p0[0]), min(int(b_off[1]), tr_p0[1]))
    p_max = (max(int(b_off[0]) + nx, tr_p0[0] + tr_u.shape[0]),
             max(int(b_off[1]) + ny, tr_p0[1] + tr_u.shape[1]))
    shape = (p_max[0] - p_min[0], p_max[1] - p_min[1], ntu)

    out_shape = shape
    reco = np.zeros(out_shape)
    ox, oy = int(b_off[0]) - p_min[0], int(b_off[1]) - p_min[1]
    if deposit_shape == "gaussian" and "deconv_q_sharp" in f.files:
        import math

        q_sharp = np.asarray(f["deconv_q_sharp"], dtype=np.float64)
        sig_ticks = 1.0 / (2.0 * np.pi * float(sigma_time))
        edges = (np.arange(u_min, u_max + 2) * B + phi).astype(np.float64)
        erf = np.vectorize(math.erf)
        if time_offsets is None:
            # weight of fit-bin k in universal bin m: Gaussian mass
            # between edges
            z = (edges[None, :] - centers[:, None]) / (np.sqrt(2.0) * sig_ticks)
            cdf = 0.5 * (1.0 + erf(z))
            W = cdf[:, 1:] - cdf[:, :-1]              # (ntq, ntu)
            reco[ox:ox + nx, oy:oy + ny, :] = np.einsum(
                "xyk,km->xym", q_sharp, W)
        else:
            off = np.asarray(time_offsets, dtype=np.float64)
            if off.shape != q_sharp.shape:
                raise ValueError("time_offsets must match deconv_q_sharp")
            xs, ys, ks = np.nonzero(q_sharp > 1e-6)
            reach = int(np.ceil(6.0 * sig_ticks / B)) + 1
            for x, y, k in zip(xs, ys, ks):
                c = centers[k] + off[x, y, k]
                m_c = int(np.floor((c - phi) / B)) - u_min
                m0 = max(m_c - reach, 0)
                m1 = min(m_c + reach + 1, ntu)
                if m1 <= m0:
                    continue
                z = (edges[m0:m1 + 1] - c) / (np.sqrt(2.0) * sig_ticks)
                cdf = 0.5 * (1.0 + erf(z))
                reco[ox + x, oy + y, m0:m1] += (
                    q_sharp[x, y, k] * (cdf[1:] - cdf[:-1]))
        # spatial analysis Gaussian on the pixel axes (same convention as
        # gaussian_filter_3d): full-array FFT so truth-side pixels align
        fx = np.fft.fftfreq(out_shape[0])
        fy = np.fft.fftfreq(out_shape[1])
        gx = np.exp(-0.5 * fx**2 / float(sigma_pxl) ** 2)
        gy = np.exp(-0.5 * fy**2 / float(sigma_pxl) ** 2)
        R = np.fft.fftn(reco, axes=(0, 1))
        reco = np.real(np.fft.ifftn(
            R * gx[:, None, None] * gy[None, :, None], axes=(0, 1)))
    else:
        for k in range(ntq):
            col = dq[:, :, k]
            b = i0[k] - u_min
            reco[ox:ox + nx, oy:oy + ny, b] += col * (1.0 - frac[k])
            reco[ox:ox + nx, oy:oy + ny, b + 1] += col * frac[k]
    truth = np.zeros(shape)
    tx, ty = tr_p0[0] - p_min[0], tr_p0[1] - p_min[1]
    truth[tx:tx + tr_u.shape[0], ty:ty + tr_u.shape[1],
          tr_t0 - u_min: tr_t0 - u_min + tr_u.shape[2]] = tr_u
    if return_origin:
        # the universal grid's own origin, so a caller can map a physical
        # (pixel, latch instant) onto these blocks instead of recomputing the
        # alignment -- which is how two callers came to use two formulas
        return truth, reco, {"u_min": int(u_min), "p_min": [int(p_min[0]),
                                                           int(p_min[1])],
                             "bin_ticks": int(B), "phi": float(phi),
                             # the offset THIS grid was built with.  It is the
                             # charge-centre one, half a bin from the fit
                             # grid's raw corner, and mixing the two shifts
                             # every latch bin by B/2.
                             "b_off": [float(x) for x in b_off]}
    return truth, reco


def metrics_from_blocks(smear_summed: np.ndarray, aligned_dq: np.ndarray,
                        corr_threshold: float = 0.5) -> dict:
    """All scalar metrics from an aligned (truth, reco) block pair."""
    sum_dq = float(aligned_dq.sum())
    sum_truth = float(smear_summed.sum())

    # Fig-3 style correlation on voxels above threshold.
    mask = aligned_dq > corr_threshold
    x = smear_summed[mask]
    y = aligned_dq[mask]
    if x.size > 2 and np.std(x) > 0 and np.std(y) > 0:
        pearson_r = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        pearson_r, slope = float("nan"), float("nan")

    ghost_frac = float((x < corr_threshold).sum() / max(x.size, 1))
    killed_mask = (smear_summed > corr_threshold) & ~mask
    true_killed = float(smear_summed[killed_mask].sum())

    # Ghost decomposition: a ghost voxel adjacent (Chebyshev distance 1) to
    # truth above the cut is a one-voxel offset of the smeared truth
    # (tolerable); an isolated ghost injects unphysical charge (disapproved).
    truth_near = smear_summed > corr_threshold
    for ax in range(truth_near.ndim):
        truth_near = (truth_near
                      | np.roll(truth_near, 1, axis=ax)
                      | np.roll(truth_near, -1, axis=ax))
    ghost_mask = mask & (smear_summed < corr_threshold)
    ghost_adj = ghost_mask & truth_near
    ghost_iso = ghost_mask & ~truth_near
    n_sel = max(int(mask.sum()), 1)
    ghost_adj_frac = float(ghost_adj.sum() / n_sel)
    ghost_iso_frac = float(ghost_iso.sum() / n_sel)
    ghost_iso_charge = float(aligned_dq[ghost_iso].sum())

    # Per-voxel residual reco - truth [ke-] over the SIGNAL REGION: every
    # voxel where either side is above the cut, so the sample contains the
    # ghosts (reco-only) and the killed truth (truth-only) as well as the
    # matched voxels.  Restricting to reco > cut alone would hide the killed
    # truth, which is half the failure mode.
    sig = mask | (smear_summed > corr_threshold)
    resid = (aligned_dq - smear_summed)[sig]
    if resid.size:
        resid_mean = float(resid.mean())
        resid_rms = float(np.sqrt((resid ** 2).mean()))   # about zero
        resid_sd = float(resid.std())                     # about the mean
    else:
        resid_mean = resid_rms = resid_sd = float("nan")

    return {
        "sum_deconv_q": round(sum_dq, 2),
        "sum_truth": round(sum_truth, 2),
        "integral_pct": round(100.0 * (sum_dq / sum_truth - 1.0), 3),
        "pearson_r": round(pearson_r, 5),
        "slope": round(slope, 5),
        "ghost_frac": round(ghost_frac, 5),
        "ghost_adj_frac": round(ghost_adj_frac, 5),
        "ghost_iso_frac": round(ghost_iso_frac, 5),
        "ghost_iso_charge": round(ghost_iso_charge, 2),
        "true_killed": round(true_killed, 2),
        "resid_mean": round(resid_mean, 5),
        "resid_rms": round(resid_rms, 5),
        "resid_sd": round(resid_sd, 5),
        "n_voxels_signal": int(resid.size),
        "n_voxels_gt_thr": int(x.size),
        "corr_threshold": corr_threshold,
    }

