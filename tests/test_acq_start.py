"""Acquisition-start edge of the first latch window: legacy -inf, uniform
scalar, and channel-wise callable (global pixel coordinates)."""
import numpy as np

from unfoldlarpix.constrained_solver import build_latch_windows

# two pixels, one single-burst sequence each; cumulative charge below the
# split threshold so each sequence yields exactly one window
LOC = np.array([[10, 3, 200, 230, 260],
                [11, 3, 500, 530, 560]])
DAT = np.array([[0.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 4.0]])
BOFF = np.array([5, 1, 100])   # block offset (px, py, ticks)


def first_windows(**kw):
    wins = build_latch_windows(LOC, DAT, 30, BOFF, csa_reset_time=2,
                               split_threshold=5.0, **kw)
    out = {}
    for w in wins:
        out.setdefault((w.px, w.py), w)     # first window per pixel
    return out


def test_default_is_minus_inf():
    w = first_windows()
    assert all(np.isneginf(v.t_lo) for v in w.values())


def test_scalar_uniform_edge():
    w = first_windows(acq_start=130.0)      # absolute ticks
    # block-local: 130 - block_offset[2]
    assert all(v.t_lo == 30.0 for v in w.values())


def test_channel_wise_callable_global_coords():
    seen = []

    def per_channel(px, py):
        seen.append((px, py))
        return 100.0 + 10.0 * px            # ticks, varies by channel

    w = first_windows(acq_start=per_channel)
    # callable receives GLOBAL pixel coordinates
    assert set(seen) == {(10, 3), (11, 3)}
    assert w[(5, 2)].t_lo == (100.0 + 10.0 * 10) - 100.0   # px_global=10
    assert w[(6, 2)].t_lo == (100.0 + 10.0 * 11) - 100.0   # px_global=11
