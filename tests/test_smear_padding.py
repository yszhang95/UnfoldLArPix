"""The truth smearing must not wrap around its own array.

`gaus_smear_true_3d` convolves by FFT, which is circular.  The time axis was
always padded; the PIXEL axes were not, and the array was opened at exactly the
charge's extent.  A track narrow in pixels therefore folded into itself --
measured on `mu_a50` (three occupied pixel columns): 83.5 ke belonging in
column 141 appeared in column 138, inflating it from 87.0 to 167.8 ke and
leaving 141 empty.  Charge is conserved by a circular convolution, so
`sum_truth` and `integral_pct` could not see it while `true_killed` and the
ghost fractions were wrong by tens of ke.

These tests fail loudly if the padding is ever removed.
"""
import numpy as np
import pytest
from unfoldlarpix.smear_truth import gaus_smear_true_3d

W = np.array([0.5, 0.5, 0.005])          # the adopted analysis widths
SIG_PIX = 1.0 / (2.0 * np.pi * W[0])     # 0.318 pixel in real space


def _smear(points, charges, width=W):
    """points: (n,3) integer locations.  charges: (n,) ke."""
    loc = np.asarray(points, dtype=np.int64)
    dat = np.zeros((loc.shape[0], 4))
    dat[:, -1] = charges
    return gaus_smear_true_3d(loc, dat, width)


def _columns(off, arr, axis=0):
    """charge per index along `axis`, keyed by absolute index."""
    other = tuple(a for a in range(3) if a != axis)
    prof = arr.sum(axis=other)
    return {int(off[axis]) + i: float(prof[i]) for i in range(prof.size)}


def test_single_point_charge_spreads_across_pixels():
    """The decisive case: with no padding the array is one pixel wide, the
    pixel filter is identically 1, and the charge cannot spread at all."""
    off, s = _smear([[100, 200, 0]], [10.0])
    cols = _columns(off, s, axis=0)
    peak = max(cols, key=lambda k: cols[k])
    assert peak == 100
    # it must actually reach the neighbours
    assert cols[99] > 0.01 * cols[100]
    assert cols[101] > 0.01 * cols[100]
    # and it must be symmetric about the charge
    for d in (1, 2, 3):
        assert cols[100 - d] == pytest.approx(cols[100 + d], abs=1e-9)


def test_narrow_track_does_not_wrap():
    """Three occupied columns -- the mu_a50 geometry.  Charge leaving the right
    edge must appear to the RIGHT, not fold onto the left-most column."""
    pts = [[138, 50, 0], [139, 50, 0], [140, 50, 0]]
    q = [13.25, 1137.05, 1215.63]        # mu_a50's own column totals
    off, s = _smear(pts, q)
    c = _columns(off, s, axis=0)
    assert c[141] > 50.0, "column 141 is empty: the +1 tap wrapped away"
    # 141 is fed by 140 the same way 138's own charge is topped up from 139,
    # so both flanks must be populated -- a wrap empties exactly one of them
    assert c[137] != pytest.approx(0.0, abs=1e-6)
    assert c[142] != pytest.approx(0.0, abs=1e-6)
    # column 138 held 13.25 ke of charge; smearing pulls in ~0.084 of 139 and
    # -0.016 of 140, so it must land near 87, NOT near 168 (the wrapped value)
    assert 70.0 < c[138] < 105.0, f"column 138 = {c[138]:.1f}, wrap-around?"
    # the two central columns keep the bulk
    assert c[139] > 1000.0 and c[140] > 1000.0


def test_charge_is_conserved():
    pts = [[138, 50, 0], [139, 50, 0], [140, 51, 7]]
    q = [13.25, 1137.05, 1215.63]
    off, s = _smear(pts, q)
    assert float(s.sum()) == pytest.approx(sum(q), rel=1e-9)


def test_padding_is_wide_enough_for_the_kernel():
    """A wider array must give the same answer as a narrow one, up to the
    kernel's own tail: padding must not be a free parameter that changes the
    result.

    The kernel is an aliased Gaussian whose taps fall only as ~1/d^2, so the
    residual converges as 1/pad rather than vanishing.  Measured max deviation
    per unit charge: 2.5e-3 at pad 4, 7.7e-4 at 8, 2.1e-4 at 16.  The shipped
    floor is 8, so the tolerance here is 1e-3 -- tight enough to catch the pad
    dropping to 4, loose enough not to demand exactness the kernel cannot give.
    """
    narrow = _smear([[100, 200, 0]], [10.0])
    # the same charge with two distant zero-charge points forcing a big array
    wide = _smear([[100, 200, 0], [80, 180, 0], [120, 220, 0]],
                  [10.0, 0.0, 0.0])
    cn = _columns(*narrow, axis=0)
    cw = _columns(*wide, axis=0)
    for k in range(96, 105):
        assert cn[k] == pytest.approx(cw[k], abs=1e-3 * 10.0), \
            f"column {k}: narrow {cn[k]:.5f} vs wide {cw[k]:.5f}"


def test_time_axis_behaviour_is_unchanged():
    """The time axis was already padded; the fix must not have touched it.
    A single charge must spread over ~sigma_time and stay symmetric."""
    off, s = _smear([[100, 200, 500]], [10.0])
    t = _columns(off, s, axis=2)
    peak = max(t, key=lambda k: t[k])
    assert peak == 500
    sig_t = 1.0 / (2.0 * np.pi * W[2])          # 31.8 fine ticks
    for d in (10, 30, 60):
        assert t[500 - d] == pytest.approx(t[500 + d], rel=1e-6)
    assert t[500 + int(sig_t)] == pytest.approx(t[500] * np.exp(-0.5), rel=0.05)
