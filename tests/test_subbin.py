"""Sub-bin (finer fitting time bin) primitives: kernel re-binning and window
overlap sampling must be consistent when the operator bin is subdivided."""
import numpy as np

from unfoldlarpix.constrained_solver import LatchWindow, windows_to_sampling
from unfoldlarpix.deconv_workflow import integrate_kernel_over_time


def test_integrate_kernel_subbin_conserves_total_and_refines():
    k = np.zeros((1, 1, 60))
    k[0, 0, :30] = 1.0
    kB = integrate_kernel_over_time(k, 30)      # bin = 30 ticks -> 2 bins
    kB2 = integrate_kernel_over_time(k, 15)     # bin = 15 ticks -> 4 bins
    assert kB.shape[-1] == 2 and kB2.shape[-1] == 4
    assert np.isclose(kB.sum(), 30.0)           # total charge preserved
    assert np.isclose(kB.sum(), kB2.sum())      # invariant under refinement


def test_windows_to_sampling_finer_bin_doubles_coverage():
    # a latch window integrating ticks [0,30): at B=30 it is exactly one block
    # bin; on a B/2=15 grid it overlaps two block bins (overlap fractions x2).
    w = LatchWindow(px=0, py=0, t_lo=0.0, t_hi=30.0, value=5.0)
    _, c1, wt1 = windows_to_sampling([w], (1, 1, 4), 30)
    _, c2, wt2 = windows_to_sampling([w], (1, 1, 8), 15)
    assert len(wt1) == 1 and np.isclose(sum(wt1), 1.0)
    assert len(wt2) == 2 and np.isclose(sum(wt2), 2.0)
