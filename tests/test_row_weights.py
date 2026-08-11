"""Row metadata and the diagonal measurement-noise weighting.

The noise model lives in model/noise.py; the row kinds are emitted by the
same loop that builds the windows (build_latch_rows), so they cannot
drift.  Scales come from the readout config, never from code constants.
"""
import numpy as np
import pytest

from unfoldlarpix.constrained_solver import (build_latch_rows,
                                             build_latch_windows)
from unfoldlarpix.model.noise import row_variances, row_weights

B, THR, RESET = 30, 5.0, 2
BOFF = np.array([0, 0, 0])


class _RC:
    adc_hold_delay, adc_down_time, one_tick = B, 24, 2
    uncorr_noise, thres_noise, reset_noise = 0.5, 0.65, 0.9


class _RCNoiseless(_RC):
    uncorr_noise = None


def _seq(trigger, nburst=1, q=20.0):
    loc = [0, 0, trigger, trigger + B, trigger + B * nburst + 26]
    dat = [0.0, 0.0, 0.0] + [q * (k + 1) for k in range(nburst)]
    return loc, dat


def _rows(triggers, nburst=1, split=THR):
    loc, dat = zip(*(_seq(t, nburst) for t in triggers))
    return build_latch_rows(np.array(loc), np.array(dat, dtype=float), B,
                            BOFF, csa_reset_time=RESET,
                            split_threshold=split)


def test_windows_wrapper_unchanged():
    """build_latch_windows must stay bit-identical to build_latch_rows[0]."""
    loc, dat = zip(*(_seq(t, 2) for t in [100, 400]))
    loc, dat = np.array(loc), np.array(dat, dtype=float)
    assert build_latch_windows(loc, dat, B, BOFF, csa_reset_time=RESET,
                               split_threshold=THR) == \
        build_latch_rows(loc, dat, B, BOFF, csa_reset_time=RESET,
                         split_threshold=THR)[0]


def test_meta_kinds_split_sequence():
    wins, metas = _rows([100], nburst=3)
    assert [m.kind for m in metas] == ["pseudo", "remainder", "diff", "diff"]
    assert len(wins) == len(metas)


def test_meta_kinds_lumped_when_below_threshold():
    wins, metas = _rows([100], nburst=2, split=1e9)
    assert [m.kind for m in metas] == ["lumped", "diff"]


def test_post_reset_flag():
    _, metas = _rows([100, 800], nburst=1)
    # first sequence virgin, second sequence post-reset
    assert [m.post_reset for m in metas] == [False, False, True, True]


def test_variances_match_model():
    _, metas = _rows([100, 800], nburst=2)
    v = row_variances(metas, _RC())
    su2, st2, sr2 = 0.25, 0.4225, 0.81
    expect = [st2 + su2, 2 * su2 + st2, 2 * su2,          # virgin seq
              st2 + su2 + sr2, 2 * su2 + st2, 2 * su2]    # post-reset seq
    assert v == pytest.approx(expect)


def test_split_mode_keeps_real_rows_at_unity():
    _, metas = _rows([100, 800], nburst=2)
    w = row_weights(metas, _RC(), mode="split")
    kinds = [m.kind for m in metas]
    for k, wi in zip(kinds, w):
        if k in ("lumped", "diff"):
            assert wi == 1.0
        else:
            assert wi < 1.0


def test_diag_mode_references_diff_row():
    _, metas = _rows([100], nburst=2)
    w = row_weights(metas, _RC(), mode="diag")
    kinds = [m.kind for m in metas]
    assert w[kinds.index("diff")] == pytest.approx(1.0)


def test_noiseless_config_rejected():
    _, metas = _rows([100])
    with pytest.raises(ValueError):
        row_variances(metas, _RCNoiseless())


def test_unknown_mode_rejected():
    _, metas = _rows([100])
    with pytest.raises(ValueError):
        row_weights(metas, _RC(), mode="full")


def test_diag_mean_normalised_when_no_diff_rows():
    """nb1 events have no burst-diff anchor; the reference falls back to
    the mean variance so the average weight is 1, not ~0.5."""
    _, metas = _rows([100, 800], nburst=1)
    assert not any(m.kind == "diff" for m in metas)
    w = row_weights(metas, _RC(), mode="diag")
    import numpy as np
    v = row_variances(metas, _RC())
    assert (1.0 / w) == pytest.approx(v / v.mean())
    assert np.average(v * w) == pytest.approx(v.mean())


class TestHitsSupport:
    """source=hits support: amplitude-blind, anchored on fired pixels."""

    def test_covers_hit_neighbourhood(self):
        import numpy as np
        from unfoldlarpix.algs.reco_algs import BuildSupport
        from unfoldlarpix.io.hits import HitsView

        class _Op:
            q_shape = (10, 10, 20)

        class _Store(dict):
            def get(self, k): return self[k]

        loc = np.array([[5, 5, 60, 90, 120]])          # trigger 60, latch 90
        dat = np.array([[0., 0., 0., 20.]])
        st = _Store()
        st['readout_config'] = _RC()
        st['op'] = _Op()
        st['time_subbin'] = 1
        st['hits_view'] = HitsView(loc, dat, B)
        st['block_offset'] = np.array([0, 0, 0], float)
        st['warm.deconv_q'] = np.zeros((10, 10, 20))
        alg = BuildSupport(source='hits', hits_dilate=2, t_pad=2)
        out = {}
        alg.put = lambda store, k, v: out.__setitem__(k, v)
        alg.execute(st)
        s = out['support']
        # trigger bin 2, latch bin 3 -> padded window [0, 5]
        assert s[5, 5, 2] and s[5, 5, 3]
        assert s[3, 3, 2] and s[7, 7, 3]      # Chebyshev-2 neighbourhood
        assert not s[5, 5, 8]                 # outside padded time window
        assert not s[2, 5, 2]                 # outside pixel neighbourhood
