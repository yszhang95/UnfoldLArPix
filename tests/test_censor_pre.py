"""Pre-trigger censoring: silence BEFORE each trigger.

The post-latch term covers only the interval after a pixel's last burst;
``split_trigger`` states the pre-trigger interval only at its endpoint.
These tests pin the boundary arithmetic (reference / arm / close), the
suppression case the burst gate exists for, and the gradient.
"""
import numpy as np
import pytest
import torch

from unfoldlarpix.constrained_solver import LatchWindow
from unfoldlarpix.io.hits import HitsView
from unfoldlarpix.model.operator import ZSOperator
from unfoldlarpix.terms.base import IterCtx
from unfoldlarpix.terms.censor import CensorRunningMax, pre_trigger_censors

B = 30
NT = 40
DOWN, ONE_TICK, CSA = 20, 2, 5


def make_op(nx=5, ny=5, nt=NT):
    rng = np.random.default_rng(1)
    k = rng.uniform(0.1, 1.0, (3, 3, 5))
    k[1, 1] += 3.0
    k = k / k.sum()
    windows = [LatchWindow(1, 1, 300.0, 330.0, 5.0)]
    return ZSOperator(k, (nx, ny, nt), windows, B, device="cpu",
                      dtype=torch.float64)


def hits(rows, nburst=1):
    """rows: list of (px, py, trigger). Latch/re-arm follow the readout."""
    loc, dat = [], []
    for px, py, trig in rows:
        last = trig + nburst * B
        loc.append([px, py, trig, trig + B, last + DOWN + ONE_TICK])
        dat.append([0.0, 0.0, 0.0] + [10.0 * (j + 1) for j in range(nburst)])
    return HitsView(np.asarray(loc, float), np.asarray(dat, float), B)


def build(rows, *, nburst=1, acq=0.0, boff=(0, 0, 0), **kw):
    op = make_op()
    return pre_trigger_censors(op, hits(rows, nburst), np.asarray(boff, float),
                              csa_reset_time=CSA, threshold=5.0, acq_start=acq,
                              margin=3.0, one_tick=ONE_TICK, **kw), op


class TestScalarEndBackCompat:
    def test_scalar_end_matches_the_original_mask(self):
        """b + 1 <= end selects the same bins as the original b < end."""
        op = make_op()
        nx, ny, _ = op.block_shape
        z, a = np.zeros((nx, ny)), np.zeros((nx, ny))
        t = CensorRunningMax(op, z, a, censor_end=NT - 5, threshold=0.0,
                             norm="l1")
        expected = np.arange(NT) < NT - 5
        assert np.array_equal(t.armed[0, 0].cpu().numpy(), expected)

    def test_per_pixel_end_shape_is_validated(self):
        op = make_op()
        nx, ny, _ = op.block_shape
        with pytest.raises(ValueError, match="censor_end must be"):
            CensorRunningMax(op, np.zeros((nx, ny)), np.zeros((nx, ny)),
                             censor_end=np.zeros((nx, ny, 3)), threshold=0.0)


class TestIntervals:
    def test_one_term_for_a_single_trigger(self):
        terms, _ = build([(2, 2, 900.0)])
        assert len(terms) == 1

    def test_window_closes_before_the_trigger(self):
        """Last checked bin end is at or before trigger - one_tick."""
        terms, _ = build([(2, 2, 900.0)], close_back=0.0)  # 900/30 = bin 30
        armed = terms[0].armed[2, 2].cpu().numpy()
        last = np.flatnonzero(armed)[-1]
        assert (last + 1) * B <= 900.0 - ONE_TICK
        assert (last + 2) * B > 900.0 - ONE_TICK

    def test_close_back_drops_the_checks_nearest_the_crossing(self):
        """Backing the close off removes bin ends near the trigger only.

        The checks nearest the crossing have ~0 slack, so the operator's
        within-bin model error shows there as a violation; measured, 20 ticks
        is what the truth-feasibility gate needs.
        """
        trig = 915.0            # bin end 900 sits 15 ticks before it
        near, _ = build([(2, 2, trig)], close_back=0.0)
        back, _ = build([(2, 2, trig)], close_back=20.0)
        a0 = np.flatnonzero(near[0].armed[2, 2].cpu().numpy())
        a1 = np.flatnonzero(back[0].armed[2, 2].cpu().numpy())
        assert (a0[-1] + 1) * B == 900.0            # the near-crossing check
        assert a1[-1] == a0[-1] - 1                 # dropped by the back-off
        assert a1[0] == a0[0]                       # opening unchanged

    def test_close_back_can_empty_a_short_interval(self):
        """A short post-reset gap loses its only check and emits no row."""
        trig0 = 300.0
        rearm = trig0 + B + DOWN + ONE_TICK
        rows = [(2, 2, trig0), (2, 2, rearm + 25.0)]
        assert len(build(rows, include_post_reset=True, close_back=0.0)[0]) == 2
        assert len(build(rows, include_post_reset=True, close_back=40.0)[0]) == 1

    def test_pre_trigger_reference_is_the_acquisition_edge(self):
        terms, _ = build([(2, 2, 900.0)], acq=150.0)   # bin 5
        w = terms[0].w[2, 2].cpu().numpy()
        assert w[4] == 0.0 and w[5] == 1.0             # full bins after 150
        # a pixel that never fired carries no weight at all
        assert float(terms[0].w[0, 0].abs().sum()) == 0.0
        assert not bool(terms[0].armed[0, 0].any())

    def test_fractional_acquisition_edge_weights_the_boundary_bin(self):
        terms, _ = build([(2, 2, 900.0)], acq=165.0)   # mid bin 5
        w = terms[0].w[2, 2].cpu().numpy()
        assert w[4] == 0.0 and w[5] == pytest.approx(0.5) and w[6] == 1.0

    def test_post_reset_intervals_are_off_by_default(self):
        """Only the pre-trigger interval ships on by default."""
        terms, _ = build([(2, 2, 300.0), (2, 2, 900.0)])
        assert len(terms) == 1
        with_all, _ = build([(2, 2, 300.0), (2, 2, 900.0)], include_post_reset=True)
        assert len(with_all) == 2

    def test_post_reset_opens_at_the_previous_rearm(self):
        """A post-reset interval references the CSA restart, opens at re-arm."""
        trig0, trig1 = 300.0, 900.0
        terms, _ = build([(2, 2, trig0), (2, 2, trig1)], include_post_reset=True)
        assert len(terms) == 2
        last_latch = trig0 + B
        w = terms[1].w[2, 2].cpu().numpy()
        ref_bin = (last_latch + CSA) / B
        assert w[int(np.floor(ref_bin)) - 1] == 0.0
        armed = terms[1].armed[2, 2].cpu().numpy()
        first = np.flatnonzero(armed)[0]
        rearm = last_latch + DOWN + ONE_TICK
        assert (first + 1) * B >= rearm and first * B < rearm

    def test_suppression_limited_retrigger_emits_no_interval(self):
        """re-arm at or after the next trigger -> nothing to constrain.

        This is the case the burst gate exists for; here it falls out of the
        boundary arithmetic instead of needing a separate tau.
        """
        trig0 = 300.0
        rearm = trig0 + B + DOWN + ONE_TICK
        terms, _ = build([(2, 2, trig0), (2, 2, rearm - 1.0)])
        assert len(terms) == 1          # only the pre-trigger interval

    def test_pixels_are_independent(self):
        terms, _ = build([(1, 1, 600.0), (3, 3, 1050.0)])
        a1 = terms[0].armed[1, 1].cpu().numpy()
        a3 = terms[0].armed[3, 3].cpu().numpy()
        assert np.flatnonzero(a1)[-1] < np.flatnonzero(a3)[-1]


class TestGradient:
    @pytest.mark.parametrize("norm", ["l1", "l2"])
    def test_gradient_matches_autograd(self, norm):
        terms, op = build([(2, 2, 900.0), (1, 3, 1080.0)], norm=norm)
        term = terms[0]
        g = torch.Generator().manual_seed(3)
        q = torch.rand(op.q_shape, generator=g, dtype=torch.float64) * 4.0
        grad = torch.zeros_like(q)
        term.grad_into(IterCtx(q, op), grad)
        q2 = q.clone().requires_grad_(True)
        term.value(IterCtx(q2, op)).backward()
        assert torch.allclose(grad, q2.grad, atol=1e-8), \
            f"max diff {(grad - q2.grad).abs().max()}"

    def test_l1_norm_adds_no_curvature(self):
        terms, _ = build([(2, 2, 900.0)], norm="l1")
        assert terms[0].curvature() == 0.0

    def test_inactive_below_threshold(self):
        """A solution whose accumulator stays low pays nothing."""
        terms, op = build([(2, 2, 900.0)])
        q = torch.zeros(op.q_shape, dtype=torch.float64)
        assert float(terms[0].value(IterCtx(q, op))) == 0.0

    def test_bites_on_an_early_excursion(self):
        """Charge early in the armed window is penalised."""
        terms, op = build([(2, 2, 900.0)])
        q = torch.zeros(op.q_shape, dtype=torch.float64)
        q[2, 2, 3] = 500.0
        assert float(terms[0].value(IterCtx(q, op))) > 0.0
