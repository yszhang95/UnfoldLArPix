"""Tests for the phase-aware (linear) charge deposit in merged_sequences_to_block."""

import numpy as np
import pytest

from unfoldlarpix.burst_processor import MergedSequence, merged_sequences_to_block


def _make_seq(pixel, times, charges):
    times = np.asarray(times, dtype=float)
    charges = np.asarray(charges, dtype=float)
    return MergedSequence(
        pixel_x=pixel[0],
        pixel_y=pixel[1],
        times=times,
        charges=charges,
        cumulative=np.concatenate([[0.0], np.cumsum(charges)]),
    )


BIN = 30


class TestLinearDeposit:
    def test_rejects_unknown_mode(self):
        seqs = {(0, 0): _make_seq((0, 0), [0.0], [1.0])}
        with pytest.raises(ValueError, match="deposit_mode"):
            merged_sequences_to_block(seqs, BIN, npadbin=2, deposit_mode="nearest")

    def test_integer_bin_times_match_floor_mode(self):
        """Times exactly on the bin grid: linear must reproduce floor exactly."""
        times = [0.0, BIN, 2 * BIN]
        charges = [1.0, 2.0, 3.0]
        seqs = {(3, 4): _make_seq((3, 4), times, charges)}
        off_f, blk_f = merged_sequences_to_block(seqs, BIN, npadbin=2, deposit_mode="floor")
        off_l, blk_l = merged_sequences_to_block(seqs, BIN, npadbin=2, deposit_mode="linear")
        np.testing.assert_array_equal(off_f, off_l)
        np.testing.assert_allclose(blk_f, blk_l, atol=1e-12)

    def test_charge_conservation_any_phase(self):
        rng = np.random.default_rng(42)
        phase = 17.3
        times = phase + BIN * np.arange(6)
        charges = rng.uniform(0.5, 5.0, size=6)
        seqs = {(0, 0): _make_seq((0, 0), times, charges)}
        _, blk = merged_sequences_to_block(seqs, BIN, npadbin=2, deposit_mode="linear")
        assert blk.sum() == pytest.approx(charges.sum(), rel=1e-12)

    def test_split_fractions(self):
        """A sample 40% into a bin leaves 60% in that bin, 40% in the next."""
        # Two entries: the first pins tmin (fpos = npadbin exactly), the
        # second sits 0.4 bins later inside the grid.
        times = [0.0, BIN + 0.4 * BIN]
        charges = [0.0, 10.0]
        seqs = {(0, 0): _make_seq((0, 0), times, charges)}
        _, blk = merged_sequences_to_block(seqs, BIN, npadbin=2, deposit_mode="linear")
        trace = blk[0, 0]
        npad = 2
        assert trace[npad + 1] == pytest.approx(6.0)
        assert trace[npad + 2] == pytest.approx(4.0)

    def test_mean_position_preserved(self):
        """Linear deposit keeps the charge-weighted mean time exact."""
        phase = 11.7
        times = phase + BIN * np.arange(4)
        charges = np.array([1.0, 4.0, 2.0, 0.5])
        seqs = {(0, 0): _make_seq((0, 0), times, charges)}
        offset, blk = merged_sequences_to_block(seqs, BIN, npadbin=2, deposit_mode="linear")
        trace = blk[0, 0]
        bins = np.arange(len(trace))
        mean_bin = (trace * bins).sum() / trace.sum()
        expected = ((times - offset[2]) / BIN * charges).sum() / charges.sum()
        assert mean_bin == pytest.approx(expected, rel=1e-12)

    def test_deposit_phase_shifts_content(self):
        times = [0.0, BIN]
        charges = [2.0, 2.0]
        seqs = {(0, 0): _make_seq((0, 0), times, charges)}
        _, blk0 = merged_sequences_to_block(
            seqs, BIN, npadbin=2, deposit_mode="linear", deposit_phase=0.0)
        _, blkm = merged_sequences_to_block(
            seqs, BIN, npadbin=2, deposit_mode="linear", deposit_phase=-1.0)
        np.testing.assert_allclose(blkm[0, 0, :-1], blk0[0, 0, 1:], atol=1e-12)

    def test_multi_pixel_independence(self):
        seqs = {
            (0, 0): _make_seq((0, 0), [10.0], [3.0]),
            (1, 2): _make_seq((1, 2), [25.0], [7.0]),
        }
        _, blk = merged_sequences_to_block(seqs, BIN, npadbin=2, deposit_mode="linear")
        assert blk[0, 0].sum() == pytest.approx(3.0)
        assert blk[1, 2].sum() == pytest.approx(7.0)
        assert blk[0, 2].sum() == pytest.approx(0.0)
        assert blk[1, 0].sum() == pytest.approx(0.0)

    def test_pad_pixels_expands_spatial_extent(self):
        seqs = {
            (10, 20): _make_seq((10, 20), [0.0], [3.0]),
            (12, 21): _make_seq((12, 21), [30.0], [5.0]),
        }
        off0, blk0 = merged_sequences_to_block(seqs, BIN, npadbin=1)
        off, blk = merged_sequences_to_block(seqs, BIN, npadbin=1, pad_pixels=4)
        assert off[0] == off0[0] - 4 and off[1] == off0[1] - 4
        assert blk.shape[0] == blk0.shape[0] + 8
        assert blk.shape[1] == blk0.shape[1] + 8
        assert blk.shape[2] == blk0.shape[2]
        # charges land at the same physical pixels (shifted indices)
        np.testing.assert_allclose(blk[4:-4, 4:-4, :], blk0, atol=1e-12)
        assert blk[:4].sum() == 0 and blk[-4:].sum() == 0

    def test_floor_mode_unchanged_regression(self):
        """Floor mode still errors on in-bin collisions and drops phase."""
        times = [5.0, 5.0 + BIN]
        charges = [1.0, 2.0]
        seqs = {(0, 0): _make_seq((0, 0), times, charges)}
        _, blk = merged_sequences_to_block(seqs, BIN, npadbin=1, deposit_mode="floor")
        trace = blk[0, 0]
        assert trace[1] == pytest.approx(1.0)
        assert trace[2] == pytest.approx(2.0)
