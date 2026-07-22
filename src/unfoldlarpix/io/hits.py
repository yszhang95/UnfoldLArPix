"""Typed access to hits arrays — column semantics enforced in one place.

Never index ``hits.location`` with bare integers elsewhere: the col3
misreading ("last latch") caused the censor reset-reference bug
(FINDINGS item 19).  Column meanings are documented in
``model/conventions.py`` and VALIDATED here at construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HitsView:
    """Read-only, semantics-checked view of an event's hits arrays.

    location columns: [pixel_x, pixel_y, trigger, first_latch, rearm]
    data columns:     [x, y, z, q1..q_nburst]  (cumulative per burst)
    """

    location: np.ndarray
    data: np.ndarray
    adc_hold_delay: int

    def __post_init__(self):
        loc = np.asarray(self.location)
        dat = np.asarray(self.data)
        if loc.ndim != 2 or loc.shape[1] != 5:
            raise ValueError(f"hits.location must be (n, 5), got {loc.shape}")
        if dat.ndim != 2 or dat.shape[1] < 4:
            raise ValueError(f"hits.data must be (n, 3+nburst), got {dat.shape}")
        if len(loc) != len(dat):
            raise ValueError("hits.location / hits.data length mismatch")
        # enforce the col3 semantics: first latch == trigger + B, always
        if len(loc) and not np.all(loc[:, 3] - loc[:, 2] == self.adc_hold_delay):
            raise ValueError(
                "hits col3 is not trigger + adc_hold_delay everywhere — "
                "column semantics violated (see conventions.py)")

    # -- location accessors (global fine ticks / pixel indices) ----------
    @property
    def pixel_x(self) -> np.ndarray:
        return self.location[:, 0]

    @property
    def pixel_y(self) -> np.ndarray:
        return self.location[:, 1]

    @property
    def trigger(self) -> np.ndarray:
        return self.location[:, 2]

    @property
    def first_latch(self) -> np.ndarray:
        """= trigger + B.  Equals the LAST latch only for nburst = 1."""
        return self.location[:, 3]

    @property
    def rearm(self) -> np.ndarray:
        """Discriminator re-arm AFTER the last burst (col4)."""
        return self.location[:, 4]

    @property
    def nburst(self) -> int:
        return self.data.shape[1] - 3

    def latch(self, k: int) -> np.ndarray:
        """k-th latch time (1-based): trigger + k*B."""
        if not (1 <= k <= self.nburst):
            raise IndexError(f"latch k={k} outside 1..{self.nburst}")
        return self.trigger + k * self.adc_hold_delay

    @property
    def last_latch(self) -> np.ndarray:
        return self.trigger + self.nburst * self.adc_hold_delay

    # -- charge accessors -------------------------------------------------
    @property
    def cumulative_charges(self) -> np.ndarray:
        """(n, nburst) cumulative charge at each latch."""
        return self.data[:, 3:]

    @property
    def burst_charges(self) -> np.ndarray:
        """(n, nburst) per-burst charges (differenced across columns)."""
        c = self.cumulative_charges
        return np.diff(np.concatenate(
            [np.zeros((len(c), 1), dtype=c.dtype), c], axis=1), axis=1)
