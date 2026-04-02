"""BurstSequenceProcessorV2 — fractional phase-shift alignment processor."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .burst_processor import (
    BurstSequence,
    MergedSequence,
    TemplateCompensationAnchor,
)
from .data_containers import Hits


class BurstSequenceProcessorV2:
    """Process burst sequences with first-interval scaling and fractional phase-shift alignment.

    Differences from V1 (BurstSequenceProcessor):
    - Dead-time slope compensation is replaced by a simpler rule:
        * gap < tau  → signal dipped below threshold and recovered; no template is
                       inserted, the next sequence's charges are fractional-shift-
                       corrected and appended directly.
        * gap >= tau → signal was absent long enough to warrant template gap-fill,
                       followed by the fractional-shifted next sequence.
    - First ADC window charge is normalised to an equivalent adc_hold_delay window.
    - Template gap-fill stops at the bin that would collide with the next sequence
      (collision-stopping filter on candidate_times).
    - Each sequence's charges are fractional-shift-corrected in the frequency domain
      to align sub-sample trigger jitter to the common time grid.
    """

    def __init__(
        self,
        adc_hold_delay: float,
        tau: float,
        deadtime: float,
        template: np.ndarray = None,
        threshold: float = None,
    ):
        """Initialise the processor.

        Args:
            adc_hold_delay: Duration of each ADC hold window (ticks or samples).
            tau: Gap threshold.  When gap = next.t_first - last_time < tau the
                 signal is assumed to have merely dipped below threshold; no template
                 is inserted.  When gap >= tau the template is used to fill the gap.
            deadtime: Physical hardware dead-time between consecutive sequences
                      (same units as adc_hold_delay).  Subtracted from the gap when
                      computing how many template ticks fit.
            template: Monotonically-increasing cumulative template waveform used to
                      interpolate charge in gaps between sequences.
            threshold: Charge threshold that defines the template truncation point.
        """
        self.adc_hold_delay = adc_hold_delay
        self.tau = tau
        self.deadtime = deadtime
        self.template = np.asarray(
            template if template is not None else self._default_template(), dtype=float
        )
        if self.template.size == 0:
            raise ValueError("Template cannot be empty.")
        if not np.all(np.diff(self.template) >= 0):
            raise ValueError("template must be monotonically increasing")
        if threshold is None:
            raise ValueError("Threshold value must be provided for template compensation.")
        self.threshold = threshold

        self.totq_per_pix: Dict[Tuple[int, int], float] = {}
        self.template_compensation_anchors: list[TemplateCompensationAnchor] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_template(self) -> np.ndarray:
        return np.array([1, 2, 3, 4, 6, 8, 16, 36], dtype=float)

    def _record_template_compensation_anchor(
        self,
        seq: BurstSequence,
        *,
        transit_threshold_idx: int,
        transit_fraction: float,
        is_bootstrap: bool,
    ) -> None:
        """Record the sequence-local peak used to anchor template compensation."""
        peak_index = int(np.argmax(seq.charges))
        peak_time = float(seq.t_first + peak_index * self.adc_hold_delay)
        peak_charge = float(seq.charges[peak_index])
        self.template_compensation_anchors.append(
            TemplateCompensationAnchor(
                pixel_x=seq.pixel_x,
                pixel_y=seq.pixel_y,
                trigger_time_idx=seq.trigger_time_idx,
                trigger_timestamp=float(seq.trigger_time_idx),
                sequence_peak_index=peak_index,
                sequence_peak_time=peak_time,
                sequence_peak_charge=peak_charge,
                transit_threshold_idx=int(transit_threshold_idx),
                transit_fraction=float(transit_fraction),
                is_bootstrap=is_bootstrap,
            )
        )

    def _fractional_shift(self, charges: np.ndarray, delta_T: float) -> np.ndarray:
        """Shift *charges* by delta_T / adc_hold_delay fractional samples via FFT.

        Args:
            charges: 1-D charge array to shift.
            delta_T: Sub-sample jitter in the same units as adc_hold_delay.

        Returns:
            Real-valued shifted array of the same length.
        """
        D = delta_T / self.adc_hold_delay  # fractional ADC samples
        N = len(charges)
        if N <= 1 or np.isclose(D, 0.0):
            return charges.copy()
        pad = N
        padded = np.concatenate([np.zeros(pad), charges, np.zeros(pad)])
        M = len(padded)
        X = np.fft.fft(padded)
        k = np.fft.fftfreq(M) * M
        phase = np.exp(-1j * 2 * np.pi * k * D / M)
        return np.real(np.fft.ifft(X * phase))[pad : pad + N]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_sequences_from_hits(
        self, hits: Hits
    ) -> Dict[Tuple[int, int], List[BurstSequence]]:
        """Extract burst sequences from a Hits container, grouped by pixel.

        In addition to converting cumulative hit data to per-burst charges, the first
        burst charge of each sequence is normalised for the extended first integration
        window caused by trigger jitter:

            delta_T             = trigger_time_idx % adc_hold_delay
            active_time_first   = adc_hold_delay + delta_T
            charges[0]         *= adc_hold_delay / active_time_first

        Args:
            hits: Hits data container.

        Returns:
            Dict mapping (pixel_x, pixel_y) to a time-sorted list of BurstSequence.
        """
        sequences: Dict[Tuple[int, int], List[BurstSequence]] = {}

        for i in range(len(hits)):
            pixel_x = int(hits.location[i, 0])
            pixel_y = int(hits.location[i, 1])
            trigger_time_idx = int(hits.location[i, 2])
            last_adc_latch = int(hits.location[i, 3])
            next_integration_start = int(hits.location[i, 4])

            # Convert cumulative hit data to per-burst charges
            raw = hits.data[i, 3:]
            charges = np.array([raw[0]] + np.diff(raw).tolist(), dtype=float)

            nburst = len(charges)
            t_start = trigger_time_idx + self.adc_hold_delay
            t_end = trigger_time_idx + self.adc_hold_delay * nburst

            seq = BurstSequence(
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                trigger_time_idx=trigger_time_idx,
                t_first=t_start,
                t_last=t_end,
                charges=charges,
                last_adc_latch=last_adc_latch,
                next_integration_start=next_integration_start,
            )

            pixel_key = (pixel_x, pixel_y)
            sequences.setdefault(pixel_key, []).append(seq)

            if pixel_key not in self.totq_per_pix:
                self.totq_per_pix[pixel_key] = float(np.sum(seq.charges))

        for pixel_key, seqs in sequences.items():
            seqs.sort(key=lambda s: s.trigger_time_idx)

            # Validate no duplicate trigger times
            trigger_times = [s.trigger_time_idx for s in seqs]
            if len(trigger_times) != len(set(trigger_times)):
                raise ValueError(
                    f"Duplicate trigger times found for pixel {pixel_key}"
                )

            # Validate sequences do not overlap (they may touch)
            for j in range(1, len(seqs)):
                prev_seq = seqs[j - 1]
                curr_seq = seqs[j]
                if curr_seq.t_first < prev_seq.t_last:
                    raise ValueError(
                        f"Invalid sequence ordering for pixel {pixel_key}: "
                        f"sequence {j} starts at {curr_seq.t_first} but previous "
                        f"ends at {prev_seq.t_last}"
                    )

        return sequences

    def _template_compensation(
        self,
        cumulative: Optional[np.ndarray],
        times: Optional[np.ndarray],
        deadtime: float,
        next_seq: BurstSequence,
        threshold: float,
        template_cumulative: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """Fill the gap before *next_seq* using template interpolation.

        Fractional phase-shift alignment is NOT applied here; it is deferred to the
        final alignment pass in process_pixel_sequences so that the original times
        are preserved during the merge (collision detection must use real times).

        Key differences from V1:
        - *deadtime* is still subtracted from the gap when computing tlength, so the
          template section is sized to the time that was actually integrating signal.
        - Template candidate times are filtered to the open interval
          (last_time, next_seq.t_first) — stopping collision with the next sequence.
        - When the collision filter yields zero valid template points (gap after
          deadtime is at most one ADC window), the method falls through gracefully
          and appends the raw next sequence charges.

        Args:
            cumulative: Current cumulative charge array (prepended zero included),
                        or None when processing the very first sequence.
            times: Current time points (length = len(cumulative) - 1),
                   or None when processing the very first sequence.
            deadtime: Physical hardware dead-time (same units as adc_hold_delay).
            next_seq: Next sequence to append.
            threshold: Charge threshold for template truncation.
            template_cumulative: Monotonically increasing cumulative template.

        Returns:
            Tuple (updated_times, updated_cumulative, transit_threshold_idx, transit_fraction).
        """
        if threshold is None:
            raise ValueError("Template compensation requires a threshold value.")
        if times is not None and len(times) == 0:
            raise ValueError("Template compensation requires existing time points.")

        first_seq = times is None or cumulative is None
        last_time = float(times[-1]) if not first_seq else 0.0
        last_cumulative = float(cumulative[-1]) if not first_seq else 0.0
        # For the collision filter we want all pre-sequence template points to
        # pass the lower bound during the bootstrap, so use -inf there.
        filter_lower = last_time if not first_seq else -np.inf

        template_cumulative = np.asarray(template_cumulative, dtype=float)
        if template_cumulative.size == 0:
            raise ValueError("Template compensation requires a non-empty cumulative template.")

        transit = threshold / np.max(np.cumsum(next_seq.charges))
        transit = min(transit, 1.0)

        # deadtime is subtracted so tlength covers only the signal-bearing portion
        # of the gap (same convention as V1).
        tlength = int(np.round(next_seq.t_first - last_time - deadtime))
        if not first_seq and tlength <= 1:
            raise ValueError(
                f"Not enough time for template compensation, available time {tlength} is too short."
            )

        # Build template section
        step = int(self.adc_hold_delay)
        if first_seq:
            threshold_idx = int(np.searchsorted(template_cumulative, transit, side="left"))
            template_section = template_cumulative[: threshold_idx + 1][::-1][::step][::-1]
        else:
            threshold_idx = None
            for jidx in range(tlength, len(template_cumulative)):
                if template_cumulative[jidx] - template_cumulative[jidx - tlength] >= transit:
                    threshold_idx = jidx
                    break
            if threshold_idx is None:
                # Fallback: transit exceeds the maximum window rise (e.g. total charge < threshold).
                # Use the end of the template — the best available rising edge.
                threshold_idx = len(template_cumulative) - 1
            template_section = template_cumulative[threshold_idx - tlength : threshold_idx + 1]
            template_section = template_section[::-1][::step][::-1]

        threshold_time = next_seq.trigger_time_idx
        n_template = len(template_section)
        offsets = np.arange(n_template)
        candidate_times = threshold_time - (n_template - 1 - offsets) * self.adc_hold_delay

        # V2: collision-stopping filter — keep only (filter_lower, next_seq.t_first).
        # filter_lower is last_time for non-bootstrap calls; -inf for the bootstrap
        # so that all pre-sequence template points are preserved.
        valid_mask = (candidate_times > filter_lower) & (candidate_times < next_seq.t_first)

        next_seq_times = np.array(
            [next_seq.t_first + i * self.adc_hold_delay for i in range(len(next_seq.charges))]
        )

        if not np.any(valid_mask):
            # No template points fit in the gap — append fractional-shifted sequence only
            next_seq_cumulative = np.cumsum(next_seq.charges) + last_cumulative
            if first_seq:
                updated_cumulative = np.insert(next_seq_cumulative, 0, 0.0)
                updated_times = next_seq_times
            else:
                updated_times = np.concatenate([times, next_seq_times])
                updated_cumulative = np.concatenate([cumulative, next_seq_cumulative])
            if len(updated_cumulative) != len(updated_times) + 1:
                raise ValueError(
                    f"Length mismatch: {len(updated_times)} times and "
                    f"{len(updated_cumulative)} cumulative values."
                )
                return updated_times, updated_cumulative, int(threshold_idx), float(transit)

        if not np.isclose(candidate_times[valid_mask][-1], threshold_time):
            raise ValueError(
                "Template compensation requires the last valid template time to be "
                "at the trigger time."
            )

        template_times = candidate_times[valid_mask]
        template_section = template_section[valid_mask]
        template_section = template_section * (threshold / template_section[-1])
        template_section_diff = np.diff(template_section, prepend=0.0)

        # Combine template interval charges with fractional-shifted next_seq charges
        chgs = (
            template_section_diff[1:].tolist()
            + [next_seq.charges[0] - threshold]
            + next_seq.charges[1:].tolist()
        )

        # Correct the first template charge when the template window straddles last_time
        trigger_time_start = template_times[0] - self.adc_hold_delay
        if not first_seq and trigger_time_start < last_time:
            delta_t_adj = last_time - trigger_time_start
            chgs[0] = template_section_diff[0] * delta_t_adj / self.adc_hold_delay + chgs[0]

        if not np.isclose(template_times[-1], threshold_time):
            raise ValueError(
                "Template compensation requires the last template time to be at the trigger time."
            )

        next_seq_cumulative = np.cumsum(chgs) + last_cumulative

        if first_seq:
            updated_cumulative = np.insert(next_seq_cumulative, 0, 0.0)
            updated_times = np.concatenate([template_times[1:], next_seq_times])
        else:
            updated_times = np.concatenate([times, template_times[1:], next_seq_times])
            updated_cumulative = np.concatenate([cumulative, next_seq_cumulative])

        if len(updated_cumulative) != len(updated_times) + 1:
            raise ValueError(
                f"Length mismatch: {len(updated_times)} times and "
                f"{len(updated_cumulative)} cumulative values."
            )
        return updated_times, updated_cumulative, int(threshold_idx), float(transit)

    def _append_shifted(
        self,
        cumulative: np.ndarray,
        times: np.ndarray,
        seq: BurstSequence,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Append *seq* to the running (times, cumulative) with first-interval scaling.

        Used when gap < tau: the signal merely dipped below threshold, so no
        template interpolation is needed.

        Only first-interval scaling is applied here.  Fractional phase-shift
        alignment is deferred to the final pass in process_pixel_sequences so
        that the collision filter continues to operate on the original (unshifted)
        times during the merge.

            active_time_first = (next.t_first - curr.t_last) - deadtime
                              = gap - deadtime

        For gap >= tau the template models the rising edge from first principles,
        so no scaling is applied to charges[0].
        """
        charges = seq.charges.copy()
        gap = seq.t_first - times[-1]
        active_time_first = gap - self.deadtime
        if active_time_first <= 0:
            raise ValueError(
                f"active_time_first={active_time_first} <= 0 "
                f"(gap={gap}, deadtime={self.deadtime}). "
                "deadtime must be smaller than the inter-sequence gap."
            )
        charges[0] *= self.adc_hold_delay / active_time_first
        seq_times = np.array(
            [seq.t_first + i * self.adc_hold_delay for i in range(len(seq.charges))]
        )
        seq_cumulative = np.cumsum(charges) + cumulative[-1]
        return (
            np.concatenate([times, seq_times]),
            np.concatenate([cumulative, seq_cumulative]),
        )

    def process_pixel_sequences(
        self, sequences: List[BurstSequence]
    ) -> MergedSequence:
        """Process all sequences for a single pixel.

        **Two-phase design**

        Phase 1 — merge (original times preserved):
          * gap < tau  — signal dipped below threshold; first-interval scaling
                         applied then the raw charges are appended directly.
          * gap >= tau — signal was absent; template gap-fill then raw charges.
          Original times are kept throughout so collision detection works correctly.

        Phase 2 — grid alignment (applied after all merging):
          Each original BurstSequence has a sub-sample jitter
          ``delta_T = trigger_time_idx % adc_hold_delay``.  All times that came from
          the same source sequence share the same ``delta_T`` offset relative to the
          common ADC grid.  After merging, we:
            1. Apply a fractional phase-shift to each contiguous same-delta_T block
               of charges to redistribute them onto the common grid.
            2. Subtract each block's ``delta_T`` from its times.
          The result is a merged sequence whose times lie exactly on the
          ``n * adc_hold_delay`` grid with no fractional offsets.

        Args:
            sequences: Time-sorted BurstSequence list for one pixel.

        Returns:
            MergedSequence with grid-aligned times and fractional-shift-corrected charges.
        """
        if len(sequences) == 0:
            raise ValueError("sequences list cannot be empty")

        first_seq = sequences[0]
        dT_first = first_seq.trigger_time_idx % self.adc_hold_delay

        # Bootstrap: build the rising-edge template before the first sequence.
        times, cumulative, threshold_idx, transit_fraction = self._template_compensation(
            None,
            None,
            0,           # no deadtime offset for the first-sequence bootstrap
            first_seq,
            self.threshold,
            self.template,
        )
        self._record_template_compensation_anchor(
            first_seq,
            transit_threshold_idx=threshold_idx,
            transit_fraction=transit_fraction,
            is_bootstrap=True,
        )
        # All times from bootstrap (template + first burst) share first_seq's delta_T
        delta_T_per_time = np.full(len(times), dT_first)

        for curr_seq in sequences[1:]:
            gap = curr_seq.t_first - times[-1]
            n_before = len(times)
            dT_curr = curr_seq.trigger_time_idx % self.adc_hold_delay

            if gap < self.tau:
                # Signal dipped below threshold — no template needed.
                # FIXME: BUG! this is not cumulative because we correct the first charge. It is an average...
                times, cumulative = self._append_shifted(cumulative, times, curr_seq)
            else:
                # Signal was absent long enough to warrant template gap-fill.
                times, cumulative, threshold_idx, transit_fraction = self._template_compensation(
                    cumulative,
                    times,
                    self.deadtime,
                    curr_seq,
                    self.threshold,
                    self.template,
                )
                self._record_template_compensation_anchor(
                    curr_seq,
                    transit_threshold_idx=threshold_idx,
                    transit_fraction=transit_fraction,
                    is_bootstrap=False,
                )

            # New time points (template + burst) all carry curr_seq's delta_T
            n_added = len(times) - n_before
            delta_T_per_time = np.concatenate(
                [delta_T_per_time, np.full(n_added, dT_curr)]
            )

        # ------------------------------------------------------------------
        # Phase 2: grid alignment
        # Apply fractional phase-shift per contiguous same-delta_T block, then
        # snap times to the common n * adc_hold_delay grid.
        # ------------------------------------------------------------------
        charges = np.diff(cumulative)
        aligned_charges = charges.copy()

        change_pts = np.where(np.abs(np.diff(delta_T_per_time)) > 1e-9)[0] + 1
        block_starts = np.concatenate([[0], change_pts])
        block_ends = np.concatenate([change_pts, [len(times)]])

        for s, e in zip(block_starts, block_ends):
            dT = delta_T_per_time[s]
            aligned_charges[s:e] = self._fractional_shift(charges[s:e], dT)

        aligned_times = times - delta_T_per_time

        # Remove duplicates at block boundaries: keep the former time point,
        # drop the colliding one from the next block.  Guaranteed by ADC > tau
        # that the step is always 0 (duplicate) or ADC (perfect) — never more.
        keep = np.concatenate([[True], np.diff(aligned_times) > 1e-9])
        aligned_times = aligned_times[keep]
        aligned_charges = aligned_charges[keep]
        aligned_cumulative = np.concatenate([[0], np.cumsum(aligned_charges)])

        return MergedSequence(
            pixel_x=sequences[0].pixel_x,
            pixel_y=sequences[0].pixel_y,
            times=aligned_times,
            charges=aligned_charges,
            cumulative=aligned_cumulative,
        )

    def process_hits(
        self, hits: Hits
    ) -> Dict[Tuple[int, int], MergedSequence]:
        """Process all burst sequences in a Hits container.

        Args:
            hits: Hits data container.

        Returns:
            Dict mapping (pixel_x, pixel_y) to MergedSequence.
        """
        self.template_compensation_anchors = []
        sequences_by_pixel = self.extract_sequences_from_hits(hits)
        return {
            pixel_key: self.process_pixel_sequences(seqs)
            for pixel_key, seqs in sequences_by_pixel.items()
        }
