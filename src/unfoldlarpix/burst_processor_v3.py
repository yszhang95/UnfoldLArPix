"""BurstSequenceProcessorV3 — two-pass dead-time then template processor."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .burst_processor import (
    BurstSequence,
    BurstSequenceProcessor,
    MergedSequence,
    TemplateCompensationAnchor,
)


@dataclass
class MergedBurstGroup:
    """Intermediate first-pass result with explicit times and cumulative charge."""

    pixel_x: int
    pixel_y: int
    trigger_time_idx: int
    times: np.ndarray
    charges: np.ndarray
    cumulative: np.ndarray

    def __post_init__(self):
        if len(self.times) != len(self.charges):
            raise ValueError(
                "MergedBurstGroup times and charges must have the same length."
            )
        if len(self.cumulative) != len(self.charges) + 1:
            raise ValueError(
                "MergedBurstGroup cumulative must be one element longer than charges."
            )
        if len(self.times) == 0:
            raise ValueError("MergedBurstGroup cannot be empty.")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("MergedBurstGroup times must be strictly increasing.")

    @property
    def t_first(self) -> float:
        return float(self.times[0])

    @property
    def t_last(self) -> float:
        return float(self.times[-1])


class BurstSequenceProcessorV3(BurstSequenceProcessor):
    """Two-pass burst processor that preserves V1 compensation rules.

    Pass 1:
    - Merge consecutive close-enough burst sequences using V1 dead-time
      compensation only.

    Pass 2:
    - Treat each pass-1 merged group as a block and connect the blocks with the
      V1 template-compensation logic, allowing the template section to truncate
      naturally to the gap available before the next merged group.
    """

    def _record_group_template_compensation_anchor(
        self,
        group: MergedBurstGroup,
        *,
        transit_threshold_idx: int,
        transit_fraction: float,
        is_bootstrap: bool,
    ) -> None:
        """Record the peak in a merged group used for template compensation."""
        peak_index = int(np.argmax(group.charges))
        self.template_compensation_anchors.append(
            TemplateCompensationAnchor(
                pixel_x=group.pixel_x,
                pixel_y=group.pixel_y,
                trigger_time_idx=group.trigger_time_idx,
                trigger_timestamp=float(group.trigger_time_idx),
                sequence_peak_index=peak_index,
                sequence_peak_time=float(group.times[peak_index]),
                sequence_peak_charge=float(group.charges[peak_index]),
                transit_threshold_idx=int(transit_threshold_idx),
                transit_fraction=float(transit_fraction),
                is_bootstrap=is_bootstrap,
            )
        )

    def _sequence_to_group(self, seq: BurstSequence) -> MergedBurstGroup:
        """Convert a raw burst sequence into the explicit-time group format."""
        times = np.array(
            [seq.t_first + i * self.adc_hold_delay for i in range(len(seq.charges))],
            dtype=float,
        )
        cumulative = np.concatenate([[0.0], np.cumsum(seq.charges, dtype=float)])
        return MergedBurstGroup(
            pixel_x=seq.pixel_x,
            pixel_y=seq.pixel_y,
            trigger_time_idx=seq.trigger_time_idx,
            times=times,
            charges=np.asarray(seq.charges, dtype=float),
            cumulative=cumulative,
        )

    def _group_to_temp_sequence(self, group: MergedBurstGroup) -> BurstSequence:
        """Adapt an explicit-time group to V1 dead-time compensation inputs."""
        return BurstSequence(
            pixel_x=group.pixel_x,
            pixel_y=group.pixel_y,
            trigger_time_idx=group.trigger_time_idx,
            t_first=group.t_first,
            t_last=group.t_last,
            charges=group.charges,
            last_adc_latch=0,
            next_integration_start=0,
        )

    def _merge_close_sequences_first_pass(
        self, sequences: List[BurstSequence]
    ) -> List[MergedBurstGroup]:
        """Collapse close-enough raw sequences using dead-time compensation only."""
        if len(sequences) == 0:
            raise ValueError("sequences list cannot be empty")

        groups: List[MergedBurstGroup] = []
        current_group = self._sequence_to_group(sequences[0])

        for seq in sequences[1:]:
            gap = seq.t_first - current_group.t_last
            if 0 < gap <= self.tau:
                temp_seq = self._group_to_temp_sequence(current_group)
                times, cumulative = self._dead_time_compensation(
                    temp_seq, seq, self.deadtime
                )
                current_group = MergedBurstGroup(
                    pixel_x=current_group.pixel_x,
                    pixel_y=current_group.pixel_y,
                    trigger_time_idx=current_group.trigger_time_idx,
                    times=np.asarray(times, dtype=float),
                    charges=np.diff(cumulative),
                    cumulative=np.asarray(cumulative, dtype=float),
                )
                continue

            groups.append(current_group)
            current_group = self._sequence_to_group(seq)

        groups.append(current_group)
        return groups

    def _template_compensation_to_group(
        self,
        cumulative: np.ndarray | None,
        times: np.ndarray | None,
        deadtime: float,
        next_group: MergedBurstGroup,
        threshold: float,
        template_cumulative: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """Apply V1-style template compensation before an explicit-time group."""
        if threshold is None:
            raise ValueError("Template compensation requires a threshold value.")
        if times is not None and len(times) == 0:
            raise ValueError("Template compensation requires existing time points.")

        first_group = times is None or cumulative is None
        last_time = float(times[-1]) if not first_group else 0.0
        last_cumulative = float(cumulative[-1]) if not first_group else 0.0

        template_cumulative = np.asarray(template_cumulative, dtype=float)
        if template_cumulative.size == 0:
            raise ValueError(
                "Template compensation requires a non-empty cumulative template."
            )

        transit = threshold / np.max(np.cumsum(next_group.charges))
        transit = min(transit, 1.0)

        tlength = next_group.t_first - last_time - deadtime
        if not first_group and tlength <= 1:
            raise ValueError(
                "Not enough time for template compensation, available time "
                f"{tlength} is too short."
            )
        tlength = int(np.round(tlength))

        step = int(self.adc_hold_delay)
        threshold_idx = None
        if first_group:
            threshold_idx = int(
                np.searchsorted(template_cumulative, transit, side="left")
            )
            template_section = template_cumulative[: threshold_idx + 1][::-1][::step][
                ::-1
            ]
        else:
            for jidx in range(tlength, len(template_cumulative)):
                if (
                    template_cumulative[jidx]
                    - template_cumulative[jidx - tlength]
                    >= transit
                ):
                    threshold_idx = jidx
                    break
            if threshold_idx is None:
                threshold_idx = len(template_cumulative) - 1
            start_idx = max(threshold_idx - tlength, 0)
            template_section = template_cumulative[start_idx : threshold_idx + 1]
            template_section = template_section[::-1][::step][::-1]

        if threshold_idx is None:
            raise ValueError(
                "Template compensation requires the template to reach the threshold "
                "within the available time."
            )

        threshold_time = float(next_group.trigger_time_idx)
        offsets = np.arange(len(template_section))
        candidate_times = threshold_time - (
            len(template_section) - 1 - offsets
        ) * self.adc_hold_delay

        valid_mask = candidate_times > last_time
        if first_group:
            template_times = candidate_times
        else:
            template_times = candidate_times[valid_mask]
        if len(template_times) == 0:
            raise ValueError(
                "No valid template points found before threshold time, cannot "
                "apply template compensation."
            )
        if not np.isclose(template_times[-1], threshold_time):
            raise ValueError(
                "Template compensation requires the last valid template time to be "
                "at the trigger time."
            )

        template_section = (
            template_section if first_group else template_section[valid_mask]
        )
        template_section = template_section * (threshold / template_section[-1])
        template_charge_steps = np.diff(template_section, prepend=0.0)

        merged_group_charges = (
            template_charge_steps[1:].tolist()
            + [float(next_group.charges[0]) - threshold]
            + next_group.charges[1:].tolist()
        )

        trigger_time_idx = template_times[0] - self.adc_hold_delay
        if not first_group and trigger_time_idx <= last_time:
            delta_t = last_time - trigger_time_idx
            merged_group_charges[0] = (
                template_charge_steps[0] * delta_t / self.adc_hold_delay
                + merged_group_charges[0]
            )
        if not np.isclose(template_times[-1], threshold_time):
            raise ValueError(
                "Template compensation requires the last template time to be at the "
                "trigger time."
            )

        next_group_cumulative = np.cumsum(merged_group_charges) + last_cumulative
        if first_group:
            updated_cumulative = np.insert(next_group_cumulative, 0, 0.0)
            updated_times = np.concatenate([template_times[1:], next_group.times])
        else:
            updated_times = np.concatenate([times, template_times[1:], next_group.times])
            updated_cumulative = np.concatenate([cumulative, next_group_cumulative])

        if len(updated_cumulative) != len(updated_times) + 1:
            raise ValueError(
                "Length mismatch between updated times and cumulative after "
                "template compensation. "
                f"Got {len(updated_times)} times and "
                f"{len(updated_cumulative)} cumulative values."
            )
        return updated_times, updated_cumulative, int(threshold_idx), float(transit)

    def process_pixel_sequences(
        self, sequences: List[BurstSequence]
    ) -> MergedSequence:
        """Process one pixel in two passes: dead-time merge then template merge."""
        if len(sequences) == 0:
            raise ValueError("sequences list cannot be empty")

        first_pass_groups = self._merge_close_sequences_first_pass(sequences)

        times, cumulative, threshold_idx, transit_fraction = (
            self._template_compensation_to_group(
                None,
                None,
                0.0,
                first_pass_groups[0],
                self.threshold,
                self.template,
            )
        )
        self._record_group_template_compensation_anchor(
            first_pass_groups[0],
            transit_threshold_idx=threshold_idx,
            transit_fraction=transit_fraction,
            is_bootstrap=True,
        )

        for group in first_pass_groups[1:]:
            times, cumulative, threshold_idx, transit_fraction = (
                self._template_compensation_to_group(
                    cumulative,
                    times,
                    self.deadtime,
                    group,
                    self.threshold,
                    self.template,
                )
            )
            self._record_group_template_compensation_anchor(
                group,
                transit_threshold_idx=threshold_idx,
                transit_fraction=transit_fraction,
                is_bootstrap=False,
            )

        charges = np.diff(cumulative)
        return MergedSequence(
            pixel_x=sequences[0].pixel_x,
            pixel_y=sequences[0].pixel_y,
            times=times,
            charges=charges,
            cumulative=cumulative,
        )
