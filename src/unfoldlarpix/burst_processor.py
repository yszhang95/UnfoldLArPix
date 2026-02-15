"""Burst sequence merging and interpolation processor for UnfoldLArPix."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from .data_containers import Hits


@dataclass
class BurstSequence:
    """Represents a single burst sequence for a pixel."""

    pixel_x: int
    pixel_y: int
    trigger_time_idx: int
    t_start: float  # trigger_time_idx + adc_hold_delay
    t_end: float    # trigger_time_idx + adc_hold_delay * nburst
    charges: np.ndarray  # Array of charge values for each burst
    last_adc_latch: int
    next_integration_start: int

    def __post_init__(self):
        """Validate sequence data."""
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end ({self.t_end}) must be > t_start ({self.t_start})")
        if len(self.charges) == 0:
            raise ValueError("charges array cannot be empty")


@dataclass
class MergedSequence:
    """Represents a merged and compensated burst sequence."""

    pixel_x: int
    pixel_y: int
    times: np.ndarray        # Time points for each charge value
    charges: np.ndarray      # Charge values (after differentiation of cumulative)
    cumulative: np.ndarray   # Cumulative charge values

    def __post_init__(self):
        """Validate merged sequence data."""
        if len(self.times) != len(self.charges):
            raise ValueError("times and charges must have the same length")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("times must be strictly monotonically increasing")


class BurstSequenceProcessor:
    """Process burst sequences with dead-time and template compensation."""

    def __init__(
        self,
        adc_hold_delay: float,
        tau: float,
        delta_t: float,
        template: Optional[np.ndarray] = None,
        template_spacing: Optional[float] = None,
    ):
        """Initialize the burst sequence processor.

        Args:
            adc_hold_delay: Time duration for each ADC hold (in same units as tau, delta_t)
            tau: Threshold for determining if sequences are close enough
            delta_t: Predefined dead time (minimal time resolution unit)
            template: Optional template waveform for non-close sequence compensation.
                     Should be monotonically increasing cumulative values.
            template_spacing: Time spacing between template points (default: adc_hold_delay)
        """
        self.adc_hold_delay = adc_hold_delay
        self.tau = tau
        self.delta_t = delta_t
        self.template = template if template is not None else self._default_template()
        self.template_spacing = template_spacing if template_spacing is not None else adc_hold_delay

        # Validate template
        if not np.all(np.diff(self.template) >= 0):
            raise ValueError("template must be monotonically increasing")

    def _default_template(self) -> np.ndarray:
        """Create a default exponential-like template."""
        return np.array([1, 2, 3, 4, 6, 8, 16, 36], dtype=float)

    def extract_sequences_from_hits(self, hits: Hits) -> Dict[Tuple[int, int], List[BurstSequence]]:
        """Extract burst sequences from Hits container, grouped by pixel.

        Args:
            hits: Hits data container

        Returns:
            Dictionary mapping (pixel_x, pixel_y) to list of BurstSequence objects
        """
        sequences: Dict[Tuple[int, int], List[BurstSequence]] = {}

        for i in range(len(hits)):
            # Extract location info
            pixel_x = int(hits.location[i, 0])
            pixel_y = int(hits.location[i, 1])
            trigger_time_idx = int(hits.location[i, 2])
            last_adc_latch = int(hits.location[i, 3])
            next_integration_start = int(hits.location[i, 4])

            # Extract charge data (skip x, y, z columns)
            charges = hits.data[i, 3:]

            # Calculate times
            nburst = len(charges)
            t_start = trigger_time_idx + self.adc_hold_delay
            t_end = trigger_time_idx + self.adc_hold_delay * nburst

            # Create sequence
            seq = BurstSequence(
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                trigger_time_idx=trigger_time_idx,
                t_start=t_start,
                t_end=t_end,
                charges=charges,
                last_adc_latch=last_adc_latch,
                next_integration_start=next_integration_start,
            )

            # Group by pixel
            pixel_key = (pixel_x, pixel_y)
            if pixel_key not in sequences:
                sequences[pixel_key] = []
            sequences[pixel_key].append(seq)

        # Sort sequences for each pixel by trigger time
        for pixel_key in sequences:
            sequences[pixel_key].sort(key=lambda s: s.trigger_time_idx)

            # Validate no duplicates
            trigger_times = [s.trigger_time_idx for s in sequences[pixel_key]]
            if len(trigger_times) != len(set(trigger_times)):
                raise ValueError(f"Duplicate trigger times found for pixel {pixel_key}")

            # Validate ordering (sequences can touch but not overlap)
            for j in range(1, len(sequences[pixel_key])):
                prev_seq = sequences[pixel_key][j-1]
                curr_seq = sequences[pixel_key][j]
                if curr_seq.t_start < prev_seq.t_end:
                    raise ValueError(
                        f"Invalid sequence ordering for pixel {pixel_key}: "
                        f"sequence {j} starts at {curr_seq.t_start} but previous ends at {prev_seq.t_end}"
                    )

        return sequences

    def _dead_time_compensation(
        self,
        seq_a: BurstSequence,
        seq_b: BurstSequence
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply dead-time compensation for close sequences.

        Args:
            seq_a: First sequence
            seq_b: Second sequence (immediately following seq_a)

        Returns:
            Tuple of (times, cumulative_charges) for merged sequences
        """
        gap = seq_b.t_start - seq_a.t_end

        if gap <= 0 or gap > self.tau:
            raise ValueError(f"Dead-time compensation requires 0 < gap <= tau, got gap={gap}")

        # Compute slope from first value of B
        value_b = seq_b.charges[0]
        slope = value_b / (gap - self.delta_t)

        # Compensated value over dead time
        compensated_value = slope * self.delta_t

        # Build cumulative array
        # Start with seq_a charges
        all_charges = list(seq_a.charges)

        # Add compensated charge and seq_b charges
        all_charges.append(compensated_value + seq_b.charges[0])
        all_charges.extend(seq_b.charges[1:])

        all_charges = np.array(all_charges)

        # Prepend zero and compute cumulative
        cumulative = np.concatenate([[0], np.cumsum(all_charges)])

        # Compute time points
        times = []

        # Times for seq_a
        for i in range(len(seq_a.charges)):
            times.append(seq_a.t_start + i * self.adc_hold_delay)

        # Time for compensated + first of seq_b (at seq_b.t_start)
        times.append(seq_b.t_start)

        # Times for rest of seq_b
        for i in range(1, len(seq_b.charges)):
            times.append(seq_b.t_start + i * self.adc_hold_delay)

        times = np.array(times)

        return times, cumulative

    def _template_compensation(
        self,
        cumulative: np.ndarray,
        times: np.ndarray,
        next_seq: BurstSequence,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply template-based compensation for non-close sequences.

        Args:
            cumulative: Current cumulative charge array (with prepended zero)
            times: Current time points (one less than cumulative due to prepended zero)
            next_seq: Next sequence that is not close

        Returns:
            Tuple of (updated_times, updated_cumulative)
        """
        # Last time and cumulative value
        last_time = times[-1] + self.adc_hold_delay if len(times) > 0 else next_seq.t_start - 4 * self.template_spacing
        last_cumulative = cumulative[-1]

        # Calculate how many template points we need
        gap = next_seq.t_start - last_time
        n_template_points = int(np.ceil(gap / self.template_spacing))

        # Take appropriate number of template points
        n_use = min(n_template_points, len(self.template))
        template_section = self.template[:n_use]

        # Scale template to continue from last cumulative
        # We want: last_cumulative + scaled_template to reach next_seq.charges[0]
        max_burst = next_seq.charges[0]
        scale_factor = max_burst / self.template[-1] if self.template[-1] > 0 else 1.0
        scaled_template = template_section * scale_factor

        # Add scaled template to cumulative
        template_cumulative = last_cumulative + scaled_template

        # Generate template times
        template_times = last_time + np.arange(1, n_use + 1) * self.template_spacing

        # Check for collisions with next_seq start time
        collision_mask = template_times >= next_seq.t_start
        if np.any(collision_mask):
            # Remove colliding elements
            first_collision_idx = np.where(collision_mask)[0][0]
            template_times = template_times[:first_collision_idx]
            template_cumulative = template_cumulative[:first_collision_idx]

        # Concatenate with existing cumulative and times
        # cumulative has prepended zero, times does not
        # After template, we need to handle the transition to next_seq

        if len(template_times) > 0:
            # Get the last template cumulative value
            last_template_cumulative = template_cumulative[-1]
            last_template_time = template_times[-1]
        else:
            last_template_cumulative = last_cumulative
            last_template_time = last_time

        # Calculate modified first charge for next_seq to maintain continuity
        time_gap_to_next = next_seq.t_start - last_template_time

        # Assume constant slope from last template point to next_seq start
        if len(template_times) > 1:
            slope = (template_cumulative[-1] - template_cumulative[-2]) / self.template_spacing
        elif len(times) > 1:
            slope = (cumulative[-1] - cumulative[-2]) / self.adc_hold_delay
        else:
            slope = 0

        # Extrapolated value at next_seq.t_start
        extrapolated_cumulative = last_template_cumulative + slope * time_gap_to_next

        # Modified first charge of next_seq
        modified_first_charge = extrapolated_cumulative + next_seq.charges[0]

        # Build updated arrays
        updated_cumulative = np.concatenate([
            cumulative,
            template_cumulative,
            [modified_first_charge],
            np.cumsum(next_seq.charges[1:]) + modified_first_charge if len(next_seq.charges) > 1 else []
        ])

        # Build times
        next_seq_times = [next_seq.t_start + i * self.adc_hold_delay for i in range(len(next_seq.charges))]
        updated_times = np.concatenate([
            times,
            template_times,
            next_seq_times
        ])

        return updated_times, updated_cumulative

    def process_pixel_sequences(
        self,
        sequences: List[BurstSequence]
    ) -> MergedSequence:
        """Process all sequences for a single pixel.

        Args:
            sequences: List of BurstSequence objects for one pixel, sorted by time

        Returns:
            MergedSequence with compensated charges
        """
        if len(sequences) == 0:
            raise ValueError("sequences list cannot be empty")

        # Start with first sequence
        first_seq = sequences[0]

        # Initialize cumulative with first sequence
        cumulative = np.concatenate([[0], np.cumsum(first_seq.charges)])
        times = np.array([first_seq.t_start + i * self.adc_hold_delay
                         for i in range(len(first_seq.charges))])

        # Process remaining sequences
        i = 1
        while i < len(sequences):
            curr_seq = sequences[i]
            # Gap is measured from start of last burst to start of current burst
            # times[-1] is the time of the last charge value
            gap = curr_seq.t_start - times[-1]

            # Check if close enough for dead-time compensation
            if 0 < gap <= self.tau:
                # Need to merge with previous - create a temporary sequence from cumulative
                # Extract charges by differentiating cumulative (skip first zero)
                prev_charges = np.diff(cumulative)

                # Create temporary sequence representing merged state
                # t_end should be the start time of the last burst
                temp_seq = BurstSequence(
                    pixel_x=curr_seq.pixel_x,
                    pixel_y=curr_seq.pixel_y,
                    trigger_time_idx=0,  # Not used
                    t_start=times[0],
                    t_end=times[-1],  # Start time of last burst
                    charges=prev_charges,
                    last_adc_latch=0,  # Not used
                    next_integration_start=0,  # Not used
                )

                # Apply dead-time compensation
                times, cumulative = self._dead_time_compensation(temp_seq, curr_seq)

            else:
                # Template compensation
                times, cumulative = self._template_compensation(cumulative, times, curr_seq)

            i += 1

        # Differentiate cumulative to get final charges
        charges = np.diff(cumulative)

        return MergedSequence(
            pixel_x=sequences[0].pixel_x,
            pixel_y=sequences[0].pixel_y,
            times=times,
            charges=charges,
            cumulative=cumulative,
        )

    def process_hits(self, hits: Hits) -> Dict[Tuple[int, int], MergedSequence]:
        """Process all burst sequences in a Hits container.

        Args:
            hits: Hits data container

        Returns:
            Dictionary mapping (pixel_x, pixel_y) to MergedSequence
        """
        # Extract sequences grouped by pixel
        sequences_by_pixel = self.extract_sequences_from_hits(hits)

        # Process each pixel
        merged_sequences = {}
        for pixel_key, sequences in sequences_by_pixel.items():
            merged_sequences[pixel_key] = self.process_pixel_sequences(sequences)

        return merged_sequences
