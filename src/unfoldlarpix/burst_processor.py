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
    t_first: float  # trigger_time_idx + adc_hold_delay
    t_last: float    # trigger_time_idx + adc_hold_delay * nburst
    charges: np.ndarray  # Array of charge values for each burst
    last_adc_latch: int
    next_integration_start: int

    def __post_init__(self):
        """Validate sequence data."""
        if self.t_last <= self.t_first:
            raise ValueError(f"t_end ({self.t_last}) must be > t_start ({self.t_first})")
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
            raise ValueError(f"times and charges must have the same length. Got {len(self.times)} and {len(self.charges)}")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("times must be strictly monotonically increasing")


class BurstSequenceProcessor:
    """Process burst sequences with dead-time and template compensation."""

    def __init__(
        self,
        adc_hold_delay: float,
        tau: float,
        deadtime: float,
        template: np.ndarray = None,
        threshold: float = None,
    ):
        """Initialize the burst sequence processor.

        Args:
            adc_hold_delay: Time duration for each ADC hold (in same units as tau, delta_t)
            tau: Threshold for determining if sequences are close enough
            delta_t: Predefined dead time (minimal time resolution unit)
            template: Optional template waveform for non-close sequence compensation.
                     Should be monotonically increasing cumulative values.
            threshold: Charge threshold that defines the template truncation point
        """
        self.adc_hold_delay = adc_hold_delay
        self.tau = tau
        self.deadtime = deadtime
        self.template = np.asarray(template if template is not None else self._default_template(), dtype=float)
        if self.template.size == 0:
            raise ValueError("Template cannot be empty.")
        self.threshold = threshold
        if threshold is None:
            raise ValueError("Threshold value must be provided for template compensation.")

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
                t_first=t_start,
                t_last=t_end,
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
                if curr_seq.t_first < prev_seq.t_last:
                    raise ValueError(
                        f"Invalid sequence ordering for pixel {pixel_key}: "
                        f"sequence {j} starts at {curr_seq.t_first} but previous ends at {prev_seq.t_last}"
                    )

        return sequences

    def _dead_time_compensation(
        self,
        seq_a: BurstSequence,
        seq_b: BurstSequence,
        delta_t : float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply dead-time compensation for close sequences.

        Args:
            seq_a: First sequence
            seq_b: Second sequence (immediately following seq_a)

        Returns:
            Tuple of (times, cumulative_charges) for merged sequences
        """
        gap = seq_b.t_first - seq_a.t_last

        if gap <= 0 or gap > self.tau:
            raise ValueError(f"Dead-time compensation requires 0 < gap <= tau, got gap={gap}")

        # Compute slope from first value of B
        value_b = seq_b.charges[0]
        slope = value_b / (gap - delta_t)

        # Compensated value over dead time
        compensated_value = slope * delta_t

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
            times.append(seq_a.t_first + i * self.adc_hold_delay)

        # Time for compensated + first of seq_b (at seq_b.t_start)
        times.append(seq_b.t_first)

        # Times for rest of seq_b
        for i in range(1, len(seq_b.charges)):
            times.append(seq_b.t_first + i * self.adc_hold_delay)

        times = np.array(times)

        return times, cumulative

    def _template_compensation(
        self,
        cumulative: np.ndarray,
        times: np.ndarray,
        deadtime: float,
        next_seq: BurstSequence,
        threshold: float,
        template_cumulative: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply template-based compensation for non-close sequences.

        Args:
            cumulative: Current cumulative charge array (with prepended zero)
            times: Current time points (one less than cumulative due to prepended zero)
            next_seq: Next sequence that is not close
            threshold: Charge threshold that defines the end of the template region
            template_cumulative: Monotonically increasing cumulative template

        Returns:
            Tuple of (updated_times, updated_cumulative)
        """
        if threshold is None:
            raise ValueError("Template compensation requires a threshold value.")
        if len(times) == 0:
            raise ValueError("Template compensation requires existing time points.")

        last_time = times[-1]
        last_cumulative = cumulative[-1]

        template_cumulative = np.asarray(template_cumulative, dtype=float)
        if template_cumulative.size == 0:
            raise ValueError("Template compensation requires a non-empty cumulative template.")
        if not np.all(np.diff(template_cumulative) >= 0):
            raise ValueError("Template must be monotonically increasing.")

        # Keep only the portion of the template strictly before threshold and append the threshold point.
        transit = threshold/np.max(np.cumsum(next_seq.charges))
        transit = min(transit, 1.0)
        tlength = next_seq.t_first - last_time - deadtime
        if tlength <= 1:
            raise ValueError(f"Not enough time for template compensation, available time {tlength} is too short.")
        tlength = int(np.round(tlength))
        threshold_idx = None
        for jidx in range(tlength, len(template_cumulative)):
            if template_cumulative[jidx+tlength] - template_cumulative[jidx] >= transit:
                threshold_idx = jidx + tlength
                break
        if threshold_idx is None:
            raise ValueError("Template compensation requires the template to reach the threshold within the available time.")
        template_section = template_cumulative[threshold_idx-tlength:threshold_idx+1] # one more tick for downsampling
        template_section = template_section[::-1][::self.adc_hold_delay][::-1]  # Reverse, downsample, reverse back

        template_section = np.diff(template_section) # integral per interval

        threshold_time = next_seq.trigger_time_idx
        n_template = len(template_section)
        offsets = np.arange(n_template)
        candidate_times = threshold_time - (n_template - 1 - offsets) * self.adc_hold_delay

        valid_mask = candidate_times > last_time

        if not np.any(valid_mask):
            raise ValueError("No valid template points found before threshold time, cannot apply template compensation.")

        template_times = candidate_times[valid_mask]
        print('template_times', template_times, candidate_times, valid_mask)
        template_section = template_section[valid_mask]
        template_section *= threshold  # FIXME: Assume Cumulative Tempalte saturates at 1.

        # charge per interval
        chgs = template_section[1:].tolist() + [next_seq.charges[0] - threshold] + next_seq.charges[1:].tolist()

        trigger_time_idx = template_times[0] - self.adc_hold_delay
        if trigger_time_idx >= last_time:
            raise ValueError("TBD")
        else:
            delta_t = last_time - trigger_time_idx
            chgs[0] = template_section[0] * delta_t / self.adc_hold_delay + chgs[0]
        # Require that we can supply the threshold right before the waveform.
        if not np.isclose(template_times[-1], threshold_time):
            raise ValueError("Template compensation requires the last template time to be at the trigger time.")

        next_seq_cumulative = np.cumsum(chgs) + last_cumulative

        updated_times = np.concatenate([times, template_times[1:]])
        next_seq_times = np.array([next_seq.t_first + i * self.adc_hold_delay for i in range(len(next_seq.charges))])

        updated_times = np.concatenate([updated_times, next_seq_times])
        updated_cumulative = np.concatenate([cumulative, next_seq_cumulative])

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
        times = np.array([first_seq.t_first + i * self.adc_hold_delay
                         for i in range(len(first_seq.charges))])

        # Process remaining sequences
        i = 1
        while i < len(sequences):
            curr_seq = sequences[i]
            # Gap is measured from start of last burst to start of current burst
            # times[-1] is the time of the last charge value
            gap = curr_seq.t_first - times[-1]

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
                    t_first=times[0],
                    t_last=times[-1],  # Start time of last burst
                    charges=prev_charges,
                    last_adc_latch=0,  # Not used
                    next_integration_start=0,  # Not used
                )

                # Apply dead-time compensation
                times, cumulative = self._dead_time_compensation(temp_seq, curr_seq, self.deadtime)

                # print(
                #     'tolerance ok'
                # )

            else:
                # Template compensation
                # print(
                #     'tolerance not ok ----------------',
                #     times, cumulative
                # )

                times, cumulative = self._template_compensation(
                    cumulative,
                    times,
                    self.deadtime,
                    curr_seq,
                    self.threshold,
                    self.template,
                )
                # print(
                #     'tolerance not ok ----------------',
                #     times, cumulative
                # )


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


def merged_sequences_to_block(
    merged_seqs: Dict[tuple[int, int], MergedSequence],
    bin_size: int,
    npadbin: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a MergedSequence to block format with specified bin size.

    Args:
        merged_seqs: MergedSequence to convert
        bin_size: Number of time points to group into each block
        shift_to_center: Whether to shift time points to the center of the block

    Returns:
        Tuple of (block_times, block_charges) where:
            block_times: Array of time points for each block
            block_charges: 2D array of charges for each block (shape [n_blocks, block_size])
    """
    pad_length = npadbin * bin_size
    # calculate pixel reaches
    pixel_keys = list(merged_seqs.keys())
    pmin, pmax = np.min(pixel_keys, axis=0), np.max(pixel_keys, axis=0)
    shape = np.zeros((3,), dtype=int)
    shape[:2] = pmax - pmin + 1
    tmin, tmax = [np.min(merged_seqs[pixel_key].times) for pixel_key in pixel_keys], [np.max(merged_seqs[pixel_key].times) for pixel_key in pixel_keys]
    tmin, tmax = np.min(tmin) - pad_length, np.max(tmax) + pad_length
    shape[2] = (int(np.ceil((tmax - tmin) / bin_size)) + 1)
    offset = np.array([pmin[0], pmin[1], tmin])

    # loop over pixels and put charges into blocks
    block_charges = np.zeros(shape, dtype=float)
    for pixel_key, merged_seq in merged_seqs.items():
        times = merged_seq.times
        charges = merged_seq.charges
        tinds = (times - offset[2]) // bin_size
        if len(np.unique(tinds)) != len(tinds):
            raise ValueError(f"Duplicate time indices found for pixel {pixel_key} after binning, cannot convert to blocks."
                             f"times: {times}, tinds: {tinds}, offset: {offset[2]}, bin_size: {bin_size}")
        pix_inds = np.array(pixel_key) - offset[:2]
        block_charges[pix_inds[0], pix_inds[1], tinds.astype(int)] = charges
    block_offset = offset

    return block_offset, block_charges
