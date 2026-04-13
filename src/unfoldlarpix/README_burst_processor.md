# Burst Sequence Processor

## Overview

The `BurstSequenceProcessor` implements a signal reconstruction algorithm for LArPix burst data that:

1. Merges adjacent burst sequences per pixel
2. Applies dead-time compensation for close sequences
3. Applies template-based interpolation for gaps between distant sequences
4. Produces differentiated charge values suitable for physics analysis

## Quick Start

```python
from unfoldlarpix import DataLoader, BurstSequenceProcessor
import numpy as np

# Load data
loader = DataLoader("data/my_data.npz")
readout_config = loader.get_readout_config()

# Create processor
processor = BurstSequenceProcessor(
    adc_hold_delay=float(readout_config.adc_hold_delay),
    tau=5.0,  # Close sequence threshold
    delta_t=1.0,  # Dead time
    template=np.array([1, 2, 3, 4, 6, 8, 16, 36]),
    template_spacing=float(readout_config.adc_hold_delay),
)

# Process hits
for event in loader.iter_events():
    if event.hits:
        merged_sequences = processor.process_hits(event.hits)

        for pixel, seq in merged_sequences.items():
            print(f"Pixel {pixel}: {len(seq.times)} time points")
            print(f"  Charges: {seq.charges}")
            print(f"  Times: {seq.times}")
```

## Algorithm

### 1. Sequence Extraction

Burst sequences are extracted from hit data and grouped by pixel coordinates. Each sequence contains:
- Start time: `t_start = trigger_time + adc_hold_delay`
- End time: `t_end = trigger_time + adc_hold_delay * nburst` (start time of last burst)
- Charge values: integral values for each burst

### 2. Dead-Time Compensation (gap ≤ tau)

When two sequences are "close enough":

```
gap = t_B_start - t_A_end
```

If `0 < gap ≤ tau`:

1. Calculate slope: `slope = first_charge_B / (gap - delta_t)`
2. Calculate compensated value: `compensated = slope * delta_t`
3. Merge sequences: `merged_charge_B_first = first_charge_B + compensated`

This compensates for charge accumulated during the dead time.

### 3. Template Compensation (gap > tau)

When sequences are far apart:

1. Scale a template waveform to match the cumulative charge
2. Insert template points between sequences
3. Remove any template points that collide with the next sequence
4. Maintain strict monotonic time ordering

### 4. Output

The processor returns `MergedSequence` objects with:
- `times`: Time points for each charge value
- `charges`: Differentiated charge values (final output)
- `cumulative`: Cumulative charge array (for debugging)

## V3 Algorithm

`BurstSequenceProcessorV3` keeps the same two compensation ingredients as v1:
- dead-time compensation for close-enough neighbors
- template compensation for separated neighbors

The main change is that v3 runs in two passes per pixel instead of deciding the
final action in a single left-to-right sweep.

### Pass 1: Merge Close Raw Burst Sequences

All raw burst sequences for one pixel are visited in time order.

For each adjacent pair:
- if `0 < gap <= tau`, merge them immediately with dead-time compensation
- if `gap > tau`, stop the current merged group and start a new one

This produces a list of first-pass merged groups. Inside each group, close
neighbors have already been stitched together, and their internal times are kept
explicitly.

### Pass 2: Template-Connect the Merged Groups

The first-pass merged groups are then processed in time order.

- The first merged group is bootstrapped with the template, the same way the
  first sequence is initialized in v1.
- Each later merged group is attached to the running waveform with template
  compensation.

The important difference from v1 is that template compensation is no longer
applied between raw sequences that happen to appear after a close merge. It is
applied only between the merged groups produced by pass 1.

### Truncated Template Compensation

In pass 2, the available space before the next merged group can be shorter than
the full template support. In that case, v3 uses only the portion of the
template that fits in the open interval before the next merged group.

That means the template is naturally truncated by the inter-group gap:
- no template points are inserted inside a close-gap region already consumed by
  pass 1 dead-time merging
- template points are inserted only in the remaining gap between two first-pass
  merged groups

### Optional Collection vs Induction Template Choice

V3 can also be configured with two templates:
- `template_coll`
- `template_indu`

When pass 2 prepares to connect a merged group, it computes that group's maximum
accumulated charge. If

`max_accumulated_q > threshold`

then v3 uses `template_coll`. Otherwise it uses `template_indu`.

`template_coll` is an increasing cumulative template. `template_indu` is a
bipolar response waveform. For compensation, v3 first keeps only the leading
positive part of `template_indu`, then integrates that clipped positive segment
to build the monotone compensation template. The late negative tail is ignored
for template clipping.

Both templates are required for v3 compensation. If you construct v3 through
the workflow helpers, `center_response` is used as the collection response and
`response_indu` provides the induction waveform.

### Summary

The v3 processing order is:

1. Extract raw burst sequences per pixel.
2. First pass: merge all close-enough raw neighbors with dead-time
   compensation.
3. Second pass: connect the resulting merged groups with template or truncated
   template compensation.
4. Differentiate the final cumulative waveform to obtain output charges.

## Parameters

### `adc_hold_delay`
Duration of each ADC hold period. This is the time window over which each burst value integrates charge.

### `tau`
Threshold for determining if sequences are "close enough" for dead-time compensation.
- If `gap ≤ tau`: Apply dead-time compensation
- If `gap > tau`: Apply template compensation

Typical value: 5-10 ms

### `delta_t`
Dead time duration - the minimal time resolution of the system. Used in dead-time compensation calculation.

Typical value: 1 ms

### `template`
A monotonically increasing array representing the expected cumulative charge shape for interpolation.

Default: `[1, 2, 3, 4, 6, 8, 16, 36]`

### `template_spacing`
Time spacing between template points. Usually same as `adc_hold_delay`.

## Examples

See `examples/burst_processing_example.py` for:
1. Manual sequence creation and processing
2. Template compensation demonstration
3. Real data processing

Run with:
```bash
cd examples
python burst_processing_example.py
```

## Data Structures

### Input: `Hits` Container

```python
hits.data: np.ndarray  # Shape (N, 3+nburst): [x, y, z, charge1, charge2, ...]
hits.location: np.ndarray  # Shape (N, 5): [pixel_x, pixel_y, trigger_idx, latch, next]
```

### Output: `MergedSequence`

```python
class MergedSequence:
    pixel_x: int
    pixel_y: int
    times: np.ndarray       # Time points
    charges: np.ndarray     # Differentiated charges
    cumulative: np.ndarray  # Cumulative charges (for debugging)
```

## Implementation Notes

### Time Interpretation

⚠️ **Important**: `t_end` is the **start time of the last burst**, not the end of integration!

- Each burst integrates from its start time for `adc_hold_delay` duration
- Actual sequence ends at `t_end + adc_hold_delay`
- Gap calculation uses burst start times: `gap = t_B_start - t_A_end`

### Sequence Ordering

Sequences are validated to ensure:
- No duplicate trigger times for same pixel
- Monotonic time ordering: `t_curr_start >= t_prev_end`
- No overlapping burst start times

### Collision Handling

When template points would overlap with the next sequence:
1. Identify colliding template points (`template_time >= next_seq_start`)
2. Remove colliding points
3. Extrapolate from remaining template to connect to next sequence

## Testing

Run tests with:
```bash
pytest tests/test_burst_processor.py -v
```

Tests cover:
- Sequence validation
- Dead-time compensation (matches CLAUDE.md example)
- Template compensation
- Sequence extraction from hits
- Error handling for invalid inputs

## References

See `CLAUDE.md` for detailed algorithm specification and examples.
