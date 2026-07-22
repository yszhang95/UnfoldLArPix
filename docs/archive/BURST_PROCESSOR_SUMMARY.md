# Burst Sequence Merging and Interpolation Framework - Implementation Summary

## Completed Tasks ✓

### 1. Core Implementation

#### **BurstSequenceProcessor** (`src/unfoldlarpix/burst_processor.py`)
A complete implementation of the burst sequence merging algorithm with:

- **Data Structures**:
  - `BurstSequence`: Represents a single burst sequence with validation
  - `MergedSequence`: Represents processed output with times, charges, and cumulative values
  - `BurstSequenceProcessor`: Main processor class

- **Algorithm Components**:
  - Sequence extraction and grouping by pixel
  - Sequence ordering and validation (no duplicates, monotonic ordering)
  - Dead-time compensation for close sequences (gap ≤ tau)
  - Template-based compensation for distant sequences (gap > tau)
  - Collision handling for template points
  - Cumulative sum construction and differentiation

### 2. Integration

#### **Updated Exports** (`src/unfoldlarpix/__init__.py`)
All burst processor classes are now exported from the main package:
```python
from unfoldlarpix import BurstSequenceProcessor, BurstSequence, MergedSequence
```

### 3. Examples

#### **Comprehensive Example** (`examples/burst_processing_example.py`)
Three examples demonstrating:

1. **Manual Sequence Processing**:
   - Recreates the CLAUDE.md example
   - Shows dead-time compensation in action
   - Gap = 3ms ≤ tau, compensated value = 65
   - Output: charges [90, 100, 195, 10] ✓

2. **Template Compensation**:
   - Three sequences: two close (A-B) + one distant (C)
   - Dead-time compensation for A-B
   - Template interpolation for B-C gap
   - 11 total time points after merging

3. **Real Data Processing**:
   - Loads NPZ file from tred package
   - Processes 264 hits across 255 pixels
   - Shows per-pixel processing results
   - Generates visualization plots

**Output Plots**:
- `burst_processing_example1.png`: Dead-time compensation visualization
- `burst_processing_example2.png`: Combined dead-time + template compensation
- `burst_processing_example3.png`: Real data pixel example

### 4. Testing

#### **Comprehensive Test Suite** (`tests/test_burst_processor.py`)
13 tests covering:

- Data structure validation
- Sequence extraction from Hits containers
- Dead-time compensation (matches CLAUDE.md exactly)
- Template compensation
- Single sequence handling
- Error handling (duplicates, invalid ordering, empty sequences)
- Monotonic time validation

**All tests passing** ✓

### 5. Documentation

#### **README** (`src/unfoldlarpix/README_burst_processor.md`)
Complete documentation including:
- Quick start guide
- Algorithm explanation
- Parameter descriptions
- Usage examples
- Implementation notes and pitfalls
- Data structure reference

#### **Memory** (`~/.claude/projects/.../memory/MEMORY.md`)
Key insights for future reference:
- Time definition clarifications (t_end is start of last burst!)
- Gap calculation details
- Common pitfalls
- Data structure formats

## Key Features

### Correctness
- ✓ Matches CLAUDE.md specification exactly
- ✓ Example from specification passes unit test
- ✓ Handles edge cases (single sequence, adjacent sequences, large gaps)

### Robustness
- ✓ Input validation (no duplicates, monotonic ordering)
- ✓ Error messages with context
- ✓ Handles collision removal for template points

### Usability
- ✓ Clean API: `processor.process_hits(hits)` → dict of merged sequences
- ✓ Works with existing DataLoader and Hits containers
- ✓ Integrated into package exports
- ✓ Type hints throughout

### Testing
- ✓ 13 unit tests covering all major functionality
- ✓ Examples demonstrate real-world usage
- ✓ Visualization for validation

## Usage Example

```python
from unfoldlarpix import DataLoader, BurstSequenceProcessor
import numpy as np

# Load data
loader = DataLoader("data/my_data.npz")
config = loader.get_readout_config()

# Create processor
processor = BurstSequenceProcessor(
    adc_hold_delay=float(config.adc_hold_delay),
    tau=5.0,  # Close sequence threshold
    delta_t=1.0,  # Dead time
    template=np.array([1, 2, 3, 4, 6, 8, 16, 36]),
)

# Process all events
for event in loader.iter_events():
    if event.hits:
        # Process all pixels in one call
        merged_sequences = processor.process_hits(event.hits)

        # Access results per pixel
        for (pixel_x, pixel_y), seq in merged_sequences.items():
            print(f"Pixel ({pixel_x}, {pixel_y}):")
            print(f"  Times: {seq.times}")
            print(f"  Charges: {seq.charges}")
            print(f"  Total: {np.sum(seq.charges):.2f}")
```

## Verification

### Algorithm Verification
The CLAUDE.md example is implemented as a unit test:

**Input**:
- Sequence A: t_start=0, charges=[90, 100]
- Sequence B: t_start=13, charges=[130, 10]
- Gap = 3ms, tau = 5ms → dead-time compensation

**Expected Output**:
- Cumulative: [0, 90, 190, 385, 395]
- Charges: [90, 100, 195, 10]
- Times: [0, 10, 13, 23]

**Test Result**: ✓ PASS (exact match)

### Real Data Verification
Example 3 successfully processes:
- 264 hits across 255 unique pixels
- Multiple sequences per pixel
- Various gap sizes (close and distant)
- No errors or warnings

## Files Created/Modified

### New Files
1. `src/unfoldlarpix/burst_processor.py` - Core implementation (384 lines)
2. `examples/burst_processing_example.py` - Usage examples (338 lines)
3. `tests/test_burst_processor.py` - Test suite (267 lines)
4. `src/unfoldlarpix/README_burst_processor.md` - Documentation
5. `~/.claude/projects/.../memory/MEMORY.md` - Memory notes

### Modified Files
1. `src/unfoldlarpix/__init__.py` - Added burst processor exports

### Generated Outputs
1. `burst_processing_example1.png` - Dead-time compensation plot (132 KB)
2. `burst_processing_example2.png` - Template compensation plot (124 KB)
3. `burst_processing_example3.png` - Real data plot (76 KB)

## Next Steps (Optional Enhancements)

1. **Performance Optimization**:
   - Vectorize sequence processing for multiple pixels
   - Caching for repeated template calculations

2. **Advanced Features**:
   - Adaptive template selection based on gap size
   - Uncertainty propagation through compensation
   - Quality metrics for merged sequences

3. **Visualization**:
   - Interactive plotting tools for sequence analysis
   - Diagnostic plots for compensation quality

4. **Integration**:
   - Add burst processing to main deconvolution pipeline
   - Create processor chain: hits → burst merging → deconvolution

## Summary

The burst sequence merging and interpolation framework is **complete and fully tested**. The implementation:

- ✓ Follows CLAUDE.md specification exactly
- ✓ Handles all specified cases (dead-time, template, edge cases)
- ✓ Integrates seamlessly with existing data structures
- ✓ Includes comprehensive tests and examples
- ✓ Is documented and ready for production use

The processor is ready to be integrated into the physics analysis pipeline.
