#!/usr/bin/env python3
"""Example demonstrating burst sequence merging and interpolation."""

import numpy as np
import matplotlib.pyplot as plt
from unfoldlarpix import DataLoader
from unfoldlarpix.burst_processor import BurstSequenceProcessor, BurstSequence


def example_manual_sequences():
    """Demonstrate burst processing with manually created sequences."""
    print("=" * 60)
    print("Example 1: Manual Sequence Processing")
    print("=" * 60)

    # Parameters matching the CLAUDE.md example
    adc_hold_delay = 10.0  # ms
    tau = 5.0              # ms
    delta_t = 1.0          # ms
    template = np.array([1, 2, 3, 4, 6, 8, 16, 36], dtype=float)

    # Create processor
    processor = BurstSequenceProcessor(
        adc_hold_delay=adc_hold_delay,
        tau=tau,
        delta_t=delta_t,
        template=template,
        template_spacing=adc_hold_delay,
    )

    # Create sequences matching CLAUDE.md example
    # Sequence A: trigger at -10ms, charges [90, 100]
    seq_a = BurstSequence(
        pixel_x=0,
        pixel_y=0,
        trigger_time_idx=-10,
        t_start=0.0,     # -10 + 10
        t_end=10.0,      # -10 + 10*2
        charges=np.array([90.0, 100.0]),
        last_adc_latch=0,
        next_integration_start=0,
    )

    # Sequence B: trigger at 3ms, charges [130, 10]
    # Note: t_start = 3 + 10 = 13ms
    seq_b = BurstSequence(
        pixel_x=0,
        pixel_y=0,
        trigger_time_idx=3,
        t_start=13.0,    # 3 + 10
        t_end=33.0,      # 3 + 10*3
        charges=np.array([130.0, 10.0]),
        last_adc_latch=0,
        next_integration_start=0,
    )

    print(f"\nSequence A: t_start={seq_a.t_start}, t_end={seq_a.t_end}, charges={seq_a.charges}")
    print(f"Sequence B: t_start={seq_b.t_start}, t_end={seq_b.t_end}, charges={seq_b.charges}")

    gap = seq_b.t_start - seq_a.t_end
    print(f"\nGap between sequences: {gap} ms")
    print(f"Tau threshold: {tau} ms")
    print(f"Gap <= tau: {gap <= tau} -> Will apply dead-time compensation")

    # Process sequences
    merged = processor.process_pixel_sequences([seq_a, seq_b])

    print(f"\nMerged sequence:")
    print(f"  Times: {merged.times}")
    print(f"  Cumulative: {merged.cumulative}")
    print(f"  Charges: {merged.charges}")

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot cumulative
    ax = axes[0]
    ax.plot(merged.times, merged.cumulative[1:], 'o-', label='Cumulative (after prepended 0)')
    ax.axvline(seq_a.t_end, color='gray', linestyle='--', alpha=0.5, label='Seq A end')
    ax.axvline(seq_b.t_start, color='gray', linestyle=':', alpha=0.5, label='Seq B start')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Cumulative Charge')
    ax.set_title('Cumulative Charge After Dead-Time Compensation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot charges
    ax = axes[1]
    ax.plot(merged.times, merged.charges, 's-', label='Compensated Charges')
    ax.axvline(seq_a.t_end, color='gray', linestyle='--', alpha=0.5, label='Seq A end')
    ax.axvline(seq_b.t_start, color='gray', linestyle=':', alpha=0.5, label='Seq B start')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Charge')
    ax.set_title('Differentiated Charges (Final Output)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('burst_processing_example1.png', dpi=150)
    print("\nSaved plot to: burst_processing_example1.png")


def example_template_compensation():
    """Demonstrate template compensation for non-close sequences."""
    print("\n" + "=" * 60)
    print("Example 2: Template Compensation for Non-Close Sequences")
    print("=" * 60)

    # Parameters
    adc_hold_delay = 10.0  # ms
    tau = 5.0              # ms
    delta_t = 1.0          # ms
    template = np.array([1, 2, 3, 4, 6, 8, 16, 36], dtype=float)

    # Create processor
    processor = BurstSequenceProcessor(
        adc_hold_delay=adc_hold_delay,
        tau=tau,
        delta_t=delta_t,
        template=template,
        template_spacing=adc_hold_delay,
    )

    # Create first close pair (will be merged with dead-time compensation)
    seq_a = BurstSequence(
        pixel_x=0, pixel_y=0, trigger_time_idx=-10,
        t_start=0.0, t_end=10.0,
        charges=np.array([90.0, 100.0]),
        last_adc_latch=0, next_integration_start=0,
    )

    seq_b = BurstSequence(
        pixel_x=0, pixel_y=0, trigger_time_idx=3,
        t_start=13.0, t_end=33.0,
        charges=np.array([130.0, 10.0]),
        last_adc_latch=0, next_integration_start=0,
    )

    # Create a third sequence that is NOT close (large gap)
    # Following the CLAUDE.md example, this starts much later
    seq_c = BurstSequence(
        pixel_x=0, pixel_y=0, trigger_time_idx=57,
        t_start=67.0,   # 57 + 10
        t_end=107.0,    # 57 + 10*5
        charges=np.array([40.0, 100.0, 195.0, 10.0]),
        last_adc_latch=0, next_integration_start=0,
    )

    gap_c = seq_c.t_start - (seq_b.t_end + adc_hold_delay)
    print(f"\nSequence A: t_start={seq_a.t_start}, t_end={seq_a.t_end}")
    print(f"Sequence B: t_start={seq_b.t_start}, t_end={seq_b.t_end}")
    print(f"Sequence C: t_start={seq_c.t_start}, t_end={seq_c.t_end}")
    print(f"\nGap A-B: {seq_b.t_start - seq_a.t_end} ms <= tau -> Dead-time compensation")
    print(f"Gap B-C: {gap_c} ms > tau -> Template compensation")

    # Process all three sequences
    merged = processor.process_pixel_sequences([seq_a, seq_b, seq_c])

    print(f"\nFinal merged sequence:")
    print(f"  Number of time points: {len(merged.times)}")
    print(f"  Times: {merged.times}")
    print(f"  Charges: {merged.charges}")
    print(f"  Total charge: {np.sum(merged.charges):.2f}")

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot cumulative
    ax = axes[0]
    ax.plot(merged.times, merged.cumulative[1:], 'o-', linewidth=2, markersize=6)
    ax.axvline(seq_a.t_end, color='red', linestyle='--', alpha=0.5, label='Seq A end')
    ax.axvline(seq_b.t_start, color='blue', linestyle='--', alpha=0.5, label='Seq B start')
    ax.axvline(seq_b.t_end + adc_hold_delay, color='green', linestyle='--', alpha=0.5, label='Seq B end')
    ax.axvline(seq_c.t_start, color='orange', linestyle='--', alpha=0.5, label='Seq C start')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Cumulative Charge')
    ax.set_title('Cumulative Charge with Dead-Time and Template Compensation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot charges
    ax = axes[1]
    ax.stem(merged.times, merged.charges, basefmt=' ', linefmt='C0-', markerfmt='C0o')
    ax.axvline(seq_a.t_end, color='red', linestyle='--', alpha=0.5, label='Seq A end')
    ax.axvline(seq_b.t_start, color='blue', linestyle='--', alpha=0.5, label='Seq B start')
    ax.axvline(seq_b.t_end + adc_hold_delay, color='green', linestyle='--', alpha=0.5, label='Seq B end')
    ax.axvline(seq_c.t_start, color='orange', linestyle='--', alpha=0.5, label='Seq C start')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Charge')
    ax.set_title('Final Charges After Full Processing Pipeline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('burst_processing_example2.png', dpi=150)
    print("\nSaved plot to: burst_processing_example2.png")


def example_with_real_data():
    """Demonstrate burst processing with real data from NPZ file."""
    print("\n" + "=" * 60)
    print("Example 3: Processing Real Hit Data from NPZ File")
    print("=" * 60)

    try:
        # Load data
        loader = DataLoader("data/pgun_muplus_3gev_tred_nburst4_noises.npz")
        readout_config = loader.get_readout_config()

        print(f"\nReadout configuration:")
        print(f"  ADC hold delay: {readout_config.adc_hold_delay} time units")
        print(f"  Time spacing: {readout_config.time_spacing} μs")
        print(f"  Number of bursts: {readout_config.nburst}")
        print(f"  Threshold: {readout_config.threshold}")

        # Create processor with realistic parameters
        processor = BurstSequenceProcessor(
            adc_hold_delay=float(readout_config.adc_hold_delay),
            tau=5.0,  # Adjust based on your needs
            delta_t=1.0,
            template=np.array([1, 2, 3, 4, 6, 8, 16, 36], dtype=float),
            template_spacing=float(readout_config.adc_hold_delay),
        )

        # Process first event
        for event in loader.iter_events():
            print(f"\nProcessing TPC {event.tpc_id}, Event {event.event_id}")

            if event.hits:
                print(f"  Total hits: {len(event.hits)}")

                # Process burst sequences
                merged_sequences = processor.process_hits(event.hits)

                print(f"  Unique pixels processed: {len(merged_sequences)}")

                # Show details for a few pixels
                for i, (pixel_key, merged) in enumerate(merged_sequences.items()):
                    if i >= 3:  # Show only first 3 pixels
                        break

                    print(f"\n  Pixel {pixel_key}:")
                    print(f"    Time points: {len(merged.times)}")
                    print(f"    Total charge: {np.sum(merged.charges):.2f}")
                    print(f"    Time range: [{merged.times[0]:.1f}, {merged.times[-1]:.1f}]")

                # Plot a sample pixel
                if len(merged_sequences) > 0:
                    sample_pixel = list(merged_sequences.keys())[0]
                    sample_merged = merged_sequences[sample_pixel]

                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

                    # Cumulative
                    ax = axes[0]
                    ax.plot(sample_merged.times, sample_merged.cumulative[1:], 'o-')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Cumulative Charge')
                    ax.set_title(f'Cumulative Charge for Pixel {sample_pixel}')
                    ax.grid(True, alpha=0.3)

                    # Charges
                    ax = axes[1]
                    ax.stem(sample_merged.times, sample_merged.charges, basefmt=' ')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Charge')
                    ax.set_title(f'Differentiated Charges for Pixel {sample_pixel}')
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig('burst_processing_example3.png', dpi=150)
                    print("\n  Saved plot to: burst_processing_example3.png")

            # Process only first event
            break

    except FileNotFoundError:
        print("\nSkipping Example 3: data file not found")
        print("Place 'pgun_muplus_3gev_tred_nburst4_noises.npz' in the data/ directory")


if __name__ == "__main__":
    # Run examples
    example_manual_sequences()
    example_template_compensation()
    example_with_real_data()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
