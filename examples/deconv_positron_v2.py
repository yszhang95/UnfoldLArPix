#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from unfoldlarpix import DataLoader
from unfoldlarpix import FieldResponseProcessor

from unfoldlarpix.hit_to_wf import hits_to_bin_wf, convert_bin_wf_to_blocks
from unfoldlarpix.deconv import deconv_fft, gaussian_filter, gaussian_filter_3d

from unfoldlarpix import BurstSequenceProcessorV2, MergedSequence
from unfoldlarpix.burst_processor import merged_sequences_to_block

from unfoldlarpix.smear_truth import gaus_smear_true, gaus_smear_true_3d

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Deconvolve positron event data (V2)")
parser.add_argument("--sigma", type=float, default=0.005,
                    help="Gaussian filter sigma in time (default: 0.005)")
parser.add_argument("--sigma-pxl", type=float, default=0.2,
                    help="Gaussian filter sigma in pixel (default: 0.2)")
parser.add_argument("--input-file", default="data/pgun_positron_3gev_tred_noises_effq_nt1.npz",
                    help="Input NPZ file produced by tred")
parser.add_argument("--field-response", default="/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz",
                    help="Field response NPZ file")
parser.add_argument("--tpc-id", type=int, default=None,
                    help="Process only this TPC ID (default: process all)")
parser.add_argument("--output-suffix", default="",
                    help="Suffix to append to output filename (e.g., 's0p0005_sp10')")
args = parser.parse_args()

# Helper to format sigma values for filenames
def fmt_sigma_detailed(v: float) -> str:
    """0.0005 -> 's0p0005', 0.005 -> 's005', 0.01 -> 's010'"""
    s = f"{v:.4f}".rstrip('0').rstrip('.')
    return 's' + s.replace('.', 'p')

def fmt_sigma_pxl_detailed(v: float) -> str:
    """0.1 -> 'sp10', 0.15 -> 'sp15', 0.2 -> 'sp20'"""
    s = f"{v:.2f}".rstrip('0').rstrip('.')
    return 'sp' + s.replace('.', 'p')

# Generate output suffix if not provided
if not args.output_suffix:
    args.output_suffix = fmt_sigma_detailed(args.sigma) + '_' + fmt_sigma_pxl_detailed(args.sigma_pxl)

loader = DataLoader(args.input_file)
readout_config = loader.get_readout_config()

fr_processor = FieldResponseProcessor(args.field_response, normalized=False)
fr_full = fr_processor.process_response()
frcenter = fr_full[fr_full.shape[0]//2, fr_full.shape[1]//2, :]
fr_temp = frcenter.copy()


def sigma_simple(y, dx=1.0):
    y = np.asarray(y, dtype=float)
    s = y.sum()
    if s == 0:
        raise ValueError("In valid zero array")

    x = np.arange(y.size) * dx
    mu = (x * y).sum() / s
    sigma = np.sqrt(((x - mu)**2 * y).sum() / s)
    return sigma

def integrate_k(kernel: np.ndarray, kticks: int) -> np.ndarray:
    """Integrate kernel over time axis."""
    n = kernel.shape[-1] // kticks * kticks
    if np.any(np.abs(kernel[..., n:]) > 1E-6):
        raise ValueError("No enough zeros is padded to the kernel during run of TRED.")

    return kernel[..., :n].reshape(*kernel.shape[:-1],  n//kticks, kticks).sum(axis=-1)


fr_full_k = integrate_k(fr_full, readout_config.adc_hold_delay)

# Iterate over events grouped by (event_id, tpc_id)
for event in loader.iter_events():
    # Filter by TPC ID if specified
    if args.tpc_id is not None and event.tpc_id != args.tpc_id:
        continue

    print(f"TPC {event.tpc_id}, Event {event.event_id}")

    # Access effective charge data
    if event.effq:
        print(f"  EffQ shape: {event.effq.data.shape}")
        print(f"  EffQ location shape: {event.effq.location.shape}")

    # Access current/waveform data
    if event.current:
        print(f"  Current shape: {event.current.data.shape}")
        print(f"  Current location shape: {event.current.location.shape}")

    # Access hit data
    if not event.hits:
        print(f"  No hits, skipping event {event.event_id} TPC {event.tpc_id}")
        continue
    print(f"  Hits shape: {event.hits.data.shape}")
    print(f"  Hits location shape: {event.hits.location.shape}")

    burst_processor = BurstSequenceProcessorV2(
        readout_config.adc_hold_delay,
        tau = readout_config.adc_hold_delay,
        deadtime = readout_config.csa_reset_time,
        template = np.cumsum(fr_temp),
        threshold = readout_config.threshold
        )
    merged_seqs = burst_processor.process_hits(event.hits)
    print('compensated', sum([np.sum(m.charges) for m in merged_seqs.values()]))
    boffset, bdata = merged_sequences_to_block(merged_seqs, readout_config.adc_hold_delay, npadbin=50)
    blocks = bdata

    sigma = args.sigma
    sigma_pxl = args.sigma_pxl
    hwf_block_data = blocks
    gaussian_kernel = gaussian_filter(n=hwf_block_data.shape[-1], dt=readout_config.adc_hold_delay,
                                      sigma=sigma)

    gaussian_kernel = gaussian_filter_3d((
        hwf_block_data.shape[0]+fr_full_k.shape[0]-1,
        hwf_block_data.shape[1]+fr_full_k.shape[1]-1,
        hwf_block_data.shape[2]), dt=(1,1,readout_config.adc_hold_delay), sigma=(sigma_pxl, sigma_pxl, sigma))

    deconv_q, local_offset = deconv_fft(hwf_block_data, fr_full_k,
                                        gaussian_kernel)
    if any(list(o != 0 for o in local_offset)):
        raise ValueError()

    smear_offset, smeared_true = gaus_smear_true_3d(event.effq.location, event.effq.data, width=np.array([sigma_pxl, sigma_pxl, sigma]))
    smear_offset[-1] += readout_config.adc_hold_delay

    print(f'smear_offset: {smear_offset}, boffset: {boffset}, '
          f'sum_deconv_q: {np.sum(deconv_q)}, '
          f'sum_deconv_q_gt1: {np.sum(deconv_q[deconv_q>1])}, '
          f'sum_deconv_q_gt4: {np.sum(deconv_q[deconv_q>4])}, '
          f'sum_smeared_true: {np.sum(smeared_true)}, '
          f'sum_effq_last: {np.sum(event.effq.data[:, -1])},'
          f'sum_hits_last: {np.sum(event.hits.data[:, -1])}')


    geometry = loader.get_geometry(event.tpc_id)
    output_filename = f"deconv_positron_v2_{args.output_suffix}_event_{event.tpc_id}_{event.event_id}.npz"
    print(f"Saving to: {output_filename}")
    np.savez(output_filename,
             deconv_q=deconv_q, boffset=boffset,
             hwf_block=hwf_block_data,
             hwf_block_offset=boffset,
             smeared_true=smeared_true, smear_offset=smear_offset,
             effq_location=event.effq.location, effq_data=event.effq.data,
             hits_location=event.hits.location, hits_data=event.hits.data,
             adc_hold_delay=readout_config.adc_hold_delay,
             csa_reset_time=readout_config.csa_reset_time,
             adc_downsample_factor=readout_config.adc_hold_delay,
             anode_position=geometry.anode_position,
             drift_direction=geometry.drift_direction,
             global_tref=event.global_tref,
             tpc_lower=geometry.lower,
             drtoa=float(np.squeeze(fr_processor.get_metadata()['drift_length'])))

# Get geometry information
geometry = loader.get_geometry(0)
print(f"TPC 0 geometry: {geometry.lower} to {geometry.upper}")

# Get readout configuration
config = loader.get_readout_config()
print(f"Time spacing: {config.time_spacing} μs")
