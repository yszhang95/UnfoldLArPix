#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from unfoldlarpix import DataLoader
from unfoldlarpix import FieldResponseProcessor

from unfoldlarpix.hit_to_wf import hits_to_bin_wf, convert_bin_wf_to_blocks
from unfoldlarpix.deconv import deconv_fft, gaussian_filter, gaussian_filter_3d

from unfoldlarpix.smear_truth import gaus_smear_true, gaus_smear_true_3d

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Deconvolve positron event data (V3)")
parser.add_argument("--sigma", type=float, default=0.005,
                    help="Gaussian filter sigma in time (default: 0.005)")
parser.add_argument("--sigma-pxl", type=float, default=0.1,
                    help="Gaussian filter sigma in pixel (default: 0.1)")
parser.add_argument("--input-file", default="data/pgun_positron_3gev_tred_noises_effq_nt1_wf.npz",
                    help="Input NPZ file produced by tred")
parser.add_argument("--field-response", default="/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz",
                    help="Field response NPZ file")
parser.add_argument("--effq-offset", default=0, type=int, help='activate when use truncated waveform')
args = parser.parse_args()

loader = DataLoader(args.input_file)
readout_config = loader.get_readout_config()

fr_processor = FieldResponseProcessor(args.field_response, normalized=False)
fr_full = fr_processor.process_response()
frcenter = fr_full[fr_full.shape[0]//2, fr_full.shape[1]//2, :]
fr_temp = frcenter.copy()


def interval_average_to_block(ticks: np.ndarray, bursts: np.ndarray, tpad=10) -> tuple[np.ndarray, np.ndarray]:
    """Smear true charge with kernel to get smeared charge."""
    if len(ticks.shape) != 2:
        raise ValueError("ticks should be 3D array")
    bursts = np.diff(bursts, axis=-1, prepend=0)
    # get a minimum shape of charge block
    loc_min = [np.min(ticks[:, i]) for i in range(ticks.shape[1])]
    loc_max = [np.max(ticks[:, i]) for i in range(ticks.shape[1])]
    loc_min = np.array(loc_min)
    loc_max = np.array(loc_max)
    print("loc_min:", loc_min, "loc_max:", loc_max)
    shape = [loc_max[i] - loc_min[i] + 1 for i in range(ticks.shape[1])]
    print(shape)
    if shape[-1] % readout_config.adc_hold_delay != 1:
        raise ValueError("The time range of the block should be divisible by adc_hold_delay.")
    shape[-1] = shape[-1] // readout_config.adc_hold_delay + bursts.shape[-1]
    print(shape)
    data = np.zeros(shape, dtype=bursts.dtype)
    # fill data with true charge
    for i in range(ticks.shape[0]):
        # print((ticks[i, 2] - loc_min[2])//readout_config.adc_hold_delay, ticks[i,2], loc_min[2])
        if ticks[i, 0] == 36+102 and ticks[i, 1] == 97+1:
            print(ticks[i, 2], loc_min[2])
        for i3 in range(bursts.shape[-1]):
            data[ticks[i, 0] - loc_min[0],
                 ticks[i, 1] - loc_min[1],
                 (ticks[i, 2] - loc_min[2])//readout_config.adc_hold_delay + i3] += bursts[i, i3]
    loc_min[-1] -= tpad*readout_config.adc_hold_delay
    data = np.pad(data, ((0, 0), (0, 0), (tpad, tpad)))
    return loc_min, data

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

    boffset, blocks = interval_average_to_block(event.hits.location[:, :3], event.hits.data[:, 3:])

    sigma = args.sigma
    sigma_pxl = args.sigma_pxl
    hwf_block_data = blocks
    gaussian_kernel = gaussian_filter(n=hwf_block_data.shape[-1], dt=readout_config.adc_hold_delay,
                                      sigma=sigma)
    effq_offset = args.effq_offset

    gaussian_kernel = gaussian_filter_3d((
        hwf_block_data.shape[0]+fr_full_k.shape[0]-1,
        hwf_block_data.shape[1]+fr_full_k.shape[1]-1,
        hwf_block_data.shape[2]), dt=(1,1,readout_config.adc_hold_delay), sigma=(sigma_pxl, sigma_pxl, sigma))

    deconv_q, local_offset = deconv_fft(hwf_block_data, fr_full_k,
                                        gaussian_kernel)
    # check local_offset
    if any(list(o != 0 for o in local_offset)):
        raise ValueError()

    smear_offset, smeared_true = gaus_smear_true_3d(event.effq.location, event.effq.data, width=np.array([sigma_pxl, sigma_pxl, sigma]))
    smear_offset[-1] += readout_config.adc_hold_delay
    # smear_offset[-1] += effq_offset
    boffset[-1] -= readout_config.adc_hold_delay

    print(f'smear_offset: {smear_offset}, boffset: {boffset}, '
          f'sum_deconv_q: {np.sum(deconv_q)}, '
          f'sum_deconv_q_gt1: {np.sum(deconv_q[deconv_q>1])}, '
          f'sum_deconv_q_gt4: {np.sum(deconv_q[deconv_q>4])}, '
          f'sum_smeared_true: {np.sum(smeared_true)}, '
          f'sum_effq_last: {np.sum(event.effq.data[:, -1])},'
          f'sum_hits_last: {np.sum(event.hits.data[:, -1])}')


    geometry = loader.get_geometry(event.tpc_id)
    np.savez(f"deconv_positron_v3_event_{event.tpc_id}_{event.event_id}.npz",
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
