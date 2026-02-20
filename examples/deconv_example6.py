#!/usr/bin/env python3
# import numpy as np
# import unfoldlarpix
import numpy as np
import matplotlib.pyplot as plt

from unfoldlarpix import DataLoader
from unfoldlarpix import FieldResponseProcessor

from unfoldlarpix.hit_to_wf import hits_to_bin_wf, convert_bin_wf_to_blocks
from unfoldlarpix.deconv import deconv_fft, gaussian_filter, gaussian_filter_3d

from unfoldlarpix import BurstSequence, BurstSequenceProcessor, MergedSequence
from unfoldlarpix.burst_processor import merged_sequences_to_block

from unfoldlarpix.smear_truth import gaus_smear_true

# Load NPZ file produced by tred
loader = DataLoader("data/pgun_muplus_3gev_nonoises_interval_average.npz")
readout_config = loader.get_readout_config()

fr_processor = FieldResponseProcessor("data/fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npz", normalized=False)
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
    if event.hits:
        print(f"  Hits shape: {event.hits.data.shape}")
        print(f"  Hits location shape: {event.hits.location.shape}")

    # hwf = hits_to_bin_wf(event.hits, template=fr_temp, threshold=readout_config.threshold,
    #                      bin_size=readout_config.adc_hold_delay,
    #                      nburst=readout_config.nburst, npad=10)

    # hwf_block = convert_bin_wf_to_blocks(hwf, bin_size=readout_config.adc_hold_delay,
    #                                      shift_to_center=True)
    boffset, hwf_block_data = interval_average_to_block(event.hits.location[:, :3], event.hits.data[:, 3:])
    print(hwf_block_data.shape)
    print('---', boffset)

    # hwf_block_data = hwf_block.data[0, ...] # single block
    # cloc = hwf_block.location[0, :2]
    # curr_mask = np.all(event.current.location[:,:2]==cloc[None, :], axis=1)
    # curr = np.squeeze(event.current.data[curr_mask])

    sigma = 0.01

    gaussian_kernel = gaussian_filter(n=hwf_block_data.shape[-1], dt=readout_config.adc_hold_delay,
                                      sigma=sigma)

    # gaussian_kernel = gaussian_filter_3d((
    #     hwf_block_data.shape[0]+fr_full_k.shape[0]-1,
    #     hwf_block_data.shape[1]+fr_full_k.shape[1]-1,
    #     hwf_block_data.shape[2]), dt=(1,1,1), sigma=(2, 2, sigma))

    deconv_q, local_offset = deconv_fft(hwf_block_data, fr_full_k,
                                        gaussian_kernel)

    smear_offset, smeared_true = gaus_smear_true(event.effq.location, event.effq.data, width=sigma)

    print(f'smear_offset: {smear_offset}, boffset: {boffset}, '
          f'sum_deconv_q: {np.sum(deconv_q)}, '
          f'sum_deconv_q_gt1: {np.sum(deconv_q[deconv_q>1])}, '
          f'sum_deconv_q_gt4: {np.sum(deconv_q[deconv_q>4])}, '
          f'sum_smeared_true: {np.sum(smeared_true)}, '
          f'sum_effq_last: {np.sum(event.effq.data[:, -1])},'
          f'sum_hits_last: {np.sum(event.hits.data[:, -1])}')


    np.savez(f"deconv_event_{event.tpc_id}_{event.event_id}.npz",
             hwf_block_data=hwf_block_data,
             deconv_q=deconv_q, boffset=boffset,
             smeared_true=smeared_true, smear_offset=smear_offset,
             effq_location=event.effq.location, effq_data=event.effq.data,
             hits_location=event.hits.location, hits_data=event.hits.data,
             adc_hold_delay=readout_config.adc_hold_delay,
             csa_reset_time=readout_config.csa_reset_time,
             adc_downsample_factor=readout_config.adc_hold_delay)

# Get geometry information
geometry = loader.get_geometry(0)
print(f"TPC 0 geometry: {geometry.lower} to {geometry.upper}")

# Get readout configuration
config = loader.get_readout_config()
print(f"Time spacing: {config.time_spacing} μs")
