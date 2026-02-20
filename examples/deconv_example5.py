#!/usr/bin/env python3
# import numpy as np
# import unfoldlarpix
import numpy as np
import matplotlib.pyplot as plt

from unfoldlarpix import DataLoader
from unfoldlarpix import FieldResponseProcessor

from unfoldlarpix.hit_to_wf import hits_to_bin_wf, convert_bin_wf_to_blocks
from unfoldlarpix.deconv import deconv_fft, gaussian_filter

from unfoldlarpix import BurstSequence, BurstSequenceProcessor, MergedSequence
from unfoldlarpix.burst_processor import merged_sequences_to_block

from unfoldlarpix.smear_truth import gaus_smear_true

# Load NPZ file produced by tred
loader = DataLoader("data/pgun_muplus_3gev_tred_nburst4_noises.npz")
readout_config = loader.get_readout_config()

fr_processor = FieldResponseProcessor("data/fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npz", normalized=False)
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

    burst_processor = BurstSequenceProcessor(
        readout_config.adc_hold_delay,
        tau = readout_config.adc_hold_delay,
        deadtime = readout_config.csa_reset_time,
        template = np.cumsum(fr_temp),
        threshold = readout_config.threshold
        )
    merged_seqs = burst_processor.process_hits(event.hits)
    print('compensated', sum([np.sum(m.charges) for m in merged_seqs.values()]))
    boffset, bdata = merged_sequences_to_block(merged_seqs, readout_config.adc_hold_delay, npadbin=5)
    blocks = bdata


    # hwf_block_data = hwf_block.data[0, ...] # single block
    # cloc = hwf_block.location[0, :2]
    # curr_mask = np.all(event.current.location[:,:2]==cloc[None, :], axis=1)
    # curr = np.squeeze(event.current.data[curr_mask])

    sigma = 0.01
    hwf_block_data = blocks
    gaussian_kernel = gaussian_filter(n=hwf_block_data.shape[-1], dt=readout_config.adc_hold_delay,
                                      sigma=sigma)
    deconv_q, local_offset = deconv_fft(hwf_block_data, fr_full_k, gaussian_kernel)

    smear_offset, smeared_true = gaus_smear_true(event.effq.location, event.effq.data, width=sigma)

    print(f'smear_offset: {smear_offset}, boffset: {boffset}, '
          f'sum_deconv_q: {np.sum(deconv_q)}, '
          f'sum_deconv_q_gt1: {np.sum(deconv_q[deconv_q>1])}, '
          f'sum_deconv_q_gt4: {np.sum(deconv_q[deconv_q>4])}, '
          f'sum_smeared_true: {np.sum(smeared_true)}, '
          f'sum_effq_last: {np.sum(event.effq.data[:, -1])},'
          f'sum_hits_last: {np.sum(event.hits.data[:, -1])}')


    np.savez(f"deconv_event_{event.tpc_id}_{event.event_id}.npz",
             deconv_q=deconv_q, local_offset=local_offset,
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
