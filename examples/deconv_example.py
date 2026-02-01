#!/usr/bin/env python3
# import numpy as np
# import unfoldlarpix
import numpy as np
import matplotlib.pyplot as plt

from unfoldlarpix import DataLoader
from unfoldlarpix import FieldResponseProcessor

from unfoldlarpix.hit_to_wf import hits_to_bin_wf, convert_bin_wf_to_blocks
from unfoldlarpix.deconv import deconv_fft

# Load NPZ file produced by tred
loader = DataLoader("data/two_point_charges.npz")
readout_config = loader.get_readout_config()

fr_processor = FieldResponseProcessor("data/fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npz", normalized=False)
fr_full = fr_processor.process_response()
frcenter = fr_full[fr_full.shape[0]//2, fr_full.shape[1]//2, :]
fr_temp = frcenter.copy()

def integrate_k(kernel: np.ndarray, kticks: int) -> np.ndarray:
    """Integrate kernel over time axis."""
    n = kernel.shape[-1] // kticks * kticks
    print(n, '============================================================')
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

    hwf = hits_to_bin_wf(event.hits, template=fr_temp, threshold=readout_config.threshold,
                         bin_size=readout_config.adc_hold_delay,
                         nburst=readout_config.nburst, npad=10)

    hwf_block = convert_bin_wf_to_blocks(hwf, bin_size=readout_config.adc_hold_delay,
                                         shift_to_center=True)


    hwf_block_data = hwf_block.data[0, ...] # single block
    plt.plot(np.arange(0, hwf_block_data.shape[-1])*readout_config.adc_hold_delay + readout_config.adc_hold_delay + hwf_block.location[0,2],
             hwf_block_data[0, 0, :], 'o', label="hwf_block")
    cloc = hwf_block.location[0, :2]
    curr_mask = np.all(event.current.location[:,:2]==cloc[None, :], axis=1)
    curr = np.squeeze(event.current.data[curr_mask])
    plt.plot(np.arange(0, curr.shape[-1]) + event.current.location[curr_mask][0, -1], curr * readout_config.adc_hold_delay,
             'o', label="original current * adc_hold_delay")
    plt.legend()
    plt.savefig('hwf_block.png')
    # Deconvolve waveform data
    print(fr_full_k.shape)
    plt.figure()
    plt.plot(np.arange(fr_full_k.shape[-1]) * readout_config.adc_hold_delay + readout_config.adc_hold_delay/2, fr_full_k[2, 0, :], 'o', label="fr_kernel")
    plt.plot(np.arange(fr_full_k.shape[-1]) * readout_config.adc_hold_delay + readout_config.adc_hold_delay/2, fr_full_k[2, 1, :], 'o', label="fr_kernel")
    plt.plot(np.arange(fr_full_k.shape[-1]) * readout_config.adc_hold_delay + readout_config.adc_hold_delay/2, fr_full_k[2, 2, :], 'o', label="fr_kernel")
    plt.plot(np.arange(fr_full.shape[-1]), fr_full[2, 2, :]*readout_config.adc_hold_delay, 'o', label="fr_full")
    plt.xlim(1500, 1800)
    plt.legend()
    plt.savefig('fr_kernel.png')

    deconv_data, local_offset = deconv_fft(hwf_block_data, fr_full_k)
    print(f"  Deconvolved data shape: {deconv_data.shape}")
    # print(deconv_data)
    print(f"  Local offset: {local_offset}")
    for i in range(2):
        plt.figure()
        plt.plot(hwf_block.location[0, -1] - local_offset[-1] + np.arange(len(deconv_data[i, 0, :])) * readout_config.adc_hold_delay,
                 deconv_data[i, 0, :], 'o', label=f"deconv charge; sum(deconv) over t = {np.sum(deconv_data[i, 0, :]):.2f}")
        cloc = hwf_block.location[0, :2]
        cloc[0] += i
        effq_mask = np.all(event.effq.location[:, :2]==cloc[None, :], axis=1)
        effq = event.effq.data[effq_mask]
        effqloc = event.effq.location[effq_mask]
        plt.plot(effqloc[:, 2], effq[:, -1] * readout_config.adc_hold_delay, 'o', label=f"effq * adc_hold_delay; sum(effq) over t = {np.sum(effq[:, -1]):.2f}")
        plt.legend(loc='upper left')
        plt.title(f"at block index {i}; pixel index = {cloc}")
        plt.savefig(f'deconv_block_{i}.png')

    print('input shape', hwf_block_data.shape, 'deconv shape', deconv_data.shape, 'fr shape', fr_full_k.shape)



# Get geometry information
geometry = loader.get_geometry(0)
print(f"TPC 0 geometry: {geometry.lower} to {geometry.upper}")

# Get readout configuration
config = loader.get_readout_config()
print(f"Time spacing: {config.time_spacing} μs")
