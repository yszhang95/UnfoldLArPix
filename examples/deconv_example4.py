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
loader = DataLoader("data/two_point_charges_noises.npz")
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

def gaus_smear_true(ticks: np.ndarray, true_charge: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Smear true charge with kernel to get smeared charge."""
    reordered = np.argsort(ticks)
    true_charge = true_charge[reordered]
    ticks = ticks[reordered]
    n = true_charge.shape[0]
    n_single_side = int((8*1/2/np.pi/width) // n + 1)
    ktimes = n_single_side * 2 + 1
    m = int(ktimes * n)
    smeared = np.zeros((m,))
    smeared = np.fft.ifft(np.fft.fft(true_charge, n=m) *
                          np.exp(-np.fft.fftfreq(n=m, d=1)**2/width**2/2), n=m).real
    smeared = np.roll(smeared, n_single_side*n)
    ticks = ticks[0] + np.arange(m) - n_single_side*n
    return ticks, smeared

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
    hwf_block_data = blocks
    hwf_block_location = boffset
    print(boffset, blocks)

    plt.plot(np.arange(0, hwf_block_data.shape[-1])*readout_config.adc_hold_delay + readout_config.adc_hold_delay + hwf_block.location[0,2],
             hwf_block_data[0, 0, :], 'o', label="hwf_block")
    cloc = hwf_block_location[:2]
    curr_mask = np.all(event.current.location[:,:2]==cloc[None, :], axis=1)
    curr = np.squeeze(event.current.data[curr_mask])
    plt.plot(np.arange(0, curr.shape[-1]) + event.current.location[curr_mask][0, -1], curr * readout_config.adc_hold_delay,
             'o', label="original current * adc_hold_delay")
    plt.legend()
    plt.savefig('hwf_block.png')
    # Deconvolve waveform data
    plt.figure()
    plt.plot(np.arange(fr_full_k.shape[-1]) * readout_config.adc_hold_delay + readout_config.adc_hold_delay/2, fr_full_k[2, 0, :], 'o', label="fr_kernel")
    plt.plot(np.arange(fr_full_k.shape[-1]) * readout_config.adc_hold_delay + readout_config.adc_hold_delay/2, fr_full_k[2, 1, :], 'o', label="fr_kernel")
    plt.plot(np.arange(fr_full_k.shape[-1]) * readout_config.adc_hold_delay + readout_config.adc_hold_delay/2, fr_full_k[2, 2, :], 'o', label="fr_kernel")
    plt.plot(np.arange(fr_full.shape[-1]), fr_full[2, 2, :]*readout_config.adc_hold_delay, 'o', label="fr_full")
    plt.xlim(1500, 1800)
    plt.legend()
    plt.savefig('fr_kernel.png')

    sigma = 0.01

    gaussian_kernel = gaussian_filter(n=hwf_block_data.shape[-1], dt=readout_config.adc_hold_delay,
                                      sigma=sigma)


    deconv_data, local_offset = deconv_fft(hwf_block_data, fr_full_k, gaussian_kernel)
    print(f"  Deconvolved data shape: {deconv_data.shape}")
    # print(deconv_data)
    print(f"  Local offset: {local_offset}")
    for i in range(2):
        plt.figure()
        plt.plot(hwf_block_location[-1] - local_offset[-1] + np.arange(len(deconv_data[i, 0, :])) * readout_config.adc_hold_delay,
                 deconv_data[i, 0, :], 'o-', label=f"deconv charge; sum(deconv) over t = {np.sum(deconv_data[i, 0, :]):.2f}")
        cloc = hwf_block_location[:2]
        cloc[0] += i
        effq_mask = np.all(event.effq.location[:, :2]==cloc[None, :], axis=1)
        effq = event.effq.data[effq_mask]
        # if effq.shape[0] != 1:
        #     raise ValueError(f"Expected exactly one effq entry for pixel {cloc}, got {effq.shape[0]}")
        effqloc = event.effq.location[effq_mask]
        # effqloc = np.padd(effqloc[:, 2], (0, len(smeared)))
        smeared = gaus_smear_true(effqloc[:, 2], effq[:, -1], width=sigma)
        effqsigma = sigma_simple(effq[:, -1], dx=1.0)
        plt.plot(effqloc[:, 2], effq[:, -1] * readout_config.adc_hold_delay, 'o-', label=f"effq * adc_hold_delay; sum(effq) over t = {np.sum(effq[:, -1]):.2f}; sigma={effqsigma:.2f}")
        smearedsigma = sigma_simple(smeared[1], dx=1.0)
        plt.plot(smeared[0], smeared[1] * readout_config.adc_hold_delay, 'o-', label=f"smeared effq * adc_hold_delay; sigma={smearedsigma:.2f}")
        plt.legend(loc='upper left')
        plt.title(f"at block index {i}; pixel index = {cloc}")
        plt.savefig(f'deconv_block_{i}_noises.png')

    print('input shape', hwf_block_data.shape, 'deconv shape', deconv_data.shape, 'fr shape', fr_full_k.shape)



# Get geometry information
geometry = loader.get_geometry(0)
print(f"TPC 0 geometry: {geometry.lower} to {geometry.upper}")

# Get readout configuration
config = loader.get_readout_config()
print(f"Time spacing: {config.time_spacing} μs")
