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

has_noises = True
finstring = "data/pgun_muplus_3gev_tred_noises_truehits_interval_average.npz"
loader = DataLoader(finstring)
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
    boffset, noisy_meas = interval_average_to_block(event.hits.location[:, :3], event.hits.data[:, 3:])
    boffset, true_meas = interval_average_to_block(event.truehits.location[:, :3], event.truehits.data[:, 3:])
    print("noisy_meas shape:", noisy_meas.shape, "true_meas shape:", true_meas.shape)


# ---------------------------------------------------------------------------
# Power spectrum and SNR ratio analysis
# ---------------------------------------------------------------------------

def power_spectrum_3d(block: np.ndarray) -> np.ndarray:
    """3D power spectrum of a real-valued block via rfftn.

    Returns array of shape (Nx, Ny, Nt//2+1).
    """
    return np.abs(np.fft.rfftn(block))**2


def snr_ratio_3d(P_true: np.ndarray, P_noisy: np.ndarray,
                 eps: float = 1e-30) -> np.ndarray:
    """Element-wise |true|^2 / |noisy|^2, guarded against division by zero."""
    return P_true / (P_noisy + eps)


def freq_axes(shape: tuple) -> list:
    """Frequency axes matching rfftn output for a real array of given shape.

    Returns [fx, fy, ft] where fx/fy use fftfreq and ft uses rfftfreq.
    """
    return [
        np.fft.fftfreq(shape[0]),   # x: pixel^-1
        np.fft.fftfreq(shape[1]),   # y: pixel^-1
        np.fft.rfftfreq(shape[2]),  # t: tick^-1  (one-sided)
    ]


def project_mean(arr: np.ndarray, axis: int) -> np.ndarray:
    """Average a 3D array over all axes except `axis`."""
    axes = tuple(i for i in range(arr.ndim) if i != axis)
    return arr.mean(axis=axes)

def project_sum(arr: np.ndarray, axis: int) -> np.ndarray:
    """Average a 3D array over all axes except `axis`."""
    axes = tuple(i for i in range(arr.ndim) if i != axis)
    return arr.sum(axis=axes)

def plot_snr_analysis(noisy_block: np.ndarray, true_block: np.ndarray,
                      dt: float = 1.0) -> plt.Figure:
    """Plot 1D projections of power spectra and the SNR ratio.

    Parameters
    ----------
    noisy_block : (Nx, Ny, Nt) array — noisy measurement block
    true_block  : (Nx, Ny, Nt) array — true (signal-only) block
    dt          : time tick size in μs (used for axis label)
    """
    # Pad to a common shape so both FFT arrays align
    shape = tuple(max(a, b) for a, b in zip(noisy_block.shape, true_block.shape))
    def pad_to(arr):
        return np.pad(arr, [(0, s - n) for s, n in zip(shape, arr.shape)])
    noisy_block = pad_to(noisy_block)
    true_block  = pad_to(true_block)

    P_noisy = power_spectrum_3d(noisy_block)
    P_true  = power_spectrum_3d(true_block)
    ratio   = snr_ratio_3d(P_true, P_noisy)

    freqs  = freq_axes(shape)
    labels = [r'$f_x$ [pixel$^{-1}$]',
              r'$f_y$ [pixel$^{-1}$]',
              rf'$f_t$ [tick$^{{-1}}$] ($\Delta t={dt}\,\mu$s)']
    names  = ['x', 'y', 't']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i in range(3):
        f  = freqs[i]
        noisy_proj = project_sum(noisy_block, axis=i)
        true_proj  = project_sum(true_block, axis=i)
        if i != 2:
            pn = np.abs(np.fft.fft(noisy_proj))**2
            pt = np.abs(np.fft.fft(true_proj))**2
        else:
            pn = np.abs(np.fft.rfft(noisy_proj))**2
            pt = np.abs(np.fft.rfft(true_proj))**2
        r  = pt/pn

        # fftshift for the two-sided x/y axes; rfftfreq is already one-sided
        if i < 2:
            f  = np.fft.fftshift(f)
            pn = np.fft.fftshift(pn)
            pt = np.fft.fftshift(pt)
            r  = np.fft.fftshift(r)

        # power spectra
        ax = axes[0, i]
        ax.semilogy(f, pn, label='noisy', alpha=0.8)
        ax.semilogy(f, pt, label='true',  alpha=0.8)
        ax.set_xlabel(labels[i])
        ax.set_ylabel('Power [a.u.]')
        ax.set_title(f'Power spectrum — {names[i]} projection')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # SNR ratio
        ax = axes[1, i]
        ax.plot(f, r)
        ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(r'$|\widetilde{S}|^2\,/(|\widetilde{S}|^2+|\widetilde{N}|^2)$')
        ax.set_title(f'S^2/(S^2+N^2) ratio — {names[i]} projection')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Fourier power spectra and S^2/(S^2+N^2) ratio', fontsize=13)
    fig.tight_layout()
    return fig


noise_postfix = "_with_noises" if has_noises else "_nonoises"
fig = plot_snr_analysis(noisy_meas, true_meas, dt=readout_config.time_spacing)
fig.savefig(f"snr_analysis{noise_postfix}.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# Real-space charge projections
# ---------------------------------------------------------------------------

def plot_charge_projections(noisy_block: np.ndarray, true_block: np.ndarray) -> plt.Figure:
    """Plot 1D charge projections by summing over the other two axes.

    One subplot per axis (x, y, t), both noisy and true overlaid.
    """
    axis_labels = ['x [pixel]', 'y [pixel]', 't [tick]']
    axis_names  = ['x', 'y', 't']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, ax in enumerate(axes):
        other = tuple(j for j in range(3) if j != i)
        noisy_proj = noisy_block.sum(axis=other)
        true_proj  = true_block.sum(axis=other)

        # pad the shorter projection to a common length
        n = max(len(noisy_proj), len(true_proj))
        noisy_proj = np.pad(noisy_proj, (0, n - len(noisy_proj)))
        true_proj  = np.pad(true_proj,  (0, n - len(true_proj)))

        x = np.arange(n)
        ax.step(x, noisy_proj, where='mid', label='noisy', alpha=0.8)
        ax.step(x, true_proj,  where='mid', label='true',  alpha=0.8)
        ax.set_xlabel(axis_labels[i])
        ax.set_ylabel('Summed charge [a.u.]')
        ax.set_title(f'Projection along {axis_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Real-space charge projections', fontsize=13)
    fig.tight_layout()
    return fig


fig2 = plot_charge_projections(noisy_meas, true_meas)
fig2.savefig(f"charge_projections{noise_postfix}.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# Ratio-only plots with y-axis clipped to [0, 1.5]
# ---------------------------------------------------------------------------

def plot_ratio_clipped(noisy_block: np.ndarray, true_block: np.ndarray,
                       dt: float = 1.0) -> plt.Figure:
    """Plot the S²/N² ratio projections only, with y limited to [0, 1.5]."""
    shape = tuple(max(a, b) for a, b in zip(noisy_block.shape, true_block.shape))
    def pad_to(arr):
        return np.pad(arr, [(0, s - n) for s, n in zip(shape, arr.shape)])
    noisy_block = pad_to(noisy_block)
    true_block  = pad_to(true_block)

    P_noisy = power_spectrum_3d(noisy_block)
    P_true  = power_spectrum_3d(true_block)

    freqs  = freq_axes(shape)
    labels = [r'$f_x$ [pixel$^{-1}$]',
              r'$f_y$ [pixel$^{-1}$]',
              rf'$f_t$ [tick$^{{-1}}$] ($\Delta t={dt}\,\mu$s)']
    names  = ['x', 'y', 't']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, ax in enumerate(axes):
        f = freqs[i]
        noisy_proj = project_sum(noisy_block, axis=i)
        true_proj  = project_sum(true_block, axis=i)
        if i != 2:
            pn = np.abs(np.fft.fft(noisy_proj))**2
            pt = np.abs(np.fft.fft(true_proj))**2
        else:
            pn = np.abs(np.fft.rfft(noisy_proj))**2
            pt = np.abs(np.fft.rfft(true_proj))**2
        r  = pt/pn

        if i < 2:
            f = np.fft.fftshift(f)
            r = np.fft.fftshift(r)

        ax.plot(f, r)
        ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(r'$|\widetilde{S}|^2\,/\,|\widetilde{N}|^2$')
        ax.set_title(f'S²/(S²+N²) ratio — {names[i]} projection')
        ax.grid(True, alpha=0.3)

    fig.suptitle('S²/(S²+N²) ratio (clipped to [0, 1.5])', fontsize=13)
    fig.tight_layout()
    return fig


fig3 = plot_ratio_clipped(noisy_meas, true_meas, dt=readout_config.time_spacing)
fig3.savefig(f"snr_ratio_clipped{noise_postfix}.png", dpi=150)
plt.show()
