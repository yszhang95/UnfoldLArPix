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


def find_active_region(
    block: np.ndarray,
    threshold: float = 0.10,
) -> list[tuple[int, int]]:
    """Find the minimal bounding region along each axis above a threshold.

    For each axis the block is summed over the other two axes, giving a 1-D
    projection.  The region is then defined as the span from the first to the
    last index whose projection value exceeds ``threshold * max``.  All indices
    in between are included, even if their values fall below the threshold.

    Parameters
    ----------
    block : np.ndarray
        3-D array of shape (N0, N1, N2).
    threshold : float
        Fraction of the per-axis projection maximum used as the cutoff
        (default 0.10, i.e. 10 %).

    Returns
    -------
    regions : list of (start, stop) tuples
        One ``(start, stop)`` pair per axis, where ``stop`` is exclusive so
        the pair can be used directly as a ``slice(start, stop)``.
    """
    if block.ndim != 3:
        raise ValueError("block must be a 3-D array")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0, 1)")

    regions = []
    for axis in range(3):
        # Sum over the other two axes to get the 1-D projection
        sum_axes = tuple(i for i in range(3) if i != axis)
        proj = block.sum(axis=sum_axes)

        cutoff = threshold * proj.max()
        above = np.where(proj > cutoff)[0]

        if above.size == 0:
            regions.append((0, proj.size))
        else:
            regions.append((int(above[0]), int(above[-1]) + 1))

    return regions


def power_spectrum_projection_hist(
    block: np.ndarray,
    unchanged_axis: int = 2,
    slice_ranges: tuple | None = None,
    n_freq_bins: int = 100,
    n_power_bins: int = 100,
    log_power: bool = True,
    draw_hist: bool = True,
    mean_color: str = 'red',
    mean_label: str = 'mean power',
    ax: plt.Axes | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-trace power spectra along one axis, displayed as a 2D histogram.

    The 3D block is sliced along the two axes that are *not* ``unchanged_axis``
    using ``slice_ranges``, then those two axes are flattened.  For every
    resulting 1-D trace the one-sided power spectrum is computed via
    ``np.fft.rfft``.  All (frequency, power) pairs are accumulated into a 2-D
    histogram and plotted; the average power at each frequency is overlaid.

    Parameters
    ----------
    block : np.ndarray
        3-D array of shape (N0, N1, N2).
    unchanged_axis : int
        Axis along which the FFT is performed (0, 1, or 2).
    slice_ranges : tuple of two (start, stop) pairs, optional
        Ranges for the two axes that are *not* ``unchanged_axis``, in the
        order they appear in ``block``.  ``None`` means use the full extent.
    n_freq_bins : int
        Number of histogram bins along the frequency axis.
    n_power_bins : int
        Number of histogram bins along the power axis.
    log_power : bool
        If True, histogram and display log10 of power.
    draw_hist : bool
        If True, draw the 2-D histogram.  Set to False to overlay only the
        mean power line on an existing axes.
    mean_color : str
        Colour of the mean-power line.
    mean_label : str
        Legend label for the mean-power line.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  A new figure/axes is created if ``None``.

    Returns
    -------
    freqs : np.ndarray, shape (Nf,)
        Frequency values (cycles / sample).
    avg_power : np.ndarray, shape (Nf,)
        Mean power (or mean log10 power) at each frequency.
    hist2d : np.ndarray, shape (n_freq_bins, n_power_bins)
        The 2-D histogram counts.
    """
    if block.ndim != 3:
        raise ValueError("block must be a 3-D array")
    if unchanged_axis not in (0, 1, 2):
        raise ValueError("unchanged_axis must be 0, 1, or 2")

    # --- 1. Slice the two other axes ---
    other_axes = [i for i in range(3) if i != unchanged_axis]
    idx: list = [slice(None), slice(None), slice(None)]
    if slice_ranges is not None:
        for ax_idx, (start, stop) in zip(other_axes, slice_ranges):
            idx[ax_idx] = slice(start, stop)
    sub = block[tuple(idx)]

    # --- 2. Move unchanged_axis to last, then flatten the other two ---
    sub = np.moveaxis(sub, unchanged_axis, -1)   # shape (A, B, N_unchanged)
    n_unchanged = sub.shape[-1]
    flat = sub.reshape(-1, n_unchanged)           # shape (N_flat, N_unchanged)

    # --- 3. Compute power spectra for every trace ---
    freqs = np.fft.rfftfreq(n_unchanged)
    powers = np.abs(np.fft.rfft(flat, axis=-1)) ** 2   # (N_flat, Nf)

    if log_power:
        powers = np.log10(powers + 1e-30)

    # --- 4. Fill 2-D histogram (freq on x, power on y) ---
    freq_edges = np.linspace(freqs[0], freqs[-1], n_freq_bins + 1)
    power_edges = np.linspace(powers.min(), powers.max(), n_power_bins + 1)

    freq_rep = np.tile(freqs, flat.shape[0])   # (N_flat * Nf,)
    hist2d, _, _ = np.histogram2d(
        freq_rep, powers.ravel(),
        bins=[freq_edges, power_edges],
    )   # shape: (n_freq_bins, n_power_bins)

    # --- 5. Average power at each frequency ---
    avg_power = powers.mean(axis=0)   # (Nf,)

    # --- Plot ---
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if draw_hist:
        extent = [freq_edges[0], freq_edges[-1], power_edges[0], power_edges[-1]]
        im = ax.imshow(
            hist2d.T,      # (n_power_bins, n_freq_bins) → rows=power, cols=freq
            origin='lower',
            aspect='auto',
            extent=extent,
            cmap='viridis',
        )
        plt.colorbar(im, ax=ax, label='counts')
        ax.set_xlabel('frequency [cycles/sample]')
        ax.set_ylabel('log\u2081\u2080(power)' if log_power else 'power')
        ax.set_title(
            f'Power spectra along axis {unchanged_axis} '
            f'(projected axes {other_axes[0]}, {other_axes[1]})'
        )

    ax.plot(freqs, avg_power, color=mean_color, linewidth=1.5, label=mean_label)
    ax.legend()

    return freqs, avg_power, hist2d

# ---------------------------------------------------------------------------
# Power-spectrum plots
# ---------------------------------------------------------------------------

# Active region is defined by the true signal
regions = find_active_region(true_meas)
print("Active regions (true_meas):", regions)

slice_ranges_per_axis = [
    (regions[1], regions[2]),   # unchanged_axis = 0
    (regions[0], regions[2]),   # unchanged_axis = 1
    (regions[0], regions[1]),   # unchanged_axis = 2
]
axis_labels = ['x (pixel)', 'y (pixel)', 't (tick)']

# Collect mean powers for the ratio plot
avg_noisy = {}
avg_true  = {}

# --- Figure 1: noisy_meas 2D histogram + mean ---
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle('Power spectra — noisy_meas')

for uax, (ax_plot, sr) in enumerate(zip(axes1, slice_ranges_per_axis)):
    freqs, avg, _ = power_spectrum_projection_hist(
        noisy_meas,
        unchanged_axis=uax,
        slice_ranges=sr,
        draw_hist=True,
        mean_color='red',
        mean_label='noisy mean',
        ax=ax_plot,
    )
    avg_noisy[uax] = (freqs, avg)
    other = [axis_labels[i] for i in range(3) if i != uax]
    ax_plot.set_title(
        f'FFT along {axis_labels[uax]}\n'
        f'(slice: {other[0]} {sr[0]}, {other[1]} {sr[1]})'
    )

plt.tight_layout()
plt.savefig('power_spectra_noisy.png', dpi=150)
plt.show()

# --- Figure 2: true_meas 2D histogram + mean ---
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('Power spectra — true_meas')

for uax, (ax_plot, sr) in enumerate(zip(axes2, slice_ranges_per_axis)):
    freqs, avg, _ = power_spectrum_projection_hist(
        true_meas,
        unchanged_axis=uax,
        slice_ranges=sr,
        draw_hist=True,
        mean_color='cyan',
        mean_label='true mean',
        ax=ax_plot,
    )
    avg_true[uax] = (freqs, avg)
    other = [axis_labels[i] for i in range(3) if i != uax]
    ax_plot.set_title(
        f'FFT along {axis_labels[uax]}\n'
        f'(slice: {other[0]} {sr[0]}, {other[1]} {sr[1]})'
    )

plt.tight_layout()
plt.savefig('power_spectra_true.png', dpi=150)
plt.show()

# --- Figure 3: mean powers overlaid + ratio ---
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle('Mean power comparison and ratio (noisy / true)')

for uax in range(3):
    freqs_n, avg_n = avg_noisy[uax]
    freqs_t, avg_t = avg_true[uax]

    ax_top = axes3[0, uax]
    ax_top.plot(freqs_n, avg_n, color='red',  linewidth=1.5, label='noisy mean')
    ax_top.plot(freqs_t, avg_t, color='cyan', linewidth=1.5, label='true mean')
    ax_top.set_xlabel('frequency [cycles/sample]')
    ax_top.set_ylabel('log\u2081\u2080(power)')
    ax_top.set_title(f'Mean power along {axis_labels[uax]}')
    ax_top.legend()

    ax_bot = axes3[1, uax]
    log_ratio = avg_t - avg_n   # difference in log10 = log10(noisy/true)
    ratio = 10 ** log_ratio          # convert back to linear ratio
    ax_bot.plot(freqs_n, ratio, color='orange', linewidth=1.5)
    ax_bot.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax_bot.set_xlabel('frequency [cycles/sample]')
    ax_bot.set_ylabel('log\u2081\u2080(noisy / true)')
    ax_bot.set_title(f'Ratio along {axis_labels[uax]}')

plt.tight_layout()
plt.savefig('power_spectra_ratio.png', dpi=150)
plt.show()
