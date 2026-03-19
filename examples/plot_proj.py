#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Input NPZ file')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--prefix', type=str, default=None,
                    help='Output filename prefix (default: input stem)')
args = parser.parse_args()
threshold = args.threshold
prefix = args.prefix if args.prefix is not None else args.input.removesuffix('.npz')

filtered_deconv_q = []
filtered_smeared_true = []
filtered_totQ = []

def align_voxel_blocks(
    fine_lower_corner: np.ndarray,
    coarse_lower_corner: np.ndarray,
    fine_voxels: np.ndarray,
    coarse_voxels: np.ndarray,
    bin_size: int | np.ndarray,
    bound_to_upper: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Pad and align fine/coarse voxel blocks so they share a lower corner and have
  compatible shapes, then sum the fine block inside each coarse voxel.
  """
  fine_voxels = np.asarray(fine_voxels)
  coarse_voxels = np.asarray(coarse_voxels)
  ndims = fine_voxels.ndim
  if ndims != coarse_voxels.ndim:
    raise ValueError("Fine and coarse blocks must have the same dimensionality.")
  if bound_to_upper:
      raise ValueError("bound_to_upper is deprecated. The shift was done in deconv script.")

  fine_lower = np.asarray(fine_lower_corner, dtype=int)
  coarse_lower = np.asarray(coarse_lower_corner, dtype=int)
  if fine_lower.shape[0] != ndims or coarse_lower.shape[0] != ndims:
    raise ValueError("Lower corner coordinates must match the voxel dimensions.")

  bin_size = np.asarray(bin_size, dtype=int)
  if bin_size.ndim == 0:
      bin_size = np.full((ndims,), bin_size, dtype=int)
      bin_size[:-1] = 1
  if bin_size.shape[0] != ndims:
    raise ValueError("bin_size must be broadcastable to each voxel dimension.")
  if np.any(bin_size <= 0):
    raise ValueError("bin_size must be positive.")

  if bound_to_upper:
      coarse_lower[-1] = coarse_lower[-1] - bin_size[-1]

  fine_shape = np.array(fine_voxels.shape, dtype=int)
  coarse_shape = np.array(coarse_voxels.shape, dtype=int)

  target_lower = coarse_lower.copy()
  # Using min instead of np.clip for a_max=0 to avoid array/scalar broadcast issues with None
  # target_lower must be <= fine_lower and <= coarse_lower to prevent negative padding
  diff_bins = ((fine_lower - target_lower) // bin_size) * bin_size
  target_lower += np.minimum(diff_bins, 0)
  print("target_lower:", target_lower, "fine_lower:", fine_lower, "coarse_lower:", coarse_lower)
  fine_upper = fine_lower + fine_shape
  coarse_upper = coarse_lower + coarse_shape * bin_size
  target_upper = coarse_upper.copy()
  target_upper += np.clip(np.ceil((fine_upper - target_upper)/bin_size) * bin_size, min=0).astype(int)
  print("target_upper:", target_upper, "fine_upper:", fine_upper, "coarse_upper:", coarse_upper)
  fine_padding_lower = fine_lower - target_lower
  coarse_padding_lower = (coarse_lower - target_lower) // bin_size
  fine_padding_upper = target_upper - fine_upper
  coarse_padding_upper = (target_upper - coarse_upper) // bin_size
  print("fine_padding_lower:", fine_padding_lower, "coarse_padding_lower:", coarse_padding_lower)
  print("fine_padding_upper:", fine_padding_upper, "coarse_padding_upper:", coarse_padding_upper)
  fine_padding = tuple((int(pre), int(post)) for pre, post in zip(fine_padding_lower, fine_padding_upper))
  coarse_padding = tuple((int(pre), int(post)) for pre, post in zip(coarse_padding_lower, coarse_padding_upper))
  print("fine_padding:", fine_padding, "coarse_padding:", coarse_padding)
  aligned_fine = np.pad(fine_voxels, pad_width=tuple((int(pre), int(
      post)) for pre, post in zip(fine_padding_lower, fine_padding_upper)), mode='constant')
  aligned_coarse = np.pad(coarse_voxels, pad_width=tuple((int(pre), int(
      post)) for pre, post in zip(coarse_padding_lower, coarse_padding_upper)), mode='constant')

  refine_factor = []
  sub_axes = []
  for i in range(ndims):
      refine_factor.append(aligned_coarse.shape[i])
      refine_factor.append(bin_size[i])
      sub_axes.append(2*i+1)
      print("aligned_coarse.shape[{}]:".format(i), aligned_coarse.shape[i], "bin_size[{}]:".format(i), bin_size[i], "aligned_fine.shape[{}]:".format(i), aligned_fine.shape[i])
  reshaped = aligned_fine.reshape(refine_factor)
  print(sub_axes)
  fine_summed = reshaped.sum(axis=tuple(sub_axes))

  output_offset = target_lower + bin_size if bound_to_upper else target_lower
  return aligned_fine, aligned_coarse, fine_summed, output_offset

for ievent in range(1):
  # f = np.load('deconv_event_0_0.npz')
  f = np.load(args.input)
  # f = np.load('deconv_positron_v2_event_0_0.npz')
  smeared_true = f['smeared_true']
  deconv_q = f['deconv_q'] * (f['deconv_q'] > threshold)
  effq_proj = np.sum(smeared_true, axis=-1)
  deconv_proj = np.sum(deconv_q, axis=-1)
  print(np.sum(deconv_proj), deconv_proj[deconv_proj > 0].shape)
  print((deconv_proj > 0).shape, deconv_proj.shape, deconv_q.shape, 'check', (deconv_q > 3.5).shape, np.sum(deconv_q > 3.5),
        )

  # start hits
  hl = f['hits_location']
  hd = f['hits_data']
  totQ = {}
  for i, (l, d) in enumerate(zip(hl, hd)):
      q = totQ.get((l[0], l[1]), 0)
      q += d[-1]
      totQ[(l[0], l[1])] = q

  totQ = np.array(list(totQ.values()))

  filtered_deconv_q.extend(deconv_proj[deconv_proj > threshold].tolist())
  filtered_smeared_true.extend(effq_proj[effq_proj > threshold].tolist())
  filtered_totQ.extend(totQ[totQ > 1].tolist())

  aligned_smear, aligned_deconv_q, smear_summed, new_offset = align_voxel_blocks(
      fine_lower_corner=f['smear_offset'],
      coarse_lower_corner=f['boffset'],
      fine_voxels=smeared_true,
      coarse_voxels=deconv_q,
      bin_size=f['adc_hold_delay'],
      bound_to_upper=False
  )
  dtus = 0.05 * f['adc_hold_delay']
  print(new_offset, f['boffset'], f['smear_offset'], aligned_smear.shape, aligned_deconv_q.shape, smear_summed.shape)
  from matplotlib.ticker import MultipleLocator
  fig, axs = plt.subplots(1, 3, figsize=(18, 6))
  for ax in axs:
      ax.grid(True, which='both', linestyle='--', alpha=0.5)
      ax.xaxis.set_major_locator(MultipleLocator(2.0))
      ax.xaxis.set_minor_locator(MultipleLocator(0.5))

  axs[0].hist(smear_summed.flatten() - aligned_deconv_q.flatten(), bins=50, range=(-5, 5), alpha=0.5)
  axs[0].set_xlabel(f'Smeared - Deconvolved [ke-/pixel/{dtus}us]')
  axs[0].set_title('All padded hits')

  axs[1].hist((smear_summed - aligned_deconv_q)[smear_summed > threshold].flatten(), bins=50, range=(-5, 5), alpha=0.5, label='Smear sum > {}'.format(threshold))
  axs[1].legend()
  axs[1].set_xlabel(f'Smeared - Deconvolved [ke-/pixel/{dtus}us]')

  axs[2].hist((smear_summed - aligned_deconv_q)[(smear_summed < threshold) & (smear_summed > 0.1)].flatten(), bins=50, range=(-5, 5), alpha=0.5, label='Smear sum > 0.1 & < {}'.format(threshold))
  axs[2].legend()
  axs[2].set_xlabel(f'Smeared - Deconvolved [ke-/pixel/{dtus}us]')
  plt.tight_layout()
  fig.savefig(f'{prefix}_hist_diff.png')
  plt.close(fig)

  # New plot: Difference for deconv_q > threshold
  fig_diff_deconv, ax_diff_deconv = plt.subplots(figsize=(8, 6))
  mask_deconv = (aligned_deconv_q > threshold)
  if np.any(mask_deconv):
      ax_diff_deconv.hist((smear_summed - aligned_deconv_q)[mask_deconv].flatten(), bins=50, range=(-5, 5), alpha=0.7)
      ax_diff_deconv.set_title(f'Smeared - Deconvolved (for Deconv > {threshold})')
      ax_diff_deconv.set_xlabel(f'Smeared Sum - Deconvolved Q [ke-/pixel/{dtus}us]')
      ax_diff_deconv.set_ylabel('Count')
      ax_diff_deconv.grid(True, which='both', linestyle='--', alpha=0.5)
      ax_diff_deconv.xaxis.set_major_locator(MultipleLocator(2.0))
      ax_diff_deconv.xaxis.set_minor_locator(MultipleLocator(0.5))
      fig_diff_deconv.tight_layout()
      fig_diff_deconv.savefig(f'{prefix}_hist_diff_deconv_mask.png')
  plt.close(fig_diff_deconv)

  from matplotlib.colors import LogNorm
  # fig2d, ax2d = plt.subplots(figsize=(8, 6))
  # h, xedges, yedges, img = ax2d.hist2d(smear_summed.flatten(), aligned_deconv_q.flatten(),
  #                                       bins=40, range=[[0, 10], [0, 10]], norm=LogNorm())
  # fig2d.colorbar(img, ax=ax2d)
  # ax2d.set_xlabel('Smeared True (summed)')
  # ax2d.set_ylabel('Deconvolved')
  # ax2d.set_title('2D Histogram: Smeared True vs Deconvolved (aligned)')
  # fig2d.savefig(f'{prefix}_hist_2d.png')
  # plt.close(fig2d)

  # Build hits grid on same coarse grid as aligned_deconv_q
  adc_hold_delay = int(f['adc_hold_delay'])
  mask = aligned_deconv_q > threshold

  fig2dh, ax2dh = plt.subplots(figsize=(8, 6))
  h, xedges, yedges, img = ax2dh.hist2d(
      smear_summed[mask], aligned_deconv_q[mask],
      bins=40, range=[[0, 10], [0, 10]], norm=LogNorm())
  fig2dh.colorbar(img, ax=ax2dh)
  ax2dh.set_xlabel(f'True Charge (smeared) [ke-/pixel/{dtus}us]')
  ax2dh.set_ylabel(f'Hits Charge [ke-/pixel/{dtus}us]')
  ax2dh.set_title(f'True Charge vs Hits (voxels with hits > {threshold})')
  fig2dh.savefig(f'{prefix}_hist_2d_hits.png')
  plt.close(fig2dh)

  # New plot: histogram of distances between peak time indices of aligned smeared true and aligned deconv
  # Consider only spatial voxels where the smeared true sequence exceeds the threshold (at any time).
  # We compute argmax along the time axis (assumed to be the last axis).
  try:
      smear_arr = np.asarray(smear_summed)
      deconv_arr = np.asarray(aligned_deconv_q)
      if smear_arr.ndim < 1 or deconv_arr.ndim < 1:
          raise ValueError("Unexpected array dimensionality for peak analysis.")
      # Determine time axis (last axis)
      time_axis = -1
      # mask pixels where smeared true has any sample > threshold
      smear_peak_mask = smear_arr.max(axis=time_axis) > threshold
      n_masked = np.count_nonzero(smear_peak_mask)
      if n_masked == 0:
          print("No smeared-true sequences exceed threshold; skipping peak-distance histogram.")
      else:
          smear_peaks = np.argmax(smear_arr, axis=time_axis)
          deconv_peaks = np.argmax(deconv_arr, axis=time_axis)
          # Extract peaks for masked spatial positions and compute signed distances (smear - deconv)
          # Flatten spatial dims to 1D for ease. Keep absolute and signed histograms.
          # Prepare flattened arrays with shape (n_spatial, n_time)
          if smear_arr.ndim == 1:
              # 1D time series only (rare)
              masked_smear_peaks = smear_peaks[smear_peak_mask]
              masked_deconv_peaks = deconv_peaks[smear_peak_mask]
              signed_dists = masked_smear_peaks.astype(int) - masked_deconv_peaks.astype(int)
              true_charge_at_peak = np.array([smear_arr[idx] for idx in np.nonzero(smear_peak_mask)[0]])
          else:
              # flatten spatial dims to (n_spatial, n_time)
              ntime = smear_arr.shape[-1]
              smear_flat = smear_arr.reshape(-1, ntime)
              deconv_flat = deconv_arr.reshape(-1, ntime)
              smear_peaks_flat = smear_peaks.reshape(-1)
              deconv_peaks_flat = deconv_peaks.reshape(-1)
              mask_flat = smear_peak_mask.reshape(-1)
              rows = np.nonzero(mask_flat)[0]
              masked_smear_peaks = smear_peaks_flat[rows]
              masked_deconv_peaks = deconv_peaks_flat[rows]
              signed_dists = masked_smear_peaks.astype(int) - masked_deconv_peaks.astype(int)
              # true charge in smeared true at the smeared_peak index for each masked spatial position
              true_charge_at_peak = smear_flat[rows, masked_smear_peaks]

          abs_dists = np.abs(signed_dists)

          # Plot histogram of signed distances
          fig_peak_signed, ax_peak_signed = plt.subplots(figsize=(8, 6))
          bins_signed = np.arange(signed_dists.min() - 0.5, signed_dists.max() + 1.5, 1.0) if signed_dists.size > 0 else [0]
          ax_peak_signed.hist(signed_dists, bins=bins_signed, alpha=0.7)
          ax_peak_signed.set_xlabel('Signed peak index distance (smeared_peak - deconv_peak) [coarse bins]')
          ax_peak_signed.set_ylabel('Count')
          ax_peak_signed.set_title(f'Peak index signed distance (n={signed_dists.size}) for smeared_true > {threshold}')

          # Add inset plot excluding masked_deconv_peaks < 10
          from mpl_toolkits.axes_grid1.inset_locator import inset_axes
          mask_nozero = (masked_deconv_peaks >= 10)
          if np.any(mask_nozero):
              ax_inset = inset_axes(ax_peak_signed, width="40%", height="40%", loc="upper right", borderpad=2)
              dists_nozero = signed_dists[mask_nozero]
              bins_nozero = np.arange(dists_nozero.min() - 0.5, dists_nozero.max() + 1.5, 1.0)
              ax_inset.hist(dists_nozero, bins=bins_nozero, alpha=0.7, color='orange')
              ax_inset.set_title("Excluding deconv_peak < 10", fontsize=10)
              ax_inset.tick_params(labelsize=8)

          fig_peak_signed.tight_layout()
          fig_peak_signed.savefig(f'{prefix}_hist_peak_signed_dist.png')
          plt.close(fig_peak_signed)

          # New 2D plot: signed distance vs true charge at smeared peak
          try:
              if signed_dists.size == 0:
                  print("No signed distances to plot for signed_vs_true_charge.")
              else:
                  # Scatter plot
                  fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
                  ax_scatter.scatter(signed_dists, true_charge_at_peak, alpha=0.6)
                  ax_scatter.set_xlabel('Signed peak index distance (smeared_peak - deconv_peak) [coarse bins]')
                  ax_scatter.set_ylabel('Smeared true charge at smeared peak [units]')
                  ax_scatter.set_title(f'Signed distance vs smeared-true charge at peak (n={signed_dists.size})')
                  fig_scatter.tight_layout()
                  fig_scatter.savefig(f'{prefix}_signed_vs_true_scatter.png')
                  plt.close(fig_scatter)

                  # Scatter plot: distances_scatter
                  fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
                  ax_scatter.scatter(masked_smear_peaks, masked_deconv_peaks, alpha=0.6)
                  ax_scatter.set_xlabel('smeared peak index (coarse bins)')
                  ax_scatter.set_ylabel('deconv peak index (coarse bins)')
                  ax_scatter.set_title(f'Peak index correlation (n={signed_dists.size})')
                  fig_scatter.tight_layout()
                  fig_scatter.savefig(f'{prefix}_distances_scatter.png')
                  plt.close(fig_scatter)

                  # 2D histogram (log color)
                  fig2dh_sd_tc, ax2dh_sd_tc = plt.subplots(figsize=(8, 6))
                  # bins for signed distances (integer offsets)
                  bins_x = np.arange(signed_dists.min() - 0.5, signed_dists.max() + 1.5, 1.0)
                  # dynamic bins for true charge
                  y_max = float(np.max(true_charge_at_peak)) if true_charge_at_peak.size > 0 else 1.0
                  bins_y = np.linspace(0.0, max(1.0, y_max), 40)
                  h2, xedges2, yedges2, img2 = ax2dh_sd_tc.hist2d(signed_dists, true_charge_at_peak, bins=[bins_x, bins_y], norm=LogNorm())
                  fig2dh_sd_tc.colorbar(img2, ax=ax2dh_sd_tc)
                  ax2dh_sd_tc.set_xlabel('Signed peak index distance (smeared_peak - deconv_peak) [coarse bins]')
                  ax2dh_sd_tc.set_ylabel('Smeared true charge at smeared peak [units]')
                  ax2dh_sd_tc.set_title(f'2D: Signed distance vs smeared-true charge at peak (n={signed_dists.size})')
                  fig2dh_sd_tc.tight_layout()
                  fig2dh_sd_tc.savefig(f'{prefix}_hist2d_signed_vs_true_charge.png')
                  plt.close(fig2dh_sd_tc)
          except Exception as e2:
              print("Failed to produce signed_vs_true_charge plots:", e2)

  except Exception as e:
      print("Failed to compute peak-distance histogram:", e)

