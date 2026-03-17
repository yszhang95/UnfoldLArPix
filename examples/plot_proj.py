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

  fine_shape = np.array(fine_voxels.shape, dtype=int)
  coarse_shape = np.array(coarse_voxels.shape, dtype=int)

  target_lower = coarse_lower.copy()
  target_lower += np.clip(np.floor((fine_lower - target_lower)//bin_size * bin_size), max=0)
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
  return aligned_fine, aligned_coarse, fine_summed, target_lower

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
      bin_size=f['adc_hold_delay']
  )
  print(new_offset, f['boffset'], f['smear_offset'], aligned_smear.shape, aligned_deconv_q.shape, smear_summed.shape)
  fig, axs = plt.subplots(1, 3, figsize=(18, 6))
  axs[0].hist(smear_summed.flatten() - aligned_deconv_q.flatten(), bins=40, range=(-5, 5), alpha=0.5)
  axs[0].set_xlabel('Smeared - Deconvolved')
  axs[0].set_title('All padded hits')
  axs[1].hist((smear_summed - aligned_deconv_q)[smear_summed > threshold].flatten(), bins=40, range=(-5, 5), alpha=0.5, label='Smear sum > {}'.format(threshold))
  axs[1].legend()
  axs[1].set_xlabel('Smeared - Deconvolved')
  axs[2].hist((smear_summed - aligned_deconv_q)[(smear_summed < threshold) & (smear_summed > 0.1)].flatten(), bins=40, range=(-5, 5), alpha=0.5, label='Smear sum > 0.1 & < {}'.format(threshold))
  axs[2].legend()
  axs[2].set_xlabel('Smeared - Deconvolved')
  plt.tight_layout()
  fig.savefig(f'{prefix}_hist_diff.png')

  from matplotlib.colors import LogNorm
  fig2d, ax2d = plt.subplots(figsize=(8, 6))
  h, xedges, yedges, img = ax2d.hist2d(smear_summed.flatten(), aligned_deconv_q.flatten(),
                                        bins=40, range=[[0, 10], [0, 10]], norm=LogNorm())
  fig2d.colorbar(img, ax=ax2d)
  ax2d.set_xlabel('Smeared True (summed)')
  ax2d.set_ylabel('Deconvolved')
  ax2d.set_title('2D Histogram: Smeared True vs Deconvolved (aligned)')
  fig2d.savefig(f'{prefix}_hist_2d.png')

  # Build hits grid on same coarse grid as aligned_deconv_q
  adc_hold_delay = int(f['adc_hold_delay'])
  mask = aligned_deconv_q > threshold

  fig2dh, ax2dh = plt.subplots(figsize=(8, 6))
  h, xedges, yedges, img = ax2dh.hist2d(
      smear_summed[mask], aligned_deconv_q[mask],
      bins=40, range=[[0, 10], [0, 10]], norm=LogNorm())
  fig2dh.colorbar(img, ax=ax2dh)
  ax2dh.set_xlabel('True Charge (smeared)')
  ax2dh.set_ylabel('Hits Charge')
  ax2dh.set_title('True Charge vs Hits (voxels with hits > f{threshold})')
  fig2dh.savefig(f'{prefix}_hist_2d_hits.png')


plt.figure(figsize=(10, 6))
plt.hist(filtered_smeared_true, label='Smeared True Projection', range=(0, 40), bins=40, alpha=0.5)
plt.hist(filtered_deconv_q, label='Deconvolved Projection', range=(0, 40), bins=40, alpha=0.5)
#plt.hist(filtered_totQ, label='Total Charge from Hits', range=(0, 40), bins=40, alpha=0.5)
plt.xlabel('Charge Projection')
plt.ylabel('Count')
plt.title('Comparison of Smeared True Projection and Deconvolved Projection')
plt.legend()
plt.grid()
plt.savefig(f'{prefix}_hist_deconv_q.png')
