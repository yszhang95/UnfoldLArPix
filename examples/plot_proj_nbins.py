#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input NPZ file")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix (default: input stem)",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=2,
        help="Number of consecutive coarse time bins to sum together.",
    )
    return parser.parse_args()


def align_voxel_blocks(
    fine_lower_corner: np.ndarray,
    coarse_lower_corner: np.ndarray,
    fine_voxels: np.ndarray,
    coarse_voxels: np.ndarray,
    bin_size: int | np.ndarray,
    bound_to_upper: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align fine and coarse blocks onto the same spatial/time support."""
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

    if bound_to_upper:
        coarse_lower = coarse_lower - bin_size

    fine_shape = np.array(fine_voxels.shape, dtype=int)
    coarse_shape = np.array(coarse_voxels.shape, dtype=int)

    target_lower = coarse_lower.copy()
    diff_bins = ((fine_lower - target_lower) // bin_size) * bin_size
    target_lower += np.minimum(diff_bins, 0)

    fine_upper = fine_lower + fine_shape
    coarse_upper = coarse_lower + coarse_shape * bin_size
    target_upper = coarse_upper.copy()
    target_upper += np.clip(
        np.ceil((fine_upper - target_upper) / bin_size) * bin_size,
        a_min=0,
        a_max=None,
    ).astype(int)

    fine_padding_lower = fine_lower - target_lower
    coarse_padding_lower = (coarse_lower - target_lower) // bin_size
    fine_padding_upper = target_upper - fine_upper
    coarse_padding_upper = (target_upper - coarse_upper) // bin_size

    fine_padding = tuple(
        (int(pre), int(post))
        for pre, post in zip(fine_padding_lower, fine_padding_upper)
    )
    coarse_padding = tuple(
        (int(pre), int(post))
        for pre, post in zip(coarse_padding_lower, coarse_padding_upper)
    )

    aligned_fine = np.pad(fine_voxels, pad_width=fine_padding, mode="constant")
    aligned_coarse = np.pad(coarse_voxels, pad_width=coarse_padding, mode="constant")

    refine_factor: list[int] = []
    sub_axes: list[int] = []
    for axis in range(ndims):
        refine_factor.append(aligned_coarse.shape[axis])
        refine_factor.append(bin_size[axis])
        sub_axes.append(2 * axis + 1)

    reshaped = aligned_fine.reshape(refine_factor)
    fine_summed = reshaped.sum(axis=tuple(sub_axes))

    output_offset = target_lower + bin_size if bound_to_upper else target_lower
    return aligned_fine, aligned_coarse, fine_summed, output_offset


def group_nbins(arr: np.ndarray, nbins: int) -> np.ndarray:
    """Sum every ``nbins`` consecutive coarse time bins along the last axis."""
    arr = np.asarray(arr)
    if nbins <= 0:
        raise ValueError("nbins must be positive.")
    if nbins == 1:
        return arr

    n_time = arr.shape[-1]
    n_grouped = (n_time // nbins) * nbins
    if n_grouped == 0:
        raise ValueError(
            f"nbins={nbins} exceeds the time axis length {n_time}."
        )

    trimmed = arr[..., :n_grouped]
    new_shape = (*trimmed.shape[:-1], n_grouped // nbins, nbins)
    return trimmed.reshape(new_shape).sum(axis=-1)


def main() -> None:
    args = parse_args()
    if args.nbins <= 0:
        raise ValueError("--nbins must be positive.")

    threshold = args.threshold
    prefix = args.prefix if args.prefix is not None else args.input.removesuffix(".npz")
    suffix = f"_{args.nbins}bin"

    f = np.load(args.input)
    smeared_true = f["smeared_true"]
    deconv_q = f["deconv_q"] * (f["deconv_q"] > threshold)

    _, aligned_deconv_q, smear_summed, _ = align_voxel_blocks(
        fine_lower_corner=f["smear_offset"],
        coarse_lower_corner=f["boffset"],
        fine_voxels=smeared_true,
        coarse_voxels=deconv_q,
        bin_size=f["adc_hold_delay"],
        bound_to_upper=False,
    )

    smear_grouped = group_nbins(smear_summed, args.nbins)
    deconv_grouped = group_nbins(aligned_deconv_q, args.nbins)

    print(
        f"Original shape: {smear_summed.shape}, "
        f"grouped-{args.nbins} shape: {smear_grouped.shape}"
    )

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    diff = smear_grouped.flatten() - deconv_grouped.flatten()
    hist_range = (-5.0 * args.nbins, 5.0 * args.nbins)
    charge_range = (0.0, 10.0 * args.nbins)

    axs[0].hist(diff, bins=40, range=hist_range, alpha=0.5)
    axs[0].set_xlabel(f"Smeared - Deconvolved ({args.nbins}-bin sum)")
    axs[0].set_title(f"All padded hits ({args.nbins}-bin sum)")

    mask_threshold = smear_grouped > threshold
    axs[1].hist(
        diff[mask_threshold.flatten()],
        bins=40,
        range=hist_range,
        alpha=0.5,
        label=f"Smear sum > {threshold}",
    )
    axs[1].legend()
    axs[1].set_xlabel(f"Smeared - Deconvolved ({args.nbins}-bin sum)")

    mask_low = (smear_grouped < threshold) & (smear_grouped > 0.1)
    axs[2].hist(
        diff[mask_low.flatten()],
        bins=40,
        range=hist_range,
        alpha=0.5,
        label=f"Smear sum > 0.1 & < {threshold}",
    )
    axs[2].legend()
    axs[2].set_xlabel(f"Smeared - Deconvolved ({args.nbins}-bin sum)")
    plt.tight_layout()
    fig.savefig(f"{prefix}{suffix}_hist_diff.png")
    plt.close(fig)

    fig2d, ax2d = plt.subplots(figsize=(8, 6))
    ax2d.hist2d(
        smear_grouped.flatten(),
        deconv_grouped.flatten(),
        bins=40,
        range=[charge_range, charge_range],
        norm=LogNorm(),
    )
    ax2d.set_xlabel(f"Smeared True ({args.nbins}-bin sum)")
    ax2d.set_ylabel(f"Deconvolved ({args.nbins}-bin sum)")
    ax2d.set_title(f"2D Histogram: {args.nbins}-bin Summed Alignment")
    fig2d.savefig(f"{prefix}{suffix}_hist_2d.png")
    plt.close(fig2d)

    print(
        f"Created {prefix}{suffix}_hist_diff.png and "
        f"{prefix}{suffix}_hist_2d.png"
    )


if __name__ == "__main__":
    main()
