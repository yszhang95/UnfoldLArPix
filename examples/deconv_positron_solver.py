#!/usr/bin/env python3
"""Constrained-solver deconvolution of zero-suppressed positron events.

Instead of template-compensating the gaps and inverse-filtering, this
driver fits the ionization charge directly to the recorded burst
integrals (see ``unfoldlarpix.constrained_solver``):

    min_q ||A q - d||^2 + beta*||relu(S_quiet(Kq) - thr)||^2 + alpha*||q||_1,
    q >= 0

The template-compensated pipeline is still run first — its block geometry
defines the solution grid and its (positivity-clipped) deconvolution is
the warm start.  The solver output is post-smoothed with the same
Gaussian filter so it is directly comparable to ``smeared_true``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from unfoldlarpix import (
    BurstSequenceProcessorV3,
    DataLoader,
    build_event_output_payload,
    prepare_field_response,
    process_event_deconvolution,
)
from unfoldlarpix.constrained_solver import (
    ZSOperator,
    build_latch_windows,
    gaussian_post_smooth,
    smear_kernel_gaussian,
    solve_fista,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", required=True)
    p.add_argument("--field-response", required=True)
    p.add_argument("--tpc-id", type=int, default=None)
    p.add_argument("--sigma", type=float, default=0.005)
    p.add_argument("--sigma-pxl", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.01,
                   help="L1 weight [ke-]. Must be of order the per-window "
                        "noise scale to remove the positivity noise-"
                        "rectification bias.")
    p.add_argument("--beta-quiet", type=float, default=1.0,
                   help="Quiet-bin inequality penalty weight.")
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--support-eps", type=float, default=None,
                   help="When set, restrict the solution to voxels where "
                        "the smoothed warm start exceeds this value [ke-] "
                        "(dilated ROI-style support projection).")
    p.add_argument("--support-dilate", type=int, default=2,
                   help="Dilation radius (voxels) of the support mask.")
    p.add_argument("--support-source", choices=["deconv", "hits"],
                   default="deconv",
                   help="Where the base support comes from: 'deconv' = "
                        "smoothed FFT-deconv warm start > --support-eps; "
                        "'hits' = union of the recorded latch windows "
                        "(fired pixel, window time bins), extended by "
                        "--hits-pre-bins/--hits-post-bins and then dilated "
                        "by --support-dilate like the deconv support.")
    p.add_argument("--hits-pre-bins", type=int, default=2,
                   help="Extra bins BEFORE each latch window included in "
                        "the hits support (charge can accumulate below "
                        "threshold for several bins before latching).")
    p.add_argument("--hits-post-bins", type=int, default=1,
                   help="Extra bins AFTER each latch window included in "
                        "the hits support.")
    p.add_argument("--debias-iter", type=int, default=0,
                   help="When > 0, refit with alpha=0 on the L1 active set "
                        "for this many iterations (two-stage debias).")
    p.add_argument("--debias-eps", type=float, default=0.01,
                   help="Active-set threshold [ke-] for the debias refit.")
    p.add_argument("--gaussian-basis", action="store_true",
                   help="Fit Gaussian-blob coefficients (q = G * c) by "
                        "folding the analysis Gaussian into the response "
                        "kernel. The unknown signal is smeared by "
                        "construction; L1 deghosts in blob space.")
    p.add_argument("--cold-start", action="store_true",
                   help="Start FISTA from zeros instead of the warm-start "
                        "deconvolution. Recommended with --gaussian-basis: "
                        "the linear deconv_q estimates q = G*c, not c, so "
                        "using it as c double-smears the initial state.")
    p.add_argument("--pad-pixels", type=int, default=0,
                   help="Spatial zero-padding (pixels) added around the hit "
                        "bounding box. The response couples +-12 pixels, so "
                        "true charge outside the box otherwise gets parked "
                        "on the boundary rows (edge ghosts).")
    p.add_argument("--warm-sigma", type=float, default=None,
                   help="Temporal Gaussian sigma (frequency units) used ONLY "
                        "for the warm-start pass and support construction. "
                        "Heavier smoothing (smaller value, e.g. 0.002 ~ 4us) "
                        "makes the seed more conservative; the fit and the "
                        "output stay at --sigma. Default: same as --sigma.")
    p.add_argument("--lean-output", action="store_true",
                   help="Save a minimal NPZ (deconv_q, offsets, params) "
                        "without the ~1.9 GB smeared_true / hwf_block "
                        "arrays. Evaluate with eval_deconv_metrics.py "
                        "--truth-npz <reference run>.")
    p.add_argument("--backend", choices=("numpy", "torch"), default="numpy",
                   help="Solver backend. 'torch' runs all FFTs and "
                        "scatter/gather on --device (float32).")
    p.add_argument("--device", default="cuda",
                   help="Torch device for --backend torch (default: cuda).")
    p.add_argument("--alpha-ladder", type=float, nargs="+", default=None,
                   help="Descending alpha stages for strong-charge-first "
                        "homotopy (e.g. 3.0 1.0 0.3). Overrides --alpha.")
    p.add_argument("--seed-cut", type=float, default=None,
                   help="Charge cut [ke-] seeding each ladder stage's "
                        "support from the previous stage's strong charges "
                        "(dilated). None = anneal alpha on fixed support.")
    p.add_argument("--seed-dilate", type=int, default=2,
                   help="Dilation radius of the strong-charge seed "
                        "(default: 2).")
    p.add_argument("--final-refit-eps", type=float, default=None,
                   help="After the ladder/DR, refit ONLY voxels above this "
                        "charge [ke-] with --final-refit-alpha, treating all "
                        "fainter charges as FROZEN background (their forward "
                        "contribution is subtracted from the data). Gain can "
                        "therefore not promote faint voxels above the "
                        "selection cut.")
    p.add_argument("--final-refit-alpha", type=float, default=0.0)
    p.add_argument("--final-refit-iters", type=int, default=150)
    p.add_argument("--spectral-l2", type=float, default=0.0,
                   help="Weight of the spectra-aware Wiener prior "
                        "lam*sum_f w(f)|Q(f)|^2 (w from the muon filter NPZ, "
                        "sharp-space corrected).")
    p.add_argument("--spectral-where", choices=("refit", "ladder", "both"),
                   default="refit",
                   help="Where the spectral prior acts: in the final refit, "
                        "inside every ladder/DR stage, or both.")
    p.add_argument("--spectral-shape", choices=("wiener", "d1", "d2"),
                   default="wiener",
                   help="Shape of the spectral weight: measured muon Wiener, "
                        "or analytic difference seminorms |D q|^2 / |D^2 q|^2 "
                        "(zero at DC: no charge-sum shrinkage; geometry-"
                        "blind, time axis only).")
    p.add_argument("--probe-conditioning", action="store_true",
                   help="Measure (lam_max, lam_min) of the data operator on "
                        "the deghosted support (conditioning of the "
                        "unregularized amplitude fit).")
    p.add_argument("--spectral-cap", type=float, default=100.0,
                   help="Cap on the Wiener weight w(f) (default: 100).")
    p.add_argument("--dr-rounds", type=int, default=0,
                   help="When > 0, run the deghost/regress alternation for "
                        "this many rounds instead of the alpha ladder: "
                        "strong-L1 position selection, then near-unbiased "
                        "amplitude regression under the exponential soft "
                        "prior, repeated.")
    p.add_argument("--alpha-deghost", type=float, default=0.5,
                   help="L1 weight of the deghost phase (default: 0.5).")
    p.add_argument("--alpha-regress", type=float, default=0.02,
                   help="On-skeleton L1 weight of the regression phase "
                        "(default: 0.02).")
    p.add_argument("--soft-seed-exponent", type=float, default=1.0,
                   help="Profile of the soft seed prior alpha*exp((d/len)^p): "
                        "1 = exponential (Laplace tail), 2 = Gaussian tail "
                        "(diffusion-motivated displacement prior).")
    p.add_argument("--soft-seed-len", type=float, default=None,
                   help="Replace the hard seeded support with a soft "
                        "exponential prior: per-voxel L1 weight alpha * "
                        "exp(d / len), d = Manhattan distance (voxels) from "
                        "the strong-charge skeleton. Charge far from the "
                        "skeleton needs exponentially stronger evidence.")
    p.add_argument("--ladder-iters", type=int, default=150,
                   help="FISTA iterations per ladder stage (default: 150).")
    p.add_argument("--lam-l2", type=float, default=0.0,
                   help="Ridge (L2) weight on the charge (fit_deconv3d-style).")
    p.add_argument("--lam-tv", type=float, default=0.0,
                   help="Isotropic gradient-norm smoothness weight "
                        "(fit_deconv3d-style lam_dx).")
    p.add_argument("--data-space", choices=("diff", "cumulative"),
                   default="diff",
                   help="Data model: per-burst window differences (historic; "
                        "ignores the MA(1) anti-correlation that differencing "
                        "creates) or cumulative latch values (i.i.d. per-tick "
                        "latch noise -> exact likelihood).")
    p.add_argument("--thr-weight", type=float, default=1.0,
                   help="Relative weight of the trigger-split pseudo-"
                        "measurement rows (their error is crossing "
                        "overshoot, not latch noise).")
    p.add_argument("--split-trigger", action="store_true",
                   help="Split each sequence's first window at the trigger, "
                        "using the readout threshold as an equality pseudo-"
                        "measurement of the pre-trigger charge "
                        "(fit_deconv3d-style).")
    p.add_argument("--basis-frac", type=float, default=1.0,
                   help="Fraction of the analysis Gaussian width used in "
                        "the fit basis (time-domain sense). 1.0 = full "
                        "analysis smearing inside the model; smaller values "
                        "fit a sharper basis and apply the residual "
                        "smearing to the output, avoiding model mismatch "
                        "with sharp window data. 0 -> spike basis.")
    p.add_argument("--deposit-phase", type=float, default=-0.5)
    p.add_argument("--warm-iter-recomp", type=int, default=2,
                   help="iter-recomp passes for the warm start.")
    p.add_argument("--time-filter-npz", default=None,
                   help="Optional muon filter for the warm-start pass.")
    p.add_argument("--output-suffix", default="")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()

    time_filter = None
    if args.time_filter_npz is not None:
        filt = np.load(args.time_filter_npz)
        key = "H_complex" if "H_complex" in filt.files else "H_mag"
        time_filter = (
            np.asarray(filt[key]),
            np.asarray(filt["freqs_cycles_per_sample"], dtype=np.float64),
        )

    loader = DataLoader(args.input_file)
    readout_config = loader.get_readout_config()
    prepared = prepare_field_response(
        args.field_response, readout_config.adc_hold_delay, normalized=False
    )
    kernel = prepared.integrated_response
    B = readout_config.adc_hold_delay

    for event in loader.iter_events():
        if args.tpc_id is not None and event.tpc_id != args.tpc_id:
            continue
        if not event.hits:
            continue
        print(f"TPC {event.tpc_id}, Event {event.event_id}")

        warm_sigma = args.warm_sigma if args.warm_sigma is not None else args.sigma
        result = process_event_deconvolution(
            event,
            readout_config,
            prepared,
            sigma_time=warm_sigma,
            sigma_pixel=args.sigma_pxl,
            processor_cls=BurstSequenceProcessorV3,
            tau=B,
            npadbin=50,
            require_zero_local_offset=True,
            deposit_mode="linear",
            deposit_phase=args.deposit_phase,
            iter_recomp=args.warm_iter_recomp,
            time_filter=time_filter,
            pad_pixels=args.pad_pixels,
        )
        block_offset = np.asarray(result.hwf_block_offset)
        block_shape = result.hwf_block.shape

        split_thr = (
            float(readout_config.threshold) if args.split_trigger else None
        )
        row_weights = None
        if args.data_space == "cumulative":
            from unfoldlarpix.constrained_solver import (
                build_cumulative_windows,
            )

            windows, is_pseudo = build_cumulative_windows(
                event.hits.location, event.hits.data, B, block_offset,
                csa_reset_time=readout_config.csa_reset_time,
                split_threshold=split_thr,
            )
            if is_pseudo.any() and args.thr_weight != 1.0:
                row_weights = np.where(is_pseudo, args.thr_weight, 1.0)
        else:
            windows = build_latch_windows(
                event.hits.location, event.hits.data, B, block_offset,
                csa_reset_time=readout_config.csa_reset_time,
                split_threshold=split_thr,
            )
        time_shift = 0
        fit_kernel = kernel
        residual_sigmas = None
        if args.gaussian_basis:
            frac = float(args.basis_frac)
            if not (0.0 < frac <= 1.0):
                raise ValueError("--basis-frac must be in (0, 1].")
            # Frequency-domain Gaussians multiply: 1/s_tot^2 = 1/s_fit^2 +
            # 1/s_res^2.  A time-domain width fraction `frac` corresponds to
            # a frequency-domain sigma s_fit = s / frac.
            sig_t_fit = args.sigma / frac
            sig_p_fit = args.sigma_pxl / frac
            if frac < 1.0:
                res_scale = 1.0 / np.sqrt(1.0 - frac * frac)
                residual_sigmas = (args.sigma * res_scale,
                                   args.sigma_pxl * res_scale)
            fit_kernel, time_shift = smear_kernel_gaussian(
                kernel, B, sig_t_fit, sig_p_fit
            )
            print(f"  gaussian basis (frac {frac}): kernel {kernel.shape} -> "
                  f"{fit_kernel.shape}, time shift {time_shift}")
        if args.backend == "torch":
            from unfoldlarpix import constrained_solver_torch as solver_mod

            op = solver_mod.TorchZSOperator(
                fit_kernel, block_shape, windows, adc_hold_delay=B,
                device=args.device, row_weights=row_weights,
            )
        else:
            from unfoldlarpix import constrained_solver as solver_mod

            op = ZSOperator(fit_kernel, block_shape, windows,
                            adc_hold_delay=B, row_weights=row_weights)
        print(f"  block {block_shape}, q-grid {op.q_shape}, "
              f"{op.n_data} burst integrals, backend {args.backend}")

        # Quiet pixels: inside the block bounding box but never triggered.
        quiet_mask = np.ones(block_shape, dtype=bool)
        for row in event.hits.location:
            px = int(row[0] - block_offset[0])
            py = int(row[1] - block_offset[1])
            if 0 <= px < block_shape[0] and 0 <= py < block_shape[1]:
                quiet_mask[px, py, :] = False
        thr = float(readout_config.threshold)

        def to_fit_grid(arr: np.ndarray) -> np.ndarray:
            """Map a physical-grid array onto the (shifted, shorter) fit grid."""
            if time_shift:
                arr = np.roll(arr, -time_shift, axis=2)
            return arr[:, :, : op.q_shape[2]]

        support = None
        if args.support_source == "hits":
            nx, ny, nt = block_shape
            support = np.zeros(block_shape, dtype=bool)
            for w in windows:
                if not (0 <= w.px < nx and 0 <= w.py < ny):
                    continue
                b0 = int(np.floor(max(w.t_lo, 0.0) / B)) - args.hits_pre_bins
                b1 = int(np.ceil(w.t_hi / B)) + args.hits_post_bins
                b0 = max(b0, 0)
                b1 = min(b1, nt)
                if b1 > b0:
                    support[w.px, w.py, b0:b1] = True
        elif args.support_eps is not None:
            warm_smooth = gaussian_post_smooth(
                np.clip(result.deconv_q, 0.0, None), B,
                warm_sigma, args.sigma_pxl,
            )
            support = warm_smooth > args.support_eps
        if support is not None:
            for _ in range(args.support_dilate):
                grown = support.copy()
                for ax in range(3):
                    for shift in (-1, 1):
                        grown |= np.roll(support, shift, axis=ax)
                support = grown
            support = to_fit_grid(support)
            print(f"  support: {support.mean() * 100:.2f}% of q voxels")

        spectral_kwargs_all = {}
        if args.spectral_l2 > 0:
            if args.spectral_shape == "wiener":
                from unfoldlarpix.constrained_solver import (
                    wiener_spectral_weight,
                )

                if args.time_filter_npz is None:
                    raise SystemExit(
                        "--spectral-l2 with shape 'wiener' needs "
                        "--time-filter-npz (truth mode)."
                    )
                filt = np.load(args.time_filter_npz)
                w_spec = wiener_spectral_weight(
                    np.asarray(filt["freqs_cycles_per_sample"], dtype=float),
                    np.asarray(filt["P_truth"], dtype=float),
                    np.asarray(filt["P_deconv"], dtype=float),
                    n_time=op.q_shape[2],
                    cap=args.spectral_cap,
                    smear_sigma_f=args.sigma * B,
                )
            else:
                from unfoldlarpix.constrained_solver import (
                    difference_spectral_weight,
                )

                order = 1 if args.spectral_shape == "d1" else 2
                w_spec = difference_spectral_weight(op.q_shape[2], order)
            spectral_kwargs_all = {
                "lam_spectral": args.spectral_l2,
                "spectral_weight": w_spec,
            }
            print(f"  spectral prior [{args.spectral_shape}]: "
                  f"lam={args.spectral_l2} ({args.spectral_where}), "
                  f"w range [{w_spec.min():.3g}, {w_spec.max():.3g}]")
        ladder_spectral = (
            spectral_kwargs_all if args.spectral_where in ("ladder", "both")
            else {}
        )
        refit_spectral = (
            spectral_kwargs_all if args.spectral_where in ("refit", "both")
            else {}
        )

        q0 = None if args.cold_start else to_fit_grid(result.deconv_q)
        if args.dr_rounds > 0:
            q_hat = solver_mod.solve_deghost_regress(
                op,
                n_rounds=args.dr_rounds,
                alpha_deghost=args.alpha_deghost,
                alpha_regress=args.alpha_regress,
                seed_cut=args.seed_cut if args.seed_cut is not None else 0.5,
                decay_len=args.soft_seed_len if args.soft_seed_len else 2.0,
                base_support=support,
                n_iter_deghost=args.ladder_iters,
                n_iter_regress=args.ladder_iters,
                q0=q0,
                beta_quiet=args.beta_quiet,
                quiet_mask=quiet_mask,
                quiet_threshold=thr,
                lam_l2=args.lam_l2,
                lam_tv=args.lam_tv,
                verbose=True,
                **ladder_spectral,
            )
        elif args.alpha_ladder:
            q_hat = solver_mod.solve_fista_ladder(
                op,
                args.alpha_ladder,
                base_support=support,
                seed_cut=args.seed_cut,
                seed_dilate=args.seed_dilate,
                soft_decay_len=args.soft_seed_len,
                soft_exponent=args.soft_seed_exponent,
                n_iter_per_stage=args.ladder_iters,
                q0=q0,
                beta_quiet=args.beta_quiet,
                quiet_mask=quiet_mask,
                quiet_threshold=thr,
                lam_l2=args.lam_l2,
                lam_tv=args.lam_tv,
                verbose=True,
                **ladder_spectral,
            )
        else:
            q_hat = solver_mod.solve_fista(
                op,
                alpha=args.alpha,
                beta_quiet=args.beta_quiet,
                quiet_mask=quiet_mask,
                quiet_threshold=thr,
                n_iter=args.n_iter,
                q0=q0,
                support_mask=support,
                lam_l2=args.lam_l2,
                lam_tv=args.lam_tv,
                verbose=True,
            )
        if args.probe_conditioning:
            from unfoldlarpix.constrained_solver import (
                probe_support_conditioning,
            )

            probe_eps = args.final_refit_eps or 0.5
            probe_support = q_hat > probe_eps
            lam_max, lam_min = probe_support_conditioning(op, probe_support)
            cond = lam_max / lam_min if lam_min > 0 else float("inf")
            print(f"  conditioning on support (> {probe_eps} ke-, "
                  f"{int(probe_support.sum())} voxels): "
                  f"lam_max={lam_max:.4g}, lam_min={lam_min:.4g}, "
                  f"cond(A^T A)={cond:.4g}")

        if args.final_refit_eps is not None:
            strong = q_hat > args.final_refit_eps
            q_faint = np.where(strong, 0.0, q_hat)
            print(f"  final refit: alpha={args.final_refit_alpha} on "
                  f"{100 * float(strong.mean()):.3f}% of voxels "
                  f"(strong > {args.final_refit_eps} ke-); "
                  f"{float(q_faint.sum()):.1f} ke- frozen as background")
            # subtract the frozen-faint forward contribution from the data
            d_faint = op.forward(q_faint)
            d_orig = op.d
            if hasattr(d_orig, "cpu"):
                op.d = d_orig - op.to_tensor(d_faint)
            else:
                op.d = d_orig - d_faint
            try:
                q_strong = solver_mod.solve_fista(
                    op,
                    alpha=args.final_refit_alpha,
                    beta_quiet=args.beta_quiet,
                    quiet_mask=quiet_mask,
                    quiet_threshold=thr,
                    n_iter=args.final_refit_iters,
                    q0=np.where(strong, q_hat, 0.0),
                    support_mask=strong,
                    lam_l2=args.lam_l2,
                    lam_tv=args.lam_tv,
                    verbose=True,
                    **refit_spectral,
                )
            finally:
                op.d = d_orig
            q_hat = q_strong + q_faint

        if args.debias_iter > 0:
            if args.backend == "torch":
                raise SystemExit(
                    "--debias-iter is not supported with --backend torch yet."
                )
            from unfoldlarpix.constrained_solver import debias_on_support

            q_hat = debias_on_support(
                op,
                q_hat,
                support_eps=args.debias_eps,
                n_iter=args.debias_iter,
                beta_quiet=args.beta_quiet,
                quiet_mask=quiet_mask,
                quiet_threshold=thr,
            )
            print(f"  debiased on active set: total q {q_hat.sum():.1f} ke-")

        d_ref = op.d.cpu().numpy() if hasattr(op.d, "cpu") else op.d
        resid = op.forward(q_hat) - d_ref
        print(f"  data residual rms {np.sqrt(np.mean(resid**2)):.4f} ke-, "
              f"total q {q_hat.sum():.1f} ke- "
              f"(warm start {np.clip(result.deconv_q, 0, None).sum():.1f})")

        if time_shift:
            # blob coefficient at fit index t sits at physical index
            # t + time_shift
            q_hat = np.roll(q_hat, time_shift, axis=2)
        if args.gaussian_basis and residual_sigmas is None:
            # full-width basis: model already carries the analysis smearing
            q_smooth = gaussian_post_smooth(
                q_hat, B, args.sigma, args.sigma_pxl
            )
        elif args.gaussian_basis:
            # partial basis: apply only the residual smearing
            q_smooth = gaussian_post_smooth(
                q_hat, B, residual_sigmas[0], residual_sigmas[1]
            )
        else:
            q_smooth = gaussian_post_smooth(
                q_hat, B, args.sigma, args.sigma_pxl
            )

        # Reuse the standard payload, replacing the deconvolution product.
        object.__setattr__(result, "deconv_q", q_smooth)
        geometry = loader.get_geometry(event.tpc_id)
        # Solver time declaration: the window->bin overlap convention of
        # the fit sits half a bin later than the linear pipeline's deposit
        # (phase -0.5) convention, so the solver output is declared at
        # -B + B//2.  Measured on nb4 (Phase-0 centroid diagnostic): the
        # -B declaration leaves a systematic -0.55 bin reco-early offset;
        # +B//2 removes it (r 0.944->0.980, ghost 10.4->5.4%).
        solver_tshift = -B + B // 2

        # Structured per-charge output, anchored to the GLOBAL tick axis
        # (grid-independent: downstream analysis can rebin to any width or
        # phase).  Physical bin-center of bin k on the declared grid is
        # boffset_raw[2] + k*B (fine ticks).  Columns:
        # [pixel_x, pixel_y, t_center_tick, charge_ke, on_skeleton]
        raw_off = np.asarray(result.hwf_block_offset, dtype=float)
        ci, cj, ck = np.where(q_hat > 0.01)
        seedcut = args.seed_cut if args.seed_cut is not None else 0.5
        charges_list = np.stack(
            [
                raw_off[0] + ci,
                raw_off[1] + cj,
                raw_off[2] + ck * float(B),
                q_hat[ci, cj, ck],
                (q_hat[ci, cj, ck] > seedcut).astype(float),
            ],
            axis=1,
        )
        if args.lean_output:
            boffset = np.array(result.hwf_block_offset, copy=True)
            boffset[-1] += solver_tshift
            payload = {
                "deconv_q": q_smooth,
                "deconv_q_sharp": q_hat.astype(np.float32),
                "boffset": boffset,
                "adc_hold_delay": readout_config.adc_hold_delay,
                "readout_nburst": event.hits.data.shape[1] - 3,
                "readout_threshold": float(readout_config.threshold),
                "lean_output": True,
            }
        else:
            payload = build_event_output_payload(
                event,
                geometry,
                readout_config,
                result,
                drift_length=prepared.drift_length,
                boffset_time_shift=solver_tshift,
                include_hwf_block=True,
            )
            payload["deconv_q_sharp"] = q_hat
        payload["solver_alpha"] = args.alpha
        payload["solver_beta_quiet"] = args.beta_quiet
        payload["solver_n_iter"] = args.n_iter
        payload["solver_warm_sigma"] = warm_sigma
        payload["solver_lam_tv"] = args.lam_tv
        payload["solver_lam_l2"] = args.lam_l2
        payload["charges"] = charges_list
        payload["charges_columns"] = (
            "pixel_x pixel_y t_center_tick charge_ke on_skeleton"
        )
        payload["boffset_raw"] = raw_off

        suffix = args.output_suffix or (
            f"a{args.alpha:g}_b{args.beta_quiet:g}".replace(".", "p")
        )
        out = Path(args.output_dir).expanduser() / (
            f"deconv_positron_solver_{suffix}_"
            f"event_{event.tpc_id}_{event.event_id}.npz"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **payload)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
