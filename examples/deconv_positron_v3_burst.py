#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from unfoldlarpix import (
    BurstSequenceProcessorV3,
    DataLoader,
    build_event_output_payload,
    prepare_field_response,
    process_event_deconvolution,
    shift_time_offset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deconvolve positron event data with BurstSequenceProcessorV3"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.005,
        help="Gaussian filter sigma in time (default: 0.005)",
    )
    parser.add_argument(
        "--sigma-pxl",
        type=float,
        default=0.2,
        help="Gaussian filter sigma in pixel (default: 0.2)",
    )
    parser.add_argument(
        "--input-file",
        default="data/pgun_positron_3gev_tred_noises_effq_nt1.npz",
        help="Input NPZ file produced by tred",
    )
    parser.add_argument(
        "--field-response",
        default="/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz",
        help="Field response NPZ file",
    )
    parser.add_argument(
        "--tpc-id",
        type=int,
        default=None,
        help="Process only this TPC ID (default: process all)",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix to append to output filename (e.g., 's0p005_sp0p2')",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for the output NPZ file",
    )
    parser.add_argument(
        "--response-template",
        choices=("center", "collection", "collection-plus-neighbors"),
        default="center",
        help="Field-response template used for burst compensation.",
    )
    parser.add_argument(
        "--enable-wiener-roi",
        action="store_true",
        help="Run a Wiener-inspired deconvolution to identify ROIs and "
        "emit a ROI-masked deconv_q.",
    )
    parser.add_argument(
        "--wiener-omega-c",
        type=float,
        default=None,
        help="Wiener time-axis cutoff frequency (cycles per ADC tick); "
        "required when --enable-wiener-roi is set.",
    )
    parser.add_argument(
        "--wiener-b",
        type=float,
        default=2.0,
        help="Wiener rolloff exponent b in F(f)=exp(-0.5*(f/omega_c)^b).",
    )
    parser.add_argument(
        "--roi-threshold-sigma",
        type=float,
        default=5.0,
        help="ROI threshold in units of quiet-pixel noise RMS.",
    )
    parser.add_argument(
        "--roi-merge-gap",
        type=int,
        default=2,
        help="Merge ROI segments separated by <= this many below-threshold bins.",
    )
    parser.add_argument(
        "--roi-expand",
        type=int,
        default=2,
        help="Expand each ROI by this many bins on each side.",
    )
    return parser.parse_args()


def fmt_sigma_detailed(value: float) -> str:
    """0.0005 -> 's0p0005', 0.005 -> 's0p005', 0.01 -> 's0p01'."""
    return "s" + f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def fmt_sigma_pxl_detailed(value: float) -> str:
    """0.1 -> 'sp0p1', 0.15 -> 'sp0p15', 0.2 -> 'sp0p2'."""
    return "sp" + f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> None:
    args = parse_args()
    if not args.output_suffix:
        args.output_suffix = (
            f"{fmt_sigma_detailed(args.sigma)}_"
            f"{fmt_sigma_pxl_detailed(args.sigma_pxl)}"
        )

    loader = DataLoader(args.input_file)
    readout_config = loader.get_readout_config()
    prepared_response = prepare_field_response(
        args.field_response,
        readout_config.adc_hold_delay,
        normalized=False,
        response_template=args.response_template.replace("-", "_"),
    )

    for event in loader.iter_events():
        if args.tpc_id is not None and event.tpc_id != args.tpc_id:
            continue

        print(f"TPC {event.tpc_id}, Event {event.event_id}")

        if event.effq:
            print(f"  EffQ shape: {event.effq.data.shape}")
            print(f"  EffQ location shape: {event.effq.location.shape}")

        if event.current:
            print(f"  Current shape: {event.current.data.shape}")
            print(f"  Current location shape: {event.current.location.shape}")

        if not event.hits:
            print(f"  No hits, skipping event {event.event_id} TPC {event.tpc_id}")
            continue
        print(f"  Hits shape: {event.hits.data.shape}")
        print(f"  Hits location shape: {event.hits.location.shape}")

        if args.enable_wiener_roi and args.wiener_omega_c is None:
            raise ValueError(
                "--wiener-omega-c is required when --enable-wiener-roi is set."
            )
        result = process_event_deconvolution(
            event,
            readout_config,
            prepared_response,
            sigma_time=args.sigma,
            sigma_pixel=args.sigma_pxl,
            processor_cls=BurstSequenceProcessorV3,
            tau=readout_config.adc_hold_delay,
            npadbin=50,
            require_zero_local_offset=True,
            enable_wiener_roi=args.enable_wiener_roi,
            wiener_omega_c=args.wiener_omega_c,
            wiener_b=args.wiener_b,
            roi_threshold_sigma=args.roi_threshold_sigma,
            roi_merge_gap=args.roi_merge_gap,
            roi_expand=args.roi_expand,
        )
        boffset = shift_time_offset(
            result.hwf_block_offset,
            -readout_config.adc_hold_delay,
        )

        print("compensated", result.compensated_charge)
        print(
            f"smear_offset: {result.smear_offset}, boffset: {boffset}, "
            f"sum_deconv_q: {np.sum(result.deconv_q)}, "
            f"sum_deconv_q_gt1: {np.sum(result.deconv_q[result.deconv_q > 1])}, "
            f"sum_deconv_q_gt4: {np.sum(result.deconv_q[result.deconv_q > 4])}, "
            f"sum_smeared_true: {np.sum(result.smeared_true)}, "
            f"sum_effq_last: {np.sum(event.effq.data[:, -1])},"
            f"sum_hits_last: {np.sum(event.hits.data[:, -1])}"
        )
        if result.roi_mask is not None:
            print(
                f"  ROI: noise_rms={result.roi_noise_rms:.4g}, "
                f"threshold={result.roi_threshold_sigma * result.roi_noise_rms:.4g}, "
                f"n_roi_bins={int(result.roi_mask.sum())}, "
                f"sum_deconv_q_roi={float(np.sum(result.deconv_q_roi)):.4g}"
            )

        output_filename = (
            f"deconv_positron_v3_burst_{args.output_suffix}_"
            f"event_{event.tpc_id}_{event.event_id}.npz"
        )
        output_path = Path(args.output_dir).expanduser() / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {output_path}")
        geometry = loader.get_geometry(event.tpc_id)
        np.savez(
            output_path,
            **build_event_output_payload(
                event,
                geometry,
                readout_config,
                result,
                drift_length=prepared_response.drift_length,
                boffset_time_shift=-readout_config.adc_hold_delay,
                include_hwf_block=True,
            ),
        )

    geometry = loader.get_geometry(0)
    print(f"TPC 0 geometry: {geometry.lower} to {geometry.upper}")

    config = loader.get_readout_config()
    print(f"Time spacing: {config.time_spacing} μs")


if __name__ == "__main__":
    main()
