#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from unfoldlarpix import (
    BurstSequenceProcessor,
    DataLoader,
    build_event_output_payload,
    prepare_field_response,
    process_event_deconvolution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deconvolve positron event data (V1)")
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
        default=0,
        help="TPC ID to process (default: 0)",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional suffix for the output NPZ filename",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for the output NPZ file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = DataLoader(args.input_file)
    readout_config = loader.get_readout_config()
    prepared_response = prepare_field_response(
        args.field_response,
        readout_config.adc_hold_delay,
        normalized=False,
    )

    for event in loader.iter_events():
        print(f"TPC {event.tpc_id}, Event {event.event_id}")
        if event.tpc_id != args.tpc_id:
            continue

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

        result = process_event_deconvolution(
            event,
            readout_config,
            prepared_response,
            sigma_time=args.sigma,
            sigma_pixel=args.sigma_pxl,
            processor_cls=BurstSequenceProcessor,
            tau=readout_config.adc_hold_delay + 24,
            npadbin=50,
        )

        print("compensated", result.compensated_charge)
        print(
            f"smear_offset: {result.smear_offset}, boffset: {result.hwf_block_offset}, "
            f"sum_deconv_q: {np.sum(result.deconv_q)}, "
            f"sum_deconv_q_gt1: {np.sum(result.deconv_q[result.deconv_q > 1])}, "
            f"sum_deconv_q_gt4: {np.sum(result.deconv_q[result.deconv_q > 4])}, "
            f"sum_smeared_true: {np.sum(result.smeared_true)}, "
            f"sum_effq_last: {np.sum(event.effq.data[:, -1])},"
            f"sum_hits_last: {np.sum(event.hits.data[:, -1])}"
        )

        geometry = loader.get_geometry(event.tpc_id)
        if args.output_suffix:
            output_filename = (
                f"deconv_positron_{args.output_suffix}_"
                f"event_{event.tpc_id}_{event.event_id}.npz"
            )
        else:
            output_filename = f"deconv_positron_event_{event.tpc_id}_{event.event_id}.npz"

        output_path = Path(args.output_dir).expanduser() / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {output_path}")
        np.savez(
            output_path,
            **build_event_output_payload(
                event,
                geometry,
                readout_config,
                result,
                drift_length=prepared_response.drift_length,
                include_hwf_block=True,
            ),
        )

    geometry = loader.get_geometry(0)
    print(f"TPC 0 geometry: {geometry.lower} to {geometry.upper}")

    config = loader.get_readout_config()
    print(f"Time spacing: {config.time_spacing} μs")


if __name__ == "__main__":
    main()
