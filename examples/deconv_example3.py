#!/usr/bin/env python3
import numpy as np

from unfoldlarpix import (
    BurstSequenceProcessor,
    DataLoader,
    build_event_output_payload,
    prepare_field_response,
    process_event_deconvolution,
    shift_time_offset,
)


INPUT_FILE = "data/pgun_muplus_3gev_tred_nburst4_noises.npz"
FIELD_RESPONSE_FILE = "data/fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npz"
SIGMA_TIME = 0.005
SIGMA_PIXEL = 0.2


def main() -> None:
    loader = DataLoader(INPUT_FILE)
    readout_config = loader.get_readout_config()
    prepared_response = prepare_field_response(
        FIELD_RESPONSE_FILE,
        readout_config.adc_hold_delay,
        normalized=False,
    )

    for event in loader.iter_events():
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

        result = process_event_deconvolution(
            event,
            readout_config,
            prepared_response,
            sigma_time=SIGMA_TIME,
            sigma_pixel=SIGMA_PIXEL,
            processor_cls=BurstSequenceProcessor,
            tau=readout_config.adc_hold_delay,
            npadbin=50,
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

        geometry = loader.get_geometry(event.tpc_id)
        np.savez(
            f"deconv_muplus_event_{event.tpc_id}_{event.event_id}.npz",
            **build_event_output_payload(
                event,
                geometry,
                readout_config,
                result,
                drift_length=prepared_response.drift_length,
                boffset_time_shift=-readout_config.adc_hold_delay,
            ),
        )

    geometry = loader.get_geometry(0)
    print(f"TPC 0 geometry: {geometry.lower} to {geometry.upper}")

    config = loader.get_readout_config()
    print(f"Time spacing: {config.time_spacing} μs")


if __name__ == "__main__":
    main()
