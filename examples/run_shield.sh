#!/bin/bash
uv run python run_analysis.py  --input-file data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_selftrigger_shield.npz --sigmas 0.005 --sigma-pxls 0.2 --dest-dir analysis_20260318_tpc0/ --plot-dir analysis_20260318_tpc0/ --plot-threshold 0.5 --thresholds 0.5 --steps 1 2 3 4 --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_shield_500V_25x25pixel_tred.npz --versions v2

uv run python run_analysis.py  --input-file data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_selftrigger_shield.npz --sigmas 0.004 --sigma-pxls 0.2 --dest-dir analysis_20260318_tpc0/ --plot-dir analysis_20260318_tpc0/ --plot-threshold 0.5 --thresholds 0.5 --steps 1 2 3 4 --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_shield_500V_25x25pixel_tred.npz --versions v2

uv run python run_analysis.py  --input-file data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_selftrigger_shield.npz --sigmas 0.003 --sigma-pxls 0.2 --dest-dir analysis_20260318_tpc0/ --plot-dir analysis_20260318_tpc0/ --plot-threshold 0.5 --thresholds 0.5 --steps 1 2 3 4 --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_shield_500V_25x25pixel_tred.npz --versions v2
