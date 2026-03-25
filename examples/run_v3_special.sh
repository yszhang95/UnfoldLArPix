 # TAG="v3_wf_sp005_spp18"
 # SigmaT="0.005"
 # SigmaP="0.18"
 # InputFile="data/pgun_positron_3gev_tred_noises_effq_nt1_wf.npz"
 # uv run python  deconv_positron_v3.py \
 #     --sigma ${SigmaT} --sigma-pxl $SigmaP \
 #     --input-file $InputFile \
 #     --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz \
 # # json
 # uv run python deconv_xyz.py --threshold 0.6 --output-dir analysis_20260318_tpc0/ \
 #     --prefix "${TAG}_t0p5" --smeared-prefix "${TAG}_t0p5_smeared" \
 #     --event-id 0 --tpc-id 0 \
 #     deconv_positron_v3_event_0_0.npz
 # # generate histograms
 # uv run python  plot_proj.py --prefix "analysis_20260318_tpc0/${TAG}" --threshold 0.5 deconv_positron_v3_event_0_0.npz
 # # rename
 # mv deconv_positron_v3_event_0_0.npz "analysis_20260318_tpc0/deconv_positron_${TAG}_event_0_0.npz"

TAG="v3_wf_sp004_spp2"
SigmaT="0.004"
SigmaP="0.2"
InputFile="data/pgun_positron_3gev_tred_noises_effq_nt1_wf.npz"
uv run python  deconv_positron_v3.py \
    --sigma ${SigmaT} --sigma-pxl $SigmaP \
    --input-file $InputFile \
    --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz \
# json
uv run python deconv_xyz.py --threshold 0.5 --output-dir analysis_20260318_tpc0/ \
    --prefix "${TAG}_t0p5" --smeared-prefix "${TAG}_t0p5_smeared" \
    --event-id 0 --tpc-id 0 \
    deconv_positron_v3_event_0_0.npz
# generate histograms
uv run python  plot_proj.py --prefix "analysis_20260318_tpc0/${TAG}" --threshold 0.5 deconv_positron_v3_event_0_0.npz
# rename
mv deconv_positron_v3_event_0_0.npz "analysis_20260318_tpc0/deconv_positron_${TAG}_event_0_0.npz"
