#!/bin/bash
# Re-evaluate the whole scan with the corrected transverse smearing width.
#
# sigma_pixel is a FREQUENCY-domain parameter: real-space sigma = 1/(2 pi sigma).
# The shipped 0.2 means 0.796 pixels -- five times the physical transverse
# diffusion (0.16 px) and far beyond the pixel quantisation limit
# (pitch/sqrt(12) = 0.289 px).  0.5 -> 0.318 px, the quantisation limit, and the
# metrics are flat from there down to 0.08 px.
#
# Both truth smearing (WriteCharges) and the reco deposit (eval --sigma-pxl)
# must use the same value; they share the exp(-f^2/2 sigma^2) convention.
#
# usage: run_angscan_deconv_sigpx.sh <arm> <sigma_pxl>
#   arm = nosplit | tau
cd "$(dirname "$0")/.."
REPO=$(pwd)
ARM=${1:-tau}; SP=${2:-0.5}
PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
FR=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
NFS=/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield
if [ "$ARM" = tau ]; then
  BM='BuildMeasurement: {split_trigger: true, acq_start: event, burst_tau: auto}'
else
  BM='BuildMeasurement: {split_trigger: false, acq_start: event}'
fi
OUT=examples/analysis_output/angscan_${ARM}_sp${SP}
mkdir -p "$OUT"
ndone=0; nfail=0
for pp in mu positron; do
  [ "$pp" = mu ] && tp=mu || tp=pos
  for ang in 00 25 50 75; do
    for N in 1 2 4 8 16 64; do
      data="$NFS/pgun_${pp}_3gev_ang${ang}_tred_nb${N}.npz"
      [ -f "$data" ] || { echo "!! MISSING $(basename "$data")"; nfail=$((nfail+1)); continue; }
      [ "$N" -eq 1 ] && base=sparse_nb1 || base=adopted_nb4
      tag=${tp}_a${ang}_nb${N}
      cfg="$OUT/job_${tag}.yaml"
      sed -e "s|input: .*npz|input: $data|" \
          -e "s|response: [^}]*|response: $FR|" \
          -e "s|out_dir: .*|out_dir: $OUT/$tag|" \
          -e "s|prefix: .*|prefix: $tag|" \
          -e "s|BuildMeasurement: {split_trigger: true}|$BM|" \
          -e "s|WriteCharges:|WriteCharges:\n      embed_truth: true\n      sigma_pixel: $SP|" \
          "$REPO/configs/${base}.yaml" > "$cfg"
      grep -q "sigma_pixel: $SP" "$cfg" || { echo "!! sigma_pixel missing in $cfg"; exit 2; }
      if ! PYTHONPATH=src $PY -m unfoldlarpix.fwk.runner "$cfg" > "$OUT/log_${tag}.txt" 2>&1; then
        echo "!! DECONV FAIL $tag ($(tail -1 "$OUT/log_${tag}.txt"))"; nfail=$((nfail+1)); continue
      fi
      solved=$(ls -t "$OUT/$tag/${tag}_event_"*.npz 2>/dev/null | head -1)
      [ -z "$solved" ] && { echo "!! NO OUTPUT $tag"; nfail=$((nfail+1)); continue; }
      PYTHONPATH=src $PY examples/eval_deconv_metrics.py "$solved" --labels "$tag" \
        --universal-grid --deposit-shape gaussian --use-fitted-offsets \
        --sigma-pxl "$SP" --json "$OUT/metrics_${tag}.json" >> "$OUT/log_${tag}.txt" 2>&1
      ndone=$((ndone+1)); echo "[$(date +%H:%M:%S)] done $tag  ($ndone)"
    done
  done
done
echo "ALL DONE arm=$ARM sigma_pxl=$SP: $ndone ok, $nfail failed -> $OUT"
