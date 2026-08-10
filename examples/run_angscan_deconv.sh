#!/bin/bash
# Deconv + metrics + plots for the muon/positron angle x burst scan.
# 8 samples (mu/pos x ang 00/25/50/75) x nburst {1,2,4,8,16,64} = 48 configs.
# nburst=1 -> sparse_nb1 (censor, 600 it); else adopted_nb4 (150 it). FR = v2a_full 25x25.
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
FR=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
NFS=/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield
OUT=examples/analysis_output/angscan
mkdir -p "$OUT"
ndone=0; nfail=0
for pp in mu positron; do
  [ "$pp" = mu ] && tp=mu || tp=pos
  for ang in 00 25 50 75; do
    for N in 1 2 4 8 16 64; do
      data="$NFS/pgun_${pp}_3gev_ang${ang}_tred_nb${N}.npz"
      if [ ! -f "$data" ]; then echo "!! MISSING $(basename "$data")"; nfail=$((nfail+1)); continue; fi
      [ "$N" -eq 1 ] && base=sparse_nb1 || base=adopted_nb4
      tag=${tp}_a${ang}_nb${N}
      cfg="$OUT/job_${tag}.yaml"
      sed -e "s|input: .*npz|input: $data|" \
          -e "s|response: [^}]*|response: $FR|" \
          -e "s|out_dir: .*|out_dir: $OUT/$tag|" \
          -e "s|prefix: .*|prefix: $tag|" \
          -e "s|WriteCharges:|WriteCharges:\n      embed_truth: true|" \
          "$REPO/configs/${base}.yaml" > "$cfg"
      if ! PYTHONPATH=src $PY -m unfoldlarpix.fwk.runner "$cfg" > "$OUT/log_${tag}.txt" 2>&1; then
        echo "!! DECONV FAIL $tag ($(tail -1 "$OUT/log_${tag}.txt"))"; nfail=$((nfail+1)); continue
      fi
      solved=$(ls -t "$OUT/$tag/${tag}_event_"*.npz 2>/dev/null | head -1)
      [ -z "$solved" ] && { echo "!! NO OUTPUT $tag"; nfail=$((nfail+1)); continue; }
      PYTHONPATH=src $PY examples/eval_deconv_metrics.py "$solved" --labels "$tag" \
        --universal-grid --deposit-shape gaussian --use-fitted-offsets \
        --json "$OUT/metrics_${tag}.json" >> "$OUT/log_${tag}.txt" 2>&1
      PYTHONPATH=src $PY examples/report_analysis.py "$solved" "$tag" "$OUT" >> "$OUT/log_${tag}.txt" 2>&1
      ndone=$((ndone+1)); echo "[$(date +%H:%M:%S)] done $tag  ($ndone)"
    done
  done
done
echo "ALL ANGSCAN DECONV DONE: $ndone ok, $nfail missing/failed -> $OUT"
