#!/bin/bash
# Arm B minus the quiet hinge: split_trigger + acq_start:event + burst_tau,
# with the quiet term removed from the objective.
#
# The quiet term is the ONLY term in adopted_nb4 (-> terms: []), but sparse_nb1
# also carries the censor term (-> just drop the quiet line).  Handled per base.
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
FR=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
NFS=/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield
OUT=examples/analysis_output/angscan_noquiet
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
          -e "s|BuildMeasurement: {split_trigger: true}|BuildMeasurement: {split_trigger: true, acq_start: event, burst_tau: auto}|" \
          -e "s|WriteCharges:|WriteCharges:\n      embed_truth: true|" \
          "$REPO/configs/${base}.yaml" > "$cfg.tmp"
      if [ "$base" = adopted_nb4 ]; then
        # quiet was the only term -> empty the list
        python3 - "$cfg.tmp" "$cfg" <<'PYEOF'
import re, sys
s = open(sys.argv[1]).read()
s = re.sub(r"\n      terms:\n        - \{type: quiet, beta: 1\.0\}", "\n      terms: []", s)
open(sys.argv[2], "w").write(s)
PYEOF
      else
        # censor remains -> just drop the quiet entry
        grep -v -- '- {type: quiet, beta: 1.0}' "$cfg.tmp" > "$cfg"
      fi
      rm -f "$cfg.tmp"
      grep -q 'acq_start: event' "$cfg" || { echo "!! acq_start missing in $cfg"; exit 2; }
      grep -q 'type: quiet' "$cfg" && { echo "!! quiet still present in $cfg"; exit 2; }
      python3 -c "import yaml,sys; yaml.safe_load(open('$cfg'))" \
        || { echo "!! $cfg is not valid YAML"; exit 2; }
      if ! PYTHONPATH=src $PY -m unfoldlarpix.fwk.runner "$cfg" > "$OUT/log_${tag}.txt" 2>&1; then
        echo "!! DECONV FAIL $tag ($(tail -1 "$OUT/log_${tag}.txt"))"; nfail=$((nfail+1)); continue
      fi
      solved=$(ls -t "$OUT/$tag/${tag}_event_"*.npz 2>/dev/null | head -1)
      [ -z "$solved" ] && { echo "!! NO OUTPUT $tag"; nfail=$((nfail+1)); continue; }
      PYTHONPATH=src $PY examples/eval_deconv_metrics.py "$solved" --labels "$tag" \
        --universal-grid --deposit-shape gaussian --use-fitted-offsets \
        --json "$OUT/metrics_${tag}.json" >> "$OUT/log_${tag}.txt" 2>&1
      ndone=$((ndone+1)); echo "[$(date +%H:%M:%S)] done $tag  ($ndone)"
    done
  done
done
echo "ALL ANGSCAN DECONV DONE: $ndone ok, $nfail missing/failed -> $OUT"
