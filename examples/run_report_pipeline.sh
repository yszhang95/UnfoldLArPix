#!/bin/bash
# Deconv + metrics + plots for every regenerated (reset-fix) positron dataset
# that we analyze (256-burst datasets are intentionally skipped).
# Each job: <resetfix-datafile>  <base-config>  <full|shield>  <tag>
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
FR_FULL=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
FR_SHIELD=/srv/storage1/yousen/tred_workspace/response_44_v2a_shield_500V_25x25pixel_tred.npz
NFS=/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield
OUT=examples/analysis_output/report_resetfix
mkdir -p "$OUT"

jobs=(
  "pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4_resetfix.npz            adopted_nb4 full   nb4"
  "pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst16_resetfix.npz           adopted_nb4 full   nb16"
  "pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst64_resetfix.npz           adopted_nb4 full   nb64"
  "pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4_fastadc0p5_resetfix.npz adopted_nb4 full   nb4fast"
  "pgun_positron_3gev_tred_noises_effq_nt1_thres5k_selftrigger_resetfix.npz        sparse_nb1  full   nb1self"
  "pgun_positron_3gev_tred_noises_effq_nt1_nburst4_shield_resetfix.npz             adopted_nb4 shield nb4sh"
  "pgun_positron_3gev_tred_noises_effq_nt1_nburst4_shield_reset0_resetfix.npz      adopted_nb4 shield nb4sh_r0"
  "pgun_positron_3gev_tred_noises_effq_nt1_nburst8_shield_reset0_resetfix.npz      adopted_nb4 shield nb8sh_r0"
  "pgun_positron_3gev_tred_noises_effq_nt1_selftrigger_shield_resetfix.npz         sparse_nb1  shield nb1self_sh"
  "pgun_positron_3gev_tred_noises_effq_nt1_selftrigger_shield_reset0_resetfix.npz  sparse_nb1  shield nb1self_sh_r0"
)

for job in "${jobs[@]}"; do
  read -r data base fr tag <<< "$job"
  [ "$fr" = shield ] && FR=$FR_SHIELD || FR=$FR_FULL
  if [ ! -f "$NFS/$data" ]; then echo "!! MISSING DATA $data — skip $tag"; continue; fi
  echo "######## $tag  ($base, $fr FR) ########"
  cfg="$OUT/job_${tag}.yaml"
  sed -e "s|input: .*npz|input: $NFS/$data|" \
      -e "s|response: [^}]*|response: $FR|" \
      -e "s|out_dir: .*|out_dir: $OUT/$tag|" \
      -e "s|prefix: .*|prefix: $tag|" \
      -e "s|WriteCharges:|WriteCharges:\n      embed_truth: true|" \
      "$REPO/configs/${base}.yaml" > "$cfg"
  PYTHONPATH=src $PY -m unfoldlarpix.fwk.runner "$cfg" 2>&1 | tail -4
  solved=$(ls -t "$OUT/$tag/${tag}_event_"*.npz | head -1)
  PYTHONPATH=src $PY examples/eval_deconv_metrics.py "$solved" --labels "$tag" \
    --universal-grid --deposit-shape gaussian --use-fitted-offsets \
    --json "$OUT/metrics_${tag}.json" 2>&1 | tail -3
  PYTHONPATH=src $PY examples/report_analysis.py "$solved" "$tag" "$OUT" 2>&1 | tail -2
  echo ""
done
echo "ALL ANALYSIS DONE -> $OUT"
