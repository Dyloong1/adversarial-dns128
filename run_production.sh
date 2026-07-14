#!/usr/bin/env bash
# run_production.sh — one command to generate the 128^3 Re_lambda~37 dataset and verify it.
#
# Usage:
#   bash run_production.sh                 # default: 8 seeds x 120 frames
#   SEEDS=8 FRAMES=120 bash run_production.sh
#   OUT=data/mydataset bash run_production.sh
#
# On 4x H200 you can run 4 seeds in parallel (one per GPU): see the PARALLEL block below.
# Requires: torch>=2.0 numpy matplotlib pyyaml, a CUDA GPU with fp64.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE"
export KMP_DUPLICATE_LIB_OK=TRUE          # Windows OpenMP guard; harmless on Linux
PY="${PYTHON:-python}"

SEEDS="${SEEDS:-8}"
FRAMES="${FRAMES:-120}"
OUT="${OUT:-data/dns128_relam37}"

echo "=============================================================="
echo " adversarial_dns128 production"
echo "   seeds=$SEEDS  frames=$FRAMES  out=$OUT"
echo "   config: 128^3 fp64, Re_lambda~37, k_f=4, Class I (k_maxeta~1.6)"
echo "=============================================================="

echo "[0/2] self-check ..."
"$PY" selfcheck.py

echo "[1/2] generating dataset ($SEEDS seeds x $FRAMES frames) ..."
# --- serial (default; one 128^3 run already saturates a single GPU) ---
"$PY" generate_dataset.py --seeds "$SEEDS" --frames "$FRAMES" --out "$OUT"

# --- PARALLEL on multi-GPU (e.g. 4x H200): uncomment to run one seed per GPU ---
# NGPU="${NGPU:-4}"
# for sd in $(seq 0 $((SEEDS-1))); do
#   CUDA_VISIBLE_DEVICES=$((sd % NGPU)) \
#     "$PY" -c "import sys; sys.path.insert(0,'.'); import generate_dataset as g; \
#               from pathlib import Path; g.run_seed($sd, $FRAMES, Path('$OUT'))" &
#   # throttle to NGPU concurrent jobs
#   if (( (sd+1) % NGPU == 0 )); then wait; fi
# done
# wait

echo "[2/2] A+D acceptance ..."
"$PY" eval/eval_ad.py "$OUT"

echo "=============================================================="
echo " DONE. Dataset at: $OUT   (verdict + metrics in $OUT/eval_AD.json)"
echo " Adversarial API:   from advance import advance;  x_next = advance(x)"
echo "=============================================================="
