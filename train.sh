#!/usr/bin/env bash
# Train CoDeRNet with the paper settings (single GPU, deterministic, seed 42).
#
# Usage:
#   bash train.sh levir <LEVIR_MCI_ROOT>/images <LIST_PATH>                      [train.py options]
#   bash train.sh whu   <WHU_CDC_ROOT>/images   <WHU_CDC_ROOT>/whuCCcaptions.json [train.py options]
#
#   <LIST_PATH> is the output folder of preprocess_levir_mci.py (e.g. ./data/LEVIR_MCI).
#   Any further options are passed through to train.py, e.g. --gpu_id 1 --savepath ./Results/run1/
#   Outputs go to ./Results/levir_mci/ or ./Results/whu_cdc/ unless --savepath is given.
set -euo pipefail

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}
[ "$#" -ge 3 ] || usage

DATASET=$1; DATA_FOLDER=$2; ANNOT=$3; shift 3
case "$DATASET" in
    levir) DATA_ARGS=(--list_path "$ANNOT") ;;
    whu)   DATA_ARGS=(--caption_json "$ANNOT") ;;
    *)     echo "unknown dataset '$DATASET' (expected: levir | whu)" >&2; usage ;;
esac

# Determinism: both variables must be set before the Python process starts.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

python "$(dirname "$0")/train.py" \
    --dataset "$DATASET" \
    --data_folder "$DATA_FOLDER" \
    "${DATA_ARGS[@]}" \
    --deterministic --seed 42 \
    "$@"
