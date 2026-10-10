#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/training/common_train_env.sh"

OUT_NPY="${WORKSPACE}/dataset/extra_data/planning_vb/formatted_pdm_score_256.npy"
mkdir -p "$(dirname "${OUT_NPY}")"

if [[ ! -d "${NAVSIM_EXP_ROOT}/metric_cache_train/metadata" ]]; then
  echo "ERROR: 训练 metric cache 不存在，请先运行:"
  echo "  bash ${PROJECT}/scripts/data/run_metric_caching_train.sh"
  exit 1
fi

cd "${PROJECT}"

echo "[gen-pdm-score] anchors=${WORKSPACE}/dataset/extra_data/planning_vb/trajectory_anchors_256.npy"
echo "[gen-pdm-score] metric_cache=${NAVSIM_EXP_ROOT}/metric_cache_train"
echo "[gen-pdm-score] output=${OUT_NPY}"

DRIVEWORLD_WORKSPACE="${WORKSPACE}" \
DRIVEWORLD_PDM_OUTPUT="${OUT_NPY}" \
"${PYTHON}" "${PROJECT}/scripts/data/gen_formatted_pdm_score.py"

if [[ -f "${OUT_NPY}" ]]; then
  echo "[gen-pdm-score] done: ${OUT_NPY}"
else
  echo "[gen-pdm-score] FAILED: output not found"
  exit 1
fi
