#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/v2x_wam_env.sh"
WORKSPACE="${V2X_WAM_WORKSPACE}"
PROJECT="${V2X_WAM_PROJECT}"

echo "==> 1. 检查工作区"
for path in \
  "${WORKSPACE}/dataset/maps" \
  "${WORKSPACE}/dataset/navsim_logs" \
  "${WORKSPACE}/dataset/sensor_blobs" \
  "${WORKSPACE}/exp/metric_cache" \
  "${WORKSPACE}/ckpts/resnet34.pth"
do
  if [[ -e "${path}" ]]; then
    echo "  OK  ${path}"
  else
    echo "  MISSING  ${path}"
  fi
done

echo "==> 2. 检查评测必需 checkpoint"
for path in \
  "${WORKSPACE}/ckpts/best.ckpt" \
  "${WORKSPACE}/ckpts/expert_imagineworld.ckpt" \
  "${WORKSPACE}/ReCogDrive-VLM-2B" \
  "${WORKSPACE}/dataset/extra_data/planning_vb/trajectory_anchors_256.npy" \
  "${WORKSPACE}/dataset/extra_data/planning_vb/formatted_pdm_score_256.npy"
do
  if [[ -e "${path}" ]]; then
    echo "  OK  ${path}"
  else
    echo "  MISSING  ${path}"
  fi
done

echo "==> 3. conda 环境 v2x-wam"
if conda env list | awk '{print $1}' | grep -qx v2x-wam; then
  echo "  OK  ${V2X_WAM_PYTHON}"
else
  echo "  MISSING  v2x-wam"
fi

echo "==> 5. 建议写入 ~/.bashrc 的环境变量"
cat <<EOF
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${WORKSPACE}/dataset/maps"
export NAVSIM_EXP_ROOT="${WORKSPACE}/exp"
export NAVSIM_DEVKIT_ROOT="${PROJECT}/navsim"
export OPENSCENE_DATA_ROOT="${WORKSPACE}/dataset"
export HF_HOME="${WORKSPACE}/ckpts"
export HYDRA_FULL_ERROR=1
EOF

echo "==> 6. 运行评测"
echo "CUDA_VISIBLE_DEVICES=4 bash ${PROJECT}/scripts/evaluation/eval_driveworld_vla.sh"
