#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/v2x_wam_env.sh"
WORKSPACE="${V2X_WAM_WORKSPACE}"
PROJECT="${V2X_WAM_PROJECT}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${WORKSPACE}/dataset/maps"
export NAVSIM_EXP_ROOT="${WORKSPACE}/exp/"
export NAVSIM_DEVKIT_ROOT="${PROJECT}/navsim/"
export OPENSCENE_DATA_ROOT="${WORKSPACE}/dataset"
export HYDRA_FULL_ERROR=1
export HF_HOME="${WORKSPACE}/ckpts"
export PYTHONPATH="${PROJECT}"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

cd "${PROJECT}"

CONFIG_NAME=default_test
GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "[eval] workspace=${WORKSPACE}"
echo "[eval] gpu=${GPU}"
echo "[eval] checkpoint=${WORKSPACE}/ckpts/best.ckpt"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" ./navsim/planning/script/run_pdm_score_gpu.py \
agent=ImagineWorld_agent \
agent.config._target_=navsim.agents.ImagineWorld.configs.${CONFIG_NAME}.ImagineWorldConfigTest \
experiment_name=eval/WoTE/${CONFIG_NAME}/ \
split=test \
scene_filter=navtest
