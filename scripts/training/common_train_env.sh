#!/usr/bin/env bash
# Shared env for V2X-WAM training on this server.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/v2x_wam_env.sh"
WORKSPACE="${V2X_WAM_WORKSPACE}"
PROJECT="${V2X_WAM_PROJECT}"

export PYTHONPATH="${PROJECT}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${WORKSPACE}/dataset/maps"
export NAVSIM_EXP_ROOT="${WORKSPACE}/exp/"
export NAVSIM_DEVKIT_ROOT="${PROJECT}/navsim/"
export OPENSCENE_DATA_ROOT="${WORKSPACE}/dataset"
export HYDRA_FULL_ERROR=1
export HF_HOME="${WORKSPACE}/ckpts"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
