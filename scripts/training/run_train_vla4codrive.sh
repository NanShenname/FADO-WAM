#!/usr/bin/env bash
# Stage-1 ImagineWorld training on VLA4CoDrive.
# Lidar BEV is zeros and map/PDM losses are off; see vla4codrive_dataset.py.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/v2x_wam_env.sh"
PROJECT="${V2X_WAM_PROJECT}"
WORKSPACE="${V2X_WAM_WORKSPACE}"
DATA_ROOT="${V2X_WAM_DATA}"

export PYTHONPATH="${PROJECT}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${WORKSPACE}/dataset/maps"
export NAVSIM_EXP_ROOT="${WORKSPACE}/exp/"
export NAVSIM_DEVKIT_ROOT="${PROJECT}/navsim/"
export OPENSCENE_DATA_ROOT="${WORKSPACE}/dataset"
export HF_HOME="${WORKSPACE}/ckpts"
export HYDRA_FULL_ERROR=1

cd "${PROJECT}"
exec "${PYTHON}" \
  navsim/planning/script/run_training_vla4codrive.py \
  --data-root "${DATA_ROOT}" \
  "$@"
