#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/training/common_train_env.sh"

mkdir -p "${NAVSIM_EXP_ROOT}/metric_cache_train"

cd "${PROJECT}"

echo "[metric-cache-train] output=${NAVSIM_EXP_ROOT}/metric_cache_train"
echo "[metric-cache-train] split=trainval scene_filter=navtrain"

"${PYTHON}" ./navsim/planning/script/run_metric_caching.py \
  split=trainval \
  scene_filter=navtrain \
  cache.cache_path="${NAVSIM_EXP_ROOT}/metric_cache_train"
