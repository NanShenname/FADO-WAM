#!/usr/bin/env bash
# Download full navtrain sensor data into workspace dataset symlinks target.
# Requires ~445GB under /data/nvme1/duku_dataset/navsim_data/sensor_blobs/trainval
set -euo pipefail

TARGET="/data/nvme1/duku_dataset/navsim_data"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/v2x_wam_env.sh"
PROJECT="${V2X_WAM_PROJECT}"

echo "This will download NAVSIM navtrain splits into: ${TARGET}"
echo "Estimated size: ~445GB. Press Ctrl+C to abort, or wait 5s..."
sleep 5

mkdir -p "${TARGET}"
cd "${TARGET}"

bash "${PROJECT}/download/download_navtrain.sh"

echo "Done. Verify with: bash ${PROJECT}/scripts/setup_train_env.sh"
