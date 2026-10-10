#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/training/common_train_env.sh"

echo "==> DriveWorld-VLA 训练环境检查"
echo "workspace=${WORKSPACE}"

check_path() {
  if [[ -e "$1" ]]; then
    echo "  OK   $1"
  else
    echo "  MISS $1"
  fi
}

echo
echo "==> 1. 基础权重"
check_path "${WORKSPACE}/ckpts/resnet34.pth"
check_path "${WORKSPACE}/ReCogDrive-VLM-2B/model.safetensors"
check_path "${WORKSPACE}/dataset/extra_data/planning_vb/trajectory_anchors_256.npy"

echo
echo "==> 2. 训练数据 (navtrain / trainval)"
check_path "${WORKSPACE}/dataset/maps"
check_path "${WORKSPACE}/dataset/navsim_logs/trainval"
check_path "${WORKSPACE}/dataset/sensor_blobs/trainval"

echo
echo "==> 3. 训练专用数据 (必须生成)"
check_path "${WORKSPACE}/dataset/extra_data/planning_vb/formatted_pdm_score_256.npy"
check_path "${WORKSPACE}/exp/metric_cache_train/metadata"

echo
echo "==> 4. 各 stage checkpoint 槽位 (训练过程中产出)"
for ckpt in stage1.ckpt freeze_stage2.ckpt "epoch=19_stage3.ckpt" best.ckpt; do
  check_path "${WORKSPACE}/ckpts/${ckpt}"
done

echo
echo "==> 5. Python 依赖"
"${PYTHON}" - <<'PY'
import importlib
mods = ["torch", "pytorch_lightning", "transformers", "decord", "torchdiffeq", "einops", "navsim"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK   {m}")
    except Exception as e:
        print(f"  MISS {m}: {e}")
PY

echo
echo "==> 6. 当前可用 navtrain 样本量 (粗略)"
"${PYTHON}" - <<'PY'
import re
from pathlib import Path
navtrain = set(re.findall(r"^\s+- '([^']+)'", Path("/mnt/nvme1n1/nyc/V2X-WAM/navsim/planning/script/config/common/scene_filter/navtrain.yaml").read_text(), re.M))
logs = {p.stem for p in Path("/data/nvme1/duku_dataset/navsim_data/navsim_logs/trainval").glob("*.pkl")}
sensors = {p.name for p in Path("/data/nvme1/duku_dataset/navsim_data/sensor_blobs/trainval").iterdir() if p.is_dir()}
ready = navtrain & logs & sensors
print(f"  navtrain filter 条目: {len(navtrain)}")
print(f"  本地 trainval logs: {len(logs)}")
print(f"  本地 sensor 目录: {len(sensors)}")
print(f"  可训练交集: {len(ready)}")
if len(ready) < 10000:
    print("  警告: 完整 navtrain 约需 445GB，当前只是子集，完整复现需下载 navtrain")
PY

echo
echo "==> 7. 下一步命令"
cat <<EOF
# 生成训练 metric cache
bash ${PROJECT}/scripts/data/run_metric_caching_train.sh

# 生成 formatted_pdm_score_256.npy (依赖上一步)
bash ${PROJECT}/scripts/data/gen_formatted_pdm_score.sh

# 再次检查
bash ${PROJECT}/scripts/setup_train_env.sh

# 开始分 stage 训练
bash ${PROJECT}/scripts/training/run_train_stage1.sh
bash ${PROJECT}/scripts/training/run_train_stage2.sh
bash ${PROJECT}/scripts/training/run_train_stage3.sh
bash ${PROJECT}/scripts/training/run_train_stage4.sh
bash ${PROJECT}/scripts/training/run_train_stage5.sh
EOF
