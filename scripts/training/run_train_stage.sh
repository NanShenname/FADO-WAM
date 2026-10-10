#!/usr/bin/env bash
# Run one DriveWorld-VLA training stage.
# Usage: run_train_stage.sh <stage_number> [extra hydra args...]
set -euo pipefail

STAGE="${1:?stage number required (1-5)}"
shift || true

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_train_env.sh"

declare -A CONFIG_NAME=(
  [1]="default_stage1"
  [2]="default_stage2"
  [3]="default_stage3"
  [4]="default_stage4"
  [5]="default_stage5"
)
declare -A CONFIG_CLASS=(
  [1]="ImagineWorldConfig"
  [2]="ImagineWorldConfigStage4"
  [3]="ImagineWorldConfigStage3"
  [4]="ImagineWorldConfigStage4"
  [5]="ImagineWorldConfigStage5"
)
declare -A BATCH_SIZE=(
  [1]=24
  [2]=36
  [3]=24
  [4]=36
  [5]=36
)
declare -A MAX_EPOCHS=(
  [1]=100
  [2]=20
  [3]=30
  [4]=30
  [5]=30
)

CFG="${CONFIG_NAME[$STAGE]}"
CLS="${CONFIG_CLASS[$STAGE]}"
BS="${BATCH_SIZE[$STAGE]}"
EPOCHS="${MAX_EPOCHS[$STAGE]}"

cd "${PROJECT}"

echo "[train] stage=${STAGE} config=${CFG} batch=${BS} epochs=${EPOCHS}"
echo "[train] output=${NAVSIM_EXP_ROOT}/WoTE/${CFG}/"

"${PYTHON}" ./navsim/planning/script/run_training.py \
  agent=ImagineWorld_agent \
  "agent.config._target_=navsim.agents.ImagineWorld.configs.${CFG}.${CLS}" \
  "experiment_name=WoTE/${CFG}/" \
  scene_filter=navtrain \
  split=trainval \
  "dataloader.params.batch_size=${BS}" \
  "trainer.params.max_epochs=${EPOCHS}" \
  "$@"
