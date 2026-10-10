source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/v2x_wam_env.sh"
SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
split=$SPLIT \
experiment_name=cv_agent \