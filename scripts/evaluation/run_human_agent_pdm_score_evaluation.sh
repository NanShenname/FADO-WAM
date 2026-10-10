SPLIT=mini

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/v2x_wam_env.sh"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=human_agent \
experiment_name=human_agent \
split=$SPLIT \