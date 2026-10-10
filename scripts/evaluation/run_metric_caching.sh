source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/v2x_wam_env.sh"
SPLIT=trainval
# SPLIT=trainval
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/dataset/maps"
export NAVSIM_EXP_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/exp/"
export NAVSIM_DEVKIT_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/"
export OPENSCENE_DATA_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/dataset"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path='/e2e-data/evad-tech-vla/liulin/WoTE/exp/metric_cache_train'
