#!/usr/bin/env python3
"""Generate formatted_pdm_score_256.npy for DriveWorld-VLA training targets."""

import lzma
import os
import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT = Path("/mnt/nvme1n1/nyc/V2X-WAM")
sys.path.insert(0, str(PROJECT))

WORKSPACE = Path(os.environ.get("DRIVEWORLD_WORKSPACE", "/mnt/nvme1n1/nyc/driveworld_workspace"))
OUTPUT_NPY = Path(
    os.environ.get(
        "DRIVEWORLD_PDM_OUTPUT",
        WORKSPACE / "dataset/extra_data/planning_vb/formatted_pdm_score_256.npy",
    )
)
ANCHORS_NPY = WORKSPACE / "dataset/extra_data/planning_vb/trajectory_anchors_256.npy"
METRIC_CACHE = WORKSPACE / "exp/metric_cache_train"

NUM_CLUSTERS = 256
# Must match metric_cache proposal_traj_num_poses=40 (4s @ 10Hz).
NUM_HORIZONS = 4

os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(WORKSPACE / "dataset/maps"))
os.environ.setdefault("NAVSIM_EXP_ROOT", str(WORKSPACE / "exp"))
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(WORKSPACE / "navsim"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(WORKSPACE / "dataset"))

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score_multi_trajs
from navsim.planning.metric_caching.metric_cache import MetricCache


def load_scoring_cfg():
    cfg_path = PROJECT / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    return cfg


def load_data_cfg():
    return {
        "metric_cache_path": str(METRIC_CACHE),
        "split": "trainval",
        "navsim_log_path": str(WORKSPACE / "dataset/navsim_logs/trainval"),
        "sensor_blobs_path": str(WORKSPACE / "dataset/sensor_blobs/trainval"),
    }


def load_scene_filter():
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    if not GlobalHydra.instance().is_initialized():
        hydra.initialize(
            config_path="../../navsim/planning/script/config/common/scene_filter",
        )
    return instantiate(hydra.compose(config_name="navtrain"))


def main():
    if not ANCHORS_NPY.exists():
        raise FileNotFoundError(f"Missing anchors: {ANCHORS_NPY}")
    if not (METRIC_CACHE / "metadata").exists():
        raise FileNotFoundError(
            f"Missing train metric cache metadata: {METRIC_CACHE}/metadata\n"
            "Run: bash scripts/data/run_metric_caching_train.sh"
        )

    cfg = load_scoring_cfg()
    data_cfg = load_data_cfg()
    scene_filter = load_scene_filter()
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)

    proposal_sampling = simulator.proposal_sampling
    proposal_sampling.time_horizon = NUM_HORIZONS
    proposal_sampling.num_poses = int(NUM_HORIZONS * 10)
    scorer.proposal_sampling = proposal_sampling
    simulator.proposal_sampling = proposal_sampling

    predefined_trajectories = np.load(ANCHORS_NPY)
    if predefined_trajectories.shape[0] != NUM_CLUSTERS:
        raise ValueError(f"Expected {NUM_CLUSTERS} anchors, got {predefined_trajectories.shape}")

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(data_cfg["navsim_log_path"]),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(METRIC_CACHE)
    tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    print(f"Generating PDM scores for {len(tokens)} tokens")

    score_rows = []
    score_dict = {}
    for token in tqdm(tokens, desc="formatted_pdm_score"):
        row = {"token": token, "valid": True}
        try:
            with lzma.open(metric_cache_loader.metric_cache_paths[token], "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            pdm_result = pdm_score_multi_trajs(
                metric_cache=metric_cache,
                model_trajectory_list=predefined_trajectories,
                future_sampling=proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            payload = {"trajectory_scores": [asdict(pdm_result)], "valid": True}
            score_dict[token] = payload
            row.update(payload)
        except Exception as exc:
            print(f"failed {token}: {exc}")
            row["valid"] = False
        score_rows.append(row)

    OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_NPY, score_dict, allow_pickle=True)
    print(f"Saved {len(score_dict)} entries -> {OUTPUT_NPY}")

    out_dir = WORKSPACE / "exp/pdm_score_gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pd.DataFrame(score_rows).to_csv(out_dir / f"{ts}.csv", index=False)


if __name__ == "__main__":
    main()
