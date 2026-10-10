#!/usr/bin/env python3
"""Count Open-Observer samples and draw current/future BEV overlays."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from navsim.agents.ImagineWorld.configs.workspace_paths import SPD_ROOT, VLA_ROOT, WORKSPACE
from navsim.planning.training.open_observer_dataset import (
    HISTORY_TIMES_S,
    SPDOpenObserverDataset,
    VLAOpenObserverDataset,
    _nearest_index,
)

OUT = Path(WORKSPACE) / "exp" / "open_observer_wam" / "audit"
EXTENT = 40.0


def _stamp_errors(stamps, cursor, horizons):
    origin = stamps[cursor]
    errors = []
    for horizon in horizons:
        index = _nearest_index(stamps, origin + horizon, 0.2)
        errors.append(None if index is None else stamps[index] - origin)
    return errors


def _draw(path: Path, targets, times, title: str) -> None:
    size = 512
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = (24, 24, 24)

    def pixel(x, y):
        col = int((x + EXTENT) / (2 * EXTENT) * (size - 1))
        row = int((EXTENT - y) / (2 * EXTENT) * (size - 1))
        return col, row

    for meter in (-20, 0, 20):
        cv2.line(image, pixel(-EXTENT, meter), pixel(EXTENT, meter), (50, 50, 50), 1)
        cv2.line(image, pixel(meter, -EXTENT), pixel(meter, EXTENT), (50, 50, 50), 1)
    mask = targets["actor_mask"] > 0.5
    roles = targets["actor_role"]
    nodes = targets["actor_node"]
    has_ego = bool((roles == 1).any())
    boxes = targets["actor_box"]
    future = targets["future_actor_box"]
    future_valid = targets["future_actor_mask"] > 0.5
    for actor in range(int(mask.sum())):
        current = boxes[actor, :2].numpy()
        is_ego = int(roles[actor]) == 1 or (not has_ego and int(nodes[actor]) == 0)
        color = (0, 220, 255) if is_ego else (255, 180, 40)
        cv2.circle(image, pixel(float(current[0]), float(current[1])), 5, color, -1)
        yaw = float(boxes[actor, 2])
        tip = current + np.array([np.cos(yaw), np.sin(yaw)]) * 3.0
        cv2.arrowedLine(
            image,
            pixel(float(current[0]), float(current[1])),
            pixel(float(tip[0]), float(tip[1])),
            color,
            1,
            tipLength=0.35,
        )
        points = [pixel(float(current[0]), float(current[1]))]
        for step, moment in enumerate(times):
            if not bool(future_valid[step, actor]):
                continue
            xy = future[step, actor, :2].numpy()
            point = pixel(float(xy[0]), float(xy[1]))
            points.append(point)
            shade = int(80 + 175 * (step + 1) / len(times))
            cv2.circle(image, point, 3, (0, shade, shade) if is_ego else (shade, shade, 255), -1)
        if len(points) > 1:
            cv2.polylines(image, [np.array(points, dtype=np.int32)], False, color, 1)
    cv2.putText(image, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    legend = "yellow=ego  blue=other  arrow=heading" if has_ego else "yellow=default query  blue=other supervised  arrow=heading"
    cv2.putText(image, legend, (8, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"horizons_s": [0.5, 1.0, 1.5, 2.0, 2.5], "history_s": list(HISTORY_TIMES_S), "splits": {}}
    overlays = []
    for name, factory in (
        ("vla", lambda split: VLAOpenObserverDataset(VLA_ROOT, split=split)),
        ("spd", lambda split: SPDOpenObserverDataset(SPD_ROOT, split=split)),
    ):
        report["splits"][name] = {}
        for split in ("train", "val", "test"):
            data = factory(split)
            entry = {
                "samples": len(data),
                "boundary_dropped": int(data.boundary_dropped),
            }
            if name == "spd":
                entry["official_split_train_val_shared_sequences"] = data.shared_sequences
                deltas = []
                for sequence, cursor in data.samples[:200]:
                    rows = data.sequences[sequence]
                    stamps = [row["time_s"] for row in rows]
                    deltas.append(_stamp_errors(stamps, cursor, data.horizons_s))
                if deltas:
                    array = np.asarray(deltas, dtype=np.float64)
                    entry["future_time_mean_s"] = array.mean(axis=0).round(4).tolist()
                    entry["future_time_max_abs_error_s"] = np.max(np.abs(array - np.array(data.horizons_s)), axis=0).round(4).tolist()
            else:
                deltas = []
                for weather, scene, current in data.samples[:200]:
                    timeline = data._scene_timeline(weather, scene, 1)
                    stamps = [row["timestamp_s"] for row in timeline]
                    deltas.append(_stamp_errors(stamps, current, data.horizons_s))
                if deltas:
                    array = np.asarray(deltas, dtype=np.float64)
                    entry["future_time_mean_s"] = array.mean(axis=0).round(4).tolist()
                    entry["future_time_max_abs_error_s"] = np.max(np.abs(array - np.array(data.horizons_s)), axis=0).round(4).tolist()
            report["splits"][name][split] = entry
            print(name, split, entry["samples"], "dropped", entry["boundary_dropped"], flush=True)
            if split in {"train", "val"} and len(data):
                _, targets = data[0]
                image_path = OUT / f"{name}_{split}_bev.png"
                _draw(image_path, targets, data.horizons_s, f"{name} {split}  t=0 plus {list(data.horizons_s)}")
                overlays.append(str(image_path))
    report["overlays"] = overlays
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
