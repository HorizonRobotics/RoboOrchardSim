# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Three ways to load a LeRobot v2 dataset produced by convert.py.

Run::

    python3 -m tools.mcap_to_lerobot.example_load \
        --dataset-root logs/lerobot_test \
        --repo-id robo_orchard/piperx_pick_attribute
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def way_1_lerobot_api(root: Path, repo_id: str) -> None:
    """The official way. Returns torch.Tensor per frame, handles video decode.

    Caveat: video decode goes through torchcodec.  If your environment has a
    broken libnvrtc, reading image fields will raise — use way 2/3 instead.
    """
    print("\n=== way 1: LeRobotDataset (official API) ===")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=root)
    print(f"  episodes : {ds.meta.total_episodes}")
    print(f"  frs   : {ds.meta.total_frames}")
    print(f"  fps      : {ds.fps}")

    sample = ds[0]
    obs_state = sample["observation.state"]
    print(f"  sample[0] keys                  : {sorted(sample.keys())}")
    print(f"  observation.state shape         : {obs_state.shape}")
    print(f"  observation.state[:7]           : {obs_state[:7].tolist()}")
    print(f"  action shape                    : {sample['action'].shape}")
    print(f"  task                            : {sample['task']!r}")

    # Trying an image is the part most likely to fail on a borked env.
    try:
        img = sample["observation.images.static_camera"]
        img_summary = f"{type(img).__name__} {tuple(img.shape)}"
        print(f"  observation.images.static_camera: {img_summary}")
    except Exception as exc:
        print(f"  (image load failed: {exc.__class__.__name__}: {exc})")


def way_2_parquet_direct(root: Path) -> None:
    """Read joint/action/timestamp columns directly from parquet.

    No torchcodec, no PIL — works even if video decoding is broken.
    Use this for inspecting joints or building a custom loader.
    """
    print("\n=== way 2: pandas.read_parquet (no video decode) ===")
    ep0 = root / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(ep0)
    print(f"  file      : {ep0}")
    print(f"  rows      : {len(df)}")
    print(f"  columns   : {df.columns.tolist()}")
    print("  first row :")
    obs_state = np.asarray(df["observation.state"].iloc[0])
    print(f"    observation.state : {obs_state}")
    print(f"    action            : {np.asarray(df['action'].iloc[0])}")
    print(f"    timestamp         : {df['timestamp'].iloc[0]}")
    print(f"    frame_index       : {df['frame_index'].iloc[0]}")
    print(f"    task_index        : {df['task_index'].iloc[0]}")


def way_3_metadata_only(root: Path) -> None:
    """Read meta/info.json + episodes.jsonl + tasks.jsonl directly."""
    print("\n=== way 3: meta files (info.json + jsonl) ===")
    info = json.loads((root / "meta" / "info.json").read_text())
    print(f"  total_episodes : {info['total_episodes']}")
    print(f"  total_frames   : {info['total_frames']}")
    print(f"  fps            : {info['fps']}")
    print(f"  features count : {len(info['features'])}")
    cams = info.get("cameras", {})
    print(f"  cameras patched: {list(cams)}")
    if "static_camera" in cams:
        K = cams["static_camera"]["intrinsic_K"]
        ext = cams["static_camera"]["extrinsic"]
        print(f"    static_camera K (3x3 row-major): {K}")
        print(f"    static_camera extrinsic xyz    : {ext['translation']}")
        print(f"    static_camera extrinsic quat   : {ext['rotation']}")

    # tasks.jsonl: one task per line
    tasks_path = root / "meta" / "tasks.jsonl"
    tasks = [
        json.loads(line)
        for line in tasks_path.read_text().splitlines()
        if line.strip()
    ]
    print(f"  tasks          : {tasks}")

    # episodes.jsonl: one episode per line
    ep_path = root / "meta" / "episodes.jsonl"
    eps = [
        json.loads(line)
        for line in ep_path.read_text().splitlines()
        if line.strip()
    ]
    print(f"  episode 0 meta : {eps[0]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the dataset root, e.g. logs/lerobot_test.",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot repo_id used at convert time.",
    )
    p.add_argument(
        "--skip-lerobot",
        action="store_true",
        help="Skip way 1 (e.g. if torchcodec is broken).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.dataset_root.exists():
        print(f"ERROR: dataset root not found: {args.dataset_root}")
        return 1

    way_3_metadata_only(args.dataset_root)
    way_2_parquet_direct(args.dataset_root)

    if not args.skip_lerobot:
        way_1_lerobot_api(args.dataset_root, args.repo_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
