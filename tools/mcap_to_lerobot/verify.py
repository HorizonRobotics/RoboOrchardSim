# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""CLI: verify a packed LeRobot v2 dataset produced by convert.py.

Two modes
---------
smoke (default)
    Load the dataset, print summary stats, sample one episode/frame,
    validate feature shapes and dtypes, check camera calib in info.json.

cross-check (--cross-check)
    Re-read the source mcap for one episode with McapEpisodeReader and
    compare frame count / first-frame joint values / task string against
    the stored dataset.  Optionally render a short video for visual check.

Exit codes
----------
0  all checks passed
1  argument error / dataset not found
2  at least one check failed
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_dataset(root: Path, repo_id: str):
    # Shim: lerobot 0.3.3 + datasets >= 4 broke torch.stack(Column) at init
    # time. Wrap torch.stack to convert the Column to a list first.
    import torch as _t
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _real_stack = _t.stack

    def _safe_stack(x, *a, **kw):
        try:
            return _real_stack(x, *a, **kw)
        except TypeError:
            return _real_stack(list(x), *a, **kw)

    _t.stack = _safe_stack
    try:
        # torchcodec in this container is ABI-incompatible with the installed
        # torch build, so force the pyav backend for video decoding.
        return LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    finally:
        _t.stack = _real_stack


def _print_banner(title: str) -> None:
    bar = "=" * 60
    logger.info(bar)
    logger.info("  %s", title)
    logger.info(bar)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {label}"
    if detail:
        msg += f": {detail}"
    if ok:
        logger.info(msg)
    else:
        logger.error(msg)
    return ok


# ---------------------------------------------------------------------------
# smoke mode
# ---------------------------------------------------------------------------


def _run_smoke(ds, info_path: Path) -> list[bool]:
    results: list[bool] = []

    # --- summary ---
    _print_banner("Dataset summary")
    logger.info("  root           : %s", ds.root)
    logger.info("  repo_id        : %s", ds.repo_id)
    logger.info("  total_episodes : %d", ds.meta.total_episodes)
    logger.info("  total_frames   : %d", ds.meta.total_frames)
    logger.info("  fps            : %d", ds.fps)
    logger.info("  features       :")
    for k, v in ds.features.items():
        logger.info(
            "    %-45s dtype=%-8s shape=%s", k, v["dtype"], v.get("shape")
        )

    results.append(_check("total_episodes > 0", ds.meta.total_episodes > 0))
    results.append(_check("total_frames > 0", ds.meta.total_frames > 0))

    # --- sample one episode ---
    _print_banner("Sample episode 0")
    ep_idx = 0
    ep_frames = ds.episode_data_index
    start = int(ep_frames["from"][ep_idx])
    end = int(ep_frames["to"][ep_idx])
    n_ep_frames = end - start
    results.append(
        _check(
            "episode 0 has frames", n_ep_frames > 0, f"{n_ep_frames} frames"
        )
    )

    if n_ep_frames > 0:
        sample = ds[start]
        logger.info("  task           : %s", sample.get("task", "<none>"))

        for key, feat in ds.features.items():
            # Skip LeRobot-managed index/timestamp/task fields.
            if key in (
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
                "task",
                "task_index",
                "next.done",
            ):
                continue
            val = sample.get(key)
            if val is None:
                results.append(
                    _check(
                        f"frame key present: {key}", False, "missing in sample"
                    )
                )
                continue

            if feat["dtype"] in ("float32", "float64", "int32", "int64"):
                arr = np.array(val)
                ok_shape = arr.shape == tuple(feat["shape"])
                results.append(
                    _check(
                        f"shape {key}",
                        ok_shape,
                        f"got {arr.shape}, expected {feat['shape']}",
                    )
                )
                logger.info(
                    "    %-45s shape=%-15s range=[%.4f, %.4f]",
                    key,
                    str(arr.shape),
                    float(arr.min()),
                    float(arr.max()),
                )
            elif feat["dtype"] in ("image", "video"):
                import PIL.Image

                if isinstance(val, PIL.Image.Image):
                    logger.info("    %-45s PIL %s %s", key, val.mode, val.size)
                    results.append(_check(f"image type {key}", True))
                elif hasattr(val, "shape"):
                    logger.info(
                        "    %-45s array %s %s", key, val.dtype, val.shape
                    )
                    results.append(_check(f"image type {key}", True))
                else:
                    results.append(
                        _check(
                            f"image type {key}",
                            False,
                            f"unexpected type {type(val)}",
                        )
                    )

    # --- check cameras in info.json ---
    _print_banner("Camera calibration in info.json")
    ok_info = False
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            cameras = info.get("cameras", {})
            if cameras:
                for cam_name, calib in cameras.items():
                    K = calib.get("intrinsic_K", [])
                    ok_K = len(K) == 9
                    results.append(
                        _check(
                            f"calib K shape for {cam_name}",
                            ok_K,
                            f"len={len(K)}",
                        )
                    )
                    logger.info(
                        "    %s: w=%s h=%s distortion=%s",
                        cam_name,
                        calib.get("width"),
                        calib.get("height"),
                        calib.get("distortion_model"),
                    )
                ok_info = True
            else:
                logger.warning(
                    "info.json has no 'cameras' key; calib was not patched."
                )
                ok_info = True  # not a hard failure
        except Exception as exc:
            results.append(_check("info.json parseable", False, str(exc)))
    else:
        logger.warning("info.json not found at %s", info_path)

    if ok_info:
        results.append(_check("info.json accessible", True))

    return results


# ---------------------------------------------------------------------------
# cross-check mode
# ---------------------------------------------------------------------------


def _run_cross_check(
    ds,
    mcap_root: Path,
    embodiment_name: str,
    manifest_path: Path | None,
    dump_video: Path | None,
) -> list[bool]:
    from tools.mcap_to_lerobot.config_presets import EMBODIMENT_REGISTRY
    from tools.mcap_to_lerobot.convert import (
        _discover_mcaps,
        _instruction_for,
        _load_manifest,
    )
    from tools.mcap_to_lerobot.reader import McapEpisodeReader

    results: list[bool] = []
    _print_banner("Cross-check: re-reading source mcap")

    cfg = EMBODIMENT_REGISTRY[embodiment_name]
    reader = McapEpisodeReader(cfg)

    manifest: dict[str, str] = {}
    if manifest_path is not None:
        manifest = _load_manifest(manifest_path)

    mcaps = _discover_mcaps(mcap_root)
    if not mcaps:
        results.append(
            _check(
                "mcaps found for cross-check", False, f"none under {mcap_root}"
            )
        )
        return results

    # Use first episode for the cross-check.
    mcap_path = mcaps[0]
    instruction = _instruction_for(mcap_path, manifest)

    logger.info("  Re-reading: %s", mcap_path)
    try:
        ep = reader.read(mcap_path, manifest_instruction=instruction)
    except Exception as exc:
        results.append(_check("re-read mcap", False, str(exc)))
        return results

    results.append(_check("re-read mcap", True, f"{ep.num_frames} frames"))

    # Compare frame count with dataset episode 0.
    ep_frames = ds.episode_data_index
    n_stored = int(ep_frames["to"][0]) - int(ep_frames["from"][0])
    frame_match = ep.num_frames == n_stored
    results.append(
        _check(
            "frame count matches",
            frame_match,
            f"mcap={ep.num_frames} dataset={n_stored}",
        )
    )

    # Compare first-frame joint values.
    if ep.num_frames > 0 and n_stored > 0:
        stored_sample = ds[int(ep_frames["from"][0])]
        stored_state = np.array(stored_sample.get("observation.state", []))
        mcap_state = ep.frames[0].get("observation.state", np.array([]))

        if stored_state.size > 0 and mcap_state.size > 0:
            max_diff = float(np.abs(stored_state - mcap_state).max())
            results.append(
                _check(
                    "first-frame observation.state matches",
                    max_diff < 1e-4,
                    f"max_diff={max_diff:.2e}",
                )
            )
        else:
            results.append(
                _check(
                    "first-frame observation.state present",
                    False,
                    "empty array",
                )
            )

        # Compare task string.
        stored_task = stored_sample.get("task", "")
        task_match = stored_task == ep.task
        results.append(
            _check(
                "task string matches",
                task_match,
                f"dataset={stored_task!r} mcap={ep.task!r}",
            )
        )

    # --- optional video dump ---
    if dump_video is not None:
        _print_banner("Rendering sample video")
        try:
            _render_video(ds, episode_idx=0, out_path=dump_video)
            results.append(_check("video render", True, str(dump_video)))
        except Exception as exc:
            results.append(_check("video render", False, str(exc)))

    return results


def _render_video(ds, episode_idx: int, out_path: Path) -> None:
    """Render the first color camera of episode_idx to an mp4 using cv2."""
    import cv2
    import torch

    ep_frames = ds.episode_data_index
    start = int(ep_frames["from"][episode_idx])
    end = int(ep_frames["to"][episode_idx])

    # Find first color image/video feature.
    cam_key = next(
        (
            k
            for k, v in ds.features.items()
            if v["dtype"] in ("image", "video") and not k.endswith("_depth")
        ),
        None,
    )
    if cam_key is None:
        raise RuntimeError("No color image feature found in dataset.")

    import PIL.Image

    frames_rgb = []
    for idx in range(start, end):
        sample = ds[idx]
        img = sample[cam_key]
        if isinstance(img, torch.Tensor):
            # LeRobot stores as float32 CHW in [0,1] or uint8 CHW in [0,255].
            arr = img.numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)  # CHW -> HWC
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            frames_rgb.append(arr)
        elif isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            frames_rgb.append(arr)
        elif isinstance(img, PIL.Image.Image):
            frames_rgb.append(np.array(img.convert("RGB")))

    if not frames_rgb:
        raise RuntimeError(f"No frames decoded for {cam_key}.")

    h, w = frames_rgb[0].shape[:2]
    fps = ds.fps
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for arr in frames_rgb:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    logger.info(
        "Video written to %s (%d frames, %dx%d @ %d fps)",
        out_path,
        len(frames_rgb),
        w,
        h,
        fps,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify a LeRobot v2 dataset produced by convert.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the LeRobotDataset to verify.",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot repo_id matching the dataset.",
    )
    # cross-check options
    p.add_argument(
        "--cross-check",
        action="store_true",
        help="Re-read source mcap and compare against stored data.",
    )
    p.add_argument(
        "--mcap-root",
        type=Path,
        default=None,
        help="Source mcap root (required with --cross-check).",
    )
    p.add_argument(
        "--embodiment",
        default="dualarm_piperx",
        help="Embodiment preset for re-reading (required with --cross-check).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest used during packing.",
    )
    p.add_argument(
        "--dump-video-sample",
        type=Path,
        default=None,
        help="If set, render episode 0 color frames to this mp4 path.",
    )
    p.add_argument(
        "--dump-rich-video",
        type=Path,
        default=None,
        help=(
            "If set, render an episode into a rich mp4: multi-camera RGB on "
            "top row, depth (bwr colormap) on bottom, joint coord-axis "
            "overlay if URDF + per-frame camera_pose are available."
        ),
    )
    p.add_argument(
        "--rich-episode-index",
        type=int,
        default=None,
        help=(
            "[Deprecated, prefer --rich-episodes] Single episode index to "
            "render with --dump-rich-video."
        ),
    )
    p.add_argument(
        "--rich-episodes",
        type=str,
        default=None,
        help=(
            "Which episodes to render with --dump-rich-video. Accepts: "
            "a single index (e.g. '2'); a comma-separated list ('0,2,5'); "
            "an inclusive range ('0-3'); or 'all'. When more than one "
            "episode is selected, '_ep{idx:04d}' is appended to the "
            "output path stem."
        ),
    )
    p.add_argument(
        "--rich-fps",
        type=int,
        default=None,
        help="Override fps for --dump-rich-video (default = dataset fps).",
    )
    p.add_argument(
        "--no-cam-overlay",
        action="store_true",
        help=(
            "Disable joint coord-axis overlay in --dump-rich-video "
            "(still draws RGB + depth grid)."
        ),
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return p.parse_args(argv)


def _resolve_episode_spec(
    spec: str | None, single_idx: int | None, total: int
) -> list[int]:
    """Turn (--rich-episodes, --rich-episode-index, total_episodes) -> list."""
    if spec is None and single_idx is None:
        return [0] if total > 0 else []
    if spec is None:
        return [int(single_idx)]  # type: ignore[arg-type]
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(total))
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in spec:
        return [int(p) for p in spec.split(",") if p.strip()]
    return [int(spec)]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.dataset_root.exists():
        logger.error("--dataset-root not found: %s", args.dataset_root)
        return 1

    if args.cross_check and args.mcap_root is None:
        logger.error("--cross-check requires --mcap-root.")
        return 1

    # Load dataset.
    try:
        ds = _load_dataset(args.dataset_root, args.repo_id)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        return 1

    info_path = args.dataset_root / "meta" / "info.json"
    all_results: list[bool] = []

    # Always run smoke.
    all_results.extend(_run_smoke(ds, info_path))

    # Optionally run cross-check.
    if args.cross_check:
        all_results.extend(
            _run_cross_check(
                ds,
                mcap_root=args.mcap_root,
                embodiment_name=args.embodiment,
                manifest_path=args.manifest,
                dump_video=args.dump_video_sample,
            )
        )

    # Optionally render a rich multi-camera + depth + joint-overlay video.
    if args.dump_rich_video is not None:
        _print_banner("Rich video render")
        from tools.mcap_to_lerobot.visualize import render_episode

        episodes = _resolve_episode_spec(
            args.rich_episodes,
            args.rich_episode_index,
            int(ds.meta.total_episodes),
        )
        if not episodes:
            all_results.append(
                _check("rich video render", False, "no episode selected")
            )
        else:
            try:
                render_episode(
                    ds,
                    episode_index=(
                        episodes if len(episodes) > 1 else episodes[0]
                    ),
                    out_path=args.dump_rich_video,
                    fps=args.rich_fps,
                    with_overlay=not args.no_cam_overlay,
                )
                all_results.append(
                    _check(
                        "rich video render",
                        True,
                        f"{len(episodes)} episode(s) -> "
                        f"{args.dump_rich_video}",
                    )
                )
            except Exception as exc:
                all_results.append(
                    _check(
                        "rich video render",
                        False,
                        f"{type(exc).__name__}: {exc}",
                    )
                )

    _print_banner("Summary")
    n_pass = sum(all_results)
    n_fail = len(all_results) - n_pass
    logger.info("  %d passed, %d failed", n_pass, n_fail)

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
