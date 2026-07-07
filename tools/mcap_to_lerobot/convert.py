# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""CLI: convert robo_orchard_sim mcap recordings into a LeRobot v2 dataset.

Usage (dual-arm piperx, auto-discover all mcaps)::

    python -m tools.mcap_to_lerobot.convert \\
        --embodiment dualarm_piperx \\
        --mcap-root logs/anymove_figure/pick_attribute_.../data \\
        --out /tmp/piperx_lerobot \\
        --repo-id robo_orchard/piperx_pick_attribute

With an external JSON manifest (D6 option)::

    python -m tools.mcap_to_lerobot.convert \\
        --embodiment dualarm_piperx \\
        --mcap-root logs/.../data \\
        --manifest selected_records.json \\
        --out /tmp/piperx_lerobot \\
        --repo-id robo_orchard/piperx_pick_attribute

Manifest format (compatible with arrow packer)::

    [{"mcap_path": "/abs/or/rel/path/env0_data.mcap",
      "instruction": "Pick the red marker."},
     ...]
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from tools.mcap_to_lerobot.config_presets import EMBODIMENT_REGISTRY
from tools.mcap_to_lerobot.reader import McapEpisodeReader
from tools.mcap_to_lerobot.writer import LeRobotEpisodePacker

# ---------------------------------------------------------------------------
# manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict[str, str]:
    """Load JSON manifest, return {mcap_path_str: instruction}.

    Handles malformed / truncated JSON gracefully: logs a warning and
    returns an empty dict so the convert loop falls back to /meta_data.
    """
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            logging.warning("Manifest %s is empty; ignoring.", path)
            return {}
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logging.warning(
            "Manifest %s could not be parsed (JSONDecodeError: %s); "
            "falling back to /meta_data for all episodes.",
            path,
            exc,
        )
        return {}
    except OSError as exc:
        logging.warning("Cannot read manifest %s: %s; ignoring.", path, exc)
        return {}

    if not isinstance(data, list):
        logging.warning(
            "Manifest %s: expected a JSON array, got %s; ignoring.",
            path,
            type(data).__name__,
        )
        return {}

    result: dict[str, str] = {}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            logging.warning("Manifest entry %d is not a dict; skipping.", idx)
            continue
        mcap_path = item.get("mcap_path", "")
        instruction = item.get("instruction", "")
        if not mcap_path:
            logging.warning(
                "Manifest entry %d missing 'mcap_path'; skipping.", idx
            )
            continue
        result[str(mcap_path)] = str(instruction)
    return result


def _instruction_for(
    mcap_path: Path,
    manifest: dict[str, str],
) -> str | None:
    """Look up instruction in manifest by exact or stem match; ent."""
    # Try exact string match first.
    key = str(mcap_path)
    if key in manifest:
        return manifest[key] or None
    # Try matching on resolved absolute path.
    key_abs = str(mcap_path.resolve())
    if key_abs in manifest:
        return manifest[key_abs] or None
    return None


# ---------------------------------------------------------------------------
# mcap discovery
# ---------------------------------------------------------------------------


def _discover_mcaps(root: Path) -> list[Path]:
    """Recursively find all env*_data.mcap files under root, sorted."""
    if not root.exists():
        raise FileNotFoundError(f"--mcap-root does not exist: {root}")
    return sorted(root.rglob("env*_data.mcap"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert robo_orchard_sim mcap recordings to a LeRobot v2 dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--embodiment",
        choices=list(EMBODIMENT_REGISTRY),
        default="dualarm_piperx",
        help="Robot embodiment preset.",
    )
    p.add_argument(
        "--mcap-root",
        type=Path,
        default=None,
        help=(
            "Directory scanned recursively for env*_data.mcap files. "
            "Ignored when --manifest is also given (manifest wins)."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the LeRobotDataset (must not exist).",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot repo_id, e.g. 'robo_orchard/piperx_pick_attribute'.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON manifest: [{mcap_path, instruction}, ...]. "
            "When given, ONLY the mcaps listed there are packed and "
            "--mcap-root is ignored."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N mcap files (useful for smoke tests).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    p.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help=(
            "Downscale factor for color + depth images (and intrinsics). "
            "Must be in (0, 1]. 1.0 = no scaling (default, zero overhead)."
        ),
    )
    p.add_argument(
        "--include-depth",
        dest="include_depth",
        action="store_true",
        default=False,
        help=(
            "Pack per-frame depth as PNG files. Off by default — depth "
            "produces 1 file per frame per camera, which is bad for "
            "object-store backends (e.g. JuiceFS) where small-file "
            "uploads dominate I/O cost."
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes. 1 (default) runs the "
            "in-process single-process path. >1 fan-outs to shard "
            "directories then merges. 0 or negative = os.cpu_count() // 2."
        ),
    )
    p.add_argument(
        "--keep-shards",
        action="store_true",
        help="Retain tmp shard directories after merge (for debugging).",
    )
    p.add_argument(
        "--best-effort",
        action="store_true",
        help=(
            "If any mcap fails, still merge the successful shards instead "
            "of aborting. Exits with code 2 when any failure occurred."
        ),
    )
    p.add_argument(
        "--upload-to",
        type=str,
        default=None,
        help=(
            "After merge, `cp -r` the finished dataset to this destination "
            "(append-only-fs friendly; dst must not exist)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = EMBODIMENT_REGISTRY[args.embodiment]

    if not (0.0 < args.image_scale <= 1.0):
        logging.error(
            "--image-scale must be in (0, 1], got %s", args.image_scale
        )
        return 1

    # --- discover mcaps ---
    # --manifest wins; else fall back to --mcap-root rglob.
    if args.manifest is not None:
        manifest_dict = _load_manifest(args.manifest)
        if not manifest_dict:
            logging.error("Manifest %s is empty or unreadable", args.manifest)
            return 1
        mcaps = [Path(p) for p in manifest_dict.keys()]
        logging.info(
            "Manifest mode: %d mcap(s) from %s", len(mcaps), args.manifest
        )
    elif args.mcap_root is not None:
        try:
            mcaps = _discover_mcaps(args.mcap_root)
        except FileNotFoundError as exc:
            logging.error("%s", exc)
            return 1
    else:
        logging.error("Either --manifest or --mcap-root is required")
        return 1

    if args.limit is not None:
        mcaps = mcaps[: args.limit]

    if not mcaps:
        logging.error("No mcap files to pack")
        return 1

    # --- parallel path ---
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = max(1, (os.cpu_count() or 2) // 2)
    if num_workers > 1:
        from tools.mcap_to_lerobot._parallel import run_parallel

        return run_parallel(
            mcaps=mcaps,
            out=args.out,
            repo_id=args.repo_id,
            embodiment=args.embodiment,
            manifest_path=args.manifest,
            image_scale=args.image_scale,
            include_depth=args.include_depth,
            num_workers=num_workers,
            keep_shards=args.keep_shards,
            best_effort=args.best_effort,
            upload_to=args.upload_to,
        )

    logging.info("Found %d mcap(s)", len(mcaps))

    # --- load optional manifest ---
    manifest: dict[str, str] = {}
    if args.manifest is not None:
        manifest = _load_manifest(args.manifest)
        logging.info(
            "Manifest loaded: %d entries from %s", len(manifest), args.manifest
        )

    # --- create dataset (probe shapes from first readable mcap) ---
    reader = McapEpisodeReader(
        cfg,
        image_scale=args.image_scale,
        include_depth=args.include_depth,
    )

    try:
        packer = LeRobotEpisodePacker.create(
            repo_id=args.repo_id,
            root=args.out,
            cfg=cfg,
            first_mcap=mcaps[0],
            use_videos=True,
            image_scale=args.image_scale,
            include_depth=args.include_depth,
        )
    except FileExistsError as exc:
        logging.error("%s", exc)
        return 1
    except Exception:
        logging.exception("Failed to create dataset at %s.", args.out)
        return 1

    # --- convert loop ---
    n_ok = n_fail = n_frames = 0
    t0 = time.time()

    for i, mcap_path in enumerate(mcaps):
        instruction = _instruction_for(mcap_path, manifest)

        try:
            ep = reader.read(
                mcap_path,
                manifest_instruction=instruction,
                default_task="manipulation",
            )
        except Exception:
            logging.exception(
                "[%d/%d] read failed: %s", i + 1, len(mcaps), mcap_path
            )
            n_fail += 1
            continue

        if ep.num_frames == 0:
            logging.warning(
                "[%d/%d] skip %s: 0 retained frames.",
                i + 1,
                len(mcaps),
                mcap_path.name,
            )
            n_fail += 1
            continue

        try:
            packer.add_episode(ep)
        except Exception:
            logging.exception(
                "[%d/%d] write failed: %s", i + 1, len(mcaps), mcap_path
            )
            n_fail += 1
            continue

        n_ok += 1
        n_frames += ep.num_frames
        logging.info(
            "[%d/%d] %s -> %d frames (task=%r)",
            i + 1,
            len(mcaps),
            mcap_path.name,
            ep.num_frames,
            ep.task,
        )

    # --- finalize ---
    try:
        packer.finalize()
    except Exception:
        logging.exception("finalize() failed.")

    elapsed = time.time() - t0
    logging.info(
        "Done in %.1fs: %d ok, %d failed, %d total frames -> %s",
        elapsed,
        n_ok,
        n_fail,
        n_frames,
        args.out,
    )

    # Optional upload to remote / bucket (matches parallel path behaviour).
    if args.upload_to is not None and n_ok > 0:
        from tools.mcap_to_lerobot._parallel import _upload_dataset

        _upload_dataset(args.out, args.upload_to)

    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
