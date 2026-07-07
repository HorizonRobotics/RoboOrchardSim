# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""CLI: generate a `[{mnstruction}, ...]` manifest from mcaps.

Discovers ``env*_data.mcap`` files under ``--mcap-root`` (recursive), renders
one instruction per file using
``robo_orchard_sim.tasks.instructions.mcap_render.render_instruction_from_mcap``,
and writes a JSON array compatible with both the arrow packer and
``tools.mcap_to_lerobot.convert --manifest``.

Example::

    python3 -m tools.mcap_to_lerobot.gen_instructions \\
        --mcap-root logs/place_a2b_hard/.../data \\
        --template-name place_a2b_default \\
        --template-mode fixed \\
        --actor-description-mode raw \\
        --asset-root /horizon-bucket/robot_lab/assets/ROBOTS \\
        --out manifest.json
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _discover_mcaps(root: Path) -> list[Path]:
    """Recursively find all env*_data.mcap files under root, sorted."""
    if not root.exists():
        raise FileNotFoundError(f"--mcap-root does not exist: {root}")
    return sorted(root.rglob("env*_data.mcap"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render per-mcap instructions and write a packing manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mcap-root",
        type=Path,
        required=True,
        help="Directory scanned recursively for env*_data.mcap files.",
    )
    p.add_argument(
        "--template-name",
        type=str,
        required=True,
        help=(
            "Instruction template name registered in "
            "robo_orchard_sim.tasks.instructions.registry "
            "(e.g. pick_default / place_a2b_default / spatial_pick_default)."
        ),
    )
    p.add_argument(
        "--template-mode",
        choices=["fixed", "variants"],
        default="fixed",
        help=(
            "'fixed' uses the canonical template string; 'variants' samples "
            "one phrasing per mcap."
        ),
    )
    p.add_argument(
        "--actor-description-mode",
        choices=["raw", "seen", "unseen"],
        default="raw",
        help=(
            "How to render {actorN.description}: 'raw' uses asset urdf-name; "
            "'seen' / 'unseen' pull from AssetRegistry caption sets."
        ),
    )
    p.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help=(
            "Asset root for AssetRegistry. Required for 'seen' / 'unseen' "
            "actor_description_mode; may be omitted for 'raw'."
        ),
    )
    p.add_argument(
        "--template-seed",
        type=int,
        default=None,
        help="Seed for variant template sampling (template_mode='variants').",
    )
    p.add_argument(
        "--actor-description-seed",
        type=int,
        default=None,
        help="Seed for seen/unseen description sampling.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the JSON manifest.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N mcap files (useful for smoke tests).",
    )
    p.add_argument(
        "--skip-failures",
        action="store_true",
        help=(
            "Log and skip mcaps that fail instruction rendering instead of "
            "aborting the whole batch."
        ),
    )
    p.add_argument(
        "--use-relative-path",
        action="store_true",
        help=(
            "Write mcap_path entries as relative to --mcap-root instead of "
            "absolute. Absolute (default) matches convert.py's lookup most "
            "robustly."
        ),
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.actor_description_mode in ("seen", "unseen") and (
        args.asset_root is None
    ):
        logger.error(
            "--asset-root is required when --actor-description-mode is "
            "'seen' or 'unseen'."
        )
        return 1

    try:
        mcaps = _discover_mcaps(args.mcap_root)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    if args.limit is not None:
        mcaps = mcaps[: args.limit]
    if not mcaps:
        logger.error("No env*_data.mcap files found under %s", args.mcap_root)
        return 1
    logger.info("Found %d mcap(s) under %s", len(mcaps), args.mcap_root)

    # Lazy import: robo_orchard_sim has heavy deps, only load when actually
    # used so --help stays fast.
    from robo_orchard_sim.tasks.instructions.base import _resolve_registry
    from robo_orchard_sim.tasks.instructions.mcap_render import (
        render_instruction_from_mcap,
    )

    registry = (
        _resolve_registry(
            registry=None,
            asset_root=str(args.asset_root) if args.asset_root else None,
        )
        if args.asset_root is not None
        else None
    )

    records: list[dict[str, str]] = []
    n_ok = n_fail = 0
    for i, mcap_path in enumerate(mcaps):
        try:
            instruction = render_instruction_from_mcap(
                mcap_path=str(mcap_path),
                template_name=args.template_name,
                registry=registry,
                asset_root=(str(args.asset_root) if args.asset_root else None),
                template_mode=args.template_mode,
                actor_description_mode=args.actor_description_mode,
                template_seed=args.template_seed,
                actor_description_seed=args.actor_description_seed,
            )
        except Exception as exc:
            n_fail += 1
            log = logger.warning if args.skip_failures else logger.error
            log(
                "[%d/%d] %s: %s: %s",
                i + 1,
                len(mcaps),
                mcap_path,
                type(exc).__name__,
                exc,
            )
            if args.skip_failures:
                continue
            return 2

        if args.use_relative_path:
            try:
                key = str(mcap_path.relative_to(args.mcap_root.resolve()))
            except ValueError:
                key = str(mcap_path)
        else:
            key = str(mcap_path.resolve())

        records.append({"mcap_path": key, "instruction": instruction})
        n_ok += 1
        logger.info(
            "[%d/%d] %s -> %s",
            i + 1,
            len(mcaps),
            mcap_path.name,
            instruction,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Wrote %d records (%d ok, %d failed) -> %s",
        len(records),
        n_ok,
        n_fail,
        args.out,
    )
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
