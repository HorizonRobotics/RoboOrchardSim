#!/usr/bin/env python3
# Project RoboOrchard
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CLI entry for the caption_labeller skill.

Example:
    python3 tools/asset_pipeline/run_caption_labeller.py \\
        --asset-root outputs/labelled \\
        --config tools/asset_pipeline/configs/gpt_config.yaml \\
        --seen-count 15 \\
        --unseen-count 5
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asset_labeller.gpt_client import load_client_from_config  # noqa: E402
from caption_labeller.caption_labeller import (  # noqa: E402
    CaptionLabeller,
    iter_urdfs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Caption candidates labeller")
    p.add_argument(
        "--asset-root",
        required=True,
        help="Directory to scan for *.urdf",
    )
    p.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs",
            "gpt_config.yaml",
        ),
    )
    p.add_argument("--seen-count", type=int, default=15)
    p.add_argument("--unseen-count", type=int, default=5)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--error-log", default=None)
    return p.parse_args()


def _write_error(error_log_path: str, record: dict) -> None:
    with open(error_log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    args = _parse_args()
    urdfs = iter_urdfs(args.asset_root)
    logger.info("found %d URDFs under %s", len(urdfs), args.asset_root)

    if args.dry_run:
        for u in urdfs:
            print(u)
        return 0

    error_log = args.error_log or os.path.join(
        "outputs", "labeller_runs", "caption_labeller_errors.jsonl"
    )
    os.makedirs(os.path.dirname(os.path.abspath(error_log)), exist_ok=True)

    client = load_client_from_config(args.config)
    labeller = CaptionLabeller(
        gpt_client=client,
        seen_count=args.seen_count,
        unseen_count=args.unseen_count,
        force=args.force,
    )

    counts = {"written": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(labeller.process, u): u for u in urdfs}
        for fut in as_completed(futures):
            urdf = futures[fut]
            result = fut.result()
            counts[result.status] = counts.get(result.status, 0) + 1
            if result.status == "failed":
                _write_error(
                    error_log,
                    {
                        "urdf_path": urdf,
                        "error_type": result.reason or "unknown",
                    },
                )

    logger.info(
        "processed %d | written %d | skipped %d | failed %d",
        len(urdfs),
        counts["written"],
        counts["skipped"],
        counts["failed"],
    )
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
