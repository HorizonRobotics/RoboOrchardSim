#!/usr/bin/env python3
# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Command-line entry for the TagLabeller skill.

Recursively scans an asset root for *.urdf files, decides whether to skip
or process each, writes resulting tags into <extra_info><tags>, and
appends any errors to a JSONL log.

Example:
    python3 tools/asset_pipeline/run_tag_labeller.py \\
        --asset-root /path/to/labelled/assets \\
        --tags-vocab tools/asset_pipeline/tag_labeller/tags.yaml \\
        --config tools/asset_pipeline/configs/gpt_config.yaml \\
        --max-workers 4
"""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_tag_labeller")


def _add_project_paths() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


_add_project_paths()

from asset_labeller.gpt_client import load_client_from_config  # noqa: E402
from tag_labeller.tag_labeller import TagLabeller  # noqa: E402
from tag_labeller.tag_vocab import TagVocab  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-tag labelled URDFs with semantic capability tags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--asset-root",
        type=str,
        required=True,
        help="Root directory to recursively scan for *.urdf files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tools/asset_pipeline/configs/gpt_config.yaml",
        help="Path to GPT config YAML file.",
    )
    parser.add_argument(
        "--tags-vocab",
        type=str,
        default="tools/asset_pipeline/tag_labeller/tags.yaml",
        help="Path to tags.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tag assets that already have a <tags> element.",
    )
    parser.add_argument(
        "--only-tags",
        type=str,
        default=None,
        help=(
            "Comma-separated tag names to evaluate. Implies --force"
            " for those tags; preserves existing tags not listed."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel GPT calls (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan but do not call GPT or modify URDFs.",
    )
    parser.add_argument(
        "--error-log",
        type=str,
        default=None,
        help=(
            "Path to JSONL error log (append mode; each row tagged with"
            " a UTC run_id). Default: <asset-root>/tag_labeller_errors.jsonl"
        ),
    )
    parser.add_argument(
        "--no-check-connection",
        action="store_true",
        help="Do not run GPT connection check at startup.",
    )
    return parser.parse_args()


def discover_urdfs(asset_root: str) -> list[str]:
    """Return absolute paths of every *.urdf file under asset_root."""
    out: list[str] = []
    for sub_dir, _dirnames, filenames in os.walk(asset_root):
        for fname in sorted(filenames):
            if fname.lower().endswith(".urdf"):
                out.append(os.path.join(sub_dir, fname))
    return sorted(out)


def filter_vocab(full_vocab: TagVocab, only_tags: str | None) -> TagVocab:
    if only_tags is None:
        return full_vocab
    names = [n.strip() for n in only_tags.split(",") if n.strip()]
    specs = []
    for n in names:
        if not full_vocab.is_known(n):
            raise SystemExit(f"--only-tags references unknown tag: {n}")
        specs.append(full_vocab.get(n))
    return TagVocab(specs)


def main() -> None:
    args = parse_args()
    asset_root = os.path.abspath(args.asset_root)
    if not os.path.isdir(asset_root):
        raise SystemExit(f"--asset-root not found: {asset_root}")

    full_vocab = TagVocab.from_yaml(args.tags_vocab)
    active_vocab = filter_vocab(full_vocab, args.only_tags)
    logger.info(
        f"Active tags: {', '.join(active_vocab.names())}"
        f" (full vocab: {len(full_vocab)} tags)"
    )

    urdfs = discover_urdfs(asset_root)
    logger.info(f"Discovered {len(urdfs)} URDF(s) under {asset_root}")
    if not urdfs:
        return

    error_log_path = args.error_log or os.path.join(
        asset_root, "tag_labeller_errors.jsonl"
    )

    if args.dry_run:
        logger.info("[DRY RUN] would process the following URDFs:")
        for u in urdfs:
            logger.info(f"  {u}")
        return

    client = load_client_from_config(
        args.config,
        check_connection=not args.no_check_connection,
    )
    labeller = TagLabeller(gpt_client=client, vocab=active_vocab)

    force = args.force or args.only_tags is not None
    merge = args.only_tags is not None

    run_id = datetime.now(timezone.utc).isoformat()
    counts = {"ok": 0, "skipped": 0, "error": 0}
    errors: list[dict] = []

    def _process_one(path):
        return labeller.process(path, force=force, merge=merge)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as pool:
        for idx, result in enumerate(pool.map(_process_one, urdfs), start=1):
            counts[result.status] += 1
            tag_str = (
                ",".join(result.tags_written) if result.tags_written else ""
            )
            logger.info(
                f"[{idx}/{len(urdfs)}] {result.status}"
                f" {result.urdf_path} {tag_str}"
            )
            if result.status == "error":
                errors.append(
                    {
                        "run_id": run_id,
                        "urdf_path": result.urdf_path,
                        "error_msg": result.error_msg,
                    }
                )

    if errors:
        with open(error_log_path, "a", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(errors)} errors to {error_log_path}")

    logger.info(
        f"Done. ok={counts['ok']} skipped={counts['skipped']}"
        f" error={counts['error']}"
    )


if __name__ == "__main__":
    main()
