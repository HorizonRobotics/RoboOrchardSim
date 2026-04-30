# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Scan an asset root and write an index parquet file."""

from __future__ import annotations
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robo_orchard_sim.asset_manager.registry.errors import (
    DuplicateAssetIdError,
)
from robo_orchard_sim.asset_manager.registry.urdf_parser import (
    ParsedUrdf,
    parse_urdf_extra_info,
)

SCHEMA_VERSION = "3"
INDEX_FILENAME = "asset_index.parquet"

DEFAULT_CACHE_ROOT = Path("/tmp/.cache/robo_orchard_sim/asset_index")


def default_cache_index_path(asset_root: str | Path) -> Path:
    """Compute a stable per-asset-root cache path under DEFAULT_CACHE_ROOT.

    The path is keyed by sha256 of the absolute asset_root so different
    asset libraries (e.g. wuwen_0403 vs wuwen_0411) don't collide. The
    Registry never reads from this location by default — callers must
    pass the result as ``index_path`` to ``AssetRegistry`` or ``output_path``
    to ``build_asset_index``.
    """
    abs_root = str(Path(asset_root).expanduser().resolve())
    digest = hashlib.sha256(abs_root.encode()).hexdigest()[:12]
    return DEFAULT_CACHE_ROOT / digest / INDEX_FILENAME


logger = logging.getLogger(__name__)

_EMPTY_SCHEMA = pa.schema(
    [
        ("uuid", pa.string()),
        ("asset_id", pa.string()),
        ("relative_path", pa.string()),
        ("domain", pa.string()),
        ("super_category", pa.string()),
        ("category", pa.string()),
        ("name", pa.string()),
        ("description", pa.string()),
        ("color", pa.string()),
        ("shape", pa.string()),
        ("material", pa.string()),
        ("real_height", pa.float64()),
        ("real_mass", pa.float64()),
        ("min_height", pa.float64()),
        ("max_height", pa.float64()),
        ("min_mass", pa.float64()),
        ("max_mass", pa.float64()),
        ("usd_path", pa.string()),
        ("urdf_path", pa.string()),
        ("interaction_path", pa.string()),
        ("caption_path", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("version", pa.string()),
        ("generate_time", pa.string()),
    ]
)


@dataclass
class SkippedAsset:
    asset_dir: str
    reason: str


@dataclass
class BuildReport:
    total_scanned: int = 0
    total_indexed: int = 0
    skipped: list[SkippedAsset] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_path: str = ""
    elapsed_seconds: float = 0.0


def _find_asset_dirs(root: Path) -> list[Path]:
    """Every dir containing <dirname>.urdf counts as an asset."""
    dirs: list[Path] = []
    for urdf in root.rglob("*.urdf"):
        if urdf.stem == urdf.parent.name:
            dirs.append(urdf.parent)
    return sorted(dirs)


def _parse_one(
    asset_dir: Path, *, strict: bool
) -> tuple[ParsedUrdf | None, str]:
    """Return (parsed, reason_if_skipped).

    Capability tags come from URDF ``<extra_info><tags>``.
    ``interaction.json`` is still required as an existence gate (used by
    the Isaac runtime layer for grasp poses) but its contents are no
    longer parsed at index-build time.
    """
    urdf_path = asset_dir / f"{asset_dir.name}.urdf"
    interaction_path = asset_dir / "interaction.json"
    if not urdf_path.exists():
        return None, "missing_urdf"
    try:
        parsed = parse_urdf_extra_info(urdf_path.read_text(), strict=strict)
    except ValueError as e:
        if strict:
            raise
        return None, f"parse_error: {e}"
    if parsed is None:
        return None, "missing_extra_info"
    if not interaction_path.exists():
        return None, "missing_interaction"
    return parsed, ""


def _row_from_parsed(
    parsed: ParsedUrdf,
    asset_dir: Path,
    rel: str,
) -> dict:
    asset_id = asset_dir.name
    return {
        "uuid": parsed.uuid,
        "asset_id": asset_id,
        "relative_path": rel,
        "domain": parsed.domain,
        "super_category": parsed.super_category,
        "category": parsed.category,
        "name": parsed.name,
        "description": parsed.description,
        "color": parsed.color,
        "shape": parsed.shape,
        "material": parsed.material,
        "real_height": parsed.real_height,
        "real_mass": parsed.real_mass,
        "min_height": parsed.min_height,
        "max_height": parsed.max_height,
        "min_mass": parsed.min_mass,
        "max_mass": parsed.max_mass,
        "usd_path": str(asset_dir / f"{asset_id}.usd"),
        "urdf_path": str(asset_dir / f"{asset_id}.urdf"),
        "interaction_path": str(asset_dir / "interaction.json"),
        "caption_path": str(asset_dir / "caption_candidates.json"),
        "tags": sorted(parsed.tags),
        "version": parsed.version,
        "generate_time": parsed.generate_time,
    }


def _build_table(rows: list[dict]) -> pa.Table:
    if rows:
        table = pa.Table.from_pylist(rows, schema=_EMPTY_SCHEMA)
    else:
        table = _EMPTY_SCHEMA.empty_table()
    return table.replace_schema_metadata(
        {b"schema_version": SCHEMA_VERSION.encode()}
    )


def build_asset_index(
    asset_root: str,
    *,
    output_path: str | None = None,
    strict: bool = False,
) -> BuildReport:
    """Scan asset_root recursively, parse each asset, write parquet."""
    start = time.monotonic()
    root = Path(asset_root).resolve()
    if output_path is None:
        output_path = str(root / INDEX_FILENAME)

    report = BuildReport(output_path=output_path)
    asset_dirs = _find_asset_dirs(root)
    report.total_scanned = len(asset_dirs)

    # Detect duplicate asset_ids BEFORE any parsing work.
    seen_asset_ids: dict[str, list[str]] = defaultdict(list)
    for asset_dir in asset_dirs:
        rel = str(asset_dir.relative_to(root))
        seen_asset_ids[asset_dir.name].append(rel)
    dups = {
        aid: paths for aid, paths in seen_asset_ids.items() if len(paths) > 1
    }
    if dups:
        aid, paths = next(iter(dups.items()))
        raise DuplicateAssetIdError(aid, paths)

    rows: list[dict] = []
    for asset_dir in asset_dirs:
        asset_id = asset_dir.name
        rel = str(asset_dir.relative_to(root))

        parsed, reason = _parse_one(asset_dir, strict=strict)
        if parsed is None:
            report.skipped.append(SkippedAsset(rel, reason))
            logger.warning("skip %s: %s", rel, reason)
            continue
        for w in parsed.warnings:
            report.warnings.append(f"{asset_id}: {w}")

        rows.append(_row_from_parsed(parsed, asset_dir, rel))

    report.total_indexed = len(rows)

    table = _build_table(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    logger.info(
        "indexed %d/%d assets -> %s",
        report.total_indexed,
        report.total_scanned,
        output_path,
    )

    report.elapsed_seconds = time.monotonic() - start
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry: build an asset index parquet from a root directory.

    Exit codes:
        0  success
        1  --max-skipped threshold exceeded
        2  argparse usage error (reserved for argparse)
        3  duplicate asset_id detected
        4  unexpected error
    """
    import argparse
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Build parquet asset index from an asset root."
    )
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--max-skipped",
        type=int,
        default=None,
        help=("Exit nonzero if skipped count exceeds this threshold"),
    )
    args = parser.parse_args(argv)

    try:
        report = build_asset_index(
            args.asset_root,
            output_path=args.output,
            strict=args.strict,
        )
    except DuplicateAssetIdError as e:
        print(f"ERROR: {e}", file=_sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {type(e).__name__}: {e}", file=_sys.stderr)
        return 4

    logger.info(
        "indexed %d / scanned %d / skipped %d / warnings %d",
        report.total_indexed,
        report.total_scanned,
        len(report.skipped),
        len(report.warnings),
    )
    for s in report.skipped:
        logger.info("  skipped %s: %s", s.asset_dir, s.reason)

    if args.max_skipped is not None and len(report.skipped) > args.max_skipped:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
