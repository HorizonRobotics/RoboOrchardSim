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
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robo_orchard_sim.asset_manager.registry.errors import (
    DuplicateAssetIdError,
    MissingAabbError,
)
from robo_orchard_sim.asset_manager.registry.urdf_parser import (
    ParsedUrdf,
    parse_urdf_extra_info,
)

SCHEMA_VERSION = "5"
INDEX_FILENAME = "asset_index.parquet"


def _auto_default_workers() -> int:
    """Pick a thread-pool size suited to the current machine."""
    try:
        cpu = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu = os.cpu_count() or 4
    return max(4, min(32, cpu * 4))


DEFAULT_WORKERS = _auto_default_workers()

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
        ("color", pa.list_(pa.string())),
        ("shape", pa.list_(pa.string())),
        ("material", pa.list_(pa.string())),
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
        ("aabb_x_min", pa.float64()),
        ("aabb_x_max", pa.float64()),
        ("aabb_y_min", pa.float64()),
        ("aabb_y_max", pa.float64()),
        ("aabb_z_min", pa.float64()),
        ("aabb_z_max", pa.float64()),
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
    super_dirs: list[Path] = []
    for domain_dir in root.iterdir():
        if not domain_dir.is_dir():
            continue
        for super_dir in domain_dir.iterdir():
            if super_dir.is_dir():
                super_dirs.append(super_dir)

    if not super_dirs:
        return []

    def walk_super(super_dir: Path) -> list[Path]:
        return [
            urdf.parent
            for urdf in super_dir.glob("*/*.urdf")
            if urdf.stem == urdf.parent.name
        ]

    n_workers = max(1, min(DEFAULT_WORKERS, len(super_dirs)))
    asset_dirs: list[Path] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for batch in ex.map(walk_super, super_dirs):
            asset_dirs.extend(batch)
    return sorted(asset_dirs)


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
    *,
    strict: bool = False,
) -> dict:
    asset_id = asset_dir.name
    if parsed.aabb_min is None or parsed.aabb_max is None:
        if strict:
            raise MissingAabbError(
                f"URDF for {asset_id} lacks <aabb>. "
                f"Re-run the labeller (Step 2 of usd-asset-batch-labelling "
                f"skill) to populate the field from the asset's USD."
            )
        aabb_x_min = aabb_x_max = None
        aabb_y_min = aabb_y_max = None
        aabb_z_min = aabb_z_max = None
    else:
        aabb_x_min, aabb_y_min, aabb_z_min = parsed.aabb_min
        aabb_x_max, aabb_y_max, aabb_z_max = parsed.aabb_max
    return {
        "uuid": parsed.uuid,
        "asset_id": asset_id,
        "relative_path": rel,
        "domain": parsed.domain,
        "super_category": parsed.super_category,
        "category": parsed.category,
        "name": parsed.name,
        "description": parsed.description,
        "color": sorted(parsed.color) if parsed.color else None,
        "shape": sorted(parsed.shape) if parsed.shape else None,
        "material": sorted(parsed.material) if parsed.material else None,
        "real_height": parsed.real_height,
        "real_mass": parsed.real_mass,
        "min_height": parsed.min_height,
        "max_height": parsed.max_height,
        "min_mass": parsed.min_mass,
        "max_mass": parsed.max_mass,
        "usd_path": str(asset_dir / f"{asset_id}.usd"),
        "urdf_path": str(asset_dir / f"{asset_id}.urdf"),
        "interaction_path": str(asset_dir / "interaction.json"),
        "caption_path": str(
            asset_dir / parsed.caption_link
            if parsed.caption_link
            else asset_dir / "caption_candidates.json"
        ),
        "aabb_x_min": aabb_x_min,
        "aabb_x_max": aabb_x_max,
        "aabb_y_min": aabb_y_min,
        "aabb_y_max": aabb_y_max,
        "aabb_z_min": aabb_z_min,
        "aabb_z_max": aabb_z_max,
        "tags": sorted(parsed.tags),
        "version": parsed.version,
        "generate_time": parsed.generate_time,
    }


def asset_set_fingerprint(root: Path) -> str:
    """sha256 over sorted asset dir paths (add/remove/rename detection)."""
    rels: list[str] = []
    for domain in sorted(root.iterdir()):
        if not domain.is_dir():
            continue
        for super_dir in sorted(domain.iterdir()):
            if not super_dir.is_dir():
                continue
            for asset_dir in sorted(super_dir.iterdir()):
                if asset_dir.is_dir():
                    rels.append(str(asset_dir.relative_to(root)))
    return hashlib.sha256("\n".join(rels).encode()).hexdigest()


def _build_table(rows: list[dict], fingerprint: str) -> pa.Table:
    if rows:
        table = pa.Table.from_pylist(rows, schema=_EMPTY_SCHEMA)
    else:
        table = _EMPTY_SCHEMA.empty_table()
    return table.replace_schema_metadata(
        {
            b"schema_version": SCHEMA_VERSION.encode(),
            b"asset_set_fingerprint": fingerprint.encode(),
        }
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
    fingerprint = asset_set_fingerprint(root)
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
    if asset_dirs:
        n_workers = max(1, min(DEFAULT_WORKERS, len(asset_dirs)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            parsed_results = list(
                ex.map(
                    lambda d: _parse_one(d, strict=strict),
                    asset_dirs,
                )
            )

        for asset_dir, (parsed, reason) in zip(
            asset_dirs, parsed_results, strict=False
        ):
            asset_id = asset_dir.name
            rel = str(asset_dir.relative_to(root))
            if parsed is None:
                report.skipped.append(SkippedAsset(rel, reason))
                logger.warning("skip %s: %s", rel, reason)
                continue
            for w in parsed.warnings:
                report.warnings.append(f"{asset_id}: {w}")

            if (
                parsed.aabb_min is None or parsed.aabb_max is None
            ) and not strict:
                report.warnings.append(
                    f"{asset_id}: missing <aabb> in URDF; recorded as None"
                )

            rows.append(
                _row_from_parsed(parsed, asset_dir, rel, strict=strict)
            )

    report.total_indexed = len(rows)

    table = _build_table(rows, fingerprint)
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
