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

"""Tests for build_asset_index (library API and CLI entry point)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from robo_orchard_sim.asset_manager.registry.build_index import (
    DEFAULT_CACHE_ROOT,
    INDEX_FILENAME,
    SCHEMA_VERSION,
    BuildReport,
    build_asset_index,
    default_cache_index_path,
)
from robo_orchard_sim.asset_manager.registry.errors import (
    DuplicateAssetIdError,
)

# ---------------------------------------------------------------------------
# build_asset_index: library API
# ---------------------------------------------------------------------------


def test_build_report_counts(mini_asset_root: Path):
    report = build_asset_index(str(mini_asset_root))
    assert isinstance(report, BuildReport)
    # 6 indexed + 1 skipped (broken_001 no interaction)
    assert report.total_scanned == 7
    assert report.total_indexed == 6
    assert len(report.skipped) == 1
    assert report.skipped[0].asset_dir.endswith("broken_001")


def test_build_writes_parquet_at_default_path(mini_asset_root: Path):
    report = build_asset_index(str(mini_asset_root))
    parquet_path = mini_asset_root / "asset_index.parquet"
    assert parquet_path.exists()
    assert report.output_path == str(parquet_path)


def test_parquet_has_schema_version_metadata(mini_asset_root: Path):
    build_asset_index(str(mini_asset_root))
    table = pq.read_table(mini_asset_root / "asset_index.parquet")
    meta = table.schema.metadata or {}
    assert meta.get(b"schema_version") == SCHEMA_VERSION.encode()


def test_parquet_row_count_matches_indexed(mini_asset_root: Path):
    build_asset_index(str(mini_asset_root))
    table = pq.read_table(mini_asset_root / "asset_index.parquet")
    assert table.num_rows == 6


def test_parquet_columns_cover_asset_meta(mini_asset_root: Path):
    build_asset_index(str(mini_asset_root))
    table = pq.read_table(mini_asset_root / "asset_index.parquet")
    cols = set(table.column_names)
    for required in [
        "uuid",
        "asset_id",
        "relative_path",
        "domain",
        "super_category",
        "category",
        "color",
        "shape",
        "material",
        "real_height",
        "real_mass",
        "usd_path",
        "urdf_path",
        "interaction_path",
        "caption_path",
        "tags",
    ]:
        assert required in cols, f"missing column {required}"


def test_parquet_caption_path_defaults_to_asset_local_json(
    mini_asset_root: Path,
):
    build_asset_index(str(mini_asset_root))
    table = pq.read_table(mini_asset_root / "asset_index.parquet")
    df = table.to_pandas()
    row = df[df.asset_id == "apple_001"].iloc[0]
    expected = (
        mini_asset_root / "food/fruits/apple_001/caption_candidates.json"
    )
    assert row.caption_path == str(expected)


def test_box_001_has_both_tags(mini_asset_root: Path):
    build_asset_index(str(mini_asset_root))
    table = pq.read_table(mini_asset_root / "asset_index.parquet")
    df = table.to_pandas()
    row = df[df.asset_id == "box_001"].iloc[0]
    assert set(row.tags) == {"graspable", "container"}


def test_plate_001_has_only_container_tag(mini_asset_root: Path):
    build_asset_index(str(mini_asset_root))
    table = pq.read_table(mini_asset_root / "asset_index.parquet")
    df = table.to_pandas()
    row = df[df.asset_id == "plate_001"].iloc[0]
    assert list(row.tags) == ["container"]


def test_build_writes_to_explicit_output_path(
    mini_asset_root: Path, tmp_path: Path
):
    """output_path overrides the default location; parent dirs auto-created."""
    nested = tmp_path / "cache" / "nested" / "out.parquet"
    assert not nested.parent.exists()
    report = build_asset_index(str(mini_asset_root), output_path=str(nested))
    assert nested.exists()
    assert report.output_path == str(nested)
    assert not (mini_asset_root / "asset_index.parquet").exists()


def test_default_cache_index_path_under_tmp_cache(tmp_path: Path):
    path = default_cache_index_path(tmp_path)
    assert DEFAULT_CACHE_ROOT in path.parents
    assert path.name == INDEX_FILENAME


def test_default_cache_index_path_is_deterministic(tmp_path: Path):
    a = default_cache_index_path(tmp_path)
    b = default_cache_index_path(tmp_path)
    assert a == b


def test_default_cache_index_path_differs_by_asset_root(tmp_path: Path):
    a = default_cache_index_path(tmp_path / "lib_a")
    b = default_cache_index_path(tmp_path / "lib_b")
    assert a != b
    assert DEFAULT_CACHE_ROOT in a.parents
    assert DEFAULT_CACHE_ROOT in b.parents


def test_default_cache_index_path_resolves_relative(
    tmp_path: Path, monkeypatch
):
    """Relative asset_root is resolved to absolute before hashing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lib").mkdir()
    rel = default_cache_index_path("lib")
    abs_ = default_cache_index_path(tmp_path / "lib")
    assert rel == abs_


def test_duplicate_asset_id_raises(tmp_path: Path, mini_asset_root: Path):
    # Create a second apple_001 in a different subpath
    dup_dir = mini_asset_root / "duplicates/apple_001"
    dup_dir.mkdir(parents=True)
    src = mini_asset_root / "food/fruits/apple_001"
    for f in src.iterdir():
        (dup_dir / f.name).write_text(f.read_text())
    with pytest.raises(DuplicateAssetIdError):
        build_asset_index(str(mini_asset_root))


def test_build_index_missing_caption_candidates_still_indexes_asset(
    mini_asset_root: Path,
):
    report = build_asset_index(str(mini_asset_root))
    assert report.total_indexed == 6
    table = pq.read_table(mini_asset_root / "asset_index.parquet")
    df = table.to_pandas()
    assert "apple_001" in set(df.asset_id)


# ---------------------------------------------------------------------------
# build_index CLI entry point
# ---------------------------------------------------------------------------


def test_cli_runs_on_mini_asset_root(mini_asset_root: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robo_orchard_sim.asset_manager.registry.build_index",
            "--asset-root",
            str(mini_asset_root),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (mini_asset_root / "asset_index.parquet").exists()
    assert "indexed " in (result.stdout + result.stderr)


def test_cli_nonzero_on_duplicate(mini_asset_root: Path):
    dup = mini_asset_root / "dups/apple_001"
    dup.mkdir(parents=True)
    src = mini_asset_root / "food/fruits/apple_001"
    for f in src.iterdir():
        shutil.copy2(f, dup / f.name)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robo_orchard_sim.asset_manager.registry.build_index",
            "--asset-root",
            str(mini_asset_root),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3, (
        f"expected exit 3 (duplicate), got {result.returncode}; "
        f"stderr={result.stderr}"
    )
    assert "Duplicate asset_id" in result.stderr
