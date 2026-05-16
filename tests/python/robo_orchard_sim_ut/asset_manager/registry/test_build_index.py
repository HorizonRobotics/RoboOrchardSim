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

import json
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
    MissingAabbError,
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


def test_parquet_caption_path_follows_urdf_link(tmp_path: Path, make_urdf):
    """URDF <caption_candidates>./X.json</> -> caption_path under asset_dir."""
    import json

    asset_dir = tmp_path / "food" / "fruits" / "apple_001"
    asset_dir.mkdir(parents=True)
    urdf_text = make_urdf(
        uuid="u-apple-link-001",
        domain="food",
        super_category="fruits",
        category="apple",
        caption_link="./caption_candidates_updated.json",
    )
    # make_urdf hardcodes the asset name "fork_001"; rewrite to apple_001
    urdf_text = urdf_text.replace("fork_001", "apple_001")
    (asset_dir / "apple_001.urdf").write_text(urdf_text)
    (asset_dir / "apple_001.usd").write_text("fake-usd")
    (asset_dir / "interaction.json").write_text(
        json.dumps({"interaction": {}})
    )

    build_asset_index(str(tmp_path))
    table = pq.read_table(tmp_path / "asset_index.parquet")
    df = table.to_pandas()
    row = df[df.asset_id == "apple_001"].iloc[0]
    expected = asset_dir / "caption_candidates_updated.json"
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
    # Create a second apple_001 in a different super_category subpath
    dup_dir = mini_asset_root / "food/duplicates/apple_001"
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
    dup = mini_asset_root / "food/dups/apple_001"
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


# ---------------------------------------------------------------------------
# v5 schema: flattened AABB columns + strict mode
# ---------------------------------------------------------------------------


def _interaction_json() -> str:
    return json.dumps({"interaction": {"passive": {"pick": {"body": []}}}})


def test_build_writes_aabb_columns_matching_urdf_values(
    tmp_path: Path, make_urdf
):
    """URDF with <aabb> -> parquet has correct aabb_* values."""
    root = tmp_path / "lib"
    asset_dir = root / "kitchen_supplies" / "tableware" / "fork_001"
    asset_dir.mkdir(parents=True)
    (asset_dir / "fork_001.urdf").write_text(
        make_urdf(
            aabb_min=(-0.05, -0.10, -0.01),
            aabb_max=(0.05, 0.10, 0.02),
        )
    )
    (asset_dir / "interaction.json").write_text(_interaction_json())

    report = build_asset_index(str(root))
    assert report.total_indexed == 1
    table = pq.read_table(report.output_path)
    row = table.to_pylist()[0]
    expected = {
        "aabb_x_min": -0.05,
        "aabb_x_max": 0.05,
        "aabb_y_min": -0.10,
        "aabb_y_max": 0.10,
        "aabb_z_min": -0.01,
        "aabb_z_max": 0.02,
    }
    actual = {k: row[k] for k in expected}
    assert actual == pytest.approx(expected)


def test_build_strict_raises_on_missing_aabb(tmp_path: Path, make_urdf):
    """URDF without <aabb> in strict mode -> MissingAabbError."""
    root = tmp_path / "lib"
    asset_dir = root / "kitchen_supplies" / "tableware" / "fork_001"
    asset_dir.mkdir(parents=True)
    # No aabb_* kwargs -> make_urdf produces a URDF without <aabb>
    (asset_dir / "fork_001.urdf").write_text(make_urdf())
    (asset_dir / "interaction.json").write_text(_interaction_json())

    with pytest.raises(MissingAabbError) as exc:
        build_asset_index(str(root), strict=True)
    assert "fork_001" in str(exc.value)


@pytest.fixture
def _non_strict_aabbless_report(tmp_path: Path, make_urdf):
    """Build an index over one URDF without <aabb> in non-strict mode."""
    root = tmp_path / "lib"
    asset_dir = root / "kitchen_supplies" / "tableware" / "fork_001"
    asset_dir.mkdir(parents=True)
    (asset_dir / "fork_001.urdf").write_text(make_urdf())
    (asset_dir / "interaction.json").write_text(_interaction_json())
    report = build_asset_index(str(root))  # strict defaults to False
    table = pq.read_table(report.output_path)
    return report, table.to_pylist()[0]


def test_build_non_strict_missing_aabb_writes_null_columns(
    _non_strict_aabbless_report,
):
    """URDF without <aabb> in non-strict mode -> aabb_* columns are None."""
    _report, row = _non_strict_aabbless_report
    null_aabb_cols = {
        k: row[k]
        for k in [
            "aabb_x_min",
            "aabb_x_max",
            "aabb_y_min",
            "aabb_y_max",
            "aabb_z_min",
            "aabb_z_max",
        ]
    }
    assert null_aabb_cols == dict.fromkeys(null_aabb_cols, None)


def test_build_non_strict_missing_aabb_emits_warning(
    _non_strict_aabbless_report,
):
    """URDF without <aabb> in non-strict mode -> warning is emitted."""
    report, _row = _non_strict_aabbless_report
    assert any("aabb" in w.lower() for w in report.warnings)
