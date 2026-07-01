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

"""Tests for the AssetRegistry class.

Covers construction / index auto-build, lookups (uuid, asset_id, has, iter),
error paths with suggestions, schema version enforcement, and filter-based
querying.
"""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from robo_orchard_sim.asset_manager.registry.build_index import SCHEMA_VERSION
from robo_orchard_sim.asset_manager.registry.errors import (
    AssetIndexNotFoundError,
    AssetIndexVersionError,
    UnknownAssetError,
)
from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.asset_manager.registry.types import AssetFilter

# ---------------------------------------------------------------------------
# Construction + index auto-build
# ---------------------------------------------------------------------------


def test_registry_auto_builds_index_when_missing(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    assert (mini_asset_root / "asset_index.parquet").exists()
    assert len(reg) == 6


def test_registry_refuses_to_auto_build_when_disabled(
    mini_asset_root: Path,
):
    with pytest.raises(AssetIndexNotFoundError):
        AssetRegistry(str(mini_asset_root), auto_build_index=False)


def test_registry_loads_from_explicit_index_path(
    mini_asset_root: Path, tmp_path: Path
):
    """Custom index_path: registry reads the parquet there, not from asset_root."""  # noqa: E501
    custom_index = tmp_path / "cached" / "wuwen.parquet"
    # First registry auto-builds at the custom location.
    reg1 = AssetRegistry(str(mini_asset_root), index_path=str(custom_index))
    assert custom_index.exists()
    assert not (mini_asset_root / "asset_index.parquet").exists()
    assert reg1.index_path == custom_index
    assert len(reg1) == 6

    # Second registry picks up the existing parquet without rebuilding.
    mtime_before = custom_index.stat().st_mtime_ns
    reg2 = AssetRegistry(str(mini_asset_root), index_path=str(custom_index))
    assert custom_index.stat().st_mtime_ns == mtime_before
    assert len(reg2) == 6


def test_registry_auto_build_creates_parent_dirs(
    mini_asset_root: Path, tmp_path: Path
):
    """Nested parent dirs for index_path are created on auto-build."""
    nested = tmp_path / "a" / "b" / "c" / "idx.parquet"
    assert not nested.parent.exists()
    AssetRegistry(str(mini_asset_root), index_path=str(nested))
    assert nested.exists()


def test_registry_index_path_defaults_to_asset_root(mini_asset_root: Path):
    """index_path=None preserves the legacy default location."""
    reg = AssetRegistry(str(mini_asset_root))
    assert reg.index_path == mini_asset_root / "asset_index.parquet"
    assert reg.index_path.exists()


def test_registry_refuses_missing_explicit_index_path(
    mini_asset_root: Path, tmp_path: Path
):
    """auto_build_index=False with custom index_path raises on missing file."""
    missing = tmp_path / "not_built.parquet"
    with pytest.raises(AssetIndexNotFoundError, match=str(missing)):
        AssetRegistry(
            str(mini_asset_root),
            index_path=str(missing),
            auto_build_index=False,
        )


def test_schema_version_mismatch_raises(mini_asset_root: Path):
    reg_path = mini_asset_root / "asset_index.parquet"
    AssetRegistry(str(mini_asset_root))  # build first
    table = pq.read_table(reg_path)
    table = table.replace_schema_metadata({b"schema_version": b"999"})
    pq.write_table(table, reg_path)
    with pytest.raises(AssetIndexVersionError):
        AssetRegistry(str(mini_asset_root), auto_build_index=False)


def test_schema_version_mismatch_auto_rebuilds(mini_asset_root: Path):
    reg_path = mini_asset_root / "asset_index.parquet"
    AssetRegistry(str(mini_asset_root))  # build first
    table = pq.read_table(reg_path)
    table = table.replace_schema_metadata({b"schema_version": b"999"})
    pq.write_table(table, reg_path)

    reg = AssetRegistry(str(mini_asset_root), auto_build_index=True)
    rebuilt = pq.read_table(reg_path)

    assert len(reg) == 6
    assert rebuilt.schema.metadata is not None
    assert rebuilt.schema.metadata[b"schema_version"] == (
        SCHEMA_VERSION.encode()
    )


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def test_get_meta_by_uuid(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    meta = reg.get_meta("u-apple-001")
    assert meta.asset_id == "apple_001"
    assert meta.color == frozenset({"red"})
    assert "graspable" in meta.tags


def test_resolve_asset_id_to_uuid(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    assert reg.resolve_asset_id("apple_001") == "u-apple-001"


def test_get_by_asset_id(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    meta = reg.get_by_asset_id("plate_001")
    assert meta.uuid == "u-plate-001"
    assert meta.tags == frozenset({"container"})


def test_get_meta_caption_path_defaults_to_asset_local_json(
    mini_asset_root: Path,
):
    reg = AssetRegistry(str(mini_asset_root))
    meta = reg.get_by_asset_id("apple_001")
    expected = (
        mini_asset_root / "food/fruits/apple_001/caption_candidates.json"
    )
    assert meta.caption_path == str(expected)


def test_has_methods(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    assert reg.has("u-apple-001")
    assert not reg.has("u-nope")


def test_contains_operator(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    assert "u-apple-001" in reg
    assert "u-nope" not in reg


def test_iter_and_len(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    assert len(reg) == 6
    ids = sorted(m.asset_id for m in reg)
    assert "apple_001" in ids
    assert "box_001" in ids


def test_category_accessors(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    assert "apple" in reg.all_categories()
    assert "fruits" in reg.all_super_categories()
    assert set(reg.categories_in("fruits")) == {"apple", "orange"}


# ---------------------------------------------------------------------------
# Lookup error paths
# ---------------------------------------------------------------------------


def test_unknown_uuid_raises_with_suggestions(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    with pytest.raises(UnknownAssetError) as ei:
        reg.get_meta("u-apple-00X")  # close to u-apple-001/002
    assert "u-apple-00X" in str(ei.value)
    assert ei.value.closest_matches  # non-empty


def test_unknown_asset_id_raises_with_suggestions(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    with pytest.raises(UnknownAssetError) as ei:
        reg.resolve_asset_id("appel_001")
    assert "appel_001" in str(ei.value)
    assert ei.value.closest_matches  # non-empty


# ---------------------------------------------------------------------------
# Filter-based querying
# ---------------------------------------------------------------------------


def test_query_no_filter_returns_all_sorted(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    metas = reg.query(AssetFilter())
    uuids = [m.uuid for m in metas]
    assert uuids == sorted(uuids)
    assert len(uuids) == 6


def test_query_sorted_by_uuid_when_asset_id_diverges(tmp_path: Path):
    """Regression: query result sorts by uuid, not asset_id.

    Built with divergent uuid/asset_id orderings so sort-by-asset_id and
    sort-by-uuid produce different sequences; verifies query honors uuid.
    """
    import json
    from textwrap import dedent

    def _urdf(name: str, uuid: str, category: str) -> str:
        return dedent(f"""\
            <?xml version='1.0' encoding='utf-8'?>
            <robot name="{name}">
              <link name="{name}">
                <inertial>
                  <mass value="0.15"/>
                  <origin xyz="0 0 0"/>
                  <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0"
                           iyz="0.0" izz="1.0"/>
                </inertial>
                <extra_info>
                  <uuid>{uuid}</uuid>
                  <domain>food</domain>
                  <super_category>fruits</super_category>
                  <category>{category}</category>
                  <name>{category}</name>
                  <color>red</color>
                  <shape>sphere</shape>
                  <material>organic</material>
                  <description>test</description>
                  <min_height>0.05</min_height>
                  <max_height>0.10</max_height>
                  <real_height>0.08</real_height>
                  <min_mass>0.10</min_mass>
                  <max_mass>0.20</max_mass>
                  <version>v0.1.0</version>
                  <generate_time>20260519000000</generate_time>
                  <tags></tags>
                </extra_info>
              </link>
            </robot>""")

    def _write(rel: str, name: str, uuid: str, category: str) -> None:
        d = tmp_path / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.urdf").write_text(_urdf(name, uuid, category))
        (d / f"{name}.usd").write_text("fake")
        (d / "interaction.json").write_text(json.dumps({"interaction": {}}))

    # asset_id alphabetical: [apple_001, zebra_001]
    # uuid alphabetical:     [aaa-zebra, zzz-apple]  -> reversed
    _write("food/fruits/apple_001", "apple_001", "zzz-apple", "apple")
    _write("food/fruits/zebra_001", "zebra_001", "aaa-zebra", "zebra")

    reg = AssetRegistry(str(tmp_path))
    metas = reg.query(AssetFilter())

    uuids = [m.uuid for m in metas]
    asset_ids = [m.asset_id for m in metas]

    # By uuid: aaa-zebra comes first
    assert uuids == ["aaa-zebra", "zzz-apple"]
    # Sort-by-asset_id would give [apple_001, zebra_001] — the OPPOSITE order
    assert asset_ids == ["zebra_001", "apple_001"]


def test_query_by_tag_graspable(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    metas = reg.query(AssetFilter(tags=frozenset({"graspable"})))
    ids = {m.asset_id for m in metas}
    # plate_001 has only 'container' -> excluded
    assert "plate_001" not in ids
    assert ids == {
        "apple_001",
        "apple_002",
        "orange_001",
        "carrot_001",
        "box_001",
    }


def test_query_by_tag_container(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    metas = reg.query(AssetFilter(tags=frozenset({"container"})))
    ids = {m.asset_id for m in metas}
    assert ids == {"plate_001", "box_001"}


def test_query_by_multiple_tags_is_and_match(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    metas = reg.query(AssetFilter(tags=frozenset({"graspable", "container"})))
    ids = {m.asset_id for m in metas}
    # Only box_001 has BOTH tags.
    assert ids == {"box_001"}


def test_query_by_category(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    metas = reg.query(AssetFilter(category="apple"))
    assert {m.asset_id for m in metas} == {"apple_001", "apple_002"}


def test_query_by_color(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    metas = reg.query(AssetFilter(category="apple", color="red"))
    assert [m.asset_id for m in metas] == ["apple_001"]


def test_query_with_only_in(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    metas = reg.query(
        AssetFilter(only_in=frozenset({"u-apple-001", "u-orange-001"}))
    )
    assert {m.asset_id for m in metas} == {
        "apple_001",
        "orange_001",
    }


def test_query_with_exclude(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    metas = reg.query(
        AssetFilter(category="apple", exclude=frozenset({"u-apple-001"}))
    )
    assert [m.asset_id for m in metas] == ["apple_002"]


def test_registry_rebuilds_when_asset_set_changes(mini_asset_root):
    """Adding an asset after the index is built triggers an auto-rebuild."""
    import shutil

    reg = AssetRegistry(str(mini_asset_root))
    before = len(reg)
    assert not reg.has("u-apple-999")

    # Clone an existing asset into a new dir with a distinct uuid/asset_id.
    src = mini_asset_root / "food/fruits/apple_001"
    dst = mini_asset_root / "food/fruits/apple_999"
    shutil.copytree(src, dst)
    (dst / "apple_001.urdf").rename(dst / "apple_999.urdf")
    (dst / "apple_001.usd").rename(dst / "apple_999.usd")
    urdf = dst / "apple_999.urdf"
    urdf.write_text(
        urdf.read_text()
        .replace("u-apple-001", "u-apple-999")
        .replace("apple_001", "apple_999")
    )

    # auto_build_index=True (default): stale fingerprint -> rebuild -> visible.
    reg2 = AssetRegistry(str(mini_asset_root))
    assert len(reg2) == before + 1
    assert reg2.has("u-apple-999")


def test_registry_warns_but_keeps_stale_index_without_autobuild(
    mini_asset_root, caplog
):
    """auto_build_index=False: stale set warns, new asset stays invisible."""
    import logging
    import shutil

    AssetRegistry(str(mini_asset_root))  # build initial index
    src = mini_asset_root / "food/fruits/apple_001"
    dst = mini_asset_root / "food/fruits/apple_999"
    shutil.copytree(src, dst)

    with caplog.at_level(logging.WARNING):
        reg = AssetRegistry(str(mini_asset_root), auto_build_index=False)
    assert not reg.has("u-apple-999")
    assert any("stale" in r.message.lower() for r in caplog.records)
