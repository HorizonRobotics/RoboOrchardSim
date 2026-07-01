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

"""Tests for the asset_manager.splits subsystem.

Covers both the AssetSplits dataclass (construction, invariants,
immutability) and the load_asset_splits YAML loader. Splits YAML entries
are uuid-authoritative ``{uuid, asset_id}`` mappings (asset_id optional,
human-reference only); uuid is the key used to populate AssetSplits.
"""

from __future__ import annotations
import dataclasses
import logging
from pathlib import Path

import pytest

from robo_orchard_sim.asset_manager.splits.errors import (
    DuplicateUuidInSplitError,
    EmptySeenSplitError,
    InvalidSplitsYamlError,
    OverlappingSplitsError,
    UnknownUuidError,
    UnsupportedSchemaVersionError,
)
from robo_orchard_sim.asset_manager.splits.splits import (
    AssetSplits,
    load_asset_splits,
)


def _e(uuid: str, asset_id: str | None = None) -> dict:
    """Build a single split entry mapping."""
    entry = {"uuid": uuid}
    if asset_id is not None:
        entry["asset_id"] = asset_id
    return entry


# ---------------------------------------------------------------------------
# AssetSplits dataclass
# ---------------------------------------------------------------------------


class TestAssetSplitsValid:
    def test_minimal_valid(self):
        splits = AssetSplits(name="test", seen=frozenset({"uuid-a", "uuid-b"}))
        assert splits.name == "test"
        assert splits.seen == frozenset({"uuid-a", "uuid-b"})
        assert splits.unseen_category == frozenset()
        assert splits.unseen_instance == frozenset()

    def test_full_valid(self):
        splits = AssetSplits(
            name="full",
            seen=frozenset({"a"}),
            unseen_category=frozenset({"b"}),
            unseen_instance=frozenset({"c"}),
        )
        assert splits.seen == frozenset({"a"})
        assert splits.unseen_category == frozenset({"b"})
        assert splits.unseen_instance == frozenset({"c"})


class TestAssetSplitsInvariants:
    def test_empty_seen_raises(self):
        with pytest.raises(EmptySeenSplitError) as exc_info:
            AssetSplits(name="bad", seen=frozenset())
        assert exc_info.value.name == "bad"

    def test_overlap_seen_unseen_category_raises(self):
        with pytest.raises(OverlappingSplitsError) as exc_info:
            AssetSplits(
                name="bad",
                seen=frozenset({"a", "b"}),
                unseen_category=frozenset({"b", "c"}),
            )
        assert exc_info.value.seen_vs_unseen_category == frozenset({"b"})
        assert exc_info.value.seen_vs_unseen_instance == frozenset()
        assert exc_info.value.unseen_category_vs_unseen_instance == frozenset()

    def test_overlap_seen_unseen_instance_raises(self):
        with pytest.raises(OverlappingSplitsError) as exc_info:
            AssetSplits(
                name="bad",
                seen=frozenset({"x"}),
                unseen_instance=frozenset({"x"}),
            )
        assert exc_info.value.seen_vs_unseen_instance == frozenset({"x"})

    def test_overlap_unseen_category_unseen_instance_raises(self):
        with pytest.raises(OverlappingSplitsError) as exc_info:
            AssetSplits(
                name="bad",
                seen=frozenset({"a"}),
                unseen_category=frozenset({"z"}),
                unseen_instance=frozenset({"z"}),
            )
        assert exc_info.value.unseen_category_vs_unseen_instance == frozenset(
            {"z"}
        )


class TestAssetSplitsProperties:
    def test_frozen_immutable(self):
        splits = AssetSplits(name="frozen", seen=frozenset({"a"}))
        with pytest.raises(dataclasses.FrozenInstanceError):
            splits.name = "changed"  # type: ignore[misc]

    def test_hashable(self):
        splits = AssetSplits(name="hash", seen=frozenset({"a"}))
        d = {splits: 1}
        assert d[splits] == 1


# ---------------------------------------------------------------------------
# load_asset_splits YAML loader (uuid-authoritative {uuid, asset_id} entries)
# ---------------------------------------------------------------------------


class TestLoadValid:
    def test_load_minimal_valid(self, mini_registry, write_yaml):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "mini",
                "seen": [
                    _e("u-apple-001", "apple_001"),
                    _e("u-orange-001", "orange_001"),
                ],
            }
        )
        splits = load_asset_splits(path, mini_registry)
        assert splits.name == "mini"
        assert splits.seen == frozenset({"u-apple-001", "u-orange-001"})
        assert splits.unseen_category == frozenset()
        assert splits.unseen_instance == frozenset()

    def test_load_full_valid(self, mini_registry, write_yaml):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "full",
                "seen": [
                    _e("u-apple-001", "apple_001"),
                    _e("u-apple-002", "apple_002"),
                ],
                "unseen_category": [_e("u-orange-001", "orange_001")],
                "unseen_instance": [_e("u-carrot-001", "carrot_001")],
            }
        )
        splits = load_asset_splits(path, mini_registry)
        assert splits.seen == frozenset({"u-apple-001", "u-apple-002"})
        assert splits.unseen_category == frozenset({"u-orange-001"})
        assert splits.unseen_instance == frozenset({"u-carrot-001"})

    def test_asset_id_optional(self, mini_registry, write_yaml):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "no_aid",
                "seen": [_e("u-apple-001"), _e("u-orange-001")],
            }
        )
        splits = load_asset_splits(path, mini_registry)
        assert splits.seen == frozenset({"u-apple-001", "u-orange-001"})


class TestLoadFileErrors:
    def test_missing_file_raises(self, mini_registry):
        with pytest.raises(InvalidSplitsYamlError):
            load_asset_splits(Path("/nonexistent/splits.yaml"), mini_registry)

    def test_malformed_yaml_raises(self, mini_registry, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(": : : not valid yaml [[[")
        with pytest.raises(InvalidSplitsYamlError):
            load_asset_splits(path, mini_registry)


class TestLoadSchemaErrors:
    def test_unknown_top_level_key_raises(self, mini_registry, write_yaml):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "bad",
                "seen": [_e("u-apple-001", "apple_001")],
                "schema_ver": 1,
            }
        )
        with pytest.raises(InvalidSplitsYamlError, match="schema_ver"):
            load_asset_splits(path, mini_registry)

    def test_unsupported_schema_version_raises(
        self, mini_registry, write_yaml
    ):
        path = write_yaml(
            {
                "schema_version": 99,
                "name": "bad",
                "seen": [_e("u-apple-001", "apple_001")],
            }
        )
        with pytest.raises(UnsupportedSchemaVersionError) as exc_info:
            load_asset_splits(path, mini_registry)
        assert exc_info.value.found == 99
        assert exc_info.value.expected == 1

    def test_missing_required_key_raises(self, mini_registry, write_yaml):
        path = write_yaml(
            {"schema_version": 1, "seen": [_e("u-apple-001", "apple_001")]}
        )
        with pytest.raises(InvalidSplitsYamlError, match="name"):
            load_asset_splits(path, mini_registry)


class TestLoadEntryShapeErrors:
    def test_entry_not_a_mapping_raises(self, mini_registry, write_yaml):
        path = write_yaml(
            {"schema_version": 1, "name": "bad", "seen": ["apple_001"]}
        )
        with pytest.raises(InvalidSplitsYamlError, match="mapping"):
            load_asset_splits(path, mini_registry)

    def test_entry_missing_uuid_raises(self, mini_registry, write_yaml):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "bad",
                "seen": [{"asset_id": "apple_001"}],
            }
        )
        with pytest.raises(InvalidSplitsYamlError, match="uuid"):
            load_asset_splits(path, mini_registry)

    def test_entry_unknown_subkey_raises(self, mini_registry, write_yaml):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "bad",
                "seen": [{"uuid": "u-apple-001", "color": "red"}],
            }
        )
        with pytest.raises(InvalidSplitsYamlError, match="color"):
            load_asset_splits(path, mini_registry)


class TestLoadDataErrors:
    def test_duplicate_uuid_in_list_raises(self, mini_registry, write_yaml):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "dup",
                "seen": [
                    _e("u-apple-001", "apple_001"),
                    _e("u-apple-001", "apple_001"),
                ],
            }
        )
        with pytest.raises(DuplicateUuidInSplitError) as exc_info:
            load_asset_splits(path, mini_registry)
        assert exc_info.value.split_name == "seen"
        assert exc_info.value.uuid == "u-apple-001"

    def test_unknown_uuid_strict_collects_all(self, mini_registry, write_yaml):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "unk",
                "seen": [_e("u-apple-001"), _e("u-nope-x")],
                "unseen_category": [_e("u-nope-y")],
            }
        )
        with pytest.raises(UnknownUuidError) as exc_info:
            load_asset_splits(path, mini_registry)
        assert set(exc_info.value.unknown_uuids) == {"u-nope-x", "u-nope-y"}

    def test_unknown_uuid_nonstrict_skips_and_warns(
        self, mini_registry, write_yaml, caplog
    ):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "lenient",
                "seen": [_e("u-apple-001"), _e("u-nope-x")],
            }
        )
        with caplog.at_level(logging.WARNING):
            splits = load_asset_splits(path, mini_registry, strict=False)
        assert splits.seen == frozenset({"u-apple-001"})
        assert "u-nope-x" in caplog.text

    def test_stale_asset_id_warns_but_loads_from_uuid(
        self, mini_registry, write_yaml, caplog
    ):
        # uuid is valid; asset_id is stale (does not match the uuid).
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "stale",
                "seen": [_e("u-apple-001", "renamed_old_apple")],
            }
        )
        with caplog.at_level(logging.WARNING):
            splits = load_asset_splits(path, mini_registry)
        assert splits.seen == frozenset({"u-apple-001"})
        assert "renamed_old_apple" in caplog.text


class TestLoadInvariantErrors:
    def test_empty_seen_from_yaml_raises(self, mini_registry, write_yaml):
        path = write_yaml({"schema_version": 1, "name": "empty", "seen": []})
        with pytest.raises(EmptySeenSplitError):
            load_asset_splits(path, mini_registry)

    def test_overlap_from_yaml_raises(self, mini_registry, write_yaml):
        path = write_yaml(
            {
                "schema_version": 1,
                "name": "overlap",
                "seen": [
                    _e("u-apple-001", "apple_001"),
                    _e("u-orange-001", "orange_001"),
                ],
                "unseen_category": [_e("u-apple-001", "apple_001")],
            }
        )
        with pytest.raises(OverlappingSplitsError):
            load_asset_splits(path, mini_registry)
