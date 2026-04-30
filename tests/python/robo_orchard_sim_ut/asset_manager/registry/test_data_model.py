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

"""Tests for the core data model: AssetMeta and the error hierarchy."""

from dataclasses import FrozenInstanceError

import pytest

from robo_orchard_sim.asset_manager.registry.errors import (
    AssetIndexNotFoundError,
    AssetIndexVersionError,
    DuplicateAssetIdError,
    EmptyPoolError,
    InsufficientPoolError,
    UnknownAssetError,
)
from robo_orchard_sim.asset_manager.registry.types import AssetMeta

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


def test_unknown_asset_error_formats_closest_matches():
    err = UnknownAssetError(
        key="appel_001",
        closest_matches=["apple_001", "apple_002"],
    )
    msg = str(err)
    assert "appel_001" in msg
    assert "apple_001" in msg
    assert "apple_002" in msg


def test_duplicate_asset_id_error_lists_paths():
    err = DuplicateAssetIdError(
        asset_id="apple_001",
        paths=["food/fruits/apple_001", "misc/apple_001"],
    )
    msg = str(err)
    assert "apple_001" in msg
    assert "food/fruits/apple_001" in msg
    assert "misc/apple_001" in msg


def test_empty_pool_error_includes_filter_repr():
    err = EmptyPoolError(filter_repr="AssetFilter(role=PICK)")
    assert "AssetFilter(role=PICK)" in str(err)


def test_insufficient_pool_error_shows_counts():
    err = InsufficientPoolError(
        mode="SAME_SUPER_CATEGORY", available=2, requested=5
    )
    msg = str(err)
    assert "2" in msg and "5" in msg
    assert "SAME_SUPER_CATEGORY" in msg


def test_index_errors_are_distinct():
    assert AssetIndexNotFoundError is not AssetIndexVersionError


# ---------------------------------------------------------------------------
# AssetMeta
# ---------------------------------------------------------------------------


def _make(**overrides) -> AssetMeta:
    defaults = dict(
        uuid="u1",
        asset_id="lemon_002",
        relative_path="food/fruits/lemon_002",
        domain="food",
        super_category="fruits",
        category="lemon",
        name="yellow lemon",
        description="...",
        color="yellow",
        shape="ellipsoid",
        material="organic",
        real_height=0.0846,
        real_mass=0.15,
        min_height=0.05,
        max_height=0.08,
        min_mass=0.1,
        max_mass=0.2,
        usd_path="/abs/lemon.usd",
        urdf_path="/abs/lemon.urdf",
        interaction_path="/abs/interaction.json",
        caption_path="/abs/caption_candidates.json",
        tags=frozenset({"graspable"}),
        version="v0.1.0",
        generate_time="20260403174917",
    )
    defaults.update(overrides)
    return AssetMeta(**defaults)


def test_asset_meta_tags_stored_as_frozenset():
    meta = _make(tags=frozenset({"graspable", "container"}))
    assert meta.tags == frozenset({"graspable", "container"})


def test_asset_meta_tags_default_empty():
    meta = _make()
    assert isinstance(meta.tags, frozenset)


def test_asset_meta_is_frozen():
    meta = _make()
    with pytest.raises(FrozenInstanceError):
        meta.asset_id = "hack"  # type: ignore[misc]


@pytest.mark.parametrize(
    "real_height, expected_bucket",
    [
        (0.04, "small"),
        (0.05, "medium"),  # boundary: small -> medium
        (0.08, "medium"),
        (0.12, "large"),  # boundary: medium -> large
        (0.20, "large"),
    ],
)
def test_size_bucket(real_height: float, expected_bucket: str):
    assert _make(real_height=real_height).size_bucket == expected_bucket
