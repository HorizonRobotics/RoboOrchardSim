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

from pathlib import Path

import numpy as np
import pytest

from robo_orchard_sim.asset_manager.registry.errors import (
    EmptyPoolError,
    InsufficientPoolError,
)
from robo_orchard_sim.asset_manager.registry.registry import (
    AssetRegistry,
    AssetSampler,
)
from robo_orchard_sim.asset_manager.registry.types import (
    AssetFilter,
    DistractorSpec,
)


def test_sample_target_deterministic(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)

    m1 = sampler.sample_target(
        AssetFilter(category="apple"), np.random.default_rng(42)
    )
    m2 = sampler.sample_target(
        AssetFilter(category="apple"), np.random.default_rng(42)
    )
    assert m1.asset_id == m2.asset_id


def test_sample_target_respects_filter(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)

    for seed in range(10):
        m = sampler.sample_target(
            AssetFilter(tags=frozenset({"graspable"}), category="apple"),
            np.random.default_rng(seed),
        )
        assert m.category == "apple"
        assert "graspable" in m.tags


def test_sample_target_empty_pool_raises(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    with pytest.raises(EmptyPoolError):
        sampler.sample_target(
            AssetFilter(category="dragonfruit"),
            np.random.default_rng(0),
        )


def test_sample_distractors_differ_super_category(mini_asset_root: Path):
    """differ=[super_category] → distractors have a different super."""
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")  # super=fruits
    distractors = sampler.sample_distractors(
        anchor,
        DistractorSpec(min_count=2, max_count=2, differ=("super_category",)),
        np.random.default_rng(0),
    )
    assert len(distractors) == 2
    for d in distractors:
        assert d.super_category != "fruits"
        assert d.asset_id != anchor.asset_id


def test_sample_distractors_match_super_differ_category(
    mini_asset_root: Path,
):
    """match=[super_category] + differ=[category] = same super, diff cat."""
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")
    distractors = sampler.sample_distractors(
        anchor,
        DistractorSpec(
            min_count=1,
            max_count=1,
            match=("super_category",),
            differ=("category",),
        ),
        np.random.default_rng(0),
    )
    assert len(distractors) == 1
    d = distractors[0]
    assert d.super_category == "fruits"
    assert d.category != "apple"
    assert d.asset_id == "orange_001"


def test_sample_distractors_match_category_differ_color(
    mini_asset_root: Path,
):
    """match=[category], differ=[color] = same category, different color."""
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")  # color=red
    distractors = sampler.sample_distractors(
        anchor,
        DistractorSpec(
            min_count=1, max_count=1, match=("category",), differ=("color",)
        ),
        np.random.default_rng(0),
    )
    assert len(distractors) == 1
    assert distractors[0].asset_id == "apple_002"
    assert distractors[0].color == "green"


def test_sample_distractors_excludes_anchor_from_pool(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")
    for seed in range(20):
        ds = sampler.sample_distractors(
            anchor,
            DistractorSpec(
                min_count=2, max_count=2, differ=("super_category",)
            ),
            np.random.default_rng(seed),
        )
        assert all(d.asset_id != anchor.asset_id for d in ds)


def test_sample_distractors_insufficient_pool_raises(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")
    with pytest.raises(InsufficientPoolError):
        sampler.sample_distractors(
            anchor,
            DistractorSpec(
                min_count=5,
                max_count=5,
                match=("category",),
                differ=("color",),
            ),
            np.random.default_rng(0),
        )


def test_sample_distractors_respects_only_in(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")
    distractors = sampler.sample_distractors(
        anchor,
        DistractorSpec(
            min_count=1,
            max_count=1,
            differ=("super_category",),
            only_in=frozenset({"u-carrot-001"}),
        ),
        np.random.default_rng(0),
    )
    assert [d.asset_id for d in distractors] == ["carrot_001"]


def test_sample_distractors_no_duplicates_in_output(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")
    distractors = sampler.sample_distractors(
        anchor,
        DistractorSpec(min_count=3, max_count=3, differ=("super_category",)),
        np.random.default_rng(0),
    )
    assert len({d.uuid for d in distractors}) == len(distractors)


def test_sample_distractors_unknown_match_differ_field_raises(
    mini_asset_root: Path,
):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")
    with pytest.raises(ValueError, match="match/differ"):
        sampler.sample_distractors(
            anchor,
            DistractorSpec(
                min_count=1,
                max_count=1,
                differ=("colour",),  # British spelling typo
            ),
            np.random.default_rng(0),
        )


def test_sample_distractors_caps_at_max_count(mini_asset_root: Path):
    """When pool >= min_count but < max_count, return min(pool, max)."""
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")
    # Pool of same super_category fruits != apple is orange_001 only (1).
    # min_count=1 satisfied, max_count=5 capped to 1.
    distractors = sampler.sample_distractors(
        anchor,
        DistractorSpec(
            min_count=1,
            max_count=5,
            match=("super_category",),
            differ=("category",),
        ),
        np.random.default_rng(0),
    )
    assert len(distractors) == 1
    assert distractors[0].asset_id == "orange_001"


def test_sample_distractors_min_count_zero_allows_empty_pool(
    mini_asset_root: Path,
):
    """min_count=0 accepts an empty pool and returns []."""
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")
    distractors = sampler.sample_distractors(
        anchor,
        DistractorSpec(
            min_count=0,
            max_count=3,
            absolute_filter=AssetFilter(category="dragonfruit"),  # no match
        ),
        np.random.default_rng(0),
    )
    assert distractors == []


def test_distractor_spec_rejects_min_gt_max():
    with pytest.raises(ValueError, match="min_count <= max_count"):
        DistractorSpec(min_count=5, max_count=2)


def test_distractor_spec_rejects_negative_min():
    with pytest.raises(ValueError, match="min_count"):
        DistractorSpec(min_count=-1, max_count=3)


def test_sample_distractors_absolute_filter_applies(mini_asset_root: Path):
    """absolute_filter further narrows the pool without anchor reference."""
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    anchor = reg.get_by_asset_id("apple_001")
    # differ super_category → fruits excluded; absolute_filter tag=container
    # → plate_001 + box_001 remain, pick 1.
    distractors = sampler.sample_distractors(
        anchor,
        DistractorSpec(
            min_count=1,
            max_count=1,
            differ=("super_category",),
            absolute_filter=AssetFilter(tags=frozenset({"container"})),
        ),
        np.random.default_rng(0),
    )
    assert "container" in distractors[0].tags
    assert distractors[0].super_category != "fruits"


def test_sample_compatible_pair_basic(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    pick, place = sampler.sample_compatible_pair(
        pick_filter=AssetFilter(
            tags=frozenset({"graspable"}), category="apple"
        ),
        place_filter=AssetFilter(tags=frozenset({"container"})),
        rng=np.random.default_rng(0),
    )
    assert pick.category == "apple"
    assert "container" in place.tags
    assert pick.uuid != place.uuid


def test_sample_compatible_pair_empty_pick_raises(
    mini_asset_root: Path,
):
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    with pytest.raises(EmptyPoolError):
        sampler.sample_compatible_pair(
            pick_filter=AssetFilter(category="nonexistent"),
            place_filter=AssetFilter(tags=frozenset({"container"})),
            rng=np.random.default_rng(0),
        )


def test_sample_compatible_pair_exhaustion_raises_collision_exhausted(
    mini_asset_root: Path,
):
    """When pick and place filters resolve to the same single asset.

    Retries exhaust and CollisionExhaustedError is raised.
    """
    reg = AssetRegistry(str(mini_asset_root))
    sampler = AssetSampler(reg)
    # box_001 is the only dual-role asset in the fixture.
    # Constrain both filters to only it.
    only_box = frozenset({"u-box-001"})
    from robo_orchard_sim.asset_manager.registry.errors import (
        CollisionExhaustedError,
        EmptyPoolError,
    )

    with pytest.raises(CollisionExhaustedError) as ei:
        sampler.sample_compatible_pair(
            pick_filter=AssetFilter(only_in=only_box),
            place_filter=AssetFilter(only_in=only_box),
            rng=np.random.default_rng(0),
        )
    # Also verify it's a subclass of EmptyPoolError for backward compat
    assert isinstance(ei.value, EmptyPoolError)
    assert ei.value.attempts == 10
