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

from robo_orchard_sim.asset_manager.registry.types import (
    AssetFilter,
    AssetMeta,
    DistractorSpec,
)


def _meta(**overrides) -> AssetMeta:
    d = dict(
        uuid="u1",
        asset_id="apple_001",
        relative_path="food/fruits/apple_001",
        domain="food",
        super_category="fruits",
        category="apple",
        name="red apple",
        description="",
        color="red",
        shape="sphere",
        material="organic",
        real_height=0.08,
        real_mass=0.15,
        min_height=0.05,
        max_height=0.1,
        min_mass=0.1,
        max_mass=0.2,
        usd_path="",
        urdf_path="",
        interaction_path="",
        caption_path="",
        tags=frozenset({"graspable"}),
    )
    d.update(overrides)
    return AssetMeta(**d)


def test_empty_filter_matches_all():
    assert AssetFilter().matches(_meta())


def test_tags_and_match():
    """AssetFilter.tags requires ALL tags to be present on the meta."""
    f = AssetFilter(tags=frozenset({"graspable"}))
    assert f.matches(_meta())
    assert not f.matches(_meta(tags=frozenset({"container"})))

    # multi-tag filter is AND: meta must contain every tag
    multi = AssetFilter(tags=frozenset({"graspable", "container"}))
    assert multi.matches(
        _meta(tags=frozenset({"graspable", "container", "stackable"}))
    )
    assert not multi.matches(_meta(tags=frozenset({"graspable"})))


def test_tags_accepts_list_or_set():
    """Convenience: lists/sets get normalised to frozenset."""
    f = AssetFilter(tags=["graspable"])
    assert f.matches(_meta())
    assert isinstance(f.tags, frozenset)


def test_category_filter():
    f = AssetFilter(category="apple")
    assert f.matches(_meta())
    assert not f.matches(_meta(category="orange"))


def test_super_category_filter():
    f = AssetFilter(super_category="fruits")
    assert f.matches(_meta())
    assert not f.matches(_meta(super_category="vegetables"))


def test_attribute_filters():
    assert AssetFilter(color="red").matches(_meta())
    assert not AssetFilter(color="green").matches(_meta())
    assert AssetFilter(shape="sphere").matches(_meta())
    assert AssetFilter(material="organic").matches(_meta())


def test_size_bucket_filter():
    assert AssetFilter(size_bucket="medium").matches(_meta(real_height=0.08))
    assert not AssetFilter(size_bucket="large").matches(
        _meta(real_height=0.08)
    )


def test_only_in_whitelist():
    f = AssetFilter(only_in=frozenset({"u1", "u2"}))
    assert f.matches(_meta(uuid="u1"))
    assert not f.matches(_meta(uuid="u3"))


def test_exclude_blacklist():
    f = AssetFilter(exclude=frozenset({"u1"}))
    assert not f.matches(_meta(uuid="u1"))
    assert f.matches(_meta(uuid="u2"))


def test_empty_only_in_matches_nothing():
    f = AssetFilter(only_in=frozenset())
    assert not f.matches(_meta(uuid="u1"))


def test_distractor_spec_defaults():
    s = DistractorSpec(min_count=3, max_count=3)
    assert s.match == ()
    assert s.differ == ()
    assert isinstance(s.absolute_filter, AssetFilter)
    assert s.only_in is None
    assert s.exclude == frozenset()


def test_distractor_spec_accepts_match_and_differ():
    s = DistractorSpec(
        min_count=3,
        max_count=3,
        match=("super_category",),
        differ=("category",),
    )
    assert s.match == ("super_category",)
    assert s.differ == ("category",)
