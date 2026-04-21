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

"""Tests for AssetResolver.

The resolver is task-agnostic: it transforms ``dict[role, config]`` into
``dict[role, AssetSpec | list[AssetSpec]]``. Role membership semantics
(required / optional / unknown) are owned by the calling task's
``TaskAssetsBase`` subclass and tested separately.
"""

from __future__ import annotations

import pytest

from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
    AssetResolutionError,
)


class TestResolveHappyPath:
    def test_resolve_returns_specs_keyed_by_role(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"tags": ["graspable"]},
                "prim_name": "pick_object",
            },
            "place": {
                "filter": {"tags": ["container"]},
                "prim_name": "place_object",
            },
        }
        result = mini_resolver.resolve(configs)
        assert set(result.keys()) == {"pick", "place"}
        assert result["pick"].name == "pick_object"
        assert result["place"].name == "place_object"

    def test_resolve_arbitrary_role_names_are_passed_through(
        self, mini_resolver
    ):
        """Resolver does not gatekeep role names — task layer does."""
        configs = {
            "pick": {
                "filter": {"tags": ["graspable"]},
                "prim_name": "pick_object",
            },
            "place": {
                "filter": {"tags": ["container"]},
                "prim_name": "place_object",
            },
            "extra_role_unknown_to_any_task": {
                "filter": {"tags": ["graspable"]},
                "prim_name": "extra_object",
            },
        }
        result = mini_resolver.resolve(configs)
        assert set(result.keys()) == {
            "pick",
            "place",
            "extra_role_unknown_to_any_task",
        }


class TestSamplingAndSpec:
    def test_resolve_produces_correct_spec_name(self, mini_resolver):
        configs = {
            "pick": {"filter": {"category": "apple"}, "prim_name": "my_apple"},
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].name == "my_apple"

    def test_resolve_spec_defaults_only_sets_name(self, mini_resolver):
        """Resolver only sets name from prim_name; pose/physics come later."""
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
            },
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].name == "pick_object"
        # Pose left unset — other subsystems (pose_reset event, etc.) own
        # runtime placement.
        assert result["pick"].initial_pos is None
        assert result["pick"].initial_rot is None


class TestSplitsInjection:
    def test_resolve_with_splits_defaults_to_seen(
        self, mini_resolver_with_splits
    ):
        configs = {
            "pick": {
                "filter": {"tags": ["graspable"]},
                "prim_name": "pick_object",
            },
        }
        result = mini_resolver_with_splits.resolve(configs)
        assert result["pick"].name == "pick_object"

    def test_resolve_with_split_unseen_category(
        self, mini_resolver_with_splits
    ):
        configs = {
            "pick": {
                "filter": {},
                "prim_name": "unseen_pick",
                "split": "unseen_category",
            },
        }
        result = mini_resolver_with_splits.resolve(configs)
        assert result["pick"].usd_path.endswith("plate_001.usd")

    def test_resolve_without_splits_ignores_split_field(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
                "split": "unseen_category",
            },
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].name == "pick_object"


class TestErrorPaths:
    def test_resolve_empty_pool_raises(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "nonexistent_category"},
                "prim_name": "pick_object",
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "pick"
        assert exc_info.value.cause is not None

    def test_resolve_invalid_filter_field_raises(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"bogus_field": "value"},
                "prim_name": "pick_object",
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "pick"


class TestDistractors:
    def test_resolve_distractors_returns_list_of_count(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
            },
            "distractors": {
                "anchor": "pick",
                "match": ["super_category"],
                "differ": ["category"],
                "min_count": 2,
                "max_count": 2,
                "prim_name_prefix": "distractor",
            },
        }
        result = mini_resolver.resolve(configs)
        assert "distractors" in result
        assert isinstance(result["distractors"], list)
        assert len(result["distractors"]) == 2
        assert result["distractors"][0].name == "distractor_0"
        assert result["distractors"][1].name == "distractor_1"

    def test_resolve_distractors_differ_super_category(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
            },
            "distractors": {
                "anchor": "pick",
                "differ": ["super_category"],
                "min_count": 2,
                "max_count": 2,
                "prim_name_prefix": "d",
            },
        }
        result = mini_resolver.resolve(configs)
        assert len(result["distractors"]) == 2

    def test_resolve_distractors_unknown_anchor_raises(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
            },
            "distractors": {
                "anchor": "nonexistent",
                "match": ["super_category"],
                "differ": ["category"],
                "min_count": 1,
                "max_count": 1,
                "prim_name_prefix": "d",
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "distractors"
        assert "nonexistent" in str(exc_info.value.cause)

    def test_resolve_distractors_insufficient_pool_raises(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
            },
            "distractors": {
                "anchor": "pick",
                "match": ["super_category"],
                "differ": ["category"],
                # same-super-category (fruits) non-apple is orange only,
                # so asking for 5 must fail.
                "min_count": 5,
                "max_count": 5,
                "prim_name_prefix": "d",
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "distractors"

    def test_resolve_distractors_preserves_target_result(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "my_apple",
            },
            "distractors": {
                "anchor": "pick",
                "match": ["super_category"],
                "differ": ["category"],
                "min_count": 1,
                "max_count": 1,
                "prim_name_prefix": "d",
            },
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].name == "my_apple"
        assert len(result["distractors"]) == 1

    def test_resolve_distractors_with_splits_only_in(
        self, mini_resolver_with_splits
    ):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
                "split": "seen",
            },
            "distractors": {
                "anchor": "pick",
                "match": ["super_category"],
                "differ": ["category"],
                "min_count": 1,
                "max_count": 1,
                "prim_name_prefix": "d",
                "split": "seen",
            },
        }
        result = mini_resolver_with_splits.resolve(configs)
        assert len(result["distractors"]) == 1

    def test_resolve_distractors_with_absolute_filter(self, mini_resolver):
        """filter: in distractor entry further narrows the pool."""
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
            },
            "distractors": {
                "anchor": "pick",
                "differ": ["super_category"],
                "filter": {"tags": ["container"]},
                "min_count": 1,
                "max_count": 1,
                "prim_name_prefix": "d",
            },
        }
        result = mini_resolver.resolve(configs)
        assert len(result["distractors"]) == 1


class TestConfigShapeValidation:
    """Config-shape errors surface as AssetResolutionError."""

    def test_resolve_unknown_entry_key_raises(self, mini_resolver):
        """Entry-level key typos surface as AssetResolutionError."""
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
                "macth": ["super_category"],  # typo of match (target entry)
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "pick"
        assert "macth" in str(exc_info.value.cause)

    def test_resolve_unknown_distractor_entry_key_raises(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
            },
            "distractors": {
                "anchor": "pick",
                "match": ["super_category"],
                "differ": ["category"],
                "min_count": 1,
                "max_count": 1,
                "mode": "similar_semantic",  # legacy key, now unknown
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "distractors"
        assert "mode" in str(exc_info.value.cause)

    def test_resolve_target_missing_filter_wrapped(self, mini_resolver):
        configs = {
            "pick": {
                "prim_name": "pick_object",
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "pick"
        assert isinstance(exc_info.value.cause, KeyError)

    def test_resolve_target_missing_prim_name_wrapped(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "pick"
        assert isinstance(exc_info.value.cause, KeyError)

    def test_resolve_distractors_missing_count_wrapped(self, mini_resolver):
        configs = {
            "pick": {
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
            },
            "distractors": {
                "anchor": "pick",
                "match": ["super_category"],
                "differ": ["category"],
                "prim_name_prefix": "d",
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "distractors"
        assert isinstance(exc_info.value.cause, KeyError)
