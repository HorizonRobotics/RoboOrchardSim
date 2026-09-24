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
task's declared ``roles`` and tested separately.
"""

from __future__ import annotations
from unittest.mock import patch

import numpy as np
import pytest

from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
    AssetResolutionError,
    AssetResolver,
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
    def test_sample_without_replacement_same_seed_returns_same_unique_order(
        self,
        mini_registry,
    ):
        candidates = tuple("abcdefghij")
        first = AssetResolver(
            registry=mini_registry,
            rng=np.random.default_rng(42),
        ).sample_without_replacement(candidates, count=4)
        replay = AssetResolver(
            registry=mini_registry,
            rng=np.random.default_rng(42),
        ).sample_without_replacement(candidates, count=4)

        assert first == replay
        assert len(first) == len(set(first)) == 4

    @pytest.mark.parametrize("count", [-1, 11])
    def test_sample_without_replacement_invalid_count_raises_value_error(
        self,
        mini_resolver,
        count,
    ):
        with pytest.raises(ValueError, match="Cannot draw"):
            mini_resolver.sample_without_replacement(
                tuple("abcdefghij"),
                count=count,
            )

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

    def test_resolve_target_missing_filter_treated_as_match_all(
        self, mini_resolver
    ):
        """Missing `filter` key == empty == match-all (Option 2 ergonomics)."""
        configs = {
            "pick": {
                "prim_name": "pick_object",
            },
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].name == "pick_object"

    def test_resolve_target_null_filter_treated_as_match_all(
        self, mini_resolver
    ):
        """`filter: null` (YAML) == empty == match-all."""
        configs = {
            "pick": {
                "filter": None,
                "prim_name": "pick_object",
            },
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].name == "pick_object"

    def test_resolve_target_non_dict_filter_raises(self, mini_resolver):
        """Non-dict filter (e.g. string) is still a hard error."""
        configs = {
            "pick": {
                "filter": "graspable",
                "prim_name": "pick_object",
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "pick"
        assert isinstance(exc_info.value.cause, TypeError)

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


class TestSampleCount:
    """Ordinary multi-asset sampling."""

    def test_target_sample_count_returns_active_spec_list(
        self,
        mini_resolver,
    ):
        out = mini_resolver.resolve(
            {
                "pick": {
                    "filter": {"super_category": "fruits"},
                    "prim_name": "pick_object",
                    "sample_count": 3,
                },
            }
        )

        assert isinstance(out["pick"], list)
        assert [spec.name for spec in out["pick"]] == [
            "pick_object_0",
            "pick_object_1",
            "pick_object_2",
        ]

    def test_multi_sample_rejects_pinned_uuid(self, mini_resolver):
        with pytest.raises(
            AssetResolutionError,
            match="requires registry filter sampling",
        ):
            mini_resolver.resolve(
                {
                    "pick": {
                        "uuid": "apple-001",
                        "prim_name": "pick_object",
                        "sample_count": 2,
                    },
                }
            )


@pytest.mark.parametrize(
    "role,entry",
    [
        (
            "pick",
            {
                "filter": {"super_category": "fruits"},
                "prim_name": "pick_object",
                "pool_size": 2,
            },
        ),
        (
            "distractors",
            {
                "anchor": "pick",
                "min_count": 1,
                "max_count": 1,
                "pool_size": 2,
            },
        ),
    ],
)
def test_asset_resolver_pool_size_entry_raises_unknown_key(
    mini_resolver,
    role,
    entry,
):
    configs = {role: entry}
    if role == "distractors":
        configs["pick"] = {
            "filter": {"category": "apple"},
            "prim_name": "pick_object",
        }
    with pytest.raises(AssetResolutionError, match="Unknown entry key"):
        mini_resolver.resolve(configs)


class TestResolveByUuid:
    """`uuid` entry key pins a target to a specific registry asset."""

    def test_resolve_by_uuid_pins_specific_asset(self, mini_resolver):
        configs = {
            "pick": {
                "uuid": "u-banana-001",
                "prim_name": "pick_object",
            },
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].name == "pick_object"
        assert result["pick"].usd_path.endswith("banana_001.usd")

    def test_resolve_by_uuid_filter_optional(self, mini_resolver):
        """`filter` is optional when `uuid` is given."""
        configs = {
            "pick": {
                "uuid": "u-orange-001",
                "prim_name": "pick_object",
            },
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].usd_path.endswith("orange_001.usd")

    def test_resolve_by_uuid_with_consistent_filter_ok(self, mini_resolver):
        configs = {
            "pick": {
                "uuid": "u-apple-001",
                "filter": {"category": "apple", "color": "red"},
                "prim_name": "pick_object",
            },
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].usd_path.endswith("apple_001.usd")

    def test_resolve_by_uuid_with_conflicting_filter_warns_but_resolves(
        self, mini_resolver, caplog
    ):
        """Uuid takes precedence; mismatched filter only emits a warning."""
        import logging

        configs = {
            "pick": {
                "uuid": "u-apple-001",
                "filter": {"category": "orange"},
                "prim_name": "pick_object",
            },
        }
        with caplog.at_level(logging.WARNING):
            result = mini_resolver.resolve(configs)
        assert result["pick"].usd_path.endswith("apple_001.usd")
        assert any(
            "u-apple-001" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_resolve_by_uuid_with_conflicting_filter_no_warn_on_match(
        self, mini_resolver, caplog
    ):
        """No warning when uuid matches the filter."""
        import logging

        configs = {
            "pick": {
                "uuid": "u-apple-001",
                "filter": {"category": "apple"},
                "prim_name": "pick_object",
            },
        }
        with caplog.at_level(logging.WARNING):
            mini_resolver.resolve(configs)
        assert not any("u-apple-001" in r.message for r in caplog.records)

    def test_resolve_by_uuid_unknown_raises(self, mini_resolver):
        configs = {
            "pick": {
                "uuid": "u-not-real",
                "prim_name": "pick_object",
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "pick"
        assert "u-not-real" in str(exc_info.value.cause)

    def test_resolve_by_uuid_inside_split_ok(
        self, mini_resolver_with_splits, mini_registry
    ):
        apple_001_uuid = mini_registry.resolve_asset_id("apple_001")
        configs = {
            "pick": {
                "uuid": apple_001_uuid,
                "prim_name": "pick_object",
                "split": "seen",
            },
        }
        result = mini_resolver_with_splits.resolve(configs)
        assert result["pick"].usd_path.endswith("apple_001.usd")

    def test_resolve_by_uuid_outside_split_warns_but_resolves(
        self, mini_resolver_with_splits, mini_registry, caplog
    ):
        """Uuid takes precedence; mismatched split only emits a warning."""
        import logging

        plate_uuid = mini_registry.resolve_asset_id("plate_001")
        configs = {
            "pick": {
                "uuid": plate_uuid,
                "prim_name": "pick_object",
                "split": "seen",
            },
        }
        with caplog.at_level(logging.WARNING):
            result = mini_resolver_with_splits.resolve(configs)
        assert result["pick"].usd_path.endswith("plate_001.usd")
        assert any(
            plate_uuid in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_resolve_by_uuid_still_requires_prim_name(self, mini_resolver):
        configs = {
            "pick": {
                "uuid": "u-apple-001",
            },
        }
        with pytest.raises(AssetResolutionError) as exc_info:
            mini_resolver.resolve(configs)
        assert exc_info.value.role == "pick"
        assert isinstance(exc_info.value.cause, KeyError)


class TestResolveByUsdPath:
    def test_usd_path_pins_target_and_keeps_prim_name(self, mini_resolver):
        configs = {
            "pick": {
                "usd_path": "/assets/mug/variants/variants.usd",
                "prim_name": "pick_object",
            }
        }
        result = mini_resolver.resolve(configs)
        assert result["pick"].name == "pick_object"

    def test_usd_path_conflicts_with_uuid_raises(self, mini_resolver):
        configs = {
            "pick": {
                "usd_path": "/assets/mug/variants/variants.usd",
                "uuid": "deadbeef",
                "prim_name": "pick_object",
            }
        }
        with pytest.raises(AssetResolutionError):
            mini_resolver.resolve(configs)

    def test_usd_path_conflicts_with_filter_raises(self, mini_resolver):
        configs = {
            "pick": {
                "usd_path": "/assets/mug/variants/variants.usd",
                "filter": {"tags": ["graspable"]},
                "prim_name": "pick_object",
            }
        }
        with pytest.raises(AssetResolutionError):
            mini_resolver.resolve(configs)


class TestActiveSnapshot:
    """active_snapshot restricts resolved assets to the snapshot uuid set."""

    @staticmethod
    def _build_spec_passthrough(meta, **_kw):
        return meta

    def test_resolve_active_snapshot_restricts_pool_to_snapshot(
        self, mini_registry
    ):
        with patch.object(
            mini_registry,
            "build_spec",
            side_effect=self._build_spec_passthrough,
        ):
            resolver = AssetResolver(
                registry=mini_registry,
                active_snapshot=frozenset({"u-apple-001"}),
                rng=np.random.default_rng(42),
            )
            result = resolver.resolve(
                {"pick": {"filter": {}, "prim_name": "x"}}
            )
        assert result["pick"].usd_path.endswith("apple_001.usd")

    def test_resolve_active_snapshot_intersects_split_returns_overlap(
        self, mini_registry
    ):
        from robo_orchard_sim.asset_manager.splits.splits import AssetSplits

        splits = AssetSplits(
            name="t",
            seen=frozenset({"u-apple-001", "u-apple-002", "u-orange-001"}),
        )
        with patch.object(
            mini_registry,
            "build_spec",
            side_effect=self._build_spec_passthrough,
        ):
            resolver = AssetResolver(
                registry=mini_registry,
                splits=splits,
                active_snapshot=frozenset({"u-apple-002", "u-box-001"}),
                rng=np.random.default_rng(42),
            )
            result = resolver.resolve(
                {"pick": {"filter": {}, "prim_name": "x", "split": "seen"}}
            )
        # seen ∩ snapshot = {u-apple-002}
        assert result["pick"].usd_path.endswith("apple_002.usd")

    def test_resolve_active_snapshot_disjoint_from_split_raises(
        self, mini_registry
    ):
        from robo_orchard_sim.asset_manager.splits.splits import AssetSplits

        splits = AssetSplits(name="t", seen=frozenset({"u-apple-001"}))
        with patch.object(
            mini_registry,
            "build_spec",
            side_effect=self._build_spec_passthrough,
        ):
            resolver = AssetResolver(
                registry=mini_registry,
                splits=splits,
                active_snapshot=frozenset({"u-orange-001"}),
                rng=np.random.default_rng(42),
            )
            with pytest.raises(AssetResolutionError, match="snapshot"):
                resolver.resolve(
                    {
                        "pick": {
                            "filter": {},
                            "prim_name": "x",
                            "split": "seen",
                        }
                    }
                )


class TestSameAs:
    """Several slots holding one asset, so only wording tells them apart.

    Slots normally each get their own asset, which is what keeps a scene
    varied. A task that asks which of four identical objects the
    instruction means needs the opposite, and cannot get it by narrowing
    a filter: the de-duplication would still hand out four different
    ones, or run the pool dry trying.
    """

    def test_clones_receive_the_asset_their_source_drew(self, mini_resolver):
        resolved = mini_resolver.resolve(
            {
                "pick_near": {
                    "filter": {"tags": ["graspable"]},
                    "prim_name": "pick_near",
                },
                "pick_far": {"same_as": "pick_near", "prim_name": "pick_far"},
                "clutter_0": {
                    "same_as": "pick_near",
                    "prim_name": "clutter_0",
                },
            }
        )

        assert {spec.usd_path for spec in resolved.values()} == {
            resolved["pick_near"].usd_path
        }
        assert [resolved[key].name for key in sorted(resolved)] == [
            "clutter_0",
            "pick_far",
            "pick_near",
        ]

    def test_a_clone_may_be_written_before_its_source(self, mini_resolver):
        """Order in the file says nothing about which slot draws."""
        resolved = mini_resolver.resolve(
            {
                "pick_far": {"same_as": "pick_near", "prim_name": "pick_far"},
                "pick_near": {
                    "filter": {"tags": ["graspable"]},
                    "prim_name": "pick_near",
                },
            }
        )
        assert resolved["pick_far"].usd_path == resolved["pick_near"].usd_path

    def test_cloning_leaves_other_slots_their_own_assets(self, mini_resolver):
        """Sharing is confined to the slots that ask for it.

        A scene usually wants its other objects to look different, so
        cloning must not spread: the slots that named a source share
        one asset, and the slot that did not still draws its own.
        """
        resolved = mini_resolver.resolve(
            {
                "pick": {
                    "filter": {"tags": ["container"]},
                    "prim_name": "pick",
                },
                "copy_a": {"same_as": "pick", "prim_name": "copy_a"},
                "copy_b": {"same_as": "pick", "prim_name": "copy_b"},
                "other": {
                    "filter": {"tags": ["container"]},
                    "prim_name": "other",
                },
            }
        )
        assert resolved["pick"].usd_path == resolved["copy_a"].usd_path
        assert resolved["pick"].usd_path == resolved["copy_b"].usd_path
        assert resolved["other"].usd_path != resolved["pick"].usd_path

    def test_the_source_still_honours_its_split(
        self, mini_resolver_with_splits
    ):
        """Sharing does not cost the run its split.

        Pinning a uuid would also give every slot one asset, but a
        pinned uuid overrides the split, so an evaluation could no
        longer be pointed at unseen assets.
        """
        resolved = mini_resolver_with_splits.resolve(
            {
                "pick": {
                    "filter": {},
                    "prim_name": "pick",
                    "split": "unseen_instance",
                },
                "pick_copy": {"same_as": "pick", "prim_name": "pick_copy"},
            }
        )
        assert "carrot_001" in resolved["pick"].usd_path
        assert resolved["pick_copy"].usd_path == resolved["pick"].usd_path

    def test_naming_a_slot_that_draws_nothing_is_rejected(self, mini_resolver):
        with pytest.raises(AssetResolutionError, match="not a slot that"):
            mini_resolver.resolve(
                {"solo": {"same_as": "absent", "prim_name": "solo"}}
            )

    def test_naming_another_clone_is_rejected_with_a_hint(self, mini_resolver):
        """Chains are refused, and the message says where to point."""
        with pytest.raises(AssetResolutionError, match="itself a clone"):
            mini_resolver.resolve(
                {
                    "source": {
                        "filter": {"tags": ["graspable"]},
                        "prim_name": "source",
                    },
                    "middle": {"same_as": "source", "prim_name": "middle"},
                    "end": {"same_as": "middle", "prim_name": "end"},
                }
            )

    @pytest.mark.parametrize(
        "extra",
        [
            {"filter": {"tags": ["graspable"]}},
            {"uuid": "u-apple-001"},
            {"split": "seen"},
            {"sample_count": 2},
        ],
    )
    def test_a_clone_that_also_asks_to_draw_is_rejected(
        self, mini_resolver, extra
    ):
        """Silently ignoring the drawing keys would mislead the author."""
        with pytest.raises(AssetResolutionError, match="could not be"):
            mini_resolver.resolve(
                {
                    "source": {
                        "filter": {"tags": ["graspable"]},
                        "prim_name": "source",
                    },
                    "clone": {
                        "same_as": "source",
                        "prim_name": "clone",
                        **extra,
                    },
                }
            )

    def test_naming_a_source_that_draws_several_is_rejected(
        self, mini_resolver
    ):
        """With several assets drawn there is no single one to share."""
        with pytest.raises(AssetResolutionError, match="there is no"):
            mini_resolver.resolve(
                {
                    "pool": {
                        "filter": {"tags": ["graspable"]},
                        "prim_name": "pool",
                        "sample_count": 2,
                    },
                    "clone": {"same_as": "pool", "prim_name": "clone"},
                }
            )

    def test_a_clone_still_needs_a_name(self, mini_resolver):
        with pytest.raises(AssetResolutionError, match="prim_name"):
            mini_resolver.resolve(
                {
                    "source": {
                        "filter": {"tags": ["graspable"]},
                        "prim_name": "source",
                    },
                    "clone": {"same_as": "source"},
                }
            )
