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


class TestPoolSize:
    """pool_size branching in target and distractor resolution."""

    def test_resolve_target_pool_returns_pool_spec(self, mini_resolver):
        """Target entry with pool_size>1 returns PoolSpec wrapping members."""
        from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec

        out = mini_resolver.resolve(
            {
                "pick": {
                    "filter": {"super_category": "fruits"},
                    "prim_name": "pick_object",
                    "pool_size": 3,
                },
            }
        )
        assert isinstance(out["pick"], PoolSpec)
        assert out["pick"].role_id == "pick_object"
        assert {m.name for m in out["pick"].members} == {
            f"pick_object_pool_{i}" for i in range(3)
        }

    def test_resolve_distractor_pool_returns_pool_spec(self, mini_resolver):
        """Distractor entry with pool_size returns PoolSpec."""
        from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec

        out = mini_resolver.resolve(
            {
                "pick": {
                    "filter": {
                        "super_category": "fruits",
                        "category": "apple",
                    },
                    "prim_name": "pick_object",
                },
                "distractors": {
                    "anchor": "pick",
                    "match": ["super_category"],
                    "min_count": 1,
                    "max_count": 1,
                    "pool_size": 2,
                    "prim_name_prefix": "distractor",
                },
            }
        )
        pool = out["distractors"]
        assert isinstance(pool, PoolSpec)
        assert pool.role_id == "distractor"
        assert pool.active_count == 1
        assert {m.name for m in pool.members} == {
            "distractor_pool_0",
            "distractor_pool_1",
        }

    def test_distractor_pool_size_with_zero_max_count_raises(
        self, mini_resolver
    ):
        """pool_size>0 + max_count=0 must reject — would leave stray actors."""
        with pytest.raises(AssetResolutionError, match="pool_size"):
            mini_resolver.resolve(
                {
                    "pick": {
                        "filter": {"category": "apple"},
                        "prim_name": "pick_object",
                    },
                    "distractors": {
                        "anchor": "pick",
                        "match": ["super_category"],
                        "min_count": 0,
                        "max_count": 0,
                        "pool_size": 2,
                        "prim_name_prefix": "d",
                    },
                }
            )

    def test_resolve_pools_are_uuid_disjoint(self, mini_registry):
        """Cross-pool UUID disjointness holds across multiple seeds."""
        cfg = {
            "pick": {
                "filter": {"super_category": "fruits"},
                "prim_name": "pick_object",
                "pool_size": 2,
            },
            "distractors": {
                "anchor": "pick",
                "match": ["super_category"],
                "min_count": 1,
                "max_count": 1,
                "pool_size": 1,
                "prim_name_prefix": "distractor",
            },
        }

        class FakeSpec:
            def __init__(self, *, name, usd_path="", **_):
                self.name = name
                self.usd_path = usd_path

        def _fake_build_spec(meta, *, name=None, **_):
            return FakeSpec(name=name or meta.asset_id, usd_path=meta.usd_path)

        for seed in range(20):
            with patch.object(
                mini_registry, "build_spec", side_effect=_fake_build_spec
            ):
                out = AssetResolver(
                    registry=mini_registry,
                    rng=np.random.default_rng(seed),
                ).resolve(cfg)
                pick_uuids = {m.usd_path for m in out["pick"].members}
                dist_uuids = {s.usd_path for s in out["distractors"]}
                assert pick_uuids.isdisjoint(dist_uuids), f"seed {seed}"


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
