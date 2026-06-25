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

"""End-to-end smoke: resolver output matches PlaceA2BTaskAssets shape.

We do NOT construct PlaceA2BTaskAssets itself - that import pulls
in Isaac Sim via ObjectSpec's transitive deps. Instead we verify the
resolver output dict has exactly the keys PlaceA2BTaskAssets expects
(pick, place, distractors), with the right value types.
"""

from __future__ import annotations


def test_resolver_output_matches_place_a2b_assets_shape(mini_resolver):
    """Resolver output is structurally compatible with PlaceA2BTaskAssets."""
    asset_configs = {
        "pick": {
            "filter": {"tags": ["graspable"], "category": "apple"},
            "prim_name": "pick_object",
        },
        "place": {
            "filter": {"tags": ["container"]},
            "prim_name": "place_object",
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

    result = mini_resolver.resolve(asset_configs)

    # Keys exactly match PlaceA2BTaskAssets fields
    assert set(result.keys()) == {"pick", "place", "distractors"}

    # pick / place are single specs
    assert result["pick"].name == "pick_object"
    assert result["place"].name == "place_object"

    # distractors is list[spec] with the expected count and names
    assert isinstance(result["distractors"], list)
    assert len(result["distractors"]) == 2
    assert result["distractors"][0].name == "distractor_0"
    assert result["distractors"][1].name == "distractor_1"


def test_resolver_output_without_distractors_matches_required_only(
    mini_resolver,
):
    """PlaceA2BTaskAssets allows distractors=None; resolver can omit it."""
    asset_configs = {
        "pick": {
            "filter": {"tags": ["graspable"], "category": "apple"},
            "prim_name": "pick_object",
        },
        "place": {
            "filter": {"tags": ["container"]},
            "prim_name": "place_object",
        },
    }

    result = mini_resolver.resolve(asset_configs)

    assert set(result.keys()) == {"pick", "place"}
    assert "distractors" not in result
