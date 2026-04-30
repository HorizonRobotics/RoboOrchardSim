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

"""Tests for AssetRegistry.build_spec (Task 11)."""

from pathlib import Path

from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.orchard_env.assets import RigidObjectSpec


def test_build_spec_defaults_from_meta(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    meta = reg.get_by_asset_id("apple_001")
    spec = reg.build_spec(meta, role="target")
    assert isinstance(spec, RigidObjectSpec)
    assert spec.name == "apple_001"
    assert spec.uuid == meta.uuid
    assert spec.category == meta.category
    assert spec.actor_type == "target"
    assert spec.attributes == ()
    assert spec.usd_path == meta.usd_path
    assert spec.caption_path == meta.caption_path
    assert spec.interaction_path == meta.interaction_path
    # Mass flows through from URDF extra_info; pose remains unset so
    # downstream modules own runtime placement.
    assert spec.mass == meta.real_mass
    assert spec.initial_pos is None
    assert spec.initial_rot is None


def test_build_spec_name_override(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    meta = reg.get_by_asset_id("apple_001")
    spec = reg.build_spec(meta, name="pick_object", role="target")
    assert spec.name == "pick_object"


def test_build_spec_role_sets_actor_type(mini_asset_root: Path):
    reg = AssetRegistry(str(mini_asset_root))
    meta = reg.get_by_asset_id("apple_001")
    spec = reg.build_spec(meta, role="pick_target")
    assert spec.actor_type == "pick_target"
