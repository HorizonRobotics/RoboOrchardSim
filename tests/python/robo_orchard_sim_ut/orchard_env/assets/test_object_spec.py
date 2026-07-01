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

from robo_orchard_sim.orchard_env.assets import RigidObjectSpec


def test_rigid_object_spec_to_isaac_cfg_uses_object_spec_shared_fields() -> (
    None
):
    spec = RigidObjectSpec(
        name="pick_object",
        usd_path="/tmp/pick.usd",
        interaction_path="/tmp/pick_interaction.json",
        caption_path="/tmp/pick_caption.json",
        scale=(2.0, 3.0, 4.0),
        mass=0.7,
        uuid="pick-uuid",
        category="apple",
        actor_type="pick_target",
        attributes={
            "color": ("red",),
            "shape": ("round",),
            "material": ("organic",),
        },
    )

    cfg = spec.to_isaac_cfg()

    assert cfg.spawn.usd_path == "/tmp/pick.usd"
    assert cfg.spawn.scale == (2.0, 3.0, 4.0)
    assert cfg.spawn.mass_props.mass == 0.7
    assert cfg.interaction_path == "/tmp/pick_interaction.json"
    assert cfg.caption_path == "/tmp/pick_caption.json"
    assert cfg.uuid == "pick-uuid"
    assert cfg.category == "apple"
    assert cfg.actor_type == "pick_target"
    assert cfg.attributes == {
        "color": ("red",),
        "shape": ("round",),
        "material": ("organic",),
    }
    assert spec.caption_path == "/tmp/pick_caption.json"
