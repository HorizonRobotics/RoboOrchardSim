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

"""Tests for PickTask light reset task-parameter wiring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.task_templates.pick_task import (
    PickAssets,
    PickTask,
    PickTaskParams,
)
from robo_orchard_sim.orchard_env.task_templates.task_params import (
    TaskDistantLightConfig,
    TaskLightResetConfig,
    TaskPoseResetConfig,
    TaskTextureResetConfig,
)


def _obj(name: str) -> RigidObjectSpec:
    return RigidObjectSpec(name=name, usd_path=f"/tmp/{name}.usd")


def test_pick_task_get_event_cfg_light_reset_disabled_omits_light_term():
    cfg = PickTask(
        assets=PickAssets(pick=_obj("pick_object")),
        params=PickTaskParams(),
    ).get_event_cfg()

    assert "random_pose_event" in cfg.terms
    assert "light_reset_event" not in cfg.terms


def test_pick_task_get_event_cfg_light_reset_enabled_adds_light_term():
    cfg = PickTask(
        assets=PickAssets(pick=_obj("pick_object")),
        params=PickTaskParams(
            light_reset=TaskLightResetConfig(
                enabled=True,
                asset_names=["task_light"],
                randomize_color=True,
                color_temperature_range={"range": (3500.0, 6500.0)},
                rgb_noise=0.05,
            )
        ),
    ).get_event_cfg()

    assert "light_reset_event" in cfg.terms
    light_term = cfg.terms["light_reset_event"]
    assert [asset_cfg.name for asset_cfg in light_term.asset_cfgs] == [
        "task_light"
    ]
    assert light_term.trigger_topic == "reset"
    assert light_term.randomize_color is True
    assert light_term.rgb_noise == 0.05
    assert light_term.color_temperature_range.range == (3500.0, 6500.0)


def test_pick_task_get_event_cfg_texture_reset_enabled_adds_texture_term():
    cfg = PickTask(
        assets=PickAssets(pick=_obj("pick_object")),
        params=PickTaskParams(
            texture_reset=TaskTextureResetConfig(
                enabled=True,
                asset_names=["table"],
                variant_set_name="Look",
                variant_sort=True,
                variant_index_range=[1, 4],
            )
        ),
    ).get_event_cfg()

    assert "texture_reset_event" in cfg.terms
    texture_term = cfg.terms["texture_reset_event"]
    assert [asset_cfg.name for asset_cfg in texture_term.asset_cfgs] == [
        "table"
    ]
    assert texture_term.trigger_topic == "reset"
    assert texture_term.variant_set_name == "Look"
    assert texture_term.variant_sort is True
    assert texture_term.variant_index_range == [1, 4]


def test_pick_task_params_light_reset_dict_parses_nested_config():
    params = PickTaskParams(
        pose_reset={
            "mode": "drop",
            "min_separation": 0.07,
            "pose_range": {
                "x": (0.1, 0.2),
                "y": (-0.2, 0.4),
                "z": (0.01, 0.02),
                "roll": (0.0, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-1.0, 1.5),
            },
        },
        light_reset={
            "enabled": True,
            "preset": "default_distant_light",
            "randomize_intensity": True,
            "intensity_range": {"range": (1000.0, 5000.0)},
        },
        texture_reset={
            "enabled": True,
            "preset": "default_table_texture",
            "variant_index_range": [0, 3],
        },
    )

    assert isinstance(params.pose_reset, TaskPoseResetConfig)
    assert params.pose_reset.mode == "drop"
    assert params.pose_reset.min_separation == 0.07
    assert params.pose_reset.pose_range.x == (0.1, 0.2)
    assert isinstance(params.light_reset, TaskLightResetConfig)
    assert params.light_reset.enabled is True
    assert params.light_reset.preset == "default_distant_light"
    assert params.light_reset.asset_names == ["background/dis_light"]
    assert isinstance(params.light_reset.distant_light, TaskDistantLightConfig)
    assert params.light_reset.distant_light.asset_name == "dis_light"
    assert params.light_reset.randomize_intensity is True
    assert params.light_reset.intensity_range.range == (1000.0, 5000.0)
    assert isinstance(params.texture_reset, TaskTextureResetConfig)
    assert params.texture_reset.enabled is True
    assert params.texture_reset.preset == "default_table_texture"
    assert params.texture_reset.asset_names == ["background/table"]
    assert params.texture_reset.variant_index_range == [0, 3]


def test_pick_task_params_legacy_flat_pose_fields_are_ignored():
    params = PickTaskParams(
        mode="drop",
        min_separation=0.07,
        pose_range={
            "x": (0.1, 0.2),
            "y": (-0.2, 0.4),
            "z": (0.01, 0.02),
            "roll": (0.0, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-1.0, 1.5),
        },
    )

    assert params.pose_reset == TaskPoseResetConfig()
    assert not hasattr(params, "mode")
    assert not hasattr(params, "min_separation")
    assert not hasattr(params, "pose_range")


def test_pick_task_params_light_reset_missing_asset_names_raises_error():
    with pytest.raises(ValidationError):
        PickTaskParams(light_reset={"enabled": True, "asset_names": []})


def test_pick_task_params_light_reset_distant_light_name_mismatch_raises():
    with pytest.raises(ValidationError):
        PickTaskParams(
            light_reset={
                "enabled": True,
                "asset_names": ["background/light"],
                "distant_light": {"asset_name": "dis_light"},
                "randomize_intensity": True,
                "intensity_range": {"range": (1000.0, 5000.0)},
            }
        )


def test_pick_task_params_texture_reset_missing_asset_names_raises_error():
    with pytest.raises(ValidationError):
        PickTaskParams(texture_reset={"enabled": True, "asset_names": []})


def test_pick_task_params_light_reset_unknown_preset_raises_error():
    with pytest.raises(ValidationError):
        PickTaskParams(
            light_reset={"enabled": True, "preset": "unknown_preset"}
        )


def test_pick_task_params_texture_reset_unknown_preset_raises_error():
    with pytest.raises(ValidationError):
        PickTaskParams(
            texture_reset={"enabled": True, "preset": "unknown_preset"}
        )
