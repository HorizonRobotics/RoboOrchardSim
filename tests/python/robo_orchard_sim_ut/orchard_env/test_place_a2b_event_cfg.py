# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Tests for PlaceA2BTask.get_event_cfg."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_templates.place_a2b_task import (
    PlaceA2BTask,
    PlaceA2BTaskParams,
)
from robo_orchard_sim.orchard_env.task_templates.task_params import (
    PoseRangeConfig,
    TaskLightResetConfig,
    TaskTextureResetConfig,
)


def _obj(name: str) -> RigidObjectSpec:
    return RigidObjectSpec(name=name, usd_path=f"/tmp/{name}.usd")


def test_place_a2b_event_cfg_multiple_candidates_emits_each_pose_actor():
    cfg = PlaceA2BTask(
        assets=TaskAssets(
            role_candidates={
                "pick": [_obj("pick_0"), _obj("pick_1")],
                "place": [_obj("place_0"), _obj("place_1")],
                "distractors": [_obj("distractor_0")],
            }
        )
    ).get_event_cfg()
    pose_term = cfg.terms["random_pose_event"]
    assert [asset.name for asset in pose_term.asset_cfgs] == [
        "objects/place_0",
        "objects/place_1",
        "objects/pick_0",
        "objects/pick_1",
        "objects/distractor_0",
    ]


def test_get_event_cfg_enabled_light_and_texture_adds_terms():
    cfg = PlaceA2BTask(
        assets=TaskAssets(
            role_candidates={
                "pick": [_obj("pick_object")],
                "place": [_obj("place_object")],
            }
        ),
        params=PlaceA2BTaskParams(
            light_reset=TaskLightResetConfig(
                enabled=True,
                asset_names=["background/dis_light"],
                distant_light={"asset_name": "dis_light"},
                randomize_intensity=True,
                intensity_range={"range": (1000.0, 5000.0)},
            ),
            texture_reset=TaskTextureResetConfig(
                enabled=True,
                asset_names=["background/table"],
            ),
        ),
    ).get_event_cfg()

    assert "light_reset_event" in cfg.terms
    assert "texture_reset_event" in cfg.terms


def test_place_a2b_task_params_legacy_pose_fields_raise_validation_error():
    with pytest.raises(ValidationError):
        PlaceA2BTaskParams(
            mode="drop",
            min_separation=0.07,
            pose_range=PoseRangeConfig(
                x=(0.1, 0.2),
                y=(-0.2, 0.4),
            ),
        )
