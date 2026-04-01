# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

"""Place-a2b task definition built on ``TaskBase``."""

from __future__ import annotations
import math
from enum import Enum

from robo_orchard_core.envs.managers.events import EventManagerCfg

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.envs.managers.events.pose_reset import (
    PoseResetTermCfg,
)
from robo_orchard_sim.orchard_env.assets import AssetSpec, ObjectSpec
from robo_orchard_sim.orchard_env.tasks.task_base import TaskBase
from robo_orchard_sim.tasks.validators.base import Validator
from robo_orchard_sim.tasks.validators.checkers import (
    is_within_xy,
    lift,
    reach,
)


class PlaceA2BRole(str, Enum):
    """Supported asset roles for the place-a2b task."""

    PICK = "pick"
    PLACE = "place"
    OTHER = "other"


class PlaceA2BTask(TaskBase):
    """A generic place-a2b task with one pick object and one place object."""

    def __init__(
        self,
        assets: dict[PlaceA2BRole, AssetSpec] | None = None,
    ):
        if assets is None:
            raise ValueError(
                "PlaceA2BTask requires an assets dict mapping roles to "
                "AssetSpec instances."
            )
        self._validate_assets(assets)
        super().__init__(assets)
        self.pick_object = self._assets[PlaceA2BRole.PICK]
        self.place_object = self._assets[PlaceA2BRole.PLACE]

    def _validate_assets(self, assets: dict[PlaceA2BRole, AssetSpec]) -> None:
        """Require ``PICK`` and ``PLACE`` keys; other roles are allowed."""
        required = {PlaceA2BRole.PICK, PlaceA2BRole.PLACE}
        missing = required - set(assets)
        if missing:
            raise ValueError(
                "PlaceA2BTask assets must include "
                f"{PlaceA2BRole.PICK!r} and {PlaceA2BRole.PLACE!r}; "
                f"missing: {sorted(missing, key=lambda r: r.value)}."
            )
        for role in (PlaceA2BRole.PICK, PlaceA2BRole.PLACE):
            if not isinstance(assets[role], ObjectSpec):
                raise TypeError(
                    "PlaceA2BTask pick/place assets must be ObjectSpec "
                    "instances."
                )

    def get_event_cfg(self) -> EventManagerCfg:
        """Return pose-reset events for place and pick objects."""
        return EventManagerCfg(
            terms={
                "random_place_pose_event": PoseResetTermCfg(
                    asset_cfgs=[
                        SceneEntityCfg(name=self.place_object.scene_name)
                    ],
                    trigger_topic="reset",
                    mode="random_non_overlap",
                    pose_range={
                        "x": [0.5, 0.55],
                        "y": [-0.05, 0.05],
                        "z": [0.0, 0.0],
                        "roll": [0.0, 0.0],
                        "pitch": [0.0, 0.0],
                        "yaw": [math.radians(-5.0), math.radians(5.0)],
                    },
                    absolute_sampling=True,
                    min_separation=0.03,
                    max_retries=256,
                    group_key="manipulation_objects",
                    clear_cross_group_cache=True,
                ),
                "random_pick_pose_event": PoseResetTermCfg(
                    asset_cfgs=[
                        SceneEntityCfg(name=self.pick_object.scene_name)
                    ],
                    trigger_topic="reset",
                    # mode="drop",
                    mode="random_non_overlap",
                    pose_range={
                        "x": [0.25, 0.4],
                        "y": [-0.35, 0.35],
                        "z": [0.0, 0.5],
                        "roll": [0.0, 0.0],
                        "pitch": [0.0, 0.0],
                        "yaw": [math.radians(-180.0), math.radians(180.0)],
                    },
                    absolute_sampling=True,
                    min_separation=0.03,
                    max_retries=256,
                    group_key="manipulation_objects",
                    clear_cross_group_cache=False,
                ),
            }
        )

    def build_validator(self) -> Validator:
        """Build the task validator for place-a2b evaluation.

        Returns:
            Validator: Task-specific success/progress validator.
        """
        pick_name = self.pick_object.scene_name
        place_name = self.place_object.scene_name
        return Validator(
            actors=[pick_name, place_name],
            criteria=[
                reach(pick_name, 0.2),
                (lift(pick_name, 0.03), [0]),
                (is_within_xy(pick_name, place_name), [1]),
                (
                    is_within_xy(
                        pick_name, place_name, open_gripper_threshold=0.04
                    ),
                    [2],
                ),
            ],
            criteria_name=[
                "reach_pick",
                "lift_pick",
                "reach_place",
                "place_within_xy",
            ],
        )
