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

from robo_orchard_core.envs.managers.events import EventManagerCfg

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.envs.managers.events.pose_reset import (
    PoseResetTermCfg,
)
from robo_orchard_sim.orchard_env.assets import ObjectSpec
from robo_orchard_sim.orchard_env.tasks.task_base import (
    TaskAssetsBase,
    TaskBase,
)
from robo_orchard_sim.tasks.validators.base import Validator
from robo_orchard_sim.tasks.validators.checkers import (
    is_within_xy,
    lift,
    reach,
)


class PlaceA2BTaskAssets(TaskAssetsBase):
    """Task-specific asset schema for place-a2b scenes."""

    required_object_fields = ("pick", "place")

    pick: ObjectSpec
    place: ObjectSpec

    def flatten(self) -> dict[str, ObjectSpec]:
        """Return task assets in the flattened shape expected by TaskBase."""
        flattened: dict[str, ObjectSpec] = {
            "pick": self.pick,
            "place": self.place,
        }
        flattened.update(self.flatten_distractors())
        return flattened


class PlaceA2BTask(TaskBase):
    """A generic place-a2b task with one pick object and one place object."""

    def __init__(self, assets: PlaceA2BTaskAssets):
        self.assets = assets
        flattened_assets = assets.flatten()
        super().__init__(flattened_assets)

        self.pick_object = self._assets["pick"]
        self.place_object = self._assets["place"]
        self.distractors = [
            self._assets[role]
            for role in flattened_assets
            if role.startswith("distractor_")
        ]

    def get_event_cfg(self) -> EventManagerCfg:
        """Return a shared pose-reset event for task objects."""
        asset_cfgs = [
            SceneEntityCfg(name=self.place_object.scene_name),
            SceneEntityCfg(name=self.pick_object.scene_name),
        ]
        asset_cfgs.extend(
            SceneEntityCfg(name=spec.scene_name) for spec in self.distractors
        )
        return EventManagerCfg(
            terms={
                "random_pose_event": PoseResetTermCfg(
                    asset_cfgs=asset_cfgs,
                    trigger_topic="reset",
                    mode="random_non_overlap",
                    pose_range={
                        "x": [0.25, 0.55],
                        "y": [-0.35, 0.35],
                        "z": [0.0, 0.0],
                        "roll": [0.0, 0.0],
                        "pitch": [0.0, 0.0],
                        "yaw": [math.radians(-180.0), math.radians(180.0)],
                    },
                    absolute_sampling=True,
                    min_separation=0.03,
                    max_retries=256,
                    group_key="manipulation_objects",
                    clear_cross_group_cache=True,
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
