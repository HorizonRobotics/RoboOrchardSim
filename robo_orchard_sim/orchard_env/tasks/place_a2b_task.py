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
from typing import Any

from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.utils.config import Config
from typing_extensions import Literal

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.envs.managers.events.pose_reset import (
    PoseResetTermCfg,
)
from robo_orchard_sim.envs.managers.record import (
    RecordTermBaseCfg,
)
from robo_orchard_sim.envs.managers.record.mcap import McapDictTermCfg
from robo_orchard_sim.orchard_env.assets import ObjectSpec
from robo_orchard_sim.orchard_env.tasks.task_base import (
    TaskAssetsBase,
    TaskBase,
)
from robo_orchard_sim.orchard_env.tasks.task_params import PoseRangeConfig
from robo_orchard_sim.tasks.instructions.base import (
    InstructionActor,
    InstructionWrapper,
)
from robo_orchard_sim.tasks.validators.base import Validator, ValidatorActor
from robo_orchard_sim.tasks.validators.checkers import (
    is_within_xy,
    lift,
    reach,
)


class PlaceA2BTaskParams(Config):
    """Task-level parameters for place-a2b."""

    mode: Literal[
        "random",
        "random_non_overlap",
        "orderly",
        "default",
        "drop",
    ] = "random_non_overlap"
    pose_range: PoseRangeConfig = PoseRangeConfig()
    min_separation: float = 0.03


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

    def __init__(
        self,
        assets: PlaceA2BTaskAssets,
        params: PlaceA2BTaskParams | None = None,
        instruction: InstructionWrapper | None = None,
    ):
        self.assets = assets
        self.params = params or PlaceA2BTaskParams()
        flattened_assets = assets.flatten()
        super().__init__(flattened_assets, instruction=instruction)

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
                    mode=self.params.mode,
                    pose_range=dict(self.params.pose_range),
                    absolute_sampling=True,
                    min_separation=self.params.min_separation,
                    max_retries=256,
                    group_key="manipulation_objects",
                    clear_cross_group_cache=True,
                ),
            }
        )

    def get_record_terms(self) -> dict[str, RecordTermBaseCfg]:
        return {
            "meta_dict_term": McapDictTermCfg(
                topic="/meta_data",
                fps=1.0,
                # Use the task-level metadata record key contract.
                key=TaskBase.EPISODE_META_RECORD_KEY,
                record_mode="once",
            )
        }

    def get_validator_actor_names(self) -> list[str]:
        """Return scene actors used by the place-a2b validator."""
        return [
            self.pick_object.scene_name,
            self.place_object.scene_name,
        ]

    def build_validator(self, actors: list[ValidatorActor]) -> Validator:
        """Build the task validator for place-a2b evaluation.

        Returns:
            Validator: Task-specific success/progress validator.
        """
        pick_name = self.pick_object.scene_name
        place_name = self.place_object.scene_name
        return Validator(
            actors=actors,
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

    def build_instruction_context(
        self,
        env: Any,
        *,
        actor_description_seed: int,
    ) -> dict[str, InstructionActor]:
        if self.instruction is None:
            return {}

        return {
            "actor1": InstructionActor.from_rigid_object(
                env.scene[self.pick_object.scene_name],
                actor_description_mode=self.instruction.actor_description_mode,
                actor_description_seed=actor_description_seed,
            ),
            "actor2": InstructionActor.from_rigid_object(
                env.scene[self.place_object.scene_name],
                actor_description_mode=self.instruction.actor_description_mode,
                actor_description_seed=actor_description_seed,
            ),
        }
