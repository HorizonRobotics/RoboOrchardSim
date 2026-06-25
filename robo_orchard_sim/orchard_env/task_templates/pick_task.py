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

"""Pick task definition built on ``TaskBase``."""

from __future__ import annotations
from typing import Any

from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.utils.config import Config

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.ext.envs.managers.events.light_reset import (
    LightResetTermCfg,
)
from robo_orchard_sim.ext.envs.managers.events.pose_reset import (
    PoseResetTermCfg,
)
from robo_orchard_sim.ext.envs.managers.events.texture_reset import (
    TextureResetTermCfg,
)
from robo_orchard_sim.ext.envs.managers.record import (
    RecordTermBaseCfg,
)
from robo_orchard_sim.ext.envs.managers.record.mcap import McapDictTermCfg
from robo_orchard_sim.orchard_env.assets import ObjectSpec
from robo_orchard_sim.orchard_env.assets.pool_spec import PoolSpec
from robo_orchard_sim.orchard_env.task_templates.task_base import (
    TaskAssetsBase,
    TaskBase,
)
from robo_orchard_sim.orchard_env.task_templates.task_params import (
    TaskLightResetConfig,
    TaskPoseResetConfig,
    TaskTextureResetConfig,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionActor,
    InstructionWrapper,
)
from robo_orchard_sim.task_components.validators.base import (
    Validator,
    ValidatorActor,
)
from robo_orchard_sim.task_components.validators.checkers import lift, reach
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
)


class PickTaskParams(Config):
    """Task-level parameters for pick."""

    pose_reset: TaskPoseResetConfig = TaskPoseResetConfig()
    light_reset: TaskLightResetConfig | None = None
    texture_reset: TaskTextureResetConfig | None = None


class PickAssets(TaskAssetsBase):
    """Task-specific asset schema for pick scenes."""

    required_object_fields = ("pick",)

    pick: ObjectSpec | PoolSpec

    def flatten(self) -> dict[str, ObjectSpec | PoolSpec]:
        """Return task assets in the flattened shape expected by TaskBase."""
        flattened: dict[str, ObjectSpec | PoolSpec] = {
            "pick": self.pick,
        }
        flattened.update(self.flatten_distractors())
        return flattened


class PickTask(TaskBase):
    """A generic pick task with one target object and optional distractors."""

    def __init__(
        self,
        assets: PickAssets,
        params: PickTaskParams | None = None,
        instruction: InstructionWrapper | None = None,
    ):
        self.assets = assets
        self.params = params or PickTaskParams()
        flattened_assets = assets.flatten()
        super().__init__(flattened_assets, instruction=instruction)

        self.pick_object = self._assets["pick"]
        self.distractors = [
            self._assets[role]
            for role in flattened_assets
            if role.startswith("distractor_")
        ]

    def get_event_cfg(self) -> EventManagerCfg:
        """Return a shared pose-reset event for task objects."""
        asset_cfgs = [SceneEntityCfg(name=self.pick_object.scene_name)]
        asset_cfgs.extend(
            SceneEntityCfg(name=spec.scene_name) for spec in self.distractors
        )
        terms = {
            "random_pose_event": PoseResetTermCfg(
                asset_cfgs=asset_cfgs,
                trigger_topic="reset",
                mode=self.params.pose_reset.mode,
                pose_range=dict(self.params.pose_reset.pose_range),
                absolute_sampling=True,
                min_separation=self.params.pose_reset.min_separation,
                max_retries=256,
                group_key="manipulation_objects",
                clear_cross_group_cache=True,
            ),
        }
        light_reset_cfg = self.params.light_reset
        if light_reset_cfg is not None and light_reset_cfg.enabled:
            terms["light_reset_event"] = LightResetTermCfg(
                asset_cfgs=[
                    SceneEntityCfg(name=name)
                    for name in light_reset_cfg.asset_names
                ],
                trigger_topic="reset",
                randomize_color=light_reset_cfg.randomize_color,
                color_temperature_range=(
                    light_reset_cfg.color_temperature_range
                ),
                rgb_noise=light_reset_cfg.rgb_noise,
                randomize_intensity=light_reset_cfg.randomize_intensity,
                intensity_range=light_reset_cfg.intensity_range,
                randomize_position=light_reset_cfg.randomize_position,
                position_cfg=light_reset_cfg.position_cfg,
                crazy_randomization_rate=light_reset_cfg.crazy_randomization_rate,
            )
        texture_reset_cfg = self.params.texture_reset
        if texture_reset_cfg is not None and texture_reset_cfg.enabled:
            terms["texture_reset_event"] = TextureResetTermCfg(
                asset_cfgs=[
                    SceneEntityCfg(name=name)
                    for name in texture_reset_cfg.asset_names
                ],
                trigger_topic="reset",
                variant_set_name=texture_reset_cfg.variant_set_name,
                variant_sort=texture_reset_cfg.variant_sort,
                variant_index_range=texture_reset_cfg.variant_index_range,
            )
        return EventManagerCfg(terms=terms)

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
        """Return scene actors used by the pick validator."""
        return [self.pick_object.scene_name]

    def build_validator(
        self,
        actors: list[ValidatorActor],
        context: ValidatorContext | None = None,
    ) -> Validator:
        """Build the task validator for pick evaluation."""
        if context is None or context.robot is None:
            raise ValueError(
                "PickTask.build_validator() requires ValidatorContext "
                "with robot data."
            )
        actors_by_name = {actor.name: actor for actor in actors}
        pick_actor = actors_by_name[self.pick_object.scene_name]
        return Validator(
            actors=actors,
            criteria=[
                reach(
                    pick_actor.name,
                    0.2,
                    robot_name=context.robot.robot_name,
                    ee_links=context.robot.ee_links,
                ),
                (lift(pick_actor, 0.03), [0]),
            ],
            criteria_name=[
                "reach_pick",
                "lift_pick",
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

        pick_object = env.scene[self.pick_object.scene_name]
        attribute_name = self.instruction.attribute_name
        if attribute_name is not None:
            actor = InstructionActor.from_rigid_object_with_attribute(
                pick_object,
                attribute_name=attribute_name,
                actor_description_mode=self.instruction.actor_description_mode,
                actor_description_seed=actor_description_seed,
            )
        else:
            actor = InstructionActor.from_rigid_object(
                pick_object,
                actor_description_mode=self.instruction.actor_description_mode,
                actor_description_seed=actor_description_seed,
            )

        return {"actor1": actor}
