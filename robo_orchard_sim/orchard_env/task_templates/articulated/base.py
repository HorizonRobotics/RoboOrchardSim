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

"""Shared task template for articulated manipulation."""

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar

from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.utils.config import Config

from robo_orchard_sim.asset_manager.metadata import get_joint, get_operation
from robo_orchard_sim.asset_manager.metadata.joint_operations import (
    OperationStartMode,
)
from robo_orchard_sim.contracts.articulated_operation import (
    ArticulatedOperation,
)
from robo_orchard_sim.orchard_env.assets import (
    ArticulatedObjectSpec,
    JointOperationMeta,
)
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_spec import RoleSpec
from robo_orchard_sim.orchard_env.task_templates.task_base import TaskBase
from robo_orchard_sim.orchard_env.task_templates.task_params import (
    TaskLightResetConfig,
    TaskPoseResetConfig,
    TaskTextureResetConfig,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionActor,
    InstructionWrapper,
)
from robo_orchard_sim.task_components.role_registry import TargetRef
from robo_orchard_sim.task_components.validators.physical_entity import (
    SceneBodyKey,
    SceneEntityKey,
)

if TYPE_CHECKING:
    from robo_orchard_sim.task_components.validators.context import (
        ValidatorContext,
    )


class ArticulatedTaskParams(Config):
    """Parameters shared by articulated manipulation tasks."""

    pose_reset: TaskPoseResetConfig = TaskPoseResetConfig()
    light_reset: TaskLightResetConfig | None = None
    texture_reset: TaskTextureResetConfig | None = None


PRIMARY_ROLE = "primary"


class ArticulatedTaskBase(TaskBase):
    """Compose shared articulated state reset and instruction behavior."""

    OPERATION_START_MODE: ClassVar[OperationStartMode] = "metadata"

    roles: ClassVar[dict[str, RoleSpec]] = {
        PRIMARY_ROLE: RoleSpec(
            description="the articulation being operated",
            cardinality="one",
        ),
    }

    def __init__(
        self,
        *,
        assets: TaskAssets,
        allowed_operations: frozenset[ArticulatedOperation],
        params: ArticulatedTaskParams | None = None,
        instruction: InstructionWrapper | None = None,
    ) -> None:
        self.params = params or ArticulatedTaskParams()
        self.allowed_operations = allowed_operations
        super().__init__(
            assets,
            instruction=instruction
            or InstructionWrapper("articulated_operation"),
        )

        primary = self.assets.by_role(PRIMARY_ROLE)[0]
        if not isinstance(primary, ArticulatedObjectSpec):
            raise TypeError(
                f"{type(self).__name__} role '{PRIMARY_ROLE}' must resolve "
                f"to an articulation, got {type(primary).__name__}."
            )
        self.primary = primary
        self.distractors = [
            spec
            for role_id, specs in self.assets.role_candidates.items()
            if role_id != PRIMARY_ROLE
            for spec in specs
        ]

    def get_role_candidates(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list[TargetRef]:
        """Offer what this role may be bound to this episode.

        Ignores ``swap``: this task never had a target fixed at build
        time to fall back to — every episode always picked a joint and
        an operation, previously by calling ``random.Random`` directly.
        The candidates are every joint on ``primary`` crossed with every
        operation allowed both by this task and by that joint; a
        selector now does the picking instead.
        """
        del swap
        if role_id != PRIMARY_ROLE:
            return [TargetRef(self.primary.scene_name)]
        return [
            TargetRef(self.primary.scene_name, joint.semantic_name, operation)
            for joint in self.primary.joint_operations
            for operation in sorted(self._allowed_joint_operations(joint))
        ]

    def _allowed_joint_operations(
        self,
        joint: JointOperationMeta,
    ) -> frozenset[ArticulatedOperation]:
        """Return task-eligible operations for one metadata joint."""
        return frozenset(joint.operations) & self.allowed_operations

    def get_role_scene_entities(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list[SceneEntityKey]:
        """Map semantic operation candidates to physical moving bodies."""
        entities = []
        for target in self.get_role_candidates(role_id, swap=swap):
            if target.semantic_name is None:
                entities.append(target.scene_name)
                continue
            if target.operation is None:
                raise ValueError(
                    f"Target '{target}' must include an operation."
                )
            joint = get_joint(
                self.primary.joint_operations,
                target.semantic_name,
            )
            operation = get_operation(
                self.primary.joint_operations,
                target.semantic_name,
                target.operation,
            )
            entities.extend(
                (
                    *(
                        SceneBodyKey(target.scene_name, name)
                        for name in operation.effective_interaction_links
                    ),
                    SceneBodyKey(target.scene_name, joint.outcome_link),
                )
            )
        return list(dict.fromkeys(entities))

    def get_event_cfg(self) -> EventManagerCfg:
        """Reset root poses before restoring articulation joint state."""
        from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (  # noqa: E501
            SceneEntityCfg,
        )
        from robo_orchard_sim.ext.envs.managers.events.joint_state_reset import (  # noqa: E501
            JointStateResetTermCfg,
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

        asset_cfgs = [SceneEntityCfg(name=self.primary.scene_name)]
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
            "joint_state_reset_event": JointStateResetTermCfg(
                asset_cfgs=[
                    SceneEntityCfg(name=self.primary.scene_name),
                ],
                trigger_topic="reset",
                noise_std=0.0,
                operation_role_id=PRIMARY_ROLE,
                operation_start_mode=self.OPERATION_START_MODE,
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
                crazy_randomization_rate=(
                    light_reset_cfg.crazy_randomization_rate
                ),
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

    def build_instruction_context(
        self,
        env: Any,
        *,
        actor_description_seed: int,
        context: "ValidatorContext | None" = None,
    ) -> dict[str, dict[str, Any]]:
        """Describe whichever joint/operation the role names right now."""
        if context is None:
            raise ValueError(
                f"{type(self).__name__}.build_instruction_context() "
                "requires a ValidatorContext to resolve role bindings."
            )
        if self.instruction is None:
            return {}
        target = context.role_registry.resolve_one(PRIMARY_ROLE)
        joint_description = next(
            joint.semantic_name
            for joint in self.primary.joint_operations
            if joint.semantic_name == target.semantic_name
        )
        operation = target.operation
        assert operation is not None
        actor = InstructionActor.from_articulation(
            env.scene[target.scene_name],
            actor_description_mode=self.instruction.actor_description_mode,
            actor_description_seed=actor_description_seed,
        )
        actor_context = dataclasses.asdict(actor)
        actor_context.update(
            {
                "joint_description": joint_description,
                "operation": operation,
                "operation_verb": operation.capitalize(),
            }
        )
        return {
            "actor1": actor_context,
        }
