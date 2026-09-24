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

"""Instruction-ordered four-cube stacking task."""

from __future__ import annotations
from typing import Any, ClassVar

from pydantic import model_validator
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
from robo_orchard_sim.ext.envs.managers.record import RecordTermBaseCfg
from robo_orchard_sim.ext.envs.managers.record.mcap import McapDictTermCfg
from robo_orchard_sim.orchard_env.assets import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_spec import RoleSpec
from robo_orchard_sim.orchard_env.task_templates.task_base import TaskBase
from robo_orchard_sim.orchard_env.task_templates.task_params import (
    TaskLightResetConfig,
    TaskPoseResetConfig,
    TaskTextureResetConfig,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionWrapper,
)
from robo_orchard_sim.task_components.role_registry import TargetRef
from robo_orchard_sim.task_components.validators.base import Validator
from robo_orchard_sim.task_components.validators.checkers import (
    AlignmentXYZChecker,
    BothGripperOpenChecker,
    ContactChecker,
    SceneCheckerSuite,
    StationaryChecker,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
)
from robo_orchard_sim.task_components.validators.metrics import (
    AllMetricSelector,
    DwellMetricSelector,
    EntityMetricSelector,
    EntityPairMetricSelector,
    GlobalMetricSelector,
    MetricStore,
    NotMetricSelector,
)
from robo_orchard_sim.task_components.validators.role_scope import RoleScope

MEMBERS_ROLE = "members"
STACK_SIZE = 4


class StackCubesTaskParams(Config):
    """Task and validation parameters for four-cube stacking."""

    cube_size: float = 0.05
    alignment_eps: tuple[float, float, float] = (0.012, 0.012, 0.006)
    stationary_linear_threshold: float = 0.01
    stationary_angular_threshold: float = 0.1
    contact_force_threshold: float = 0.1
    success_dwell_steps: int = 15
    pose_reset: TaskPoseResetConfig = TaskPoseResetConfig()
    light_reset: TaskLightResetConfig | None = None
    texture_reset: TaskTextureResetConfig | None = None

    @model_validator(mode="after")
    def validate_stack_parameters(self) -> "StackCubesTaskParams":
        """Reject non-physical geometry and dwell parameters."""
        if self.cube_size <= 0.0:
            raise ValueError("cube_size must be positive.")
        if any(value < 0.0 for value in self.alignment_eps):
            raise ValueError("alignment_eps values must be non-negative.")
        if self.success_dwell_steps < 1:
            raise ValueError("success_dwell_steps must be at least 1.")
        return self


class StackCubesTask(TaskBase):
    """Stack four seed-selected colored cubes in their generated order."""

    roles: ClassVar[dict[str, RoleSpec]] = {
        MEMBERS_ROLE: RoleSpec(
            description="cubes ordered from bottom to top",
            cardinality="many",
            required_traits=("is_graspable",),
        )
    }

    def __init__(
        self,
        *,
        assets: TaskAssets,
        params: StackCubesTaskParams | None = None,
        instruction: InstructionWrapper | None = None,
    ) -> None:
        self.params = params or StackCubesTaskParams()
        super().__init__(assets, instruction=instruction)
        role_members = self.assets.by_role(MEMBERS_ROLE)
        if len(role_members) != STACK_SIZE:
            raise ValueError(
                f"StackCubesTask requires {STACK_SIZE} members, "
                f"got {len(role_members)}."
            )
        self.members: list[RigidObjectSpec] = []
        for member in role_members:
            if not isinstance(member, RigidObjectSpec):
                raise TypeError(
                    "StackCubesTask members must be rigid objects, got "
                    f"{type(member).__name__}."
                )
            self.members.append(member)
        self._color_by_scene_name = {}
        for member in self.members:
            colors = member.attributes.get("color", ())
            if len(colors) != 1:
                raise ValueError(
                    f"Stack cube '{member.scene_name}' must declare exactly "
                    f"one color, got {colors}."
                )
            self._color_by_scene_name[member.scene_name] = colors[0]
        colors = tuple(self._color_by_scene_name.values())
        if len(set(colors)) != STACK_SIZE:
            raise ValueError(
                "Stack cubes must declare one unique color per member, "
                f"got {colors}."
            )

    def get_role_candidates(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list[TargetRef]:
        """Return members in their generated bottom-to-top order."""
        del swap
        return [
            TargetRef(member.scene_name)
            for member in self.assets.by_role(role_id)
        ]

    def get_event_cfg(self) -> EventManagerCfg:
        """Scatter all cubes without overlap inside the configured range."""
        terms = {
            "random_pose_event": PoseResetTermCfg(
                asset_cfgs=[
                    SceneEntityCfg(name=member.scene_name)
                    for member in self.members
                ],
                trigger_topic="reset",
                mode=self.params.pose_reset.mode,
                pose_range=dict(self.params.pose_reset.pose_range),
                absolute_sampling=True,
                min_separation=self.params.pose_reset.min_separation,
                max_retries=256,
                group_key="manipulation_objects",
                clear_cross_group_cache=True,
            )
        }
        light_reset = self.params.light_reset
        if light_reset is not None and light_reset.enabled:
            terms["light_reset_event"] = LightResetTermCfg(
                asset_cfgs=[
                    SceneEntityCfg(name=name)
                    for name in light_reset.asset_names
                ],
                trigger_topic="reset",
                randomize_color=light_reset.randomize_color,
                color_temperature_range=light_reset.color_temperature_range,
                rgb_noise=light_reset.rgb_noise,
                randomize_intensity=light_reset.randomize_intensity,
                intensity_range=light_reset.intensity_range,
                randomize_position=light_reset.randomize_position,
                position_cfg=light_reset.position_cfg,
                crazy_randomization_rate=light_reset.crazy_randomization_rate,
            )
        texture_reset = self.params.texture_reset
        if texture_reset is not None and texture_reset.enabled:
            terms["texture_reset_event"] = TextureResetTermCfg(
                asset_cfgs=[
                    SceneEntityCfg(name=name)
                    for name in texture_reset.asset_names
                ],
                trigger_topic="reset",
                variant_set_name=texture_reset.variant_set_name,
                variant_sort=texture_reset.variant_sort,
                variant_index_range=texture_reset.variant_index_range,
            )
        return EventManagerCfg(terms=terms)

    def get_record_terms(self) -> dict[str, RecordTermBaseCfg]:
        """Record the same episode metadata contract as other tasks."""
        return {
            "meta_dict_term": McapDictTermCfg(
                topic="/meta_data",
                fps=1.0,
                key=TaskBase.EPISODE_META_RECORD_KEY,
                record_mode="once",
            )
        }

    def build_validator(
        self,
        context: ValidatorContext | None = None,
    ) -> Validator:
        """Build progressive ordered-stack validation."""
        if context is None or context.robot is None:
            raise ValueError(
                "StackCubesTask.build_validator() requires ValidatorContext "
                "with robot data."
            )
        targets = context.role_registry.resolve_many(MEMBERS_ROLE)
        if len(targets) != STACK_SIZE:
            raise ValueError(
                f"Role '{MEMBERS_ROLE}' must bind {STACK_SIZE} cubes, "
                f"got {len(targets)}."
            )
        names = [target.scene_name for target in targets]
        metric_store = MetricStore(RoleScope.from_task(self))
        checker_suite = SceneCheckerSuite(
            (
                AlignmentXYZChecker(
                    MEMBERS_ROLE,
                    MEMBERS_ROLE,
                    metric_name="stack_alignment",
                    eps=self.params.alignment_eps,
                    target_height_offset=self.params.cube_size,
                ),
                StationaryChecker(
                    linear_threshold=(self.params.stationary_linear_threshold),
                    angular_threshold=(
                        self.params.stationary_angular_threshold
                    ),
                ),
                ContactChecker(
                    force_threshold=self.params.contact_force_threshold,
                ),
                BothGripperOpenChecker(),
            )
        )
        adjacent = tuple(
            EntityPairMetricSelector(
                metric_store=metric_store,
                subject_id=lower,
                object_id=upper,
                metric="stack_alignment",
            )
            for lower, upper in zip(names, names[1:], strict=False)
        )
        stationary = tuple(
            EntityMetricSelector(
                metric_store=metric_store,
                entity_id=name,
                metric="stationary",
            )
            for name in names
        )
        gripper_open = GlobalMetricSelector(
            metric_store=metric_store,
            metric="gripper_open",
        )
        released = tuple(
            NotMetricSelector(
                EntityMetricSelector(
                    metric_store=metric_store,
                    entity_id=name,
                    metric="contacted",
                )
            )
            for name in names
        )

        first_pair = adjacent[0]
        first_three = AllMetricSelector(adjacent[:2])
        full_order = AllMetricSelector(adjacent)
        settled_stack = DwellMetricSelector(
            AllMetricSelector(
                (
                    *adjacent,
                    *stationary,
                    *released,
                    gripper_open,
                )
            ),
            self.params.success_dwell_steps,
        )
        return Validator(
            context=context,
            checker_suite=checker_suite,
            metric_store=metric_store,
            criteria=[
                first_pair,
                first_three,
                full_order,
                settled_stack,
            ],
            criteria_name=[
                "first_pair_stacked",
                "first_three_stacked",
                "full_order_stacked",
                "settled_stack",
            ],
            progress_criteria=[0, 1, 2, 3],
            success_criteria=[3],
        )

    def build_instruction_context(
        self,
        env: Any,
        *,
        actor_description_seed: int,
        context: ValidatorContext | None = None,
    ) -> dict[str, dict[str, str]]:
        """Render colors in the same bottom-to-top order as validation."""
        del env, actor_description_seed
        if context is None:
            raise ValueError(
                "StackCubesTask.build_instruction_context() requires a "
                "ValidatorContext to resolve role bindings."
            )
        targets = context.role_registry.resolve_many(MEMBERS_ROLE)
        return {
            f"actor{index}": {
                "color": self._color_by_scene_name[target.scene_name]
            }
            for index, target in enumerate(targets, start=1)
        }


__all__ = [
    "MEMBERS_ROLE",
    "STACK_SIZE",
    "StackCubesTask",
    "StackCubesTaskParams",
]
