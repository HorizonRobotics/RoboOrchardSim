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
from collections.abc import Mapping
from typing import Any, ClassVar

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
from robo_orchard_sim.orchard_env.assets import ObjectSpec
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
from robo_orchard_sim.task_components.instructions.counterfactual import (
    CounterfactualCondition,
    counterfactual_condition_for_task,
    counterfactual_instruction_actors,
)
from robo_orchard_sim.task_components.role_registry import TargetRef
from robo_orchard_sim.task_components.validators.base import (
    Validator,
)
from robo_orchard_sim.task_components.validators.checkers import (
    BothGripperOpenChecker,
    ContactChecker,
    LiftChecker,
    ReachChecker,
    SceneCheckerSuite,
    WithinXYChecker,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
)
from robo_orchard_sim.task_components.validators.counterfactual import (
    CounterfactualSpec,
    CounterfactualStageRelationSelector,
    CounterfactualStageSelector,
    CounterfactualStageSpec,
    CounterfactualStageTracker,
    CounterfactualValidator,
)
from robo_orchard_sim.task_components.validators.metrics import (
    AllMetricSelector,
    AnyRoleMetricSelector,
    BoundRoleMetricSelector,
    BoundRolePairMetricSelector,
    DwellMetricSelector,
    GlobalMetricSelector,
    MetricStore,
)
from robo_orchard_sim.task_components.validators.role_scope import RoleScope


class PlaceA2BTaskParams(Config):
    """Task-level parameters for place-a2b."""

    pose_reset: TaskPoseResetConfig = TaskPoseResetConfig()
    light_reset: TaskLightResetConfig | None = None
    texture_reset: TaskTextureResetConfig | None = None
    reach_dwell_steps: int = 15


PICK_ROLE = "pick"
PLACE_ROLE = "place"

_SWAP_POOLS: dict[str, tuple[str, ...]] = {
    PICK_ROLE: (PICK_ROLE, "distractors_pick"),
    PLACE_ROLE: (PLACE_ROLE, "distractors_place"),
}
"""Asset groups each role may draw from under swap.

The clutter around the pick object was sampled to resemble it, and the
same holds for the place target, so each role rotates within its own
group and never into the other's — one is grasped, the other receives.
"""


class PlaceA2BTask(TaskBase):
    """A generic place-a2b task with one pick object and one place object."""

    roles: ClassVar[dict[str, RoleSpec]] = {
        PICK_ROLE: RoleSpec(
            description="the object to pick up",
            cardinality="one",
            required_traits=("is_graspable",),
        ),
        PLACE_ROLE: RoleSpec(
            description="where the picked object should end up",
            cardinality="one",
        ),
    }

    def __init__(
        self,
        assets: TaskAssets,
        params: PlaceA2BTaskParams | None = None,
        instruction: InstructionWrapper | None = None,
    ):
        self.params = params or PlaceA2BTaskParams()
        super().__init__(assets, instruction=instruction)

        self.pick_object = self.assets.by_role(PICK_ROLE)[0]
        self.place_object = self.assets.by_role(PLACE_ROLE)[0]

        self.distractors: list[ObjectSpec] = [
            spec
            for role_id, specs in self.assets.role_candidates.items()
            if role_id not in self.roles
            for spec in specs
        ]

    def get_role_candidates(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list[TargetRef]:
        """Offer the objects this role may point at.

        Under swap each role also rotates through the clutter drawn to
        resemble it, but never through the other role's: one object is
        grasped, the other receives it. A YAML that declares no matching
        clutter yields a single candidate and nothing rotates.
        """
        groups = _SWAP_POOLS.get(role_id, (role_id,)) if swap else (role_id,)
        return [
            TargetRef(spec.scene_name)
            for group in groups
            for spec in self.assets.by_role(group)
        ]

    def get_event_cfg(self) -> EventManagerCfg:
        """Return reset events for all task objects."""
        terms: dict = {}

        # Order: place first (largest target), then pick, so the smaller
        # pick has more room when sampling around it. Distractors come last.
        actor_names = [
            *(spec.scene_name for spec in self.assets.by_role(PLACE_ROLE)),
            *(spec.scene_name for spec in self.assets.by_role(PICK_ROLE)),
            *(spec.scene_name for spec in self.distractors),
        ]

        if actor_names:
            terms["random_pose_event"] = PoseResetTermCfg(
                asset_cfgs=[SceneEntityCfg(name=name) for name in actor_names],
                trigger_topic="reset",
                mode=self.params.pose_reset.mode,
                pose_range=dict(self.params.pose_reset.pose_range),
                absolute_sampling=True,
                min_separation=self.params.pose_reset.min_separation,
                max_retries=256,
                group_key="manipulation_objects",
                clear_cross_group_cache=True,
            )

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

    def build_validator(
        self,
        context: ValidatorContext | None = None,
    ) -> Validator:
        """Build the task validator for place-a2b evaluation.

        Scores whichever objects the roles name right now, so rebinding
        them between episodes moves both targets without rebuilding
        anything.

        Returns:
            Validator: Task-specific success/progress validator.
        """
        if context is None or context.robot is None:
            raise ValueError(
                "PlaceA2BTask.build_validator() requires ValidatorContext "
                "with robot data."
            )
        metric_store = MetricStore(RoleScope.from_task(self))
        checker_suite = SceneCheckerSuite(
            (
                ReachChecker(threshold=0.2),
                ContactChecker(force_threshold=0.1),
                LiftChecker(threshold=0.03),
                WithinXYChecker(PICK_ROLE, PLACE_ROLE),
                BothGripperOpenChecker(),
            )
        )
        condition = counterfactual_condition_for_task(self)
        if condition is not None:
            spec = self._get_counterfactual_spec(
                condition=condition,
                metric_store=metric_store,
            )
            return CounterfactualValidator(
                context=context,
                checker_suite=checker_suite,
                metric_store=metric_store,
                spec=spec,
            )

        reach_pick = BoundRoleMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PICK_ROLE,
            metric="reached",
        )
        contact_pick = BoundRoleMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PICK_ROLE,
            metric="contacted",
        )
        grasp_pick = BoundRoleMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PICK_ROLE,
            metric="grasped",
        )
        lift_pick = BoundRoleMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PICK_ROLE,
            metric="lifted",
        )
        lifted_while_grasped = AllMetricSelector((lift_pick, grasp_pick))
        within_place = BoundRolePairMetricSelector(
            metric_store=metric_store,
            context=context,
            subject_role=PICK_ROLE,
            object_role=PLACE_ROLE,
            metric="within_xy",
        )
        gripper_open = GlobalMetricSelector(
            metric_store=metric_store,
            metric="gripper_open",
        )
        placed_and_released = AllMetricSelector((within_place, gripper_open))
        return Validator(
            context=context,
            checker_suite=checker_suite,
            metric_store=metric_store,
            criteria=[
                DwellMetricSelector(
                    reach_pick,
                    self.params.reach_dwell_steps,
                ),
                (
                    contact_pick,
                    [0],
                ),
                (
                    grasp_pick,
                    [1],
                ),
                (
                    lifted_while_grasped,
                    [2],
                ),
                (within_place, [3]),
                (
                    placed_and_released,
                    [4],
                ),
            ],
            criteria_name=[
                "reach_pick",
                "contact_pick",
                "grasp_pick",
                "lift_pick",
                "reach_place",
                "place_within_xy",
            ],
        )

    def _get_counterfactual_spec(
        self,
        *,
        condition: CounterfactualCondition,
        metric_store: MetricStore,
    ) -> CounterfactualSpec:
        """Return PlaceA2B-specific counterfactual metrics and rubric."""
        if (
            condition == "generic_object"
            and self.instruction is not None
            and self.instruction.template != "place_a2b_generic_object"
        ):
            raise ValueError(
                "PlaceA2B generic counterfactual evaluation requires the "
                "'place_a2b_generic_object' instruction template."
            )

        def rubric(reached: Mapping[str, bool]) -> tuple[bool, float]:
            success = (
                reached["any_placed"]
                if condition == "generic_object"
                else not any(reached.values())
            )
            return success, 1.0 if success else 0.0

        tracker = CounterfactualStageTracker(
            metric_store=metric_store,
            role_id=PICK_ROLE,
            stages=(
                CounterfactualStageSpec(
                    "reach",
                    ("reached",),
                    self.params.reach_dwell_steps,
                ),
                CounterfactualStageSpec("contact", ("contacted",)),
                CounterfactualStageSpec("grasp", ("grasped",)),
                CounterfactualStageSpec("lift", ("lifted", "grasped")),
            ),
        )
        picked_within_place = CounterfactualStageRelationSelector(
            tracker=tracker,
            completed_stage="lift",
            object_role=PLACE_ROLE,
            relation_metric="within_xy",
        )
        placed_and_released = AllMetricSelector(
            (
                picked_within_place,
                GlobalMetricSelector(
                    metric_store=metric_store,
                    metric="gripper_open",
                ),
            )
        )
        if condition == "generic_object":
            criteria = (
                *(
                    CounterfactualStageSelector(tracker, stage)
                    for stage in ("reach", "contact", "grasp", "lift")
                ),
                (placed_and_released, [3]),
            )
        else:
            criteria = (
                CounterfactualStageSelector(tracker, "reach"),
                AnyRoleMetricSelector(
                    metric_store=metric_store,
                    role_id=PICK_ROLE,
                    metric="contacted",
                ),
                AnyRoleMetricSelector(
                    metric_store=metric_store,
                    role_id=PICK_ROLE,
                    metric="grasped",
                ),
                AnyRoleMetricSelector(
                    metric_store=metric_store,
                    role_id=PICK_ROLE,
                    metric="lifted",
                ),
                placed_and_released,
            )
        return CounterfactualSpec(
            criteria=criteria,
            criteria_name=(
                "any_reached",
                "any_contacted",
                "any_grasped",
                "any_lifted",
                "any_placed",
            ),
            rubric=rubric,
        )

    def build_instruction_context(
        self,
        env: Any,
        *,
        actor_description_seed: int,
        context: ValidatorContext | None = None,
    ) -> dict[str, InstructionActor]:
        """Describe whichever objects the roles name right now."""
        if context is None:
            raise ValueError(
                "PlaceA2BTask.build_instruction_context() requires a "
                "ValidatorContext to resolve role bindings."
            )
        if self.instruction is None:
            return {}
        counterfactual_actors = counterfactual_instruction_actors(self)
        if counterfactual_actors is not None:
            return counterfactual_actors

        pick_target = context.role_registry.resolve_one(PICK_ROLE)
        place_target = context.role_registry.resolve_one(PLACE_ROLE)
        return {
            "actor1": InstructionActor.from_rigid_object(
                env.scene[pick_target.scene_name],
                actor_description_mode=self.instruction.actor_description_mode,
                actor_description_seed=actor_description_seed,
            ),
            "actor2": InstructionActor.from_rigid_object(
                env.scene[place_target.scene_name],
                actor_description_mode=self.instruction.actor_description_mode,
                actor_description_seed=actor_description_seed,
            ),
        }
