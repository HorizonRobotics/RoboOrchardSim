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

"""Semantic-affordance articulated manipulation task."""

import math
import re
from typing import TYPE_CHECKING, ClassVar

from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.utils.config import Config

from robo_orchard_sim.contracts.articulated_operation import (
    SUPPORTED_ARTICULATED_OPERATIONS,
    ArticulatedOperation,
)
from robo_orchard_sim.orchard_env.assets import JointOperationMeta
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_templates.articulated.base import (
    PRIMARY_ROLE,
    ArticulatedTaskBase,
    ArticulatedTaskParams,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionWrapper,
)
from robo_orchard_sim.task_components.instructions.counterfactual import (
    counterfactual_condition_for_task,
)
from robo_orchard_sim.task_components.role_registry import TargetRef

if TYPE_CHECKING:
    from robo_orchard_sim.task_components.validators.base import Validator
    from robo_orchard_sim.task_components.validators.context import (
        ValidatorContext,
    )


class AffordanceValidatorConfig(Config):
    """Thresholds for affordance evaluation."""

    reach_threshold: float = 0.2
    contact_force_threshold: float = 0.01
    min_opposition_deg: float = 120.0
    min_inward_cosine: float = 0.3
    joint_min_delta: float | None = None
    joint_min_linear_delta: float = 0.0001  # meters: 0.1 mm
    joint_min_angular_delta: float = 0.017453292519943295  # radians: 1 degree
    target_fraction_tolerance: float = 0.05
    stationary_linear_threshold: float = 0.02
    stationary_angular_threshold: float = 0.15
    reach_dwell_steps: int = 15
    dwell_steps: int = 5


class AffordanceTaskParams(ArticulatedTaskParams):
    """Parameters for single-target affordance evaluation."""

    validator: AffordanceValidatorConfig = AffordanceValidatorConfig()


class AffordanceTask(ArticulatedTaskBase):
    """Evaluate each joint's operation from the asset's default state."""

    params: AffordanceTaskParams
    ALLOWED_OPERATIONS: ClassVar[frozenset[ArticulatedOperation]] = (
        SUPPORTED_ARTICULATED_OPERATIONS
    )

    def __init__(
        self,
        *,
        assets: TaskAssets,
        params: AffordanceTaskParams | None = None,
        instruction: InstructionWrapper | None = None,
    ) -> None:
        params = params or AffordanceTaskParams()
        super().__init__(
            assets=assets,
            allowed_operations=self.ALLOWED_OPERATIONS,
            params=params,
            instruction=instruction,
        )

    def get_role_candidates(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list[TargetRef]:
        """Offer joints with an operation matching their default state."""
        del swap
        if role_id != PRIMARY_ROLE:
            return super().get_role_candidates(role_id)
        candidates = []
        for joint in self.primary.joint_operations:
            operation = self._default_operation(joint)
            if operation is None:
                continue
            candidates.append(
                TargetRef(
                    self.primary.scene_name,
                    joint.semantic_name,
                    operation,
                )
            )
        return candidates

    def _default_joint_position(self, joint_name: str) -> float:
        """Resolve one joint's configured asset-default position."""
        configured = self.primary.joint_pos
        if configured is None:
            configured = self.primary.template_cfg.init_state.joint_pos
        if configured is None:
            return 0.0
        if joint_name in configured:
            return float(configured[joint_name])

        matches = [
            float(position)
            for pattern, position in configured.items()
            if re.fullmatch(pattern, joint_name)
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            return 0.0
        raise ValueError(
            f"Joint '{joint_name}' matches multiple default-position "
            f"patterns in {configured}."
        )

    def _default_operation(
        self,
        joint: JointOperationMeta,
    ) -> ArticulatedOperation | None:
        """Find a unique default-state operation, or None if none matches."""
        default_position = self._default_joint_position(joint.joint_name)
        matches: list[ArticulatedOperation] = [
            operation
            for operation in sorted(self._allowed_joint_operations(joint))
            if math.isclose(
                joint.operations[operation].initial_joint_position,
                default_position,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"Joint '{joint.semantic_name}' default position "
                f"{default_position} must match exactly one operation, "
                f"matched {matches}."
            )
        return matches[0]

    def get_event_cfg(self) -> EventManagerCfg:
        """Keep the articulation at its asset-declared default joint state."""
        event_cfg = super().get_event_cfg()
        joint_reset = event_cfg.terms["joint_state_reset_event"]
        joint_reset.operation_role_id = None
        return event_cfg

    def build_validator(
        self,
        context: "ValidatorContext | None" = None,
    ) -> "Validator":
        """Build the affordance rubric."""
        if counterfactual_condition_for_task(self) is not None:
            raise NotImplementedError(
                "AffordanceTask does not support counterfactual evaluation."
            )

        from robo_orchard_sim.task_components.validators.base import Validator
        from robo_orchard_sim.task_components.validators.checkers import (
            ContactChecker,
            JointStateChecker,
            ReachChecker,
            SceneCheckerSuite,
            StationaryChecker,
        )
        from robo_orchard_sim.task_components.validators.metrics import (
            AllMetricSelector,
            AnyMetricSelector,
            BoundRoleInteractionBodyMetricSelector,
            BoundRoleJointDeltaSelector,
            BoundRoleJointFractionSelector,
            BoundRoleJointMovedSelector,
            BoundRoleOutcomeBodyMetricSelector,
            DwellMetricSelector,
            MetricStore,
        )
        from robo_orchard_sim.task_components.validators.role_scope import (
            RoleScope,
        )

        if context is None or context.robot is None:
            raise ValueError(
                "AffordanceTask.build_validator() requires a "
                "ValidatorContext carrying robot data."
            )
        validation = self.params.validator
        metric_store = MetricStore(RoleScope.from_task(self))
        reach = BoundRoleInteractionBodyMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
            metric="reached",
        )
        grasp = BoundRoleInteractionBodyMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
            metric="grasped",
        )
        contact = BoundRoleInteractionBodyMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
            metric="contacted",
        )
        moved = BoundRoleJointMovedSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
            min_delta=validation.joint_min_delta,
            min_linear_delta=validation.joint_min_linear_delta,
            min_angular_delta=validation.joint_min_angular_delta,
        )
        target_fraction = BoundRoleJointFractionSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
            tolerance=validation.target_fraction_tolerance,
        )
        target_delta = BoundRoleJointDeltaSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
        )
        target_joint_state = AnyMetricSelector((target_fraction, target_delta))
        stationary = BoundRoleOutcomeBodyMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
            metric="stationary",
        )
        # The evaluator binds roles before building each episode's validator.
        is_press = (
            context.role_registry.resolve_one(PRIMARY_ROLE, 0).operation
            == "press"
        )
        return Validator(
            context=context,
            checker_suite=SceneCheckerSuite(
                (
                    ReachChecker(threshold=validation.reach_threshold),
                    ContactChecker(
                        force_threshold=validation.contact_force_threshold,
                        min_opposition_deg=validation.min_opposition_deg,
                        min_inward_cosine=validation.min_inward_cosine,
                    ),
                    JointStateChecker(),
                    StationaryChecker(
                        linear_threshold=(
                            validation.stationary_linear_threshold
                        ),
                        angular_threshold=(
                            validation.stationary_angular_threshold
                        ),
                    ),
                )
            ),
            metric_store=metric_store,
            criteria=[
                DwellMetricSelector(reach, validation.reach_dwell_steps),
                contact,
                grasp,
                AllMetricSelector((contact, moved)) if is_press else moved,
                DwellMetricSelector(
                    target_joint_state, validation.dwell_steps
                ),
                DwellMetricSelector(
                    AllMetricSelector((target_joint_state, stationary)),
                    validation.dwell_steps,
                ),
            ],
            independent_criteria=[
                DwellMetricSelector(reach, validation.reach_dwell_steps),
                contact,
                grasp,
                moved,
                DwellMetricSelector(
                    target_joint_state, validation.dwell_steps
                ),
                DwellMetricSelector(
                    AllMetricSelector((target_joint_state, stationary)),
                    validation.dwell_steps,
                ),
            ],
            criteria_name=[
                "reach",
                "contact",
                "grasp",
                "moved",
                "target_joint_state",
                "settled_at_target",
            ],
            success_criteria=[3] if is_press else [5],
            progress_criteria=[1, 3] if is_press else [0, 1, 3, 4, 5],
        )
