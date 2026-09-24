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

"""Joint-direction articulated manipulation task."""

from typing import TYPE_CHECKING, ClassVar

from robo_orchard_core.utils.config import Config

from robo_orchard_sim.asset_manager.metadata.joint_operations import (
    OPPOSING_OPERATION_PAIRS,
    OperationStartMode,
)
from robo_orchard_sim.contracts.articulated_operation import (
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

if TYPE_CHECKING:
    from robo_orchard_sim.task_components.validators.base import Validator
    from robo_orchard_sim.task_components.validators.context import (
        ValidatorContext,
    )


class JointOperationValidatorConfig(Config):
    """Thresholds for joint-operation evaluation."""

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


class JointOperationTaskParams(ArticulatedTaskParams):
    """Parameters for single-target joint-operation evaluation."""

    validator: JointOperationValidatorConfig = JointOperationValidatorConfig()


class JointDirectionTaskParams(JointOperationTaskParams):
    """Parameters for single-target joint-direction evaluation."""


class JointOperationTask(ArticulatedTaskBase):
    """Evaluate one seeded operation on an eligible articulation joint.

    Subclasses narrow ``ALLOWED_OPERATIONS`` to the directions they
    grade; the shared rubric below is direction-agnostic because it
    reads the target joint state off whichever operation the role
    registry bound this episode.
    """

    params: JointOperationTaskParams
    ALLOWED_OPERATIONS: ClassVar[frozenset[ArticulatedOperation]]

    def __init__(
        self,
        *,
        assets: TaskAssets,
        params: JointOperationTaskParams | None = None,
        instruction: InstructionWrapper | None = None,
    ) -> None:
        params = params or JointOperationTaskParams()
        super().__init__(
            assets=assets,
            allowed_operations=self.ALLOWED_OPERATIONS,
            params=params,
            instruction=instruction,
        )

    def build_validator(
        self,
        context: "ValidatorContext | None" = None,
    ) -> "Validator":
        """Build the joint-operation rubric."""
        if counterfactual_condition_for_task(self) is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support counterfactual "
                "evaluation."
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
                f"{type(self).__name__}.build_validator() requires a "
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
            operation_start_mode=self.OPERATION_START_MODE,
            min_delta=validation.joint_min_delta,
            min_linear_delta=validation.joint_min_linear_delta,
            min_angular_delta=validation.joint_min_angular_delta,
        )
        target_fraction = BoundRoleJointFractionSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
            operation_start_mode=self.OPERATION_START_MODE,
            tolerance=validation.target_fraction_tolerance,
        )
        target_delta = BoundRoleJointDeltaSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
            operation_start_mode=self.OPERATION_START_MODE,
        )
        target_joint_state = AnyMetricSelector((target_fraction, target_delta))
        stationary = BoundRoleOutcomeBodyMetricSelector(
            metric_store=metric_store,
            context=context,
            role_id=PRIMARY_ROLE,
            metric="stationary",
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
                moved,
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
            success_criteria=[5],
            progress_criteria=[0, 1, 3, 4, 5],
        )


class JointDirectionTask(JointOperationTask):
    """Grade either direction from the opposing operations' midpoint.

    Both annotated starts belong to the selected joint. Reset and evaluation
    use their average, clipped to that joint's runtime soft limits.
    """

    OPERATION_START_MODE: ClassVar[OperationStartMode] = "opposing_midpoint"

    ALLOWED_OPERATIONS: ClassVar[frozenset[ArticulatedOperation]] = frozenset(
        {"open", "close", "pull", "push"}
    )
    OPPOSING_OPERATION_PAIRS: ClassVar[
        tuple[tuple[ArticulatedOperation, ArticulatedOperation], ...]
    ] = OPPOSING_OPERATION_PAIRS

    def _allowed_joint_operations(
        self,
        joint: JointOperationMeta,
    ) -> frozenset[ArticulatedOperation]:
        """Keep only complete opposing operation pairs for this joint.

        Asking for a direction only makes sense when the joint can go
        both ways, so a joint annotated with just one half of a pair
        contributes no candidates here.
        """
        supported = frozenset(joint.operations) & self.allowed_operations
        return frozenset(
            operation
            for pair in self.OPPOSING_OPERATION_PAIRS
            if set(pair) <= supported
            for operation in pair
        )
