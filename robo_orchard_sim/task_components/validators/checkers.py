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

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np
import torch

from robo_orchard_sim.task_components.validators.base import GripperRange
from robo_orchard_sim.task_components.validators.contact_binding import (
    ContactBinding,
    ContactFilterKey,
    GripperContactPair,
    gripper_contact_sensors,
)
from robo_orchard_sim.task_components.validators.metrics import SceneMetrics
from robo_orchard_sim.task_components.validators.physical_entity import (
    CollisionBodyBounds,
    SceneBodyKey,
    SceneEntityKey,
    SceneJointKey,
    entity_pose_array,
    entity_position,
    entity_prim_path,
    entity_quaternion,
    has_articulation_components,
    iter_articulation_bodies,
    iter_articulation_joints,
    resolve_body_id,
    resolve_rigid_body_prim_path,
)
from robo_orchard_sim.task_components.validators.role_scope import (
    EntityId,
    RoleScope,
)

if TYPE_CHECKING:
    from robo_orchard_sim.task_components.validators.context import (
        ValidatorContext,
    )

# Fraction of the close->open joint travel required to count as "open".
DEFAULT_GRIPPER_OPEN_RATIO = 0.8

# Keeps the boundary inclusive despite float rounding (0.04/0.05 < 0.8).
_OPEN_RATIO_TOL = 1e-6


def _gripper_joint_is_open(
    value: float, gripper_range: GripperRange, open_ratio: float
) -> bool:
    """Return whether a joint position is open enough.

    Direction-agnostic: open_val may be greater or less than close_val. A
    degenerate range (open == close) cannot gate and counts as open.
    """
    span = gripper_range.open_val - gripper_range.close_val
    if span == 0:
        return True
    ratio = (value - gripper_range.close_val) / span
    return ratio >= open_ratio - _OPEN_RATIO_TOL


# ---------------------------------------------------------------------
# Checker contracts and scene-wide composition
# ---------------------------------------------------------------------


class SceneChecker(Protocol):
    """Observe one kind of fact over a complete interaction scope."""

    def evaluate(
        self,
        env,
        context: ValidatorContext,
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return facts observed on the current simulation step."""
        ...

    def reset(self) -> None:
        """Drop episode-scoped checker state."""
        ...


class SceneCheckerSuite:
    """Evaluate scene checkers and merge their disjoint metric outputs."""

    def __init__(self, checkers: Sequence[SceneChecker]) -> None:
        self.checkers = tuple(checkers)

    def evaluate(
        self,
        env,
        context: ValidatorContext,
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Evaluate every checker exactly once and merge its metrics."""
        unary: dict[str, dict[EntityId, bool]] = {}
        binary: dict[str, dict[tuple[EntityId, EntityId], bool]] = {}
        global_values: dict[str, bool] = {}
        values: dict[str, dict[EntityId, float]] = {}
        for checker in self.checkers:
            observed_metrics = checker.evaluate(
                env,
                context,
                scope,
                env_idx=env_idx,
            )
            self._merge_metrics(unary, observed_metrics.unary, "unary")
            self._merge_metrics(binary, observed_metrics.binary, "binary")
            self._merge_metrics(
                global_values,
                observed_metrics.global_values,
                "global",
            )
            self._merge_metrics(
                values,
                observed_metrics.values,
                "value",
            )
        return SceneMetrics(
            unary=unary,
            binary=binary,
            global_values=global_values,
            values=values,
        )

    @staticmethod
    def _merge_metrics(target: dict, source, domain: str) -> None:
        duplicate = set(target).intersection(source)
        if duplicate:
            raise ValueError(
                f"Duplicate {domain} metrics from scene checkers: "
                f"{sorted(duplicate)}."
            )
        target.update(source)

    def reset(self) -> None:
        """Reset every child checker."""
        for checker in self.checkers:
            checker.reset()


def _entity_is_reached(
    env,
    entity_id: EntityId,
    *,
    threshold: float,
    tcp_positions: torch.Tensor,
    collision_bounds: CollisionBodyBounds,
    env_idx: int = 0,
) -> bool:
    """Check TCP proximity using link bounds or a root entity's position."""
    if isinstance(entity_id, SceneBodyKey):
        distances = collision_bounds.distances(
            env, entity_id, tcp_positions, env_idx
        )
    else:
        target_position = entity_position(env, entity_id, env_idx)
        distances = torch.linalg.vector_norm(
            tcp_positions - target_position, dim=-1
        )
    return bool((distances < threshold).any())


def _entity_is_lifted(
    env,
    entity_id: EntityId,
    *,
    context: "ValidatorContext",
    threshold: float,
    env_idx: int = 0,
) -> bool:
    """Return whether one entity rose above its settled initial height."""
    if not isinstance(entity_id, str):
        raise TypeError("Lift checking currently requires a root entity.")
    target_position = entity_position(env, entity_id, env_idx)
    init_height = context.init_state_of(entity_id, env_idx)[2]
    return bool((target_position[2] - init_height).item() > threshold)


# ---------------------------------------------------------------------
# Scene-wide unary interaction checkers
# ---------------------------------------------------------------------


class ReachChecker:
    """Observe TCP proximity, using collider bounds for articulated links.

    Root-entity facts retain their point-distance semantics. Per-link facts
    use live body poses and cached local collision-geometry bounding boxes.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self.collision_bounds = CollisionBodyBounds()

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return one ``reached`` value per scoped entity."""
        robot = context.robot
        if robot is None:
            raise ValueError("Reach checking requires robot metadata.")
        if not robot.robot_name:
            raise ValueError("robot_name is required for reach checker.")
        tcp_positions = robot.tcp_positions_w(env, env_idx)

        reached: dict[EntityId, bool] = {}
        for entity_id in scope.entities:
            if isinstance(entity_id, SceneJointKey):
                continue
            reached[entity_id] = _entity_is_reached(
                env,
                entity_id,
                threshold=self.threshold,
                tcp_positions=tcp_positions,
                collision_bounds=self.collision_bounds,
                env_idx=env_idx,
            )
        for component in iter_articulation_bodies(env, scope):
            reached[component.body_key] = _entity_is_reached(
                env,
                component.body_key,
                threshold=self.threshold,
                tcp_positions=tcp_positions,
                collision_bounds=self.collision_bounds,
                env_idx=env_idx,
            )
        return SceneMetrics(unary={"reached": reached})

    def reset(self) -> None:
        """Clear cached geometry for the next scene or episode."""
        self.collision_bounds.reset()


class LiftChecker:
    """Observe lift facts for every entity in an interaction scope."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return one ``lifted`` value per scoped entity."""
        return SceneMetrics(
            unary={
                "lifted": {
                    entity_id: _entity_is_lifted(
                        env,
                        entity_id,
                        context=context,
                        threshold=self.threshold,
                        env_idx=env_idx,
                    )
                    for entity_id in scope.entities
                }
            }
        )

    def reset(self) -> None:
        """Lift checking is stateless."""


# ---------------------------------------------------------------------
# Scene-wide joint component checker
# ---------------------------------------------------------------------


class JointStateChecker:
    """Observe position and, for bounded joints, normalized fraction."""

    def __init__(
        self,
        *,
        position_metric: str = "joint_position",
        fraction_metric: str = "joint_fraction",
    ) -> None:
        self.position_metric = position_metric
        self.fraction_metric = fraction_metric

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return operation-independent joint measurements for the scene."""
        del context
        positions: dict[EntityId, float] = {}
        fractions: dict[EntityId, float] = {}
        for component in iter_articulation_joints(env, scope):
            data = component.articulation.data
            position = float(data.joint_pos[env_idx, component.joint_id])
            limits = data.joint_pos_limits[env_idx, component.joint_id]
            lower = float(limits[0])
            upper = float(limits[1])
            positions[component.joint_key] = position
            if not math.isfinite(lower) or not math.isfinite(upper):
                continue
            span = upper - lower
            if span <= 0.0:
                raise ValueError(
                    f"Joint '{component.joint_key.joint_name}' on "
                    f"'{component.joint_key.scene_name}' has invalid limits "
                    f"[{lower}, {upper}]."
                )
            fractions[component.joint_key] = (position - lower) / span
        return SceneMetrics(
            values={
                self.position_metric: positions,
                self.fraction_metric: fractions,
            }
        )

    def reset(self) -> None:
        """Articulation joint state checking is stateless."""


def _relation_pairs(
    scope: RoleScope,
    subject_role: str,
    object_role: str,
) -> tuple[tuple[EntityId, EntityId], ...]:
    if not scope.entities_for_role(subject_role):
        raise ValueError(
            f"Subject role '{subject_role}' has no interaction entities."
        )
    if not scope.entities_for_role(object_role):
        raise ValueError(
            f"Object role '{object_role}' has no interaction entities."
        )
    return scope.relation_pairs(subject_role, object_role)


def _gripper_joint_open(
    env,
    robot_name: str,
    gripper_range: GripperRange,
    *,
    open_ratio: float,
    env_idx: int,
) -> bool:
    robot = env.scene[robot_name]
    joint_ids, _ = robot.find_joints(gripper_range.name)
    if len(joint_ids) == 0:
        raise ValueError(
            f"Gripper joint '{gripper_range.name}' "
            f"not found in robot '{robot_name}'."
        )
    value = robot.data.joint_pos[env_idx][joint_ids[0]].item()
    return _gripper_joint_is_open(value, gripper_range, open_ratio)


def _is_within_xy(
    env,
    subject_id: EntityId,
    object_id: EntityId,
    *,
    env_idx: int,
) -> bool:
    from robo_orchard_sim.task_components.validators.utils import (
        is_object_center_in_obb,
    )

    subject_pos = entity_position(env, subject_id, env_idx).cpu().numpy()
    object_pose_array = entity_pose_array(env, object_id)
    object_prim_path = entity_prim_path(env, object_id, env_idx)
    return is_object_center_in_obb(
        env.scene.stage,
        object_prim_path,
        object_pose_array,
        subject_pos,
        idx_env=env_idx,
    )


def _is_aligned_xy(
    env,
    subject_id: EntityId,
    object_id: EntityId,
    *,
    eps: np.ndarray,
    env_idx: int,
) -> bool:
    subject_pos = entity_position(env, subject_id, env_idx).cpu().numpy()
    object_pos = entity_position(env, object_id, env_idx).cpu().numpy()
    return bool(np.all(abs(subject_pos[:2] - object_pos[:2]) < eps))


def _is_aligned_xyz(
    env,
    subject_id: EntityId,
    object_id: EntityId,
    *,
    eps: np.ndarray,
    target_height_offset: float,
    env_idx: int,
) -> bool:
    subject_pos = entity_position(env, subject_id, env_idx).cpu().numpy()
    object_pos = entity_position(env, object_id, env_idx).cpu().numpy()
    target_pos = np.array(
        subject_pos[:2].tolist() + [subject_pos[2] + target_height_offset]
    )
    return bool(np.all(abs(object_pos - target_pos) < eps))


# ---------------------------------------------------------------------
# Scene-wide binary interaction checkers
# ---------------------------------------------------------------------


class WithinXYChecker:
    """Observe XY containment over a role-filtered relation domain."""

    def __init__(
        self,
        subject_role: str,
        object_role: str,
        *,
        metric_name: str = "within_xy",
    ) -> None:
        self.subject_role = subject_role
        self.object_role = object_role
        self.metric_name = metric_name

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return containment for every subject-role/object-role pair."""
        pairs = _relation_pairs(
            scope,
            self.subject_role,
            self.object_role,
        )
        del context
        return SceneMetrics(
            binary={
                self.metric_name: {
                    pair: _is_within_xy(
                        env,
                        pair[0],
                        pair[1],
                        env_idx=env_idx,
                    )
                    for pair in pairs
                }
            }
        )

    def reset(self) -> None:
        """XY containment checking is stateless."""


class AlignmentXYChecker:
    """Observe XY alignment over a role-filtered relation domain."""

    def __init__(
        self,
        subject_role: str,
        object_role: str,
        *,
        metric_name: str = "alignment_xy",
        eps: tuple[float, float] = (0.02, 0.02),
    ) -> None:
        self.subject_role = subject_role
        self.object_role = object_role
        self.metric_name = metric_name
        self.eps = np.array(eps)

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return alignment for every subject-role/object-role pair."""
        pairs = _relation_pairs(
            scope,
            self.subject_role,
            self.object_role,
        )
        del context
        return SceneMetrics(
            binary={
                self.metric_name: {
                    pair: _is_aligned_xy(
                        env,
                        pair[0],
                        pair[1],
                        eps=self.eps,
                        env_idx=env_idx,
                    )
                    for pair in pairs
                }
            }
        )

    def reset(self) -> None:
        """XY alignment checking is stateless."""


class AlignmentXYZChecker:
    """Observe offset XYZ alignment over a role-filtered relation domain."""

    def __init__(
        self,
        subject_role: str,
        object_role: str,
        *,
        metric_name: str = "alignment_xyz",
        eps: tuple[float, float, float] = (0.025, 0.025, 0.0120),
        target_height_offset: float = 0.04,
    ) -> None:
        self.subject_role = subject_role
        self.object_role = object_role
        self.metric_name = metric_name
        self.eps = np.array(eps)
        self.target_height_offset = target_height_offset

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return alignment for every subject-role/object-role pair."""
        pairs = _relation_pairs(
            scope,
            self.subject_role,
            self.object_role,
        )
        del context
        return SceneMetrics(
            binary={
                self.metric_name: {
                    pair: _is_aligned_xyz(
                        env,
                        pair[0],
                        pair[1],
                        eps=self.eps,
                        target_height_offset=self.target_height_offset,
                        env_idx=env_idx,
                    )
                    for pair in pairs
                }
            }
        )

    def reset(self) -> None:
        """XYZ alignment checking is stateless."""


def _rotate_by_quat_wxyz(
    quat_wxyz: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
    """Rotate a vector by a scalar-first quaternion."""
    w, xyz = quat_wxyz[0], quat_wxyz[1:]
    t = 2.0 * torch.linalg.cross(xyz, vec)
    return vec + w * t + torch.linalg.cross(xyz, t)


def _angle_between_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the unsigned angle between vectors in degrees."""
    a = a / (torch.linalg.norm(a) + 1e-12)
    b = b / (torch.linalg.norm(b) + 1e-12)
    cos_a = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_a)).item()


class AxisAlignChecker:
    """Observe axis alignment for every entity in an interaction scope."""

    def __init__(
        self,
        local_axis: Sequence[float],
        world_target: Sequence[float],
        max_deg: float,
        *,
        metric_name: str = "axis_aligned",
    ) -> None:
        if len(local_axis) != 3:
            raise ValueError(
                f"local_axis must have 3 components, got {len(local_axis)}."
            )
        if len(world_target) != 3:
            raise ValueError(
                f"world_target must have 3 components, "
                f"got {len(world_target)}."
            )
        if max_deg < 0.0:
            raise ValueError(f"max_deg must be non-negative, got {max_deg}.")

        self._local_axis = torch.tensor(list(local_axis), dtype=torch.float32)
        self._world_target = torch.tensor(
            list(world_target), dtype=torch.float32
        )
        self.max_deg = max_deg
        self.metric_name = metric_name

    def _entity_is_aligned(
        self,
        env,
        entity_id: EntityId,
        env_idx: int,
    ) -> bool:
        quat_w = entity_quaternion(env, entity_id, env_idx)
        local_axis = self._local_axis.to(
            device=quat_w.device, dtype=quat_w.dtype
        )
        world_target = self._world_target.to(
            device=quat_w.device, dtype=quat_w.dtype
        )
        world_axis = _rotate_by_quat_wxyz(quat_w, local_axis)
        return _angle_between_deg(world_axis, world_target) <= self.max_deg

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return one axis-alignment value per scoped entity."""
        del context
        return SceneMetrics(
            unary={
                self.metric_name: {
                    entity_id: self._entity_is_aligned(
                        env,
                        entity_id,
                        env_idx,
                    )
                    for entity_id in scope.entities
                }
            }
        )

    def reset(self) -> None:
        """Axis alignment checking is stateless."""


class GripperOpenChecker:
    """Observe one configured gripper joint as a global metric."""

    def __init__(
        self,
        joint_name: str,
        *,
        metric_name: str | None = None,
        open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
    ) -> None:
        if not joint_name:
            raise ValueError("joint_name is required for gripper checker.")
        self.joint_name = joint_name
        self.metric_name = metric_name or f"gripper_open:{joint_name}"
        self.open_ratio = open_ratio

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return the configured joint's current open state."""
        del scope
        robot = context.robot
        if robot is None:
            raise ValueError("Gripper checking requires robot metadata.")
        gripper_range = next(
            (
                spec
                for spec in robot.gripper_joints
                if spec.name == self.joint_name
            ),
            None,
        )
        if gripper_range is None:
            raise ValueError(
                f"Gripper joint '{self.joint_name}' is not configured "
                f"for robot '{robot.robot_name}'."
            )
        return SceneMetrics(
            global_values={
                self.metric_name: _gripper_joint_open(
                    env,
                    robot.robot_name,
                    gripper_range,
                    open_ratio=self.open_ratio,
                    env_idx=env_idx,
                )
            }
        )

    def reset(self) -> None:
        """Single-joint gripper checking is stateless."""


class BothGripperOpenChecker:
    """Observe whether every configured gripper joint is open."""

    def __init__(
        self,
        *,
        metric_name: str = "gripper_open",
        open_ratio: float = DEFAULT_GRIPPER_OPEN_RATIO,
    ) -> None:
        self.metric_name = metric_name
        self.open_ratio = open_ratio

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return whether every context-defined gripper joint is open."""
        del scope
        robot = context.robot
        if robot is None:
            raise ValueError("Gripper checking requires robot metadata.")
        if not robot.gripper_joints:
            raise ValueError("Gripper checking requires gripper joints.")
        return SceneMetrics(
            global_values={
                self.metric_name: all(
                    _gripper_joint_open(
                        env,
                        robot.robot_name,
                        gripper_range,
                        open_ratio=self.open_ratio,
                        env_idx=env_idx,
                    )
                    for gripper_range in robot.gripper_joints
                )
            }
        )

    def reset(self) -> None:
        """Multi-joint gripper checking is stateless."""


# ---------------------------------------------------------------------
# Rigid settled-state check
# ---------------------------------------------------------------------


class _SpeedThresholds:
    """Velocity thresholds for one rigid settled-state check."""

    def __init__(
        self,
        linear_threshold: float = 0.01,
        angular_threshold: float = 0.1,
        check_angular: bool = True,
    ) -> None:
        if linear_threshold < 0.0:
            raise ValueError(
                f"linear_threshold must be non-negative, "
                f"got {linear_threshold}."
            )
        if angular_threshold < 0.0:
            raise ValueError(
                f"angular_threshold must be non-negative, "
                f"got {angular_threshold}."
            )
        self.linear_threshold = linear_threshold
        self.angular_threshold = angular_threshold
        self.check_angular = check_angular

    def is_settled(self, linear: torch.Tensor, angular: torch.Tensor) -> bool:
        """Return whether both speeds sit within their thresholds."""
        if float(torch.linalg.norm(linear)) > self.linear_threshold:
            return False
        if not self.check_angular:
            return True
        return float(torch.linalg.norm(angular)) <= self.angular_threshold


class StationaryChecker:
    """Observe stationary state for every entity in the interaction scope."""

    def __init__(
        self,
        linear_threshold: float = 0.01,
        angular_threshold: float = 0.1,
        check_angular: bool = True,
        *,
        metric_name: str = "stationary",
    ) -> None:
        self.speed = _SpeedThresholds(
            linear_threshold=linear_threshold,
            angular_threshold=angular_threshold,
            check_angular=check_angular,
        )
        self.metric_name = metric_name

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return one stationary value per scoped entity."""
        del context
        stationary: dict[EntityId, bool] = {}
        for entity_id in scope.entities:
            if isinstance(entity_id, SceneJointKey):
                continue
            if isinstance(entity_id, str):
                linear = env.scene[entity_id].data.root_lin_vel_w[env_idx]
                angular = env.scene[entity_id].data.root_ang_vel_w[env_idx]
            else:
                body_id = resolve_body_id(env, entity_id)
                data = env.scene[entity_id.scene_name].data
                linear = data.body_lin_vel_w[env_idx, body_id]
                angular = data.body_ang_vel_w[env_idx, body_id]
            stationary[entity_id] = self.speed.is_settled(linear, angular)
        for component in iter_articulation_bodies(env, scope):
            data = component.articulation.data
            stationary[component.body_key] = self.speed.is_settled(
                data.body_lin_vel_w[env_idx, component.body_id],
                data.body_ang_vel_w[env_idx, component.body_id],
            )
        return SceneMetrics(unary={self.metric_name: stationary})

    def reset(self) -> None:
        """Stationary checking is stateless."""


class ContactChecker:
    """Observe gripper contact and grasp for scoped scene entities."""

    def __init__(
        self,
        force_threshold: float = 0.01,
        min_opposition_deg: float = 120.0,
        min_inward_cosine: float = 0.3,
        *,
        contact_metric: str = "contacted",
        grasp_metric: str = "grasped",
    ) -> None:
        if force_threshold < 0.0:
            raise ValueError(
                f"force_threshold must be non-negative, got {force_threshold}."
            )
        if not 0.0 <= min_opposition_deg <= 180.0:
            raise ValueError(
                "min_opposition_deg must be within [0, 180], "
                f"got {min_opposition_deg}."
            )
        if not 0.0 <= min_inward_cosine <= 1.0:
            raise ValueError(
                "min_inward_cosine must be within [0, 1], "
                f"got {min_inward_cosine}."
            )
        self.force_threshold = force_threshold
        self.max_opposition_cosine = math.cos(math.radians(min_opposition_deg))
        self.min_inward_cosine = min_inward_cosine
        self.contact_metric = contact_metric
        self.grasp_metric = grasp_metric
        self._bindings: dict[str, ContactBinding] = {}

    @staticmethod
    def _target_paths(
        env,
        scope: RoleScope,
        components,
    ) -> dict[ContactFilterKey, str]:
        targets = {}
        for env_idx in range(env.num_envs):
            for entity_id in scope.entities:
                if isinstance(entity_id, SceneBodyKey):
                    targets[(env_idx, entity_id)] = entity_prim_path(
                        env,
                        entity_id,
                        env_idx,
                    )
                elif isinstance(
                    entity_id, str
                ) and not has_articulation_components(env.scene[entity_id]):
                    targets[(env_idx, entity_id)] = (
                        resolve_rigid_body_prim_path(
                            env.scene[entity_id],
                            env_idx,
                        )
                    )
            for component in components:
                articulation = component.articulation
                try:
                    path = articulation.root_physx_view.link_paths[env_idx][
                        component.body_id
                    ]
                except (AttributeError, IndexError, TypeError):
                    raise ValueError(
                        f"Could not resolve body {component.body_id} on "
                        f"'{component.body_key.scene_name}' to a PhysX path "
                        f"for env {env_idx}."
                    ) from None
                targets[(env_idx, component.body_key)] = str(path)
        return targets

    def _binding(self, sensor_name: str) -> ContactBinding:
        if sensor_name not in self._bindings:
            self._bindings[sensor_name] = ContactBinding(sensor_name)
        return self._bindings[sensor_name]

    def _pair_contact_state(
        self,
        env,
        pair: GripperContactPair,
        target: SceneEntityKey,
        env_idx: int,
    ) -> tuple[bool, bool]:
        first = self._binding(pair.sensor_names[0]).finger_force(
            env,
            target,
            env_idx=env_idx,
        )
        second = self._binding(pair.sensor_names[1]).finger_force(
            env,
            target,
            env_idx=env_idx,
        )
        first_norm = torch.linalg.norm(first)
        second_norm = torch.linalg.norm(second)
        contacted = bool(
            first_norm.item() > self.force_threshold
            or second_norm.item() > self.force_threshold
        )
        if (
            first_norm.item() <= self.force_threshold
            or second_norm.item() <= self.force_threshold
        ):
            return contacted, False

        first_dir = first / first_norm
        second_dir = second / second_norm
        if (
            torch.dot(first_dir, second_dir).item()
            > self.max_opposition_cosine
        ):
            return contacted, False

        first_pos = self._binding(
            pair.sensor_names[0]
        ).finger_contact_position(env, target, env_idx=env_idx)
        second_pos = self._binding(
            pair.sensor_names[1]
        ).finger_contact_position(env, target, env_idx=env_idx)
        finger_axis = second_pos - first_pos
        if not torch.isfinite(finger_axis).all().item():
            return contacted, False
        finger_distance = torch.linalg.norm(finger_axis)
        if finger_distance.item() <= torch.finfo(finger_axis.dtype).eps:
            return contacted, False
        finger_axis = finger_axis / finger_distance
        # Sensors report forces on the fingers; negate for forces on target.
        grasped = (
            torch.dot(-first_dir, finger_axis).item() >= self.min_inward_cosine
            and torch.dot(-second_dir, -finger_axis).item()
            >= self.min_inward_cosine
        )
        return contacted, grasped

    def evaluate(
        self,
        env,
        context: "ValidatorContext",
        scope: RoleScope,
        env_idx: int = 0,
    ) -> SceneMetrics:
        """Return contact and grasp facts for all physical targets in scope."""
        robot = context.robot
        if robot is None:
            raise ValueError("Contact checking requires robot metadata.")
        pairs = gripper_contact_sensors(context)
        components = tuple(iter_articulation_bodies(env, scope))
        targets = self._target_paths(env, scope, components)
        for pair in pairs:
            for sensor_name in pair.sensor_names:
                self._binding(sensor_name).rebind(env, targets)

        target_keys = tuple(
            dict.fromkeys(
                [
                    *(
                        key
                        for key in scope.entities
                        if isinstance(key, SceneBodyKey)
                        or (
                            isinstance(key, str)
                            and not has_articulation_components(env.scene[key])
                        )
                    ),
                    *(component.body_key for component in components),
                ]
            )
        )
        pair_states = {
            target: tuple(
                self._pair_contact_state(
                    env,
                    pair,
                    target,
                    env_idx,
                )
                for pair in pairs
            )
            for target in target_keys
        }
        return SceneMetrics(
            unary={
                self.contact_metric: {
                    target: any(contacted for contacted, _ in states)
                    for target, states in pair_states.items()
                },
                self.grasp_metric: {
                    target: any(grasped for _, grasped in states)
                    for target, states in pair_states.items()
                },
            }
        )

    def reset(self) -> None:
        """Contact filters remain valid for the stable scene scope."""
