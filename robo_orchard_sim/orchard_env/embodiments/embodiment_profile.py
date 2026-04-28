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

"""Embodiment profile metadata used by trajectory generation executors."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from robo_orchard_sim.controllers.curobo_planner.curobo import (
        ArticulationJointCuroboTrajPlanner,
        ArticulationJointCuroboTrajPlannerCfg,
    )


def _to_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Normalize optional string sequences into tuples."""
    if values is None:
        return ()
    return tuple(values)


def _find_one_body(
    articulation: Any,
    body_name: str,
    field_name: str,
) -> tuple[int, str]:
    """Resolve one body name and fail with a precise error message."""
    body_ids, body_names = articulation.find_bodies(body_name)
    if len(body_ids) != 1:
        raise ValueError(
            f"Expected exactly one body for {field_name}='{body_name}', "
            f"but found {len(body_ids)} matches: {body_names}."
        )
    return int(body_ids[0]), body_names[0]


@dataclass(frozen=True)
class ManipulatorProfile:
    """Semantic manipulator description owned by an embodiment."""

    arm_joint_names: tuple[str, ...]
    ee_body_name: str
    gripper_joint_names: tuple[str, ...] = ()
    body_names: tuple[str, ...] = ()
    base_body_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arm_joint_names",
            _to_tuple(self.arm_joint_names),
        )
        object.__setattr__(
            self,
            "gripper_joint_names",
            _to_tuple(self.gripper_joint_names),
        )
        object.__setattr__(self, "body_names", _to_tuple(self.body_names))
        if not self.arm_joint_names:
            raise ValueError("arm_joint_names must not be empty.")
        if not self.ee_body_name:
            raise ValueError("ee_body_name must not be empty.")


@dataclass(frozen=True)
class RobotInfoCfg:
    """Executor-facing concrete robot and manipulator configuration."""

    robot_name: str
    manipulator_name: str
    gripper_open_val: list[float]  # should match number of gripper joints
    gripper_close_val: list[float]  # should match number of gripper joints
    t_standard_tcp_to_robot_ee: np.ndarray
    planner: ArticulationJointCuroboTrajPlannerCfg
    manipulator_profile: ManipulatorProfile | None = None

    def __post_init__(self) -> None:
        if self.planner is None:
            raise ValueError("RobotInfoCfg.planner must not be None.")

    def with_robot_name(self, robot_name: str) -> "RobotInfoCfg":
        """Return a copy bound to a concrete scene robot name."""
        return replace(self, robot_name=robot_name)

    def resolve(
        self,
        env: Any,
        context: Any = None,
    ) -> "ResolvedManipulatorProfile":
        """Resolve the configured manipulator into runtime ids."""
        robot_info = self

        manipulator_profile = robot_info.manipulator_profile
        robot_name = robot_info.robot_name

        if manipulator_profile is None:
            raise ValueError("RobotInfoCfg requires manipulator_profile.")
        if not robot_name:
            raise ValueError("RobotInfoCfg.robot_name must not be empty.")
        if not robot_info.manipulator_name:
            raise ValueError(
                "RobotInfoCfg.manipulator_name must not be empty."
            )

        if context is None:
            raise ValueError(
                "RobotInfoCfg planner resolution requires a manipulator "
                "context."
            )
        planner = context.resolve_planner_instance(
            robot_name=robot_name,
            manipulator_name=robot_info.manipulator_name,
            planner_cfg=robot_info.planner,
            env_nums=env.num_envs,
        )

        articulation = env.scene[robot_name]
        return ResolvedManipulatorProfile.from_articulation(
            articulation=articulation,
            robot_name=robot_name,
            manipulator_name=robot_info.manipulator_name,
            cfg=manipulator_profile,
            gripper_open_val=robot_info.gripper_open_val,
            gripper_close_val=robot_info.gripper_close_val,
            t_standard_tcp_to_robot_ee=robot_info.t_standard_tcp_to_robot_ee,
            planner=planner,
        )


@dataclass(frozen=True)
class ResolvedManipulatorProfile:
    """Runtime manipulator information resolved from an articulation."""

    robot_name: str
    manipulator_name: str
    joint_ids: tuple[int, ...]
    joint_names: tuple[str, ...]
    gripper_joint_ids: tuple[int, ...]
    gripper_joint_names: tuple[str, ...]
    body_ids: tuple[int, ...]
    body_names: tuple[str, ...]
    ee_body_id: int
    ee_body_name: str
    planner: ArticulationJointCuroboTrajPlanner
    gripper_open_val: list[float] | None = None
    gripper_close_val: list[float] | None = None
    t_standard_tcp_to_robot_ee: np.ndarray | None = None
    base_body_id: int | None = None
    base_body_name: str | None = None

    def __post_init__(self) -> None:
        if self.planner is None:
            raise ValueError(
                "ResolvedManipulatorProfile.planner must not be None."
            )

    @property
    def gripper_ids(self) -> tuple[int, ...]:
        """Compatibility alias for older executor code."""
        return self.gripper_joint_ids

    @property
    def ee_id(self) -> int:
        """Compatibility alias for code that expects ee_id."""
        return self.ee_body_id

    @classmethod
    def from_articulation(
        cls,
        articulation: Any,
        robot_name: str,
        manipulator_name: str,
        cfg: ManipulatorProfile,
        planner: ArticulationJointCuroboTrajPlanner,
        gripper_open_val: list[float] | None = None,
        gripper_close_val: list[float] | None = None,
        t_standard_tcp_to_robot_ee: np.ndarray | None = None,
    ) -> "ResolvedManipulatorProfile":
        """Resolve semantic manipulator metadata into runtime indices."""
        joint_ids, joint_names = articulation.find_joints(
            list(cfg.arm_joint_names)
        )
        if len(joint_ids) == 0:
            raise ValueError(
                f"Manipulator '{manipulator_name}' on robot '{robot_name}' "
                "did not resolve any arm joints."
            )

        gripper_joint_ids: tuple[int, ...] = ()
        gripper_joint_names: tuple[str, ...] = ()
        if cfg.gripper_joint_names:
            found_ids, found_names = articulation.find_joints(
                list(cfg.gripper_joint_names)
            )
            gripper_joint_ids = tuple(int(idx) for idx in found_ids)
            gripper_joint_names = tuple(found_names)

        body_ids: tuple[int, ...] = ()
        body_names: tuple[str, ...] = ()
        if cfg.body_names:
            found_ids, found_names = articulation.find_bodies(
                list(cfg.body_names)
            )
            body_ids = tuple(int(idx) for idx in found_ids)
            body_names = tuple(found_names)

        ee_body_id, ee_body_name = _find_one_body(
            articulation,
            cfg.ee_body_name,
            "ee_body_name",
        )

        base_body_id = None
        base_body_name = None
        if cfg.base_body_name is not None:
            base_body_id, base_body_name = _find_one_body(
                articulation,
                cfg.base_body_name,
                "base_body_name",
            )

        return cls(
            robot_name=robot_name,
            manipulator_name=manipulator_name,
            joint_ids=tuple(int(idx) for idx in joint_ids),
            joint_names=tuple(joint_names),
            gripper_joint_ids=gripper_joint_ids,
            gripper_joint_names=gripper_joint_names,
            body_ids=body_ids,
            body_names=body_names,
            ee_body_id=ee_body_id,
            ee_body_name=ee_body_name,
            gripper_open_val=gripper_open_val,
            gripper_close_val=gripper_close_val,
            t_standard_tcp_to_robot_ee=t_standard_tcp_to_robot_ee,
            planner=planner,
            base_body_id=base_body_id,
            base_body_name=base_body_name,
        )
