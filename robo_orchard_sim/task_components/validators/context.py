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

"""Runtime context used when building task validators."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from robo_orchard_sim.task_components.validators.base import GripperRange

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
        EmbodimentBase,
    )


@dataclass(frozen=True, slots=True)
class ValidatorRobotContext:
    """Runtime robot data consumed by validator construction."""

    robot_name: str
    ee_links: tuple[str, ...] = ()
    gripper_joints: tuple[GripperRange, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidatorContext:
    """Runtime context used to build task validators."""

    robot: ValidatorRobotContext | None = None


def build_validator_context(
    embodiment: "EmbodimentBase",
) -> ValidatorContext:
    """Build validator runtime context from the resolved embodiment."""
    robot_info_cfgs = embodiment.get_robot_info_cfgs()
    if not robot_info_cfgs:
        return ValidatorContext()

    ee_links: list[str] = []
    gripper_joints: list[GripperRange] = []
    seen_ee_links: set[str] = set()
    seen_gripper_joints: set[str] = set()
    for robot_info in robot_info_cfgs.values():
        manipulator_profile = robot_info.manipulator_profile
        if manipulator_profile is None:
            continue
        ee_body_name = manipulator_profile.ee_body_name
        if ee_body_name and ee_body_name not in seen_ee_links:
            ee_links.append(ee_body_name)
            seen_ee_links.add(ee_body_name)

        for joint_name, open_val, close_val in zip(
            manipulator_profile.gripper_joint_names,
            robot_info.gripper_open_val,
            robot_info.gripper_close_val,
            strict=True,
        ):
            if joint_name in seen_gripper_joints:
                continue
            gripper_joints.append(
                GripperRange(
                    name=joint_name,
                    open_val=open_val,
                    close_val=close_val,
                )
            )
            seen_gripper_joints.add(joint_name)

    return ValidatorContext(
        robot=ValidatorRobotContext(
            robot_name=embodiment.scene_name,
            ee_links=tuple(ee_links),
            gripper_joints=tuple(gripper_joints),
        )
    )
