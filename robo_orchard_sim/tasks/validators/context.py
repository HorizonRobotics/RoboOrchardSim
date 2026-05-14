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

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
        EmbodimentBase,
    )


@dataclass(frozen=True, slots=True)
class ValidatorRobotContext:
    """Runtime robot data consumed by validator construction."""

    robot_name: str
    ee_links: tuple[str, ...] = ()
    gripper_links: tuple[str, ...] = ()
    open_gripper_threshold: float | None = None


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
    gripper_links: list[str] = []
    seen_ee_links: set[str] = set()
    seen_gripper_links: set[str] = set()
    for robot_info in robot_info_cfgs.values():
        manipulator_profile = robot_info.manipulator_profile
        if manipulator_profile is None:
            continue
        ee_body_name = manipulator_profile.ee_body_name
        if ee_body_name and ee_body_name not in seen_ee_links:
            ee_links.append(ee_body_name)
            seen_ee_links.add(ee_body_name)
        positive_open_joints = set()
        for joint_name, joint_open_val in zip(
            manipulator_profile.gripper_joint_names,
            robot_info.gripper_open_val,
            strict=True,
        ):
            if joint_open_val > 0:
                positive_open_joints.add(joint_name)

        for gripper_joint_name in manipulator_profile.gripper_joint_names:
            if (
                positive_open_joints
                and gripper_joint_name not in positive_open_joints
            ):
                continue
            if gripper_joint_name in seen_gripper_links:
                continue
            gripper_links.append(gripper_joint_name)
            seen_gripper_links.add(gripper_joint_name)

    return ValidatorContext(
        robot=ValidatorRobotContext(
            robot_name=embodiment.scene_name,
            ee_links=tuple(ee_links),
            gripper_links=tuple(gripper_links),
        )
    )
