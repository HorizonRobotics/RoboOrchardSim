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

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.embodiment import (
    DualArmPiperEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ManipulatorProfile,
    ResolvedManipulatorProfile,
    RobotInfoCfg,
)
from robo_orchard_sim.tasks.trajs_gen.base_executor import BaseExecutorCfg
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
)


class _FakePlanner:
    def __init__(
        self,
        cfg: "_FakePlannerCfg | None" = None,
        env_nums: int = 1,
    ) -> None:
        self.cfg = cfg
        self.env_nums = env_nums


class _FakePlannerCfg:
    class_type = _FakePlanner


class _FakeArticulation:
    def find_joints(self, names):
        mapping = {
            "left_joint[1-6]": (
                [0, 1, 2, 3, 4, 5],
                [f"left_joint{i}" for i in range(1, 7)],
            ),
            "left_joint7": ([6], ["left_joint7"]),
            "left_joint8": ([7], ["left_joint8"]),
        }
        ids = []
        resolved_names = []
        for name in names:
            if name not in mapping:
                continue
            found_ids, found_names = mapping[name]
            ids.extend(found_ids)
            resolved_names.extend(found_names)
        return ids, resolved_names

    def find_bodies(self, names):
        mapping = {
            "left_base_link": ([10], ["left_base_link"]),
            "left_link1": ([11], ["left_link1"]),
            "left_link2": ([12], ["left_link2"]),
            "left_link3": ([13], ["left_link3"]),
            "left_link4": ([14], ["left_link4"]),
            "left_link5": ([15], ["left_link5"]),
            "left_link6": ([16], ["left_link6"]),
        }
        if isinstance(names, str):
            return mapping.get(names, ([], []))

        ids = []
        resolved_names = []
        for name in names:
            if name not in mapping:
                continue
            found_ids, found_names = mapping[name]
            ids.extend(found_ids)
            resolved_names.extend(found_names)
        return ids, resolved_names


class _FakeEnv:
    def __init__(self):
        self.num_envs = 1
        self.scene = {"robots/dualarm_piper": _FakeArticulation()}


def test_robot_info_cfg_dualarm_piper_left_arm_returns_correct_identity():
    embodiment = DualArmPiperEmbodiment(enable_cameras=False)

    robot_info = embodiment.get_robot_info_cfg("left_arm")

    assert isinstance(robot_info, RobotInfoCfg)
    assert robot_info.robot_name == "robots/dualarm_piper"
    assert robot_info.manipulator_name == "left_arm"


def test_robot_info_cfg_dualarm_piper_left_arm_profile_has_correct_anatomy():
    embodiment = DualArmPiperEmbodiment(enable_cameras=False)

    robot_info = embodiment.get_robot_info_cfg("left_arm")

    assert isinstance(robot_info.manipulator_profile, ManipulatorProfile)
    assert robot_info.manipulator_profile.ee_body_name == "left_link6"
    assert robot_info.manipulator_profile.gripper_joint_names == (
        "left_joint7",
        "left_joint8",
    )


def test_resolved_manipulator_profile_from_articulation_returns_ids():
    cfg = ManipulatorProfile(
        arm_joint_names=("left_joint[1-6]",),
        gripper_joint_names=("left_joint7", "left_joint8"),
        body_names=("left_base_link", "left_link6"),
        base_body_name="left_base_link",
        ee_body_name="left_link6",
    )

    resolved = ResolvedManipulatorProfile.from_articulation(
        articulation=_FakeArticulation(),
        robot_name="robots/dualarm_piper",
        manipulator_name="left_arm",
        cfg=cfg,
        planner=cast(Any, _FakePlanner()),
    )

    assert resolved.joint_ids == (0, 1, 2, 3, 4, 5)
    assert resolved.gripper_joint_ids == (6, 7)
    assert resolved.ee_body_id == 16
    assert resolved.base_body_id == 10


def test_base_executor_cfg_resolve_with_robot_info_returns_ids():
    embodiment = DualArmPiperEmbodiment(enable_cameras=False)
    cfg = BaseExecutorCfg(
        robot_info=replace(
            embodiment.get_robot_info_cfg("left_arm"),
            planner=cast(Any, _FakePlannerCfg()),
        ),
    )

    resolved = cfg.resolve_manipulator_info(
        env=_FakeEnv(),
        context=ManipulatorBindingContext(),
    )

    assert resolved.robot_name == "robots/dualarm_piper"
    assert resolved.manipulator_name == "left_arm"
    assert resolved.ee_body_name == "left_link6"


def test_base_executor_cfg_resolve_with_robot_info_instantiates_planner():
    embodiment = DualArmPiperEmbodiment(enable_cameras=False)
    cfg = BaseExecutorCfg(
        robot_info=replace(
            embodiment.get_robot_info_cfg("left_arm"),
            planner=cast(Any, _FakePlannerCfg()),
        ),
    )

    resolved = cfg.resolve_manipulator_info(
        env=_FakeEnv(),
        context=ManipulatorBindingContext(),
    )

    assert isinstance(resolved.planner, _FakePlanner)


def test_robot_info_cfg_planner_without_context_raises_value_error():
    embodiment = DualArmPiperEmbodiment(enable_cameras=False)
    robot_info = embodiment.get_robot_info_cfg("left_arm")

    with pytest.raises(ValueError, match="requires a manipulator context"):
        robot_info.resolve(env=_FakeEnv())


def test_robot_info_cfg_missing_planner_raises_value_error():
    with pytest.raises(ValueError, match="RobotInfoCfg.planner"):
        RobotInfoCfg(
            robot_name="robots/fake",
            manipulator_name="left_arm",
            gripper_open_val=[],
            gripper_close_val=[],
            t_standard_tcp_to_robot_ee=np.eye(4),
            manipulator_profile=ManipulatorProfile(
                arm_joint_names=("joint1",),
                ee_body_name="ee",
            ),
            planner=cast(Any, None),
        )


def test_resolved_manipulator_profile_missing_planner_raises_value_error():
    with pytest.raises(ValueError, match="ResolvedManipulatorProfile.planner"):
        ResolvedManipulatorProfile(
            robot_name="robots/fake",
            manipulator_name="left_arm",
            joint_ids=(0,),
            joint_names=("joint1",),
            gripper_joint_ids=(),
            gripper_joint_names=(),
            body_ids=(),
            body_names=(),
            ee_body_id=0,
            ee_body_name="ee",
            planner=cast(Any, None),
        )
