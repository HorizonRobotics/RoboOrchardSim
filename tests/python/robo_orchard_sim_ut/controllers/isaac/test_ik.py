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

import numpy as np
import pytest
import torch
from robo_orchard_core.controllers.differential_ik import (
    DifferentialIKSolverConfig,
)
from robo_orchard_core.utils import math as math_utils

from robo_orchard_sim.controllers.differential_ik.differential_ik import (
    IsaacDifferentialIKControllerCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.envs.env_cfg import SimulationCfg
from robo_orchard_sim.ext.envs.env_base import (
    IsaacEnvContextManager,
)
from robo_orchard_sim.ext.envs.manager_based_env import (
    IsaacManagerBasedEnvCfg,
)
from robo_orchard_sim.ext.models.assets.articulation import Articulation
from robo_orchard_sim.ext.models.scenes.table_scene import (
    TableSceneCfg,
)
from robo_orchard_sim.orchard_env.embodiments.franka_panda.cfg import (
    FRANKA_PANDA_HIGH_PD_CFG,
)


@pytest.fixture()
def franka_table() -> TableSceneCfg:
    """Fixture for a table scene with a Franka Panda robot.

    We use FRANKA_PANDA_HIGH_PD_CFG for better control of the robot.
    """
    scene_cfg = TableSceneCfg(
        num_envs=1,
        env_spacing=2,
        # replace prim_path with your own robot prim_path
        robots={
            "robot_franka": FRANKA_PANDA_HIGH_PD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/robot_franka"
            )
        },
    )
    return scene_cfg


class TestDifferentialIKController:
    def test_config_init(self, app, franka_table: TableSceneCfg):
        ik_cfg = IsaacDifferentialIKControllerCfg(
            asset_name="robots/robot_franka",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            num_envs=1,
            device="cpu",
            diff_ik_solver_cfg=DifferentialIKSolverConfig(ik_method="dls"),
        )

        env_cfg = IsaacManagerBasedEnvCfg[TableSceneCfg](
            scene=franka_table,
            sim=SimulationCfg(dt=1.0 / 60),
            decimation=1,
        )

        with IsaacEnvContextManager(
            cfg=env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            _ = env.step()
            print("ik_cfg: ", ik_cfg)
            controller = ik_cfg.__call__(env)
            print("controller: ", controller)
            env.scene.delete_all_assets()

        # # with env_manager as env:
        # #     env.

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_set_goal_and_compute(
        self, app, franka_table: TableSceneCfg, device: str
    ):
        ik_cfg = IsaacDifferentialIKControllerCfg(
            asset_name="robots/robot_franka",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            num_envs=1,
            device=device,
            diff_ik_solver_cfg=DifferentialIKSolverConfig(ik_method="dls"),
        )

        env_cfg = IsaacManagerBasedEnvCfg[TableSceneCfg](
            scene=franka_table,
            sim=SimulationCfg(dt=1.0 / 60),
            decimation=1,
        )

        with IsaacEnvContextManager(
            cfg=env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            env.step()
            controller = ik_cfg(env)

            franka: Articulation = env.scene[ik_cfg.asset_name]
            print("body_names: ", franka.data.body_names)
            print("root_state_w: ", franka.data.root_state_w)
            # find the idx of "panda_hand" in body_names
            body_idx = franka.data.body_names.index("panda_hand")

            # run a few steps to stabilize the robot
            for _ in range(200):
                env.step()

            # set the goal position for the controller.
            # The goal should not be too far from the current position
            # to avoid infeasible solutions

            target_pos = torch.tensor([[0.4, 0, 0.6]], device=device)
            target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
            controller.set_goal(target_pos=target_pos, target_quat=target_quat)
            initial_body_state_w = franka.data.body_state_w[:, body_idx].to(
                device=device
            )
            for _ in range(400):
                body_state_w = franka.data.body_state_w[:, body_idx].to(
                    device=device
                )
                # since the root frame is the same as the world frame, we can
                # directly use the body_state_w as the input
                target_joint_pos = controller.calculate(
                    body_pos=body_state_w[..., :3],
                    body_quat=body_state_w[..., 3:7],
                ).to(device=franka.device)

                franka.set_joint_position_target(
                    target_joint_pos,
                    joint_ids=controller.joint_ids,
                )
                franka.write_data_to_sim()
                env.step()

            body_state_w = franka.data.body_state_w[:, body_idx].to(
                device=device
            )
            print("initial body_state_w: ", initial_body_state_w)
            print("target_pos: ", target_pos)
            print("target_quat: ", target_quat)
            print("final body_state_w: ", body_state_w)

            pos_err, quat_err = math_utils.pose_diff(
                target_pos,
                target_quat,
                body_state_w[..., :3],
                body_state_w[..., 3:7],
            )
            axis_angle_err = math_utils.quaternion_to_axis_angle(quat_err)
            print("pos_err: ", pos_err, " norm: ", pos_err.norm())
            print("quat_err: ", quat_err)
            print(
                "axis angle error: ",
                axis_angle_err,
                " norm: ",
                axis_angle_err.norm(),
            )
            np.testing.assert_array_less(
                pos_err.norm().numpy(force=True), 1e-3
            )
            np.testing.assert_array_less(
                axis_angle_err.norm().numpy(force=True), 1e-3
            )
            env.scene.delete_all_assets()
