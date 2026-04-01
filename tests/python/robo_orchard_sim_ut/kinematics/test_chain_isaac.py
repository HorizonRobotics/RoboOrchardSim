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

import os
import tempfile

import pytest
import torch
from robo_orchard_core.kinematics.chain import (
    KinematicChain,
    KinematicSerialChain,
)
from robo_orchard_core.utils import math as math_utils

from robo_orchard_sim.cfg_wrappers.envs.env_cfg import SimulationCfg
from robo_orchard_sim.envs.env_base import (
    IsaacEnvContextManager,
)
from robo_orchard_sim.envs.manager_based_env import (
    IsaacManagerBasedEnvCfg,
)
from robo_orchard_sim.models.assets.articulation import Articulation
from robo_orchard_sim.models.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from robo_orchard_sim.models.scenes.table_scene import TableSceneCfg
from robo_orchard_sim.utils.usd import usd_to_urdf


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


class TestChainConsistencyWithIsaac:
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_fk_consistency(
        self,
        franka_table: TableSceneCfg,
        device: str,
    ):
        asset_name = "robots/robot_franka"
        body_name = "panda_hand"
        usd_path = franka_table.robots["robot_franka"].spawn.usd_path  # type: ignore

        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_folder = f"{temp_dir}/panda.urdf"
            urdf_path = usd_to_urdf(usd_path, urdf_folder)
            assert os.path.exists(urdf_path)
            serial_chain = KinematicSerialChain(
                chain=KinematicChain.from_file(
                    urdf_path, "urdf", device=device
                ),
                root_frame_name="panda",
                end_frame_name=f"panda_{body_name}",
            )

        env_cfg = IsaacManagerBasedEnvCfg[TableSceneCfg](
            scene=franka_table,
            sim=SimulationCfg(dt=1.0 / 60),
            decimation=1,
        )

        def compare_fk(
            step_id: int,
            franka: Articulation,
            body_name: str,
            joint_ids,
            serial_chain: KinematicSerialChain,
            device,
        ):
            body_idx = franka.data.body_names.index(body_name)
            joint_state = franka.data.joint_pos[:, joint_ids].to(device=device)
            body_state_w = franka.data.body_state_w[:, body_idx].to(
                device=device
            )[..., 0:7]
            body_state_w[..., 3:7] = math_utils.quaternion_standardize(
                body_state_w[..., 3:7]
            )

            fk = serial_chain.forward_kinematics(joint_state)
            fk_body_state_w = fk[f"panda_{body_name}"]
            fk_body_state_w = torch.cat(
                [
                    fk_body_state_w.get_translation(),
                    math_utils.quaternion_standardize(
                        fk_body_state_w.get_rotation_quaternion()
                    ),
                ],
                dim=-1,
            )
            if not torch.allclose(fk_body_state_w, body_state_w, atol=1e-6):
                print(f"step_id: {step_id}, body_state_w: {body_state_w}")
                print(
                    f"step_id: {step_id}, fk_body_state_w: {fk_body_state_w}"
                )
                print(
                    f"step_id: {step_id}, diff: {fk_body_state_w - body_state_w}"  # noqa: E501
                )
                raise ValueError("FK is not consistent with the simulation.")

        with IsaacEnvContextManager(
            cfg=env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            env.step()
            franka: Articulation = env.scene[asset_name]
            joint_ids, joint_names = franka.find_joints(["panda_joint.*"])

            body_idx = franka.data.body_names.index(body_name)
            print("body_names: ", franka.data.body_names)
            print("frame_names: ", serial_chain.frame_names)
            print(
                "joint_parameter_names: ", serial_chain.joint_parameter_names
            )
            print("joint_names: ", joint_names)
            # make sure that the joint names are the same
            assert serial_chain.joint_parameter_names == joint_names
            assert f"panda_{body_name}" in serial_chain.frame_names
            body_state_w = franka.data.body_state_w[:, body_idx].to(
                device=device
            )
            joint_state = franka.data.joint_pos[:, joint_ids].to(device=device)
            print("joint_state.shape: ", joint_state.shape)
            print("body_state_w.shape: ", body_state_w.shape)

            for i in range(100):
                env.step()
                compare_fk(
                    step_id=i,
                    franka=franka,
                    body_name=body_name,
                    joint_ids=joint_ids,
                    serial_chain=serial_chain,
                    device=device,
                )

    @pytest.mark.parametrize("device", ["cpu"])
    def test_jacobian_consistency(
        self,
        franka_table: TableSceneCfg,
        device: str,
    ):
        asset_name = "robots/robot_franka"
        body_name = "panda_hand"
        usd_path = franka_table.robots["robot_franka"].spawn.usd_path  # type: ignore
        dtype = torch.float32
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_folder = f"{temp_dir}/panda.urdf"
            urdf_path = usd_to_urdf(usd_path, urdf_folder)
            assert os.path.exists(urdf_path)
            serial_chain = KinematicSerialChain(
                chain=KinematicChain.from_file(
                    urdf_path, "urdf", device=device, dtype=dtype
                ),
                root_frame_name="panda",
                end_frame_name=f"panda_{body_name}",
            )

        env_cfg = IsaacManagerBasedEnvCfg[TableSceneCfg](
            scene=franka_table,
            sim=SimulationCfg(dt=1.0 / 60),
            decimation=1,
        )

        # @TODO: The difference between the jacobian from the simulation and
        # the serial chain is large. It is not numerical issue. Need to
        # investigate further.
        def compare_jacobian(
            step_id: int,
            franka: Articulation,
            body_name: str,
            joint_ids,
            serial_chain: KinematicSerialChain,
            device,
            tol: float = 0.04,
        ):
            body_idx = franka.data.body_names.index(body_name)
            if franka.is_fixed_base:
                jacobi_body_idx = body_idx - 1
                jacobi_joint_ids = joint_ids
            else:
                jacobi_body_idx = body_idx
                jacobi_joint_ids = [i + 6 for i in joint_ids]
            jacobian: torch.Tensor = franka.root_physx_view.get_jacobians()[
                :, jacobi_body_idx, :, jacobi_joint_ids
            ].to(device=device, dtype=dtype)
            joint_state = franka.data.joint_pos[:, joint_ids].to(
                device=device, dtype=dtype
            )
            serial_chain_jacobian = serial_chain.jacobian(joint_state)
            if not torch.allclose(
                jacobian, serial_chain_jacobian, atol=tol, rtol=tol
            ):
                print(f"step_id: {step_id}, jacobian: {jacobian}")
                print(
                    f"step_id: {step_id}, "
                    f"serial_chain_jacobian: {serial_chain_jacobian}"
                )
                print(
                    f"step_id: {step_id}, "
                    f"diff: {serial_chain_jacobian - jacobian}"
                )
                raise ValueError(
                    "Jacobian is not consistent with the simulation."
                )

        with IsaacEnvContextManager(
            cfg=env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            env.step()
            franka: Articulation = env.scene[asset_name]
            joint_ids, joint_names = franka.find_joints(["panda_joint.*"])

            print("body_names: ", franka.data.body_names)
            print("frame_names: ", serial_chain.frame_names)
            print(
                "joint_parameter_names: ", serial_chain.joint_parameter_names
            )
            print("joint_names: ", joint_names)
            # make sure that the joint names are the same
            assert serial_chain.joint_parameter_names == joint_names
            assert f"panda_{body_name}" in serial_chain.frame_names

            for i in range(100):
                env.step()
                compare_jacobian(
                    step_id=i,
                    franka=franka,
                    body_name=body_name,
                    joint_ids=joint_ids,
                    serial_chain=serial_chain,
                    device=device,
                )
