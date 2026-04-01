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

import torch
from robo_orchard_core.envs.managers.events.event_manager import (
    EventManagerCfg,
)

from robo_orchard_sim.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.envs import (
    IsaacEnvContextManager,
    IsaacManagerBasedEnv,
    IsaacManagerBasedEnvCfg,
)
from robo_orchard_sim.envs.managers.events.default_reset import (
    DefaultResetTerm,
    DefaultResetTermCfg,
)
from robo_orchard_sim.models.scenes.table_scene import TableSceneCfg


class TestDefaultResetTerm:
    def test_default_reset_cfg_defaults_reset_joint_targets_to_false(self):
        cfg = DefaultResetTermCfg(trigger_topic="reset")

        assert cfg.class_type is DefaultResetTerm
        assert cfg.reset_joint_targets is False

    def test_default_reset_event_restores_rigid_asset_default_state(
        self,
        simple_table_scene_cfg: TableSceneCfg,
    ):
        scene_cfg = simple_table_scene_cfg.copy()
        scene_cfg.objects["cube1"].spawn.rigid_props.disable_gravity = True

        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=scene_cfg,
            events=EventManagerCfg(
                terms={
                    "default_reset": DefaultResetTermCfg(
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                    )
                }
            ),
        )

        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            origin_pose = (
                env.scene["objects/cube1"].data.root_state_w[:, :7].clone()
            )
            _ = env.step()
            env.reset()
            _ = env.step()
            after_pose = (
                env.scene["objects/cube1"].data.root_state_w[:, :7].clone()
            )

            assert torch.allclose(
                origin_pose, after_pose, rtol=1e-02, atol=1e-04
            )

    def test_default_reset_event_restores_franka_joint_targets(
        self,
        franka_table: TableSceneCfg,
    ):
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=franka_table.copy(),
            events=EventManagerCfg(
                terms={
                    "default_reset": DefaultResetTermCfg(
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                        reset_joint_targets=True,
                        asset_cfgs=[
                            SceneEntityCfg(name="robots/robot_franka")
                        ],
                    )
                }
            ),
        )

        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            robot = env.scene["robots/robot_franka"]
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()

            updated_joint_pos = default_joint_pos + 0.05
            updated_joint_vel = default_joint_vel + 0.1
            robot.set_joint_position_target(updated_joint_pos)
            robot.set_joint_velocity_target(updated_joint_vel)

            assert torch.allclose(
                robot.data.joint_pos_target, updated_joint_pos
            )
            assert torch.allclose(
                robot.data.joint_vel_target, updated_joint_vel
            )

            env.reset()
            _ = env.step()

            assert torch.allclose(
                robot.data.joint_pos_target,
                default_joint_pos,
                rtol=1e-05,
                atol=1e-06,
            )
            assert torch.allclose(
                robot.data.joint_vel_target,
                default_joint_vel,
                rtol=1e-05,
                atol=1e-06,
            )
