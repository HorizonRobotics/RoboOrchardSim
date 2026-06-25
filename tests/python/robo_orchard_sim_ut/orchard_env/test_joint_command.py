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

from types import SimpleNamespace

import pytest
import torch
from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)

from robo_orchard_sim.contracts.joint_command import (
    UnifiedJointCommand,
)
from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import ArticulationCfg
from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.ext.envs.managers.actions.articulation.joint_position import (  # noqa: E501
    ArticulationJointPositionActionTermCfg,
)
from robo_orchard_sim.orchard_env.assets import ArticulationSpec
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper.embodiment import (
    DualArmPiperEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)
from robo_orchard_sim.orchard_env.embodiments.franka_panda.embodiment import (
    FrankaPandaEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.panda_droid.embodiment import (
    PandaDroidEmbodiment,
)


class _ResolvedAssetCfg:
    name = "robots/test_robot"
    joint_ids = [0, 2]

    def resolve(self, scene):
        del scene


class _ActionCfgOnlyEmbodiment(EmbodimentBase):
    def __init__(self):
        super().__init__(
            robot=ArticulationSpec(
                name="test_robot",
                namespace="robots",
                template_cfg=ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/test_robot",
                    spawn=UsdFileCfg(usd_path="/tmp/test_robot.usd"),
                    init_state=ArticulationCfg.InitialStateCfg(joint_pos={}),
                    actuators={},
                ),
            )
        )

    def get_action_cfg(self) -> ActionManagerCfg:
        return ActionManagerCfg(
            terms={
                "arm": ArticulationJointPositionActionTermCfg(
                    asset_cfg=SceneEntityCfg(
                        name=self.scene_name,
                        joint_names=["joint[1-2]"],
                    ),
                    scale=1.0,
                    use_default_offset=False,
                ),
                "gripper": ArticulationJointPositionActionTermCfg(
                    asset_cfg=SceneEntityCfg(
                        name=self.scene_name,
                        joint_names=["finger_joint1", "finger_joint2"],
                    ),
                    scale=1.0,
                    use_default_offset=False,
                ),
            }
        )


class _WildcardActionCfgEmbodiment(_ActionCfgOnlyEmbodiment):
    def get_action_cfg(self) -> ActionManagerCfg:
        return ActionManagerCfg(
            terms={
                "arm": ArticulationJointPositionActionTermCfg(
                    asset_cfg=SceneEntityCfg(
                        name=self.scene_name,
                        joint_names=["joint.*"],
                    ),
                    scale=1.0,
                    use_default_offset=False,
                )
            }
        )


def test_unified_joint_command_compact_specs_selects_matching_columns():
    values = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    action = UnifiedJointCommand.from_specs(
        values=values,
        joint_specs=("left_joint[1-4]", "right_joint[1-4]"),
    )

    selected = action.select("right_joint[2-4]")

    assert selected.tolist() == [[5.0, 6.0, 7.0]]


def test_unified_joint_command_missing_selected_joint_raises_value_error():
    action = UnifiedJointCommand.from_specs(
        values=torch.zeros((1, 2)),
        joint_specs=("left_joint[1-2]",),
    )

    with pytest.raises(ValueError, match="left_joint3"):
        action.select("left_joint[2-3]")


def test_env_action_state_resolved_asset_cfg_returns_joint_positions():
    from robo_orchard_sim.contracts import joint_command

    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        )
    )
    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            active_terms=["robot_joint_position"],
            cfg=SimpleNamespace(
                terms={
                    "robot_joint_position": SimpleNamespace(
                        asset_cfg=_ResolvedAssetCfg()
                    )
                }
            ),
        ),
        scene={"robots/test_robot": asset},
    )

    action = joint_command.EnvActionState.build_hold_position(env)

    assert {
        term_name: term_action.tolist()
        for term_name, term_action in action.items()
    } == {"robot_joint_position": [[1.0, 3.0]]}


def test_env_action_state_partial_action_preserves_unupdated_terms():
    from robo_orchard_sim.contracts import joint_command

    env_action_state = joint_command.EnvActionState(
        {
            "left_robot_joint_position": torch.tensor([[1.0]]),
            "left_robot_gripper_control": torch.tensor([[2.0]]),
        }
    )

    action = env_action_state.update(
        {"left_robot_joint_position": torch.tensor([[9.0]])}
    )

    assert {
        term_name: term_action.tolist()
        for term_name, term_action in action.items()
    } == {
        "left_robot_joint_position": [[9.0]],
        "left_robot_gripper_control": [[2.0]],
    }


def test_env_action_state_unknown_term_raises_key_error():
    from robo_orchard_sim.contracts import joint_command

    env_action_state = joint_command.EnvActionState(
        {"left_robot_joint_position": torch.tensor([[1.0]])}
    )

    with pytest.raises(KeyError, match="right_robot_joint_position"):
        env_action_state.update(
            {"right_robot_joint_position": torch.tensor([[9.0]])}
        )


def test_embodiment_base_joint_command_action_cfg_terms_returns_env_action():
    action = UnifiedJointCommand.from_specs(
        values=torch.arange(4, dtype=torch.float32).reshape(1, 4),
        joint_specs=("joint[1-2]", "finger_joint1", "finger_joint2"),
    )
    embodiment = _ActionCfgOnlyEmbodiment()

    translated = embodiment.translate_joint_command_to_env_action(action)

    assert {key: value.tolist() for key, value in translated.items()} == {
        "arm": [[0.0, 1.0]],
        "gripper": [[2.0, 3.0]],
    }


def test_embodiment_base_joint_command_wildcard_spec_raises_value_error():
    action = UnifiedJointCommand.from_specs(
        values=torch.arange(2, dtype=torch.float32).reshape(1, 2),
        joint_specs=("joint1", "joint2"),
    )
    embodiment = _WildcardActionCfgEmbodiment()

    with pytest.raises(ValueError, match="Wildcard joint specs"):
        embodiment.translate_joint_command_to_env_action(action)


def test_dualarm_piper_joint_command_complete_action_returns_terms():
    action = UnifiedJointCommand.from_specs(
        values=torch.arange(16, dtype=torch.float32).reshape(1, 16),
        joint_specs=(
            "left_joint[1-6]",
            "left_joint[7-8]",
            "right_joint[1-6]",
            "right_joint[7-8]",
        ),
    )
    embodiment = DualArmPiperEmbodiment(enable_cameras=False)

    translated = embodiment.translate_joint_command_to_env_action(action)

    assert {key: value.tolist() for key, value in translated.items()} == {
        "left_robot_joint_position": [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
        "left_robot_gripper_control": [[6.0, 7.0]],
        "right_robot_joint_position": [[8.0, 9.0, 10.0, 11.0, 12.0, 13.0]],
        "right_robot_gripper_control": [[14.0, 15.0]],
    }


def test_dualarm_piper_joint_command_partial_action_returns_matching_terms():
    action = UnifiedJointCommand.from_specs(
        values=torch.arange(6, dtype=torch.float32).reshape(1, 6),
        joint_specs=("left_joint[1-6]",),
    )
    embodiment = DualArmPiperEmbodiment(enable_cameras=False)

    translated = embodiment.translate_joint_command_to_env_action(action)

    assert {key: value.tolist() for key, value in translated.items()} == {
        "left_robot_joint_position": [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
    }


def test_franka_panda_gripper_action_cfg_uses_explicit_joint_names():
    embodiment = FrankaPandaEmbodiment(enable_cameras=False)

    action_cfg = embodiment.get_action_cfg()

    assert action_cfg.terms["robot_gripper_control"].asset_cfg.joint_names == [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]


def test_panda_droid_gripper_action_cfg_uses_finger_joint():
    embodiment = PandaDroidEmbodiment(enable_cameras=False)

    action_cfg = embodiment.get_action_cfg()

    assert action_cfg.terms["robot_gripper_control"].asset_cfg.joint_names == [
        "finger_joint",
    ]
