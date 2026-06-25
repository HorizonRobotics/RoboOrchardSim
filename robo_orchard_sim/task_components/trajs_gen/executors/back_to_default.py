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

"""Back-to-default executor for returning to the default joint pose."""

from __future__ import annotations
from typing import Any, Literal

import torch

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.task_components.trajs_gen.base_executor import (
    BaseExecutor,
    BaseExecutorCfg,
    Trajectories,
    _ManipulatorRuntimeState,
)
from robo_orchard_sim.task_components.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
)
from robo_orchard_sim.utils.config import ClassType_co


class BackToDefaultExecutor(BaseExecutor):
    """Plan a trajectory from the current arm state to default joints."""

    cfg: "BackToDefaultExecutorCfg"

    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        """Resolve robot info and plan a move back to default pose."""
        resolved = self.cfg.resolve_manipulator_info(env, context=context)
        self.last_resolved_manipulator = resolved

        runtime_state: _ManipulatorRuntimeState = self.build_runtime_state(
            env=env,
            resolved=resolved,
        )
        target_joint_positions = self._resolve_default_joint_positions(
            env=env,
            resolved=resolved,
        )
        target_pose = resolved.planner.fk(
            target_joint_positions,
            w_first=True,
        )
        debug_target_pose = self.build_debug_target_pose(
            env=env,
            runtime_state=runtime_state,
            pose_robot_base=target_pose,
            name="back_to_default_target",
        )
        gripper_val = self._resolve_gripper_value(
            resolved,
            state=self.cfg.gripper_state,
        )

        # trajs_mode = self._resolve_trajs_mode()

        trajectories, success_flags = self.gen_to_target_joint_position_trajs(
            planner=resolved.planner,
            start_joint_positions=runtime_state.current_joint_positions,
            target_joint_positions=target_joint_positions,
            gripper_val=gripper_val,
        )
        return Trajectories(
            trajectories=trajectories,
            success=bool(torch.all(success_flags).item()),
            resolved_manipulator=resolved,
            debug_target_poses=(debug_target_pose,),
        )

    def _resolve_default_joint_positions(
        self,
        env: Any,
        resolved: ResolvedManipulatorProfile,
    ) -> torch.Tensor:
        default_joint_pos = env.scene[
            resolved.robot_name
        ].data.default_joint_pos
        return default_joint_pos[:, resolved.joint_ids]


class BackToDefaultExecutorCfg(BaseExecutorCfg):
    """Configuration for :class:`BackToDefaultExecutor`."""

    class_type: ClassType_co[BackToDefaultExecutor] = BackToDefaultExecutor
    action_type: str = "back_to_default"

    gripper_state: Literal["OPEN", "CLOSED"] = "OPEN"
