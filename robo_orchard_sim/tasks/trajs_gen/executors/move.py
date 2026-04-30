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

"""Move executor for direct end-effector target poses."""

from __future__ import annotations
from typing import Any, Literal

import torch

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.tasks.trajs_gen.base_executor import (
    BaseExecutor,
    BaseExecutorCfg,
    Trajectories,
    _ManipulatorRuntimeState,
)
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
)
from robo_orchard_sim.tasks.trajs_gen.pose_generator import (
    MotionPose,
    PoseGenerationContext,
    PoseGeneratorCfg,
)
from robo_orchard_sim.utils.config import ClassType_co


class MoveExecutor(BaseExecutor):
    """Plan arm trajectories to a configured target end-effector pose."""

    cfg: "MoveExecutorCfg"

    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        """Resolve robot info and plan to the configured target pose."""
        resolved = self.cfg.resolve_manipulator_info(
            env,
            context=context,
        )
        self.last_resolved_manipulator = resolved

        robot_state: _ManipulatorRuntimeState = self.build_runtime_state(
            env=env,
            resolved=resolved,
        )
        motion_target: MotionPose = self._generate_motion_target(
            runtime_state=robot_state,
        )
        debug_target_poses = ()
        if motion_target.type == "pose":
            debug_target_poses = (
                self.build_debug_target_pose(
                    env=env,
                    runtime_state=robot_state,
                    pose_robot_base=motion_target.data,
                    name="move_target",
                ),
            )
        elif motion_target.type == "joint":
            target_pose = resolved.planner.fk(motion_target.data)
            debug_target_poses = (
                self.build_debug_target_pose(
                    env=env,
                    runtime_state=robot_state,
                    pose_robot_base=target_pose,
                    name="move_target",
                ),
            )
        gripper_val = self._resolve_gripper_value(
            resolved,
            state=self.cfg.gripper_state,
        )

        trajectories, success_flags = self._plan_motion_target(
            resolved=resolved,
            current_joint_positions=robot_state.current_joint_positions,
            motion_target=motion_target,
            gripper_val=gripper_val,
        )
        return Trajectories(
            trajectories=trajectories,
            success=bool(torch.all(success_flags).item()),
            resolved_manipulator=resolved,
            debug_target_poses=debug_target_poses,
        )

    def _generate_motion_target(
        self,
        runtime_state: _ManipulatorRuntimeState,
    ) -> MotionPose:
        generator = self.cfg.target()
        context = PoseGenerationContext(
            robot_base_pose_w=runtime_state.robot_base_pose_w,
            ee_pose_w=runtime_state.ee_pose_w,
            current_joint_pos=runtime_state.current_joint_positions,
            executor=self,
        )
        return generator.generate(context)

    def _plan_motion_target(
        self,
        resolved: ResolvedManipulatorProfile,
        current_joint_positions: torch.Tensor,
        motion_target: MotionPose,
        gripper_val: list[float],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        trajs_mode = self._resolve_trajs_mode()
        if motion_target.type == "pose":
            return self.gen_to_target_pose_trajs(
                planner=resolved.planner,
                start_joint_positions=current_joint_positions,
                target_pose=motion_target.data,
                gripper_val=gripper_val,
                mode=trajs_mode,
            )
        if motion_target.type == "joint":
            return self.gen_to_target_joint_position_trajs(
                planner=resolved.planner,
                start_joint_positions=current_joint_positions,
                target_joint_positions=motion_target.data,
                gripper_val=gripper_val,
            )
        raise ValueError(
            "MoveExecutor target generator must return MotionPose "
            "with type 'pose' or 'joint'."
        )


class MoveExecutorCfg(BaseExecutorCfg):
    """Configuration for :class:`MoveExecutor`."""

    class_type: ClassType_co[MoveExecutor] = MoveExecutor
    action_type: str = "move"

    target: PoseGeneratorCfg
    gripper_state: Literal["OPEN", "CLOSED"] = "OPEN"
