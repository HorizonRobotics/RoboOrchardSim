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

"""Minimal base interfaces for atomic action executors."""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import robo_orchard_core.utils.math as math_utils
import torch
from robo_orchard_core.utils.config import ClassConfig

from robo_orchard_sim.controllers.curobo_planner.mixin import (
    CannotFindTrajectoryError,
    JointStateTrajetory,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
    ManipulatorResolver,
)
from robo_orchard_sim.utils.config import ClassType_co

TrajectoryMode = Literal["AvoidObs", "Simple"]


@dataclass
class ObjectInfo:
    name: str
    mode: Literal["active", "passive"]
    action: str
    part: str


@dataclass
class DebugTargetPose:
    """Debug target pose reported by an executor for optional visualization."""

    name: str
    pose_w: torch.Tensor


@dataclass
class Trajectories:
    trajectories: list[torch.Tensor]
    success: bool
    resolved_manipulator: ResolvedManipulatorProfile
    debug_target_poses: tuple[DebugTargetPose, ...] = ()


@dataclass
class _ManipulatorRuntimeState:
    frame: str
    current_joint_positions: torch.Tensor
    robot_base_pose_w: torch.Tensor
    ee_pose_w: torch.Tensor


class BaseExecutor(metaclass=ABCMeta):
    """Base executor interface for one atomic action.

    Concrete executors implement :meth:`plan`.  Callers pass the
    manipulator binding context explicitly so each planning request declares
    the sequence-scoped resolver state it should use.
    """

    cfg: "BaseExecutorCfg"

    def __init__(self, cfg: "BaseExecutorCfg") -> None:
        self.cfg = cfg
        if self.cfg.trajs_mode not in ("AvoidObs", "Simple"):
            raise ValueError(
                f"Invalid trajs_mode value: '{self.cfg.trajs_mode}'. "
                "Must be one of 'AvoidObs' or 'Simple'."
            )
        self.last_resolved_manipulator: ResolvedManipulatorProfile | None = (
            None
        )

    @abstractmethod
    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        """Plan one atomic action.

        Args:
            env: Runtime environment used by the executor during planning.
            context: Sequence-scoped manipulator binding context.

        Returns:
            tuple[AtomicActionTrajectories, bool]:
                - Planned batched trajectories. Each element is one
                  ``torch.Tensor`` trajectory for one environment.
                - Whether the planning stage succeeded.
        """

        raise NotImplementedError

    def reset(self) -> None:
        """Reset executor state for a new action sequence."""
        self.last_resolved_manipulator = None

    def build_debug_target_pose(
        self,
        *,
        env: Any,
        runtime_state: _ManipulatorRuntimeState,
        pose_robot_base: torch.Tensor,
        name: str,
    ) -> DebugTargetPose:
        """Return a robot-base-frame target pose as a world debug target."""
        base_pos = runtime_state.robot_base_pose_w[:, :3]
        base_quat = runtime_state.robot_base_pose_w[:, 3:]
        target_pos = pose_robot_base[:, :3]
        target_quat = pose_robot_base[:, 3:]
        marker_pos, marker_quat = math_utils.frame_transform_combine(
            base_pos,
            base_quat,
            target_pos,
            target_quat,
        )
        if runtime_state.frame == "env":
            marker_pos = marker_pos + env.scene.env_origins[:]

        return DebugTargetPose(
            name=name,
            pose_w=torch.cat((marker_pos, marker_quat), dim=-1),
        )

    def gen_to_target_joint_position_trajs(
        self,
        planner: Any,
        start_joint_positions: torch.Tensor,
        target_joint_positions: torch.Tensor,
        gripper_val: list[float],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Generate trajectories that move joints to a target state."""
        start_joint_positions = self._ensure_tensor(
            start_joint_positions,
            "start_joint_positions",
        )
        target_joint_positions = self._ensure_tensor(
            target_joint_positions,
            "target_joint_positions",
            device=start_joint_positions.device,
        )

        try:
            if start_joint_positions.shape[0] == 1:
                traj = planner.plan_to_target_joint_positions(
                    start_joint_positions=start_joint_positions.contiguous(),
                    target_joint_positions=target_joint_positions.contiguous(),
                )
            else:
                traj = planner.plan_to_target_ee_pose(
                    start_joint_positions=start_joint_positions.contiguous(),
                    target_poses=planner.fk(
                        target_joint_positions
                    ).contiguous(),
                )
        except CannotFindTrajectoryError as e:
            print(
                f"Cannot find trajectory to target joint positions {target_joint_positions}: {e}"  # noqa: E501
            )
            traj = self._failed_joint_state_trajectory(
                target_joint_positions,
            )

        return self._convert_trajs(traj, gripper_val)

    def _resolve_gripper_value(
        self,
        resolved: ResolvedManipulatorProfile,
        state: Literal["OPEN", "CLOSED"],
    ) -> list[float]:
        value = (
            resolved.gripper_open_val
            if state == "OPEN"
            else resolved.gripper_close_val
        )
        if value is None:
            raise ValueError(f"Resolved manipulator missing {state} gripper.")
        if len(value) != len(resolved.gripper_joint_ids):
            raise ValueError(
                f"Resolved manipulator {state} gripper value count must "
                "match the resolved gripper joint count."
            )
        return value

    def gen_to_target_pose_trajs(
        self,
        planner: Any,
        start_joint_positions: torch.Tensor,
        target_pose: torch.Tensor,
        gripper_val: list[float],
        mode: TrajectoryMode = "Simple",
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Generate trajectories that move the end effector to a pose."""
        start_joint_positions = self._ensure_tensor(
            start_joint_positions,
            "start_joint_positions",
        )
        target_pose_xyzw = self._pose_wxyz_to_xyzw(
            self._ensure_tensor(
                target_pose,
                "target_pose",
                device=start_joint_positions.device,
            )
        )

        try:
            traj = planner.plan_to_target_ee_pose(
                start_joint_positions=start_joint_positions.contiguous(),
                target_poses=target_pose_xyzw.contiguous(),
                mode=mode,
            )
        except CannotFindTrajectoryError as e:
            print(
                f"Cannot find trajectory to target pose in [x,y,z,qw,qx,qy,qz]"
                f" {target_pose}: {e}"
            )
            traj = self._failed_joint_state_trajectory(
                start_joint_positions,
            )

        return self._convert_trajs(traj, gripper_val)

    def gen_gripper_trajs(
        self,
        current_joint_positions: torch.Tensor,
        start_gripper_val: list[float],
        end_gripper_val: list[float],
        length: int = 5,
        enable_interpolation: bool = True,
    ) -> list[torch.Tensor]:
        """Generate batched gripper-only trajectories over fixed arm joints."""
        current_joint_positions = self._ensure_tensor(
            current_joint_positions,
            "current_joint_positions",
        )
        device = current_joint_positions.device
        batch_size = current_joint_positions.shape[0]

        start_gripper_tensor = torch.tensor(
            start_gripper_val,
            dtype=current_joint_positions.dtype,
            device=device,
        )
        end_gripper_tensor = torch.tensor(
            end_gripper_val,
            dtype=current_joint_positions.dtype,
            device=device,
        )

        actions = []
        for batch_index in range(batch_size):
            arm_trajectory = (
                current_joint_positions[batch_index]
                .unsqueeze(0)
                .repeat(length, 1)
            )
            if enable_interpolation:
                gripper_trajectory = torch.stack(
                    [
                        torch.linspace(
                            start_gripper_tensor[joint_index],
                            end_gripper_tensor[joint_index],
                            steps=length,
                            device=device,
                        )
                        for joint_index in range(len(start_gripper_val))
                    ],
                    dim=1,
                )
            else:
                gripper_trajectory = end_gripper_tensor.unsqueeze(0).repeat(
                    length,
                    1,
                )
            actions.append(
                torch.cat((arm_trajectory, gripper_trajectory), dim=1)
            )
        return actions

    def get_last_trajs(self, trajs: list[torch.Tensor]) -> torch.Tensor:
        """Return the last control step from each batched trajectory."""
        return torch.stack([traj[-1] for traj in trajs], dim=0)

    def build_runtime_state(
        self,
        env: Any,
        resolved: ResolvedManipulatorProfile,
        frame: Literal["world", "env"] = "world",
    ) -> _ManipulatorRuntimeState:
        articulation = env.scene[resolved.robot_name]

        current_joint_positions = articulation.data.joint_pos[
            :, resolved.joint_ids
        ]

        if resolved.base_body_id is None:
            robot_base_pos_w = articulation.data.root_pos_w
            robot_base_quat_w = articulation.data.root_quat_w
        else:
            robot_base_pos_w = articulation.data.body_link_pos_w[
                :, resolved.base_body_id
            ]
            robot_base_quat_w = articulation.data.body_link_quat_w[
                :, resolved.base_body_id
            ]

        ee_pos_w = articulation.data.body_link_pos_w[:, resolved.ee_body_id]
        ee_quat_w = articulation.data.body_link_quat_w[:, resolved.ee_body_id]

        if frame == "env":
            # Convert to environment frame by subtracting the origin
            robot_base_pos_w = robot_base_pos_w - env.scene.env_origins[:]
            ee_pos_w = ee_pos_w - env.scene.env_origins[:]

        robot_base_pose = torch.cat(
            [robot_base_pos_w, robot_base_quat_w], dim=-1
        )
        ee_pose = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
        return _ManipulatorRuntimeState(
            frame=frame,
            current_joint_positions=current_joint_positions,
            robot_base_pose_w=robot_base_pose,
            ee_pose_w=ee_pose,
        )

    def _convert_trajs(
        self,
        trajs: JointStateTrajetory,
        gripper_val: list[float],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Append gripper commands to planner trajectories."""
        position_list = self._positions_to_list(trajs.positions)
        actions = []
        for traj, success in zip(
            position_list,
            trajs.indices,
            strict=False,
        ):
            joint_action = (
                traj if bool(success.item()) else traj[0].unsqueeze(0)
            )
            gripper_control = torch.tensor(
                gripper_val,
                dtype=joint_action.dtype,
                device=joint_action.device,
            ).repeat(joint_action.shape[0], 1)
            actions.append(torch.cat((joint_action, gripper_control), dim=1))
        return actions, trajs.indices

    def _merge_trajs(self, *lists: list[torch.Tensor]) -> list[torch.Tensor]:
        """Merge batched trajectories along the time dimension."""
        if not lists:
            return []
        batch_size = len(lists[0])
        for traj_list in lists:
            if len(traj_list) != batch_size:
                raise ValueError(
                    "All trajectory lists must have the same batch size."
                )
        return [
            torch.cat(
                [traj_list[batch_index] for traj_list in lists],
                dim=0,
            )
            for batch_index in range(batch_size)
        ]

    def _failed_joint_state_trajectory(
        self,
        fallback_joint_positions: torch.Tensor,
    ) -> JointStateTrajetory:
        positions = fallback_joint_positions.unsqueeze(1)
        return JointStateTrajetory(
            positions=positions,
            indices=torch.zeros(
                positions.shape[0],
                dtype=torch.bool,
                device=positions.device,
            ),
            velocities=torch.zeros_like(positions),
        )

    def _positions_to_list(
        self,
        positions: torch.Tensor | list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if isinstance(positions, torch.Tensor):
            return [positions[index] for index in range(positions.size(0))]
        return positions

    def _ensure_tensor(
        self,
        value: Any,
        field_name: str,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.tensor(value, device=device)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
        return tensor

    def _pose_wxyz_to_xyzw(self, target_pose: torch.Tensor) -> torch.Tensor:
        if target_pose.ndim != 2 or target_pose.shape[1] != 7:
            raise ValueError(
                "target_pose must have shape [BATCH, 7] in "
                "[x, y, z, qw, qx, qy, qz] order."
            )
        return target_pose[:, [0, 1, 2, 4, 5, 6, 3]]

    def _resolve_trajs_mode(self) -> TrajectoryMode:
        if self.cfg.trajs_mode == "AvoidObs":
            print(
                f"[{type(self).__name__}] AvoidObs is not supported yet; "
                "set back to Simple trajectory planning."
            )
        return "Simple"


class BaseExecutorCfg(ClassConfig[BaseExecutor]):
    """Configuration for :class:`BaseExecutor`."""

    class_type: ClassType_co[BaseExecutor] = BaseExecutor
    robot_info: ManipulatorResolver
    priority: int = 0
    action_type: str = ""
    trajs_mode: TrajectoryMode = "Simple"

    def __call__(self, **kwargs: Any) -> BaseExecutor:
        return self.class_type(self, **kwargs)

    def resolve_manipulator_info(
        self,
        env: Any,
        context: ManipulatorBindingContext | None = None,
    ) -> ResolvedManipulatorProfile:
        """Resolve the configured manipulator into runtime ids."""
        if self.robot_info is None:
            raise ValueError("BaseExecutorCfg.robot_info must not be None.")
        return self.robot_info.resolve(env=env, context=context)
