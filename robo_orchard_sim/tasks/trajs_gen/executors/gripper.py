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

"""Gripper executor for fixed-arm open and close trajectories."""

from __future__ import annotations
from typing import Any, Literal

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.tasks.trajs_gen.base_executor import (
    BaseExecutor,
    BaseExecutorCfg,
    Trajectories,
)
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
)
from robo_orchard_sim.utils.config import ClassType_co


class GripperExecutor(BaseExecutor):
    """Plan gripper-only trajectories while keeping arm joints fixed."""

    cfg: "GripperExecutorCfg"

    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        """Resolve the manipulator and plan the gripper trajectory."""
        resolved = self.cfg.resolve_manipulator_info(
            env,
            context=context,
        )
        self.last_resolved_manipulator = resolved

        runtime_state = self.build_runtime_state(
            env=env,
            resolved=resolved,
        )
        current_gripper_val = self._current_gripper_value(env, resolved)
        target_gripper_val = self._resolve_gripper_value(
            resolved,
            state=self.cfg.gripper_state,
        )

        trajectories = self.gen_gripper_trajs(
            current_joint_positions=runtime_state.current_joint_positions,
            start_gripper_val=current_gripper_val,
            end_gripper_val=target_gripper_val,
            length=self.cfg.length,
            enable_interpolation=self.cfg.enable_interpolation,
        )
        return Trajectories(
            trajectories=trajectories,
            success=True,
            resolved_manipulator=resolved,
        )

    def _current_gripper_value(
        self,
        env: Any,
        resolved: ResolvedManipulatorProfile,
    ) -> list[float]:
        if len(resolved.gripper_joint_ids) == 0:
            raise ValueError(
                "Resolved manipulator must provide gripper joint ids."
            )
        current_gripper = env.scene[resolved.robot_name].data.joint_pos[
            0, resolved.gripper_joint_ids
        ]
        return current_gripper.detach().cpu().tolist()


class GripperExecutorCfg(BaseExecutorCfg):
    """Configuration for :class:`GripperExecutor`."""

    class_type: ClassType_co[GripperExecutor] = GripperExecutor
    action_type: str = "gripper"

    gripper_state: Literal["OPEN", "CLOSED"] = "OPEN"
    length: int = 10
    enable_interpolation: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.length <= 0:
            raise ValueError("GripperExecutorCfg.length must be positive.")
