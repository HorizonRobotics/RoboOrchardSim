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

"""Pick executor placeholder module."""

from __future__ import annotations
from typing import Any

import torch

from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.tasks.trajs_gen.base_executor import (
    BaseExecutor,
    BaseExecutorCfg,
    ObjectInfo,
    Trajectories,
)
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
)
from robo_orchard_sim.utils.config import ClassType_co


class PickExecutor(BaseExecutor):
    """Return fixed joint targets for one manipulator."""

    cfg: "PickExecutorCfg"

    def __init__(self, cfg: "PickExecutorCfg") -> None:
        super().__init__(cfg)
        self.last_resolved_manipulator: ResolvedManipulatorProfile | None = (
            None
        )

    def plan(
        self,
        env: Any,
        context: ManipulatorBindingContext,
    ) -> Trajectories:
        """Resolve robot info and emit a hard-coded arm joint target."""
        resolved = self.cfg.resolve_manipulator_info(
            env,
            context=context,
        )
        self._validate_resolved_manipulator(resolved)
        self.last_resolved_manipulator = resolved

        print(f"PickExecutor: manipulator info: {resolved.manipulator_name}")
        print(f"PickExecutor: manipulator info: {resolved.joint_ids}")

        num_envs: int = env.num_envs
        device: torch.device = env.device
        joint_target = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.04, 0.04],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00],
            ],
            dtype=torch.float32,
            device=device,
        )
        expected_joint_count = len(resolved.joint_ids) + len(
            resolved.gripper_joint_ids
        )
        if expected_joint_count != joint_target.shape[1]:
            raise ValueError(
                "Hard-coded pick output does not match the resolved "
                "arm and gripper joint count."
            )
        trajectories = [joint_target.clone() for _ in range(num_envs)]
        return Trajectories(
            trajectories=trajectories,
            success=True,
            resolved_manipulator=resolved,
        )

    def _validate_resolved_manipulator(
        self,
        resolved: ResolvedManipulatorProfile,
    ) -> None:
        if not resolved.robot_name:
            raise ValueError(
                "Resolved manipulator robot_name must not be empty."
            )
        if not resolved.joint_ids:
            raise ValueError(
                "Resolved manipulator joint_ids must not be empty."
            )
        if not resolved.ee_body_name:
            raise ValueError(
                "Resolved manipulator ee_body_name must not be empty."
            )


class PickExecutorCfg(BaseExecutorCfg):
    """Configuration for :class:`PickExecutor`."""

    class_type: ClassType_co[PickExecutor] = PickExecutor
    action_type: str = "pick"

    pick_object_info: ObjectInfo
