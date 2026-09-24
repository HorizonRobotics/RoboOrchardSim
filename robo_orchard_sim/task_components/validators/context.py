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

"""Runtime context used when building task validators."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robo_orchard_sim.task_components.validators.base import GripperRange

if TYPE_CHECKING:
    import torch

    from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
        EmbodimentBase,
    )
    from robo_orchard_sim.task_components.role_registry import (
        RoleRegistry,
    )


@dataclass(frozen=True, slots=True)
class ValidatorRobotContext:
    """Runtime robot data consumed by validator construction."""

    robot_name: str
    ee_links: tuple[str, ...] = ()
    gripper_joints: tuple[GripperRange, ...] = ()
    gripper_body_groups: tuple[tuple[str, ...], ...] = ()
    # (EE link name, TCP translation in that link frame), copied once.
    tcp_offsets: tuple[tuple[str, tuple[float, float, float]], ...] = ()

    def tcp_positions_w(self, env: Any, env_idx: int = 0) -> "torch.Tensor":
        """Transform configured TCP origins using world link poses (wxyz)."""
        import torch

        if not self.tcp_offsets:
            raise ValueError("Reach requires configured TCP frames.")
        asset = env.scene[self.robot_name]
        positions = []
        for body_name, tcp_offset in self.tcp_offsets:
            ids, names = asset.find_bodies(body_name)
            exact = [
                i
                for i, name in zip(ids, names, strict=True)
                if name == body_name
            ]
            if len(exact) != 1:
                raise ValueError(
                    f"Expected one TCP link '{body_name}', found {names}."
                )
            index = exact[0]
            origin = asset.data.body_pos_w[env_idx, index]
            quat = asset.data.body_quat_w[env_idx, index]
            offset = origin.new_tensor(tcp_offset)
            cross = 2 * torch.linalg.cross(quat[1:], offset)
            rotated = (
                offset + quat[0] * cross + torch.linalg.cross(quat[1:], cross)
            )
            positions.append(origin + rotated)
        return torch.stack(positions)


class ValidatorContext:
    """Runtime information portal for validator construction.

    Mutable by design: role bindings change every episode (via the
    referenced RoleRegistry), and object states are snapshotted once the
    scene has settled.
    """

    def __init__(
        self,
        robot: ValidatorRobotContext | None,
        role_registry: "RoleRegistry",
    ) -> None:
        self.robot = robot
        self._role_registry = role_registry
        self._init_states: dict[str, "torch.Tensor"] = {}
        self._final_states: dict[str, "torch.Tensor"] = {}

    @property
    def role_registry(self) -> "RoleRegistry":
        """Role bindings for the current episode.

        The registry itself is mutable and rebound each episode; only the
        reference is fixed for the lifetime of this context.
        """
        return self._role_registry

    @classmethod
    def from_embodiment(
        cls,
        embodiment: "EmbodimentBase",
        role_registry: "RoleRegistry",
    ) -> "ValidatorContext":
        """Build validator runtime context from the resolved embodiment."""
        robot_info_cfgs = embodiment.get_robot_info_cfgs()
        if not robot_info_cfgs:
            return cls(robot=None, role_registry=role_registry)

        ee_links: list[str] = []
        tcp_offsets: list[tuple[str, tuple[float, float, float]]] = []
        gripper_joints: list[GripperRange] = []
        gripper_body_groups: list[tuple[str, ...]] = []
        seen_ee_links: set[str] = set()
        seen_gripper_joints: set[str] = set()
        seen_gripper_body_groups: set[tuple[str, ...]] = set()
        for robot_info in robot_info_cfgs.values():
            manipulator_profile = robot_info.manipulator_profile
            if manipulator_profile is None:
                continue
            ee_body_name = manipulator_profile.ee_body_name
            if ee_body_name and ee_body_name not in seen_ee_links:
                ee_links.append(ee_body_name)
                seen_ee_links.add(ee_body_name)

            transform = np.asarray(
                robot_info.t_standard_tcp_to_robot_ee, dtype=float
            )
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(
                    transform[:3, :3].T @ transform[:3, :3], np.eye(3)
                )
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
            ):
                raise ValueError(
                    f"Invalid TCP transform for EE link '{ee_body_name}'."
                )
            # Same inverse convention as the trajectory executors:
            # T_world_tcp = T_world_ee @ inverse(T_tcp_ee).
            offset = -transform[:3, :3].T @ transform[:3, 3]
            tcp_offsets.append((ee_body_name, tuple(float(v) for v in offset)))

            body_group = manipulator_profile.gripper_body_names
            if body_group and body_group not in seen_gripper_body_groups:
                gripper_body_groups.append(body_group)
                seen_gripper_body_groups.add(body_group)

            for joint_name, open_val, close_val in zip(
                manipulator_profile.gripper_joint_names,
                robot_info.gripper_open_val,
                robot_info.gripper_close_val,
                strict=True,
            ):
                if joint_name in seen_gripper_joints:
                    continue
                gripper_joints.append(
                    GripperRange(
                        name=joint_name,
                        open_val=open_val,
                        close_val=close_val,
                    )
                )
                seen_gripper_joints.add(joint_name)

        return cls(
            robot=ValidatorRobotContext(
                robot_name=embodiment.scene_name,
                ee_links=tuple(ee_links),
                tcp_offsets=tuple(tcp_offsets),
                gripper_joints=tuple(gripper_joints),
                gripper_body_groups=tuple(gripper_body_groups),
            ),
            role_registry=role_registry,
        )

    @staticmethod
    def _snapshot(env: Any, scene_names: list[str]) -> dict[str, Any]:
        """Read pose and velocity for each entity, left on its device."""
        import torch

        states = {}
        for name in scene_names:
            d = env.scene[name].data
            states[name] = torch.cat(
                [
                    d.root_pos_w,
                    d.root_quat_w,
                    d.root_lin_vel_w,
                    d.root_ang_vel_w,
                ],
                dim=-1,
            )
        return states

    def capture_init_states(self, env: Any, scene_names: list[str]) -> None:
        """Snapshot settled state for all operable objects.

        Call once after scene settle. Each entry is shape (num_envs, 13):
        pos(3) + quat(4) + lin_vel(3) + ang_vel(3), kept on the original
        device so checkers can compare without host-device copies.
        """
        self._init_states = self._snapshot(env, scene_names)

    def capture_final_states(self, env: Any, scene_names: list[str]) -> None:
        """Snapshot where everything ended up, in the same layout."""
        self._final_states = self._snapshot(env, scene_names)

    def init_state_of(
        self,
        scene_name: str,
        env_idx: int | None = 0,
    ) -> "torch.Tensor":
        """Return the settled state for one scene entity.

        Shape is ``(13,)`` for a single env, or ``(num_envs, 13)`` when
        ``env_idx`` is None.
        """
        return self._lookup(
            self._init_states, scene_name, env_idx, "capture_init_states"
        )

    def final_state_of(
        self,
        scene_name: str,
        env_idx: int | None = 0,
    ) -> "torch.Tensor":
        """Return the end-of-episode state for one scene entity."""
        return self._lookup(
            self._final_states, scene_name, env_idx, "capture_final_states"
        )

    @staticmethod
    def _lookup(
        states: dict[str, "torch.Tensor"],
        scene_name: str,
        env_idx: int | None,
        capture_call: str,
    ) -> "torch.Tensor":
        try:
            state = states[scene_name]
        except KeyError:
            raise RuntimeError(
                f"State for '{scene_name}' not captured. "
                f"Call {capture_call}() first."
            ) from None
        return state if env_idx is None else state[env_idx]

    def scene_name_of(self, role_id: str, env_idx: int = 0) -> str:
        """Return the scene_name currently bound to a role."""
        return self.role_registry.resolve_one(role_id, env_idx).scene_name
