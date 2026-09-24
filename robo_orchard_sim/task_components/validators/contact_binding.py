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

"""Bind contact sensors to physical scene components."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from robo_orchard_sim.task_components.validators.physical_entity import (
    SceneEntityKey,
    scene_name_of,
)

GRIPPER_CONTACT_NAMESPACE = "sensors"
GRIPPER_CONTACT_SENSOR_PREFIX = "gripper_contact_"

ContactFilterKey = tuple[int, SceneEntityKey]


@dataclass(frozen=True, slots=True)
class GripperContactPair:
    """The two sensor/body pairs belonging to one manipulator gripper."""

    sensor_names: tuple[str, str]
    body_names: tuple[str, str]


def gripper_contact_sensors(context) -> tuple[GripperContactPair, ...]:
    """Return per-finger sensor names grouped by manipulator."""
    if context.robot is None:
        raise ValueError("Contact checks need robot metadata in the context.")
    groups = context.robot.gripper_body_groups
    if not groups:
        raise ValueError(
            f"Robot '{context.robot.robot_name}' declares no gripper bodies."
        )
    pairs = []
    for manipulator_idx, bodies in enumerate(groups):
        if len(bodies) != 2:
            raise ValueError(
                "Opposing contact needs exactly two gripper bodies per "
                f"manipulator, got {list(bodies)}."
            )
        pairs.append(
            GripperContactPair(
                sensor_names=(
                    f"{GRIPPER_CONTACT_NAMESPACE}/"
                    f"{GRIPPER_CONTACT_SENSOR_PREFIX}{manipulator_idx}_0",
                    f"{GRIPPER_CONTACT_NAMESPACE}/"
                    f"{GRIPPER_CONTACT_SENSOR_PREFIX}{manipulator_idx}_1",
                ),
                body_names=(bodies[0], bodies[1]),
            )
        )
    return tuple(pairs)


class ContactBinding:
    """Aim one contact sensor at physical components in the scene scope."""

    def __init__(self, sensor_name: str) -> None:
        if not sensor_name:
            raise ValueError("sensor_name must be non-empty.")
        self.sensor_name = sensor_name
        self._bound: dict[ContactFilterKey, str] = {}
        self._columns: dict[ContactFilterKey, int] = {}

    @property
    def bound_targets(self) -> Mapping[ContactFilterKey, str]:
        """Return a copy of the current physical-body filter mapping."""
        return dict(self._bound)

    def rebind(
        self,
        env,
        targets: Mapping[ContactFilterKey, str],
    ) -> None:
        """Point the sensor at every unique physical target body."""
        if not targets:
            raise ValueError(
                f"Contact binding for '{self.sensor_name}' needs targets."
            )
        if dict(targets) == self._bound:
            return

        ordered_keys = sorted(
            targets,
            key=lambda item: (
                item[0],
                scene_name_of(item[1]),
                str(item[1]),
            ),
        )
        ordered_paths = [targets[key] for key in ordered_keys]
        sensor = env.scene[self.sensor_name]
        sensor.cfg.filter_prim_paths_expr = ordered_paths
        # IsaacLab exposes no public runtime filter-refresh API.
        sensor._initialize_impl()

        self._bound = dict(targets)
        self._columns = self._resolve_columns(sensor, targets)

    def _resolve_columns(
        self,
        sensor,
        targets: Mapping[ContactFilterKey, str],
    ) -> dict[ContactFilterKey, int]:
        filter_paths = list(sensor.contact_physx_view.filter_paths[0])
        columns = {}
        for key, prim_path in targets.items():
            try:
                columns[key] = filter_paths.index(prim_path)
            except ValueError:
                raise ValueError(
                    f"Contact sensor '{self.sensor_name}' did not resolve "
                    f"'{prim_path}'. Resolved: {filter_paths}."
                ) from None
        return columns

    def finger_force(
        self,
        env,
        target: SceneEntityKey,
        *,
        env_idx: int = 0,
    ) -> torch.Tensor:
        """Return this finger's force against one physical target body."""
        key = (env_idx, target)
        if key not in self._columns:
            raise RuntimeError(
                f"Contact binding for '{self.sensor_name}' has no target "
                f"'{target}' in env {env_idx}; call rebind() first."
            )
        sensor = env.scene[self.sensor_name]
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            raise RuntimeError(
                f"Contact sensor '{self.sensor_name}' reports no filtered "
                "forces."
            )
        column = matrix[env_idx, :, self._columns[key]]
        if column.shape[0] != 1:
            raise ValueError(
                f"Contact sensor '{self.sensor_name}' reports "
                f"{column.shape[0]} bodies; expected exactly one."
            )
        return column[0]

    def finger_contact_position(
        self,
        env,
        target: SceneEntityKey,
        *,
        env_idx: int = 0,
    ) -> torch.Tensor:
        """Return the mean world contact point, or NaN for no contact."""
        key = (env_idx, target)
        if key not in self._columns:
            raise RuntimeError(
                f"Contact binding for '{self.sensor_name}' has no target "
                f"'{target}' in env {env_idx}; call rebind() first."
            )
        positions = env.scene[self.sensor_name].data.contact_pos_w
        if positions is None:
            raise RuntimeError(
                f"Contact sensor '{self.sensor_name}' reports no contact "
                "positions; enable track_contact_points."
            )
        column = positions[env_idx, :, self._columns[key]]
        if column.shape[0] != 1:
            raise ValueError(
                f"Contact sensor '{self.sensor_name}' reports "
                f"{column.shape[0]} bodies; expected exactly one."
            )
        return column[0]


__all__ = [
    "GRIPPER_CONTACT_NAMESPACE",
    "GRIPPER_CONTACT_SENSOR_PREFIX",
    "ContactBinding",
    "GripperContactPair",
    "gripper_contact_sensors",
]
