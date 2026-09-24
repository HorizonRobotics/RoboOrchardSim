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

from __future__ import annotations
from typing import Any

import torch
from isaaclab.sensors.contact_sensor import (
    ContactSensor as _ContactSensor,
    ContactSensorCfg as _ContactSensorCfg,
)
from pydantic import Field

from robo_orchard_sim.ext.cfg_wrappers.sensor_cfg import SensorBaseCfg
from robo_orchard_sim.utils.config import (
    ClassType,
    isaac_configclass2pydantic,
)

__all__ = ["ContactSensor", "ContactSensorCfg"]


class ContactSensor(_ContactSensor):
    """An IsaacLab contact sensor that rejects incomplete contact buffers."""

    def _unpack_contact_buffer_data(
        self,
        contact_data: torch.Tensor,
        buffer_count: torch.Tensor,
        buffer_start_indices: torch.Tensor,
        avg: bool = True,
        default: float = float("nan"),
    ) -> torch.Tensor:
        if buffer_count.shape != buffer_start_indices.shape:
            raise RuntimeError(
                f"Mismatched contact metadata shapes: {self.cfg.prim_path}"
            )
        if buffer_count.numel():
            counts = buffer_count.to(dtype=torch.int64)
            starts = buffer_start_indices.to(dtype=torch.int64)
            active = counts > 0
            minimum_count, minimum_start, required_capacity = torch.stack(
                (
                    counts.min(),
                    torch.where(active, starts, 0).min(),
                    torch.where(active, starts + counts, 0).max(),
                )
            ).tolist()
            if minimum_count < 0 or minimum_start < 0:
                raise RuntimeError(
                    "Negative contact count or active start: "
                    f"{self.cfg.prim_path}"
                )
            if required_capacity > contact_data.shape[0]:
                raise RuntimeError(
                    f"Contact buffer overflow for '{self.cfg.prim_path}': "
                    f"capacity={contact_data.shape[0]}, "
                    f"required={required_capacity}. "
                    "Increase max_contact_data_count_per_prim. "
                    "Contact data was not indexed or truncated."
                )
        return super()._unpack_contact_buffer_data(
            contact_data,
            buffer_count,
            buffer_start_indices,
            avg=avg,
            default=default,
        )


def _default_visualizer_cfg() -> Any:
    return _ContactSensorCfg().visualizer_cfg


class ContactSensorCfg(
    SensorBaseCfg[ContactSensor],
    isaac_configclass2pydantic(_ContactSensorCfg),
):
    """The pydantic version of isaac lab ContactSensorCfg class.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sensors.contact_sensor.ContactSensorCfg`
    """

    class_type: ClassType[ContactSensor] = ContactSensor

    max_contact_data_count_per_prim: int = Field(default=1024, ge=1)

    visualizer_cfg: Any = Field(
        default_factory=_default_visualizer_cfg, exclude=True
    )
    """Marker cfg used only when ``debug_vis`` is enabled.

    Excluded from serialization: its ``markers`` hold raw spawner dataclasses
    carrying bare ``func`` callables, which have no pydantic representation.
    Dropping it loses no state needed to reproduce a run.
    """

    def prim_path_format(self, **kwargs) -> None:
        """Expand namespaces for both the sensor and its contact filters."""
        super().prim_path_format(**kwargs)
        self.filter_prim_paths_expr = [
            path.format(**kwargs) for path in self.filter_prim_paths_expr
        ]
