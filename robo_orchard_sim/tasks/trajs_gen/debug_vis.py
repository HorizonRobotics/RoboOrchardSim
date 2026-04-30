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

"""Debug visualization helpers for atomic action executors."""

from __future__ import annotations
from typing import Any

import torch

from robo_orchard_sim.models.assets.asset_cfg import NV_ISAAC_DIR


class AtomicActionDebugVisualizer:
    """Lazy Isaac marker manager for atomic action target poses."""

    def __init__(
        self,
        *,
        root_prim_path: str = "/Visuals/atomic_actions",
        marker_scale: tuple[float, float, float] = (0.1, 0.1, 0.1),
        marker_usd_path: str | None = None,
    ) -> None:
        self._root_prim_path = root_prim_path.rstrip("/")
        self._marker_scale = marker_scale
        self._marker_usd_path = marker_usd_path or self._default_usd_path()
        self._markers: dict[str, Any] = {}

    def visualize_pose(
        self,
        *,
        marker_name: str,
        pose_w: torch.Tensor,
    ) -> None:
        """Visualize batched poses in world frame."""
        marker = self._get_marker(marker_name)
        marker.set_visibility(True)
        marker.visualize(pose_w[:, :3], pose_w[:, 3:])

    def clear(self) -> None:
        """Hide and forget all debug markers managed by this visualizer."""
        for marker in self._markers.values():
            marker.set_visibility(False)
        self._markers.clear()

    def _get_marker(self, marker_name: str) -> Any:
        marker = self._markers.get(marker_name)
        if marker is not None:
            return marker

        import isaaclab.sim as sim_utils
        from isaaclab.markers import (
            VisualizationMarkers,
            VisualizationMarkersCfg,
        )

        marker_cfg = VisualizationMarkersCfg(
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=self._marker_usd_path,
                    scale=self._marker_scale,
                )
            }
        )
        marker = VisualizationMarkers(
            marker_cfg.replace(
                prim_path=f"{self._root_prim_path}/{marker_name}"
            )
        )
        self._markers[marker_name] = marker
        return marker

    def _default_usd_path(self) -> str:
        return f"{NV_ISAAC_DIR}/Props/UIElements/frame_prim.usd"
