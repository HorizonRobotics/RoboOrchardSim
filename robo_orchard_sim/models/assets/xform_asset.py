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
#

"""Submodule for xform asset."""

from __future__ import annotations
import weakref
from typing import Sequence

import torch
from isaaclab.assets import AssetBase
from isaaclab.utils.buffers import TimestampedBuffer
from isaacsim.core.prims import XFormPrim as XFormPrimView

from robo_orchard_sim.cfg_wrappers.assets_cfg import AssetBaseCfg


class XFormPrimAsset(AssetBase):
    """XFormPrimAsset is a class for handling XForm prim assets.

    This class is a Isaac Lab style asset class of `XFromPrim` in Isaac Sim. It
    provides high level functions to deal with an Xform prim (only one Xform
    prim) and its attributes/properties.

    The XFormPrimAsset can be applied by geometry transformation, such as
    translation, rotation, and scaling.

    """

    def __init__(self, cfg: AssetBaseCfg):
        super().__init__(cfg)

    def _initialize_impl(self):
        self._xform_view = XFormPrimView(
            self.cfg.prim_path, reset_xform_properties=False
        )
        self._data = XFormPrimData(self._xform_view)

    @property
    def num_instances(self) -> int:
        return self._xform_view.count

    @property
    def data(self) -> XFormPrimData:
        """The data of the asset."""
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets all internal buffers of selected environments.

        Args:
            env_ids: The indices of the object to reset.
                Defaults to None (all instances).
        """
        # resolve all indices
        if env_ids is None:
            env_ids = slice(None)  # type: ignore
        self.data.root_state_w[env_ids] = 0.0
        return

    def write_data_to_sim(self):
        """Writes data to the simulator.

        Do nothing because XformPrimAsset is not interactive.
        """

        return

    def write_root_state_to_sim(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set the root position and orientation state into the simulation.

        The position and orientation is in the world frame.

        Note:
            State changing is usually called in the reset or initialization
            of the environment, not in the simulation step. This is because
            chaning state directly will break the physics simulation.

        Args:
            positions: Root position in world frame. Shape is
                (len(env_ids), 3). Defaults to None, which means left
                unchanged.
            orientations: Root orientation in world frame. Shape is
                (len(env_ids), 4). Orientation is in quaternion format
                (w, x, y, z). Defaults to None, which means left unchanged.
            env_ids: Environment indices. If None, then all indices are used.
        """
        if positions is None and orientations is None:
            return

        if env_ids is None:
            env_ids = None
        else:
            env_ids = list(env_ids)
        self._xform_view.set_world_poses(
            positions=positions, orientations=orientations, indices=env_ids
        )

    def update(self, dt: float):
        """Update the internal buffers.

        The time step ``dt`` is used to compute numerical derivatives of
        quantities such as joint accelerations which are not provided by
        the simulator.

        Args:
            dt: The amount of time passed from last ``update`` call.
        """
        self._data.update(dt)


class XFormPrimData:
    """Data class for XFormPrimAsset."""

    def __init__(self, xform_view: XFormPrimView):
        self._xform_view: XFormPrimView = weakref.proxy(xform_view)
        self._sim_timestamp = 0.0
        self._root_state_w = TimestampedBuffer()

    def update(self, dt: float):
        """Updates the data for prim object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        self._sim_timestamp += dt
        self._update_data()

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state in simulation world frame.

        Shape is (num_instances, 7) where the first 3 elements are the position
        and the last 4 elements are the quaternion.
        """

        self._update_data()
        return self._root_state_w.data

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame.

        This quantity is the position of the asset.
        Shape is (num_instances, 3).
        """
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame.

        This quantity is the orientation of the asset.
        Shape is (num_instances, 4).
        """
        return self.root_state_w[:, 3:7]

    def _update_data(self):
        if self._root_state_w.timestamp < self._sim_timestamp:
            pose, orien = self._xform_view.get_world_poses()
            assert isinstance(pose, torch.Tensor)
            assert isinstance(orien, torch.Tensor)
            self._root_state_w.data = torch.cat((pose, orien), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp
