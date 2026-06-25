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

"""Submodule for articulation asset."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation.articulation import (
    Articulation as _Articulation,
)
from isaacsim.core.prims import Articulation as ArticulationView
from isaacsim.core.utils.types import JointsState
from pxr import UsdPhysics

from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import (
    ArticulationCfg as _ArticulationCfg,
    SpawnerCfgType_co,
)
from robo_orchard_sim.utils.config import ClassType_co


class Articulation(_Articulation):
    """Articulation class extended from isaac lab's `Articulation`.

    This class extends the original
    :py:class:`~isaaclab.assets.articulation.articulation.Articulation`
    class to provide additional functionalities:

    - root_articulation_view to get ArticulationView from isaac sim.

    """

    @property
    def root_articulation_view(self) -> ArticulationView:
        """Get the root articulation view."""
        assert self._root_articulation_view is not None
        return self._root_articulation_view

    def _get_root_prim_path(self) -> str:
        """Get the root prim path of the articulation.

        This method is copied from the original implementation of
        :py:class:`isaaclab.assets.articulation.articulation.Articulation`.
        """
        # obtain the first prim in the regex expression
        # (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(
                f"Failed to find prim for expression: '{self.cfg.prim_path}'."
            )
        template_prim_path = template_prim.GetPath().pathString

        # find articulation root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),  # type: ignore # noqa: E501
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find an articulation when resolving '{self.cfg.prim_path}'."  # noqa: E501
                " Please ensure that the prim has 'USD ArticulationRootAPI' applied."  # noqa: E501
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single articulation when resolving '{self.cfg.prim_path}'."  # noqa: E501
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one articulation in the prim path tree."  # noqa: E501
            )

        # resolve articulation root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = (
            self.cfg.prim_path + root_prim_path[len(template_prim_path) :]  # noqa: E203
        )
        return root_prim_path_expr

    def _initialize_impl(self):
        """Initialize the articulation.

        This method overrides the original implementation to initialize the
        root articulation view.
        """

        super()._initialize_impl()
        self._initialize_root_articulate_view()

    def _initialize_root_articulate_view(self):
        """Initialize the root articulation view."""
        assert self._physics_sim_view is not None, (
            "The physics sim view must be set."
        )
        assert self._root_physx_view is not None
        assert self._root_physx_view.is_homogeneous

        root_articulation_view = ArticulationView(
            prim_paths_expr=self._get_root_prim_path(),
            reset_xform_properties=False,
        )
        self._root_articulation_view = root_articulation_view

        # Copied from isaac sim implementation.
        root_articulation_view._physics_sim_view = self._physics_sim_view
        root_articulation_view._physics_view = self._root_physx_view

        if not root_articulation_view._is_initialized:
            root_articulation_view._metadata = (
                root_articulation_view._physics_view.shared_metatype
            )
            root_articulation_view._num_dof = (
                root_articulation_view._physics_view.max_dofs
            )
            root_articulation_view._num_bodies = (
                root_articulation_view._physics_view.max_links
            )
            root_articulation_view._num_shapes = (
                root_articulation_view._physics_view.max_shapes
            )
            root_articulation_view._num_fixed_tendons = (
                root_articulation_view._physics_view.max_fixed_tendons
            )
            root_articulation_view._body_names = (
                root_articulation_view._metadata.link_names
            )
            root_articulation_view._body_indices = dict(
                zip(
                    root_articulation_view._body_names,
                    range(len(root_articulation_view._body_names)),
                    strict=False,
                )
            )
            root_articulation_view._dof_names = (
                root_articulation_view._metadata.dof_names
            )
            root_articulation_view._dof_indices = (
                root_articulation_view._metadata.dof_indices
            )
            root_articulation_view._dof_types = (
                root_articulation_view._metadata.dof_types
            )
            root_articulation_view._dof_paths = (
                root_articulation_view._physics_view.dof_paths
            )
            root_articulation_view._joint_indices = (
                root_articulation_view._metadata.joint_indices
            )
            root_articulation_view._joint_names = (
                root_articulation_view._metadata.joint_names
            )
            root_articulation_view._joint_types = (
                root_articulation_view._metadata.joint_types
            )
            root_articulation_view._num_joints = (
                root_articulation_view._metadata.joint_count
            )
            root_articulation_view._link_indices = (
                root_articulation_view._metadata.link_indices
            )
            root_articulation_view._prim_paths = (
                root_articulation_view._physics_view.prim_paths
            )
            root_articulation_view._is_initialized = True
            (
                root_articulation_view._default_kps,
                root_articulation_view._default_kds,  # type: ignore
            ) = root_articulation_view.get_gains(clone=True)
            default_actions = root_articulation_view.get_applied_actions(
                clone=True
            )

            # TODO: implement effort part
            if (
                root_articulation_view._default_state.positions is None
                or root_articulation_view._default_state.orientations is None
            ):
                default_positions, default_orientations = (
                    root_articulation_view.get_world_poses()
                )
                if root_articulation_view._default_state.positions is None:
                    root_articulation_view._default_state.positions = (  # type: ignore
                        default_positions.data
                        if self._backend == "warp"
                        else default_positions
                    )
                if root_articulation_view._default_state.orientations is None:
                    root_articulation_view._default_state.orientations = (  # type: ignore
                        default_orientations.data
                        if self._backend == "warp"
                        else default_orientations
                    )

            if root_articulation_view._default_joints_state is None:
                root_articulation_view._default_joints_state = JointsState(
                    positions=None,  # type: ignore
                    velocities=None,  # type: ignore
                    efforts=None,  # type: ignore
                )
            if root_articulation_view._default_joints_state.positions is None:
                root_articulation_view._default_joints_state.positions = (  # type: ignore # noqa: E501
                    default_actions.joint_positions
                )
            if root_articulation_view._default_joints_state.velocities is None:
                root_articulation_view._default_joints_state.velocities = (  # type: ignore # noqa: E501
                    default_actions.joint_velocities
                )
            if root_articulation_view._default_joints_state.efforts is None:
                root_articulation_view._default_joints_state.efforts = (
                    root_articulation_view._backend_utils.create_zeros_tensor(  # type: ignore # noqa: E501
                        shape=[
                            root_articulation_view.count,
                            root_articulation_view.num_dof,
                        ],
                        dtype="float32",
                        device=self._device,
                    )
                )

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements.

        Overridden to invalidate the root articulation view.
        """
        super()._invalidate_initialize_callback(event)
        self._root_articulation_view = None


class ArticulationCfg(_ArticulationCfg[SpawnerCfgType_co, Articulation]):
    class_type: ClassType_co[Articulation] = Articulation
