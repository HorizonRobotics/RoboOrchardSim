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

"""Differential IK Controller."""

from __future__ import annotations

import torch
from robo_orchard_core.controllers.differential_ik import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from robo_orchard_core.utils import math as math_utils

from robo_orchard_sim.ext.envs.env_base import IsaacEnv
from robo_orchard_sim.ext.models.assets.articulation import Articulation
from robo_orchard_sim.utils.config import ClassType_co as ClassType


class IsaacDifferentialIKController(
    DifferentialIKController["IsaacDifferentialIKControllerCfg", IsaacEnv],
):
    """The differential IK controllers.

    This class is a subclass of
    :py:class:`~robo_orchard_core.controllers.differential_ik.DifferentialIKController`
    to implement the differential IK controller for Isaac environments.

    """

    def __init__(self, cfg: IsaacDifferentialIKControllerCfg, env: IsaacEnv):
        super().__init__(cfg, env)
        self._env = env

        """Set the joint and body indices for the controller."""
        self._asset: Articulation = self._env.scene.__getitem__(cfg.asset_name)
        # initialize the controller for position and pose commands
        # separately, because the underlying Isaac controller
        # does not support changing the command type.

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self._cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self._cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self._cfg.body_name}."
                f"Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's
        # number of bodies
        #
        # The jacobian shape:
        #   * Fixed articulation base: ``(num_env, num_bodies - 1, 6, num_dof)`` # noqa: E501
        #   * Non-fixed articulation base: ``(num_env, num_bodies, 6, num_dof + 6)`` # noqa: E501
        #       - The first 6 columns are for the base.
        #
        # for more detail, please refer to
        # :py:meth:`isaacsim.core.articulations.ArticulationView.get_jacobian_shape` # noqa: E501

        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

    @property
    def joint_ids(self) -> list[int] | slice:
        """Get the joint indices."""
        return self._joint_ids

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    def set_goal(
        self,
        target_pos: torch.Tensor | None,
        target_quat: torch.Tensor | None,
    ) -> None:
        """Set the target body goal for the controller.

        If the target position or quaternion is None, the controller will keep
        the current position or orientation, which is computed from the
        articulation's state.

        Note:
            The target position and quaternion should be in the root frame, NOT
            the world frame!

        Args:
            target_pos (torch.Tensor | None): The target position of body.
                It should be a tensor of shape (num_envs, 3). If None, the
                controller will keep the current position.
            target_quat (torch.Tensor | None): The target quaternion of body.
                It should be a tensor of shape (num_envs, 4). If None,
                the controller will keep the current orientation.
        """

        if target_pos is None and target_quat is None:
            raise ValueError(
                "Both target position and quaternion cannot be None."
            )

        # set current pose if target is None
        if target_pos is None or target_quat is None:
            pose_t, pose_r = self._pose_body_wrt_root()

        self._target_pos[:] = (
            target_pos.to(device=self._device)
            if target_pos is not None
            else pose_t
        )
        self._target_quat[:] = (
            target_quat.to(device=self._device)
            if target_quat is not None
            else pose_r
        )

    def _get_joint_positions(self) -> torch.Tensor:
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if joint_pos.device != self._device:
            joint_pos = joint_pos.to(device=self._device)

        return joint_pos

    def _get_jacobian(self) -> torch.Tensor:
        # physx return the jacobian in world frame.
        jacobian: torch.Tensor = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]

        # The jacobian is in world frame, we need to transform it to the root
        # frame.
        tf_q_root_w = self._asset.data.root_quat_w[:, :4]
        tf_mat_world_r = math_utils.quaternion_to_matrix(
            math_utils.quaternion_invert(tf_q_root_w)
        )
        jacobian[:, :3, :] = torch.bmm(tf_mat_world_r, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(tf_mat_world_r, jacobian[:, 3:, :])

        if jacobian.device != self._device:
            jacobian = jacobian.to(device=self._device)

        return jacobian

    def _pose_body_wrt_root(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the pose of the body relative to the root."""
        body_pose_w = self._asset.data.body_state_w[:, self._body_idx, :7]
        root_pose_w = self._asset.data.root_state_w[:, :7]

        return math_utils.frame_transform_subtract(
            t01=root_pose_w[:, :3],
            q01=root_pose_w[:, 3:],
            t02=body_pose_w[:, :3],
            q02=body_pose_w[:, 3:],
        )


class IsaacDifferentialIKControllerCfg(
    DifferentialIKControllerCfg[IsaacDifferentialIKController]
):
    """The configuration for IsaacDifferentialIKController."""

    class_type: ClassType[IsaacDifferentialIKController] = (
        IsaacDifferentialIKController
    )

    asset_name: str
    """The name of the scene entity that needs to be controlled."""

    joint_names: list[str]
    """List of joint names or regex expressions that the action will
    be mapped to."""

    body_name: str
    """Name of the body or frame for which IK is performed."""
