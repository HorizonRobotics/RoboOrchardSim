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

import robo_orchard_core.utils.math as math_utils
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.rigid_object import RigidObject
from robo_orchard_core.datatypes.geometry import BatchFrameTransform
from robo_orchard_core.datatypes.tf_graph import BatchFrameTransformGraph
from robo_orchard_core.envs.managers.observations import (
    ObservationTermBase,
    ObservationTermCfg,
)
from typing_extensions import Sequence, TypeVar

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.ext.models.assets.articulation import (
    Articulation as ElementArticulation,
)
from robo_orchard_sim.ext.models.assets.rigid_object import (
    RigidObject as ElementRigidObject,
)
from robo_orchard_sim.ext.models.assets.xform_asset import XFormPrimAsset
from robo_orchard_sim.ext.models.sensors.camera import Camera
from robo_orchard_sim.utils.config import ClassType_co

FrameTransformTermCfg_co = TypeVar(
    "FrameTransformTermCfg_co",
    bound="FrameTransformTermCfg",
    covariant=True,
)

WORLD_FRAME_ID = "world"
"""Fixed frame id for the env-local world frame.

Not user-configurable: downstream consumers (e.g. the sim mcap packer)
already normalize parent frames to this literal name, and letting each
caller pick its own name would silently fracture the TF tree.
"""


class FrameTransformTerm(
    ObservationTermBase[
        IsaacEnvType_co, "FrameTransformTermCfg", BatchFrameTransformGraph
    ]
):
    def __init__(self, cfg: FrameTransformTermCfg, env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._env = env

        if cfg.world_parent and cfg.asset_cfg is not None:
            raise ValueError(
                "world_parent=True means the parent is the env-local "
                "world frame; asset_cfg must be left None."
            )
        if not cfg.world_parent and cfg.asset_cfg is None:
            raise ValueError("asset_cfg is required unless world_parent=True.")

        self.cfg.child_asset_cfg.resolve(env.scene)
        self._child_asset = env.scene[cfg.child_asset_cfg.name]
        self._child_names = self._get_asset_names(
            self._child_asset, self.cfg.child_asset_cfg
        )

        if cfg.world_parent:
            self._parent_asset = None
            self._parent_names = [WORLD_FRAME_ID] * len(self._child_names)
        else:
            assert cfg.asset_cfg is not None  # validated above
            self.cfg.asset_cfg.resolve(env.scene)
            self._parent_asset = env.scene[cfg.asset_cfg.name]
            self._parent_names = self._get_asset_names(
                self._parent_asset, self.cfg.asset_cfg
            )

        self._validate_and_build_name_pairs()

    def __call__(self) -> BatchFrameTransformGraph:
        type2impl = {
            XFormPrimAsset: self.__xform_impl,
            RigidObject: self.__rigid_object_impl,
            ElementRigidObject: self.__rigid_object_impl,
            Articulation: self.__rigid_object_impl,
            ElementArticulation: self.__rigid_object_impl,
            Camera: self.__camera_impl,
            type(None): self.__world_impl,
        }

        # parent_data is:(Batch, target, 3/4)
        # child_data is: (Batch, target, 3/4)
        # and tf is (Batch, target, 3/4)

        # get parent_data
        parent_type = type(self._parent_asset)
        if parent_type in type2impl:
            parent_pos, parent_quat = type2impl[parent_type](
                self._parent_asset, self.cfg.asset_cfg
            )
        else:
            raise NotImplementedError(
                f"Parent asset type {parent_type} is not supported."
            )

        # get child_data
        child_type = type(self._child_asset)
        if child_type in type2impl:
            child_pos, child_quat = type2impl[child_type](
                self._child_asset, self.cfg.child_asset_cfg
            )
        else:
            raise NotImplementedError(
                f"Child asset type {child_type} is not supported."
            )

        # get tf data: child_2_parent_pos/quat shape [batch, N_targets, 3/4]
        # child cord in parent frame
        child_2_parent_pos, child_2_parent_quat = (
            math_utils.frame_transform_subtract(
                parent_pos, parent_quat, child_pos, child_quat
            )
        )

        # build BatchFrameTransform for each (parent, child) pair
        tf_list: list[BatchFrameTransform] = []
        for i, (parent_name, child_name) in enumerate(self._name_pairs):
            tf_list.append(
                BatchFrameTransform(
                    xyz=child_2_parent_pos[:, i, :],
                    quat=child_2_parent_quat[:, i, :],
                    parent_frame_id=parent_name,
                    child_frame_id=child_name,
                )
            )

        return BatchFrameTransformGraph(
            tf_list=tf_list,
            bidirectional=self.cfg.bidirectional,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the observation term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        """
        pass

    def _get_asset_names(self, asset, cfg: LabSceneEntityCfg) -> list[str]:
        asset_type = type(asset)

        if asset_type in [XFormPrimAsset, Camera]:
            return [cfg.name]
        else:
            body_names = cfg.body_names
            if not body_names:
                body_ids = cfg.body_ids
                if body_ids == slice(None):
                    body_ids = []
                if isinstance(body_ids, slice):
                    body_ids = range(*body_ids.indices(len(asset.body_names)))
                body_names = (
                    [asset.body_names[id] for id in body_ids]
                    if body_ids
                    else []
                )
            if not body_names:
                return [cfg.name]
            return [f"{cfg.name}/{name}" for name in body_names]

    def _validate_and_build_name_pairs(self):
        parent_count = len(self._parent_names)
        child_count = len(self._child_names)

        if parent_count != child_count:
            if parent_count == 1:
                self._name_pairs = [
                    (self._parent_names[0], child_name)
                    for child_name in self._child_names
                ]
            else:
                raise ValueError(
                    f"Incompatible frame count between parent and child. "
                    f"Parent frames: {parent_count} ({self._parent_names}), "
                    f"Child frames: {child_count} ({self._child_names}). "
                    f"Parent frame count must be either 1 or equal to child "
                    f"frame count ({child_count})."
                )
        else:
            self._name_pairs = list(
                zip(self._parent_names, self._child_names, strict=False)
            )

        # print(f"Frame transform pairs: {self._name_pairs}")

    def __xform_impl(self, asset: XFormPrimAsset, cfg: LabSceneEntityCfg):
        assert isinstance(asset, XFormPrimAsset)

        pos = asset.data.root_pos_w - self._env.scene.env_origins[:]
        quat = asset.data.root_quat_w
        return pos.unsqueeze(1), quat.unsqueeze(1)

    def __rigid_object_impl(
        self, asset: RigidObject | Articulation, cfg: LabSceneEntityCfg
    ):
        """Implementation for rigid object assets.

        This function handles:
            properties: [position, orientation, linear_velocity,
                angular_velocity]
            source: [root, body]
        """
        assert isinstance(asset, (RigidObject, Articulation))
        slice_idx = (slice(None), cfg.body_ids)

        env_origins = self._env.scene.env_origins[:].unsqueeze(1)  # [2, 1, 3]

        pos = asset.data.body_pos_w[slice_idx] - env_origins
        quat = asset.data.body_quat_w[slice_idx]
        return pos, quat

    def __camera_impl(self, asset: Camera, cfg: LabSceneEntityCfg):
        assert isinstance(asset, Camera)

        pos = asset.data.pos_w - self._env.scene.env_origins[:]
        quat = asset.data.quat_w_ros
        return pos.unsqueeze(1), quat.unsqueeze(1)

    def __world_impl(self, asset: None, cfg: Any):
        """Identity pose for the env-local world frame.

        Other impls already subtract ``env_origins``, so the world
        origin in that local frame is zero -- not ``env_origins``
        again.
        """
        device = self._env.device
        num_envs = self._env.num_envs
        pos = torch.zeros(num_envs, 1, 3, device=device)
        quat = torch.zeros(num_envs, 1, 4, device=device)
        quat[..., 0] = 1.0
        return pos, quat


class FrameTransformTermCfg(ObservationTermCfg[FrameTransformTerm, Any]):
    # class_type: type = FrameTransformer
    class_type: ClassType_co[FrameTransformTerm] = FrameTransformTerm

    asset_cfg: Any = None
    """Parent asset. Required unless ``world_parent`` is True, in which
    case the parent is the env-local world frame and this must be
    left None."""

    child_asset_cfg: LabSceneEntityCfg
    """Configuration for the child asset in the frame transform term."""

    world_parent: bool = False
    """Use the env-local world frame as parent instead of ``asset_cfg``.

    Mutually required with ``asset_cfg``: exactly one of the two must
    be set."""

    bidirectional: bool = True
    """Whether to add mirrored (inverse) edges in the graph. Default True."""
