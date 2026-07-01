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

"""Observation term for observing properties of assets in the scene."""

from __future__ import annotations

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.asset_base import AssetBase
from isaaclab.assets.rigid_object import RigidObject
from robo_orchard_core.envs.managers.observations import (
    ObservationTermBase,
    ObservationTermCfg,
)
from typing_extensions import Generic, Literal, Sequence

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg as LabSceneEntityCfg,
)
from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.ext.models.assets.rigid_object import (
    RigidObject as ElementRigidObject,
)
from robo_orchard_sim.ext.models.assets.xform_asset import XFormPrimAsset
from robo_orchard_sim.utils.config import ClassType_co


class AssetObservationTerm(
    ObservationTermBase[
        IsaacEnvType_co, "AssetObservationTermCfg", torch.Tensor
    ],
    Generic[IsaacEnvType_co],
):
    """Observation term for observing properties of assets in the scene.

    See the :class:`AssetObservationTermCfg` class for more details about
    the configuration.

    """

    def __init__(self, cfg: AssetObservationTermCfg, env: IsaacEnvType_co):
        super().__init__(cfg, env)

        self.cfg.asset_cfg.resolve(env.scene)
        self._asset: AssetBase = env.scene[self.cfg.asset_cfg.name]

    def __call__(self) -> torch.Tensor:
        """Get the observation from asset."""

        if self.cfg.ref_frame != "world":
            raise ValueError(
                "Only support ref_frame = world, but get {}".format(
                    self.cfg.ref_frame
                )
            )

        type2impl = {
            XFormPrimAsset: self.__xform_impl,
            RigidObject: self.__rigid_object_impl,
            ElementRigidObject: self.__rigid_object_impl,
            Articulation: self.__articulation_impl,
        }

        asset_type = type(self._asset)
        if asset_type in type2impl:
            return type2impl[asset_type](self._asset)
        else:
            raise NotImplementedError(
                f"Asset type {asset_type} is not supported."
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the observation term."""
        # Do nothing for now
        pass

    def __xform_impl(self, asset: XFormPrimAsset):
        assert isinstance(self._asset, XFormPrimAsset)

        if self.cfg.property_name == "position":
            return asset.data.root_pos_w
        elif self.cfg.property_name == "orientation":
            return asset.data.root_quat_w
        elif self.cfg.property_name == "state":
            return asset.data.root_state_w
        else:
            raise NotImplementedError(
                f"Property name {self.cfg.property_name} is not supported "
                f"for {type(asset).__name__} class."
            )

    def __rigid_object_impl(self, asset: RigidObject | Articulation):
        """Implementation for rigid object assets.

        This function handles:
            properties: [position, orientation, linear_velocity,
                angular_velocity]
            source: [root, body]
        """
        assert isinstance(asset, (RigidObject, Articulation))
        not_implemented_error = NotImplementedError(
            f"Property name {self.cfg.property_name} of {self.cfg.property_source} "  # noqa
            f"is not supported for {type(asset).__name__} class."
        )

        def parse_root_prop() -> torch.Tensor:
            if self.cfg.property_name == "pose":
                return asset.data.root_link_state_w[:, :7]
            elif self.cfg.property_name == "state":
                return asset.data.root_link_state_w
            elif self.cfg.property_name == "position":
                return asset.data.root_pos_w
            elif self.cfg.property_name == "orientation":
                return asset.data.root_quat_w
            elif self.cfg.property_name == "linear_velocity":
                return asset.data.root_lin_vel_w
            elif self.cfg.property_name == "angular_velocity":
                return asset.data.root_ang_vel_w
            else:
                raise not_implemented_error

        def parse_body_prop() -> torch.Tensor:
            slice_idx = (slice(None), self.cfg.asset_cfg.body_ids)
            if self.cfg.property_name == "position":
                return asset.data.body_pos_w[slice_idx]
            elif self.cfg.property_name == "orientation":
                return asset.data.body_quat_w[slice_idx]
            elif self.cfg.property_name == "linear_velocity":
                return asset.data.body_lin_vel_w[slice_idx]
            elif self.cfg.property_name == "angular_velocity":
                return asset.data.body_ang_vel_w[slice_idx]
            elif self.cfg.property_name == "linear_acc":
                return asset.data.body_lin_acc_w[slice_idx]
            elif self.cfg.property_name == "angular_acc":
                return asset.data.body_ang_acc_w[slice_idx]
            else:
                raise not_implemented_error

        if self.cfg.property_source == "root":
            return parse_root_prop()
        elif self.cfg.property_source == "body":
            return parse_body_prop()
        else:
            raise not_implemented_error

    def __articulation_impl(self, asset: Articulation):
        """Implementation for articulation assets."""
        assert isinstance(asset, Articulation)
        not_implemented_error = NotImplementedError(
            f"Property name {self.cfg.property_name} of {self.cfg.property_source} "  # noqa
            f"is not supported for {type(asset).__name__} class."
        )

        def parse_joint_prop() -> torch.Tensor:
            slice_idx = (slice(None), self.cfg.asset_cfg.joint_ids)
            if self.cfg.property_name == "position":
                return asset.data.joint_pos[slice_idx]
            elif self.cfg.property_name == "position_relative":
                return (
                    asset.data.joint_pos[slice_idx]
                    - asset.data.default_joint_pos[slice_idx]
                )
            elif self.cfg.property_name == "linear_velocity":
                return asset.data.joint_vel[slice_idx]
            elif self.cfg.property_name == "linear_velocity_relative":
                return (
                    asset.data.joint_vel[slice_idx]
                    - asset.data.default_joint_vel[slice_idx]
                )
            elif self.cfg.property_name == "angular_acc":
                return asset.data.joint_acc[slice_idx]
            elif self.cfg.property_name == "effort":
                return torch.cat(
                    [
                        actuator.computed_effort.cpu()
                        for actuator in asset.actuators.values()
                    ],
                    dim=1,
                )[slice_idx]
            else:
                raise not_implemented_error

        if self.cfg.property_source == "joint":
            return parse_joint_prop()
        else:
            return self.__rigid_object_impl(asset)


class AssetObservationTermCfg(
    ObservationTermCfg[
        AssetObservationTerm[IsaacEnvType_co], LabSceneEntityCfg
    ],
    Generic[IsaacEnvType_co],
):
    """Configuration for the asset observation term."""

    class_type: ClassType_co[AssetObservationTerm[IsaacEnvType_co]] = (
        AssetObservationTerm[IsaacEnvType_co]
    )

    asset_cfg: LabSceneEntityCfg
    """The configuration of the asset."""

    property_source: Literal["root", "body", "joint"]
    """The source of the property.

    Note:
        - For rigid objects, the source can be either `root` or `body`.
        - For articulations, the source can be `joint`, `root`, or `body`.
        - For xform assets, the source is always `root`.

    """

    property_name: Literal[
        "pose",
        "state",
        "position",
        "position_relative",
        "orientation",
        "linear_velocity",
        "linear_velocity_relative",
        "angular_velocity",
        "linear_acc",
        "angular_acc",
        "effort",
    ]
    """The property name to observe.

    Note:

      - For xform assets, the property can be {position, orientation, pose}.
      - For rigid objects and articulations, the property can be
        {pose, state, position, orientation, linear_velocity, angular_velocity,
        linear_acc, angular_acc}.
      - For articulations, orientation related properties are not supported
        for joints.

    """

    ref_frame: Literal["world"] = "world"
    """The reference frame of the property."""
