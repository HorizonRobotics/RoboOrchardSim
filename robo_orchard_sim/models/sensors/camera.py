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

# Currently only Isaac Lab Camera and TiledCamera are supported.
# TODO:
# 1. Unify Different camera implementation, such as isaac.lab or isaac.sim


from __future__ import annotations
import weakref
from typing import Literal, Sequence, Tuple, TypeVar

import isaacsim.core.utils.stage as stage_utils
import torch
from isaaclab.sensors.camera import (
    Camera as _Camera,
    TiledCamera as _TiledCamera,
)
from isaaclab.sensors.camera.camera_cfg import CameraCfg as _CameraCfg
from isaaclab.sensors.camera.tiled_camera_cfg import (
    TiledCameraCfg as _TiledCameraCfg,
)
from isaacsim.core.utils.stage import get_stage_up_axis
from pydantic import AliasChoices, ConfigDict, Field
from robo_orchard_core.devices.cameras.camera import CameraBaseCfg
from robo_orchard_core.utils.config import TorchTensor
from robo_orchard_core.utils.math import math_utils

from robo_orchard_sim.cfg_wrappers.sensor_cfg import SensorBaseCfg
from robo_orchard_sim.cfg_wrappers.sim.spawners.sensors_cfg import (
    FisheyeCameraCfg,
    PinholeCameraCfg,
)
from robo_orchard_sim.models.sensors.isaac_camera import (
    IsaacCameraMixin,
)
from robo_orchard_sim.utils.config import (
    ClassType,
    Config,
)

CameraType = TypeVar("CameraType", bound="Camera | TiledCamera")


class Camera(_Camera, IsaacCameraMixin):
    """Wrapper class for isaac lab Camera.

    This class wraps the isaac lab Camera sensor and provides
    :py:class:`BatchCameraBase` properties and methods.

    In addition, this class provides a hook handler for after capture
    event (triggered by `_update_buffers_impl`).


    """

    def __init__(self, cfg: CameraCfg | SemanticCameraCfg | _CameraCfg):
        IsaacCameraMixin.__init__(self, cfg)
        if isinstance(cfg, (CameraCfg, SemanticCameraCfg)):
            dict_cfg = cfg.__dict__
            _Camera.__init__(self, _CameraCfg(**dict_cfg))
        else:
            _Camera.__init__(self, cfg)

    @property
    def local_frame_id(self) -> str:
        return self.cfg.prim_path

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        self.after_capture_hook_handler(weakref.proxy(self))

    def _initialize_impl(self):
        """Initialize the camera sensor."""

        # clear the sensor prims to avoid timeline bug.
        self._sensor_prims = []

        # # skip
        # prim_paths_expr = self.cfg.prim_path
        # if not isinstance(prim_paths_expr, list):
        #     prim_paths_expr = [prim_paths_expr]
        # prim_paths = []
        # for prim_path_expression in prim_paths_expr:
        #     prim_paths = prim_paths + find_matching_prim_paths(
        #         prim_path_expression
        #     )
        # if len(prim_paths) == 0:
        #     warnings.warn(
        #         f"Cannot find prim path for {prim_paths_expr}. Skip "
        #     )
        #     return

        return super()._initialize_impl()


class TiledCamera(_TiledCamera, IsaacCameraMixin):
    """Wrapper class for isaac lab TiledCamera.

    This class wraps the isaac lab Camera sensor and provides
    :py:class:`BatchCameraBase` properties and methods.

    In addition, this class provides a hook handler for after capture
    event (triggered by `_update_buffers_impl`).
    """

    def __init__(self, cfg: CameraCfg | _TiledCameraCfg):
        IsaacCameraMixin.__init__(self, cfg)
        if isinstance(cfg, CameraCfg):
            _TiledCamera.__init__(self, _TiledCameraCfg(**cfg.__dict__))
        else:
            _TiledCamera.__init__(self, cfg)

    @property
    def local_frame_id(self) -> str:
        return self.cfg.prim_path

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        self.after_capture_hook_handler(weakref.proxy(self))

    def _initialize_impl(self):
        # clear the sensor prims to avoid timeline bug.
        self._sensor_prims = []
        return super()._initialize_impl()


class CameraOffset(Config):
    """The offset pose of the sensor's frame from the sensor's parent frame.

    This class extend isaaclab's CameraCfg.OffsetCfg to support
    `toward_target` property, which is used to calculate the rotation
    quaternion based on the position of the camera and the target
    position to look at.

    """

    xyz: tuple[float, float, float] | TorchTensor = Field(
        default=(0.0, 0.0, 0.0),
        validation_alias=AliasChoices("xyz", "trans", "pos"),
    )
    """3D ranslation vector or position.

    Defaults to (0.0, 0.0, 0.0)."""

    quat: tuple[float, float, float, float] | TorchTensor = Field(
        default=(1.0, 0.0, 0.0, 0.0),
        validation_alias=AliasChoices("quat", "rot", "orientation"),
    )

    @property
    def pos(self):
        return self.xyz

    @pos.setter
    def pos(self, value):
        self.xyz = value

    @property
    def rot(self):
        return self.quat

    @rot.setter
    def rot(self, value):
        self.quat = value

    @property
    def trans(self):
        return self.xyz

    @trans.setter
    def trans(self, value):
        self.xyz = value

    toward_target: Tuple[float, float, float] | None = Field(
        default=None, exclude=True
    )
    """The target position to look at. This property is used only for
    initialization. If set, the `rot` property will be calculated based
    on the `pos` and `toward_target` properties.
    """

    convention: Literal["world", "ros", "opengl"] = "ros"
    """The convention of the rotation quaternion.

    This property is used to be compatible with Isaac Lab's camera.
    """

    def __post_init__(self):
        if self.toward_target is not None:
            if self.rot is not None and self.rot != (1.0, 0.0, 0.0, 0.0):
                raise ValueError(
                    "Cannot set both `toward_target` and `rot` at the same time."  # noqa
                )
        if self.toward_target is not None:
            target = torch.tensor(
                self.toward_target, dtype=torch.float32, device="cpu"
            )
            if target.dim() == 1:
                target = target.unsqueeze(0)

            if not torch.is_tensor(self.trans):
                eye = torch.tensor(
                    self.trans, dtype=torch.float32, device="cpu"
                )
            else:
                eye = self.trans.to(
                    dtype=torch.float32, device="cpu", copy=True
                )

            if eye.dim() == 1:
                eye = eye.unsqueeze(0)

            try:
                get_stage_up_axis()
            except Exception as e:
                raise RuntimeError(
                    f"Try to get UsdStage up axis failed: {e}. "
                    "Please create stage before using `toward_target`."
                    f"Current stage: {stage_utils.get_current_stage()}"
                )

            # create_rotation_matrix_from_view always return
            # orientation in opengl convention
            up_axis = stage_utils.get_stage_up_axis()
            view_convention = self.convention
            if view_convention == "ros":
                view_convention = "cam"

            if up_axis == "Z":
                rot_mat = math_utils.rotation_matrix_from_view(
                    camera_position=eye,
                    at=target,
                    up=((0, 0, 1.0),),
                    device="cpu",
                    view_convention=view_convention,
                )
            elif up_axis == "Y":
                rot_mat = math_utils.rotation_matrix_from_view(
                    camera_position=eye,
                    at=target,
                    up=((0, 1.0, 0),),
                    device="cpu",
                    view_convention=view_convention,
                )

            rot_mat = math_utils.matrix_to_quaternion(rot_mat)
            self.rot = tuple(rot_mat.flatten().tolist())
            self.toward_target = None


class CameraCfg(SensorBaseCfg[CameraType], CameraBaseCfg[CameraType]):
    """Camera configuration.

    The base class config for camera sensors. It can be used to configure
    Camera, TiledCamera, etc.

    Unlike CamaraCfg in isaac lab, this class leave attributes that are
    related to semantic segmentation to its subclass SemanticCameraCfg.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    offset: CameraOffset = CameraOffset()
    """The offset pose of the sensor's frame from the sensor's parent frame.

    Defaults to identity.
    """

    width: int
    """Width of the image in pixels."""

    height: int
    """Height of the image in pixels."""

    # spawn: IsaacConfigType[PinholeCameraCfg | FisheyeCameraCfg] | None = None
    spawn: PinholeCameraCfg | FisheyeCameraCfg | None = None
    """Spawn configuration for the asset.

    If None, then the prim is not spawned by the asset. Instead, it is assumed
    that the  asset is already present in the scene.
    """

    data_types: list[str] = ["rgb"]
    """List of sensor names/types to enable for the camera. """


class SemanticCameraCfg(CameraCfg[Camera]):
    """Configuration for a camera sensor with semantic segmentation.

    This class extends the CameraCfg class to include semantic attributes.
    It is also identical to CameraCfg class in isaac lab.
    """

    class_type: ClassType[Camera] = Camera

    colorize_semantic_segmentation: bool = True
    # for semantic segmentation only.
    colorize_instance_id_segmentation: bool = True
    # for semantic segmentation only.
    colorize_instance_segmentation: bool = True
    # for semantic segmentation only.
