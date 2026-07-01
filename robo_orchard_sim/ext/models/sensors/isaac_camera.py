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

import torch
from isaaclab.sensors.camera import CameraData as LabCameraData
from robo_orchard_core.datatypes import BatchFrameTransform
from robo_orchard_core.devices.cameras.camera import (
    BatchCameraBase,
    BatchCameraData,
)


class IsaacCameraMixin(BatchCameraBase):
    """Mixin class for isaac lab Camera.

    This mixin class provides the common properties and methods of
    :class:`BatchCameraBase` for isaac lab Camera.

    """

    @property
    def local_frame_id(self) -> str:
        """Get the local frame ID of the camera coordinate."""
        raise NotImplementedError(
            "local_frame_id property is not implemented."
        )

    @property
    def data(self) -> LabCameraData:
        raise NotImplementedError("data property is not implemented.")

    @property
    def image_shape(self) -> tuple[int, int]:
        """Get the shape(height, width) of the image.

        Returns:
            tuple[int, int]: A tuple containing (height, width) of the camera.
        """
        return self.data.image_shape

    @property
    def intrinsic_matrices(self) -> torch.Tensor:
        """Get the intrinsic matrices for the camera.

        Returns:
            torch.Tensor: The intrinsic matrices for the camera.
        """
        return self.data.intrinsic_matrices

    @property
    def pose_global(self) -> BatchFrameTransform:
        """Get the global pose of the camera.

        Returns:
            Pose6DCfg: The global pose of the camera.
        """
        return BatchFrameTransform(
            xyz=self.data.pos_w,
            quat=self.data.quat_w_ros,
            parent_frame_id="/World",
            child_frame_id=self.local_frame_id,
        )

    @property
    def sensor_data(self) -> dict[str, torch.Tensor]:
        """Get the camera data of the camera.

        Returns:
            TensorDict: The camera data.
        """
        return self.data.output

    def get_camera_data(self) -> dict[str, BatchCameraData]:
        """Get the camera data of the camera.

        Returns:
            dict[str, BatchCameraData]: The camera data, where the key is the
                topic of the camera.
        """
        ret = {}
        for key, data in self.sensor_data.items():
            if "rgb" in key:
                pix_fmt = "RGB"
            else:
                pix_fmt = None
            ret[key] = BatchCameraData(
                topic=key,
                intrinsic_matrices=self.intrinsic_matrices,
                pose=self.pose_global,
                sensor_data=data,
                image_shape=self.image_shape,
                pix_fmt=pix_fmt,
            )
        return ret
