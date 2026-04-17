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

from typing import Any, Literal, Sequence, cast

import cv2
import numpy as np
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from google.protobuf.timestamp_pb2 import Timestamp
from robo_orchard_core.datatypes.camera_data import BatchCameraData

from robo_orchard_sim.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.envs.managers.record import (
    RecordTermBase,
    RecordTermBaseCfg,
)
from robo_orchard_sim.envs.managers.record.mcap.message import Message
from robo_orchard_sim.utils.config import ClassType_co

# from robo_orchard_sim.utils.viz_utils import visualize_depth

ReturnType = dict[str, list[Message]]


def visualize_depth(
    depth,
    min_valid_depth=None,
    max_valid_depth=None,
    colormap=cv2.COLORMAP_JET,
    invalid_color=(0, 0, 0),
):
    assert min_valid_depth is not None or max_valid_depth is not None
    # 如果是uint16格式，转换为米为单位的float
    if depth.dtype == np.uint16:
        depth = depth.astype(float) / 1000.0  # 转换为米
    # 处理无效值（NaN或非正值）
    mask_valid = depth > 0
    # 归一化到[0, 1]范围
    depth_scaled = np.where(
        mask_valid,
        (depth - min_valid_depth) / (max_valid_depth - min_valid_depth),
        0,
    )
    # 裁剪并缩放到[0, 255]
    depth_scaled = np.clip(depth_scaled * 255, 0, 255).astype(np.uint8)
    # 应用颜色映射
    colored = cv2.applyColorMap(depth_scaled, colormap)
    # 标记无效区域
    colored[~mask_valid] = invalid_color
    return colored


class McapImageTerm(
    RecordTermBase[IsaacEnvType_co, "McapImageTermCfg", ReturnType]
):
    def __init__(self, cfg: "McapImageTermCfg", env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env

    def __call__(
        self, data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> ReturnType:
        """The implementation of the record term.

        All subclasses should implement this method to return the observation.

        """
        image_data = self._parse_data_from_dict(data, self._cfg.key)[
            self._cfg.key
        ]

        if self._cfg.mode == "rgb":
            return self._record_rgb(image_data, ts)
        elif self._cfg.mode == "depth":
            return self._record_depth(image_data, ts)
        elif self._cfg.mode == "color_depth":
            return self._record_color_depth(image_data, ts)
        elif self._cfg.mode == "calibration":
            return self._record_calibration(image_data, ts)
        else:
            raise ValueError(
                f"Invalid mode: {self._cfg.mode}. Supported modes are 'rgb', "
                "'depth', 'color_depth', and 'calibration'."
            )

    def _record_rgb(
        self, image_data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> ReturnType:
        """Record RGB image data."""
        image = (
            self._get_camera_data(image_data, "rgb").sensor_data.cpu().numpy()
        )
        image_format = "jpg"

        msgs = []
        for i in range(image.shape[0]):
            img = image[i]
            rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            compressed_data = cv2.imencode("." + image_format, rgb)[1]

            msg = Message(
                data=CompressedImage(
                    timestamp=ts,
                    frame_id=self._cfg.frame_id,
                    format=image_format,
                    data=compressed_data.tobytes(),
                ),
                log_time=ts,
                pub_time=ts,
            )
            msgs.append(msg)

        return {self._cfg.topic: msgs}

    def _record_depth(
        self, image_data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> ReturnType:
        """Record depth image data."""
        image = (
            self._get_camera_data(image_data, "depth")
            .sensor_data.cpu()
            .numpy()
        )
        image_format = "png"

        msgs = []
        for i in range(image.shape[0]):
            img = image[i]
            # Convert from meters to millimeters
            depth_in_mm_float = img[..., 0] * 1000
            # Replace nan with 0 and inf with the max value of uint16
            depth_in_mm_float[np.isnan(depth_in_mm_float)] = 0
            depth_in_mm_float[np.isinf(depth_in_mm_float)] = np.iinfo(
                np.uint16
            ).max
            # Cast to uint16
            depth_in_mm = depth_in_mm_float.astype(np.uint16)
            compressed_data = cv2.imencode("." + image_format, depth_in_mm)[1]

            msg = Message(
                data=CompressedImage(
                    timestamp=ts,
                    frame_id=self._cfg.frame_id,
                    format=image_format,
                    data=compressed_data.tobytes(),
                ),
                log_time=ts,
                pub_time=ts,
            )
            msgs.append(msg)

        return {self._cfg.topic: msgs}

    def _record_color_depth(
        self, image_data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> ReturnType:
        """Record colorized depth image data."""
        image = (
            self._get_camera_data(image_data, "depth")
            .sensor_data.cpu()
            .numpy()
        )
        image_format = "png"

        msgs = []
        for i in range(image.shape[0]):
            img = image[i]
            depth_img = img[..., 0].astype(np.float32)
            color_depth_img = visualize_depth(
                depth=depth_img,
                colormap=cv2.COLORMAP_PLASMA,
                min_valid_depth=0.1,
                max_valid_depth=2.0,
            )
            compressed_data = cv2.imencode(
                "." + image_format, color_depth_img
            )[1]

            msg = Message(
                data=CompressedImage(
                    timestamp=ts,
                    frame_id=self._cfg.frame_id,
                    format=image_format,
                    data=compressed_data.tobytes(),
                ),
                log_time=ts,
                pub_time=ts,
            )
            msgs.append(msg)

        return {self._cfg.topic: msgs}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the observation term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        """
        pass

    def _record_calibration(
        self, image_data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> ReturnType:
        """Record calibration data if needed."""
        rgb_data = self._get_camera_data(image_data, "rgb")
        intrinsic = rgb_data.intrinsic_matrices.cpu().numpy()
        image_shape = rgb_data.image_shape

        env_msgs = []
        for i in range(intrinsic.shape[0]):
            msg = self._convert_calibration(
                matrix=intrinsic[i],
                image_shape=image_shape,
                frame_id=self._cfg.frame_id,
                ts=ts,
            )
            env_msgs.append(msg)

        return {self._cfg.topic: env_msgs}

    def _convert_calibration(
        self,
        matrix: np.ndarray,
        image_shape: tuple[int, int],
        frame_id: str,
        ts: Timestamp,
    ):
        height, width = image_shape
        d_param = np.zeros(5)

        k_param = np.zeros(9)
        k_param[0] = matrix[0][0]  # fx
        k_param[2] = matrix[0][2]  # cx
        k_param[4] = matrix[1][1]  # fy
        k_param[5] = matrix[1][2]  # cy
        k_param[8] = 1.0

        p_param = np.zeros(12)
        p_param[0] = matrix[0][0]  # fx
        p_param[2] = matrix[0][2]  # cx
        # p_param[3] = matrix[0][3] # TX
        p_param[5] = matrix[1][1]  # fy
        p_param[6] = matrix[1][2]  # cy
        p_param[10] = 1.0

        intrinsic = CameraCalibration(
            timestamp=ts,
            frame_id=frame_id,
            width=width,
            height=height,
            distortion_model="plumb_bob",
            D=d_param,
            K=k_param,
            R=None,  # left cam only
            P=p_param,
        )
        msg = Message(
            data=intrinsic,
            log_time=ts,
            pub_time=ts,
        )

        return msg

    def _get_camera_data(
        self,
        image_data: Any,
        mode: Literal["rgb", "depth"],
    ) -> BatchCameraData:
        if isinstance(image_data, BatchCameraData):
            return image_data

        if isinstance(image_data, dict):
            if mode in image_data:
                return cast(BatchCameraData, image_data[mode])
            if "output" in image_data and mode in image_data["output"]:
                return cast(BatchCameraData, image_data["output"][mode])

        raise KeyError(
            f"Camera data for mode '{mode}' not found in "
            f"{type(image_data).__name__}."
        )


class McapImageTermCfg(RecordTermBaseCfg):
    """The configuration class for the McapImageTerm.

    Args:
        class_type: The class type of the record term. Defaults to
            McapImageTerm.
        key: The key to access image data from the observation dictionary.
        frame_id: The frame identifier for the recorded image messages.
        mode: The recording mode that determines what type of image data to
            record.

    """

    class_type: ClassType_co[McapImageTerm] = McapImageTerm

    key: str
    """The key used to extract image data from the observation dictionary.
    This should correspond to the observation group containing camera data."""

    frame_id: str
    """The frame identifier that will be included in the recorded messages.
    This is typically used to identify the camera or sensor frame."""

    mode: Literal["rgb", "depth", "color_depth", "calibration"] = "rgb"
