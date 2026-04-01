# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

import contextlib
import ctypes
import warnings
from typing import Any, Callable, Optional

import numpy as np
import omni
import omni.kit.app
import omni.kit.commands
import torch
from pxr import Gf, Sdf, Usd, UsdGeom

__all__ = ["ViewportCamera"]


def Vec3d_to_np(vec: Gf.Vec3d) -> np.ndarray:
    return np.array([vec[0], vec[1], vec[2]], dtype=np.float64)


def Vec3d_to_tuple(vec: Gf.Vec3d) -> tuple[float, float, float]:
    return (vec[0], vec[1], vec[2])


def capture_viewport_to_buffer(
    viewport_api: Any, on_capture_fn: Callable[[torch.Tensor], None]
):
    """Capture viewport to buffer.

    This function is a wrapper around
    `omni.kit.viewport.utility.capture_viewport_to_buffer` that converts
    the buffer to a numpy array and calls the given callback function.

    The capture is triggered by simulation events, so the callback function
    will be called asynchronously.

    Args:
        viewport_api: The viewport API object.
        on_capture_fn(np.ndarray): The callback function that takes the
            captured image as a numpy array.
    """

    def on_viewport_captured_as_np(
        buffer: ctypes.py_object,
        buffer_size: int,
        width: int,
        height: int,
        format: Any,
    ):
        try:
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(
                ctypes.c_byte * buffer_size
            )
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
                ctypes.py_object,
                ctypes.c_char_p,
            ]
            content = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
        except Exception as e:
            warnings.warn(f"Failed to capture viewport: {e}", stacklevel=2)
            return
        img = torch.frombuffer(content.contents, dtype=torch.uint8).reshape(
            [height, width, 4]
        )
        on_capture_fn(img)

    from omni.kit.viewport.utility import (
        capture_viewport_to_buffer as _capture_viewport_to_buffer,
    )

    return _capture_viewport_to_buffer(
        viewport_api, on_viewport_captured_as_np
    )


class ViewportCamera:
    def __init__(
        self, viewport: Any | None = None, camera_path: Optional[str] = None
    ):
        if viewport is None:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("No default or provided Viewport")

        self._viewport_api = viewport
        self._camera_path = str(
            camera_path if camera_path else viewport.camera_path
        )
        self._time_code: Usd.TimeCode = Usd.TimeCode.Default()  # type: ignore
        self._cam_prim = viewport.stage.GetPrimAtPath(self._camera_path)
        self._usd_camera = (
            UsdGeom.Camera(self._cam_prim) if self._cam_prim else None
        )
        if self._usd_camera is None:
            raise ValueError(
                f"Invalid Usd.Prim or UsdGeom.Camera: {self._camera_path}"
            )

        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        self._viewport_cam_state = ViewportCameraState(
            viewport=self._viewport_api,
            camera_path=self._camera_path,
            time=self._time_code,
        )

        # check if center of interest property exists, create if not
        coi_prop = self._cam_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            self._cam_prim.CreateAttribute(
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))

    @property
    def updates_enabled(self) -> bool:
        return self._viewport_api.updates_enabled

    @updates_enabled.setter
    def updates_enabled(self, value: bool):
        self._viewport_api.updates_enabled = value

    @contextlib.contextmanager
    def enable_updates(self):
        old_update = self.updates_enabled
        self.updates_enabled = True
        yield
        self.updates_enabled = old_update

    @property
    def usd_camera(self) -> UsdGeom.Camera:
        return self._usd_camera

    @property
    def focal_length(self) -> float:
        return self._cam_prim.GetAttribute("focalLength").Get()

    @property
    def position_world(self) -> tuple[float, float, float]:
        return Vec3d_to_tuple(self._viewport_cam_state.position_world)

    @property
    def look_target_world(self) -> tuple[float, float, float]:
        return Vec3d_to_tuple(self._viewport_cam_state.target_world)

    @property
    def rotation_quat_world(self) -> tuple[float, float, float, float]:
        """Get the camera rotation quaternion in world coordinates.

        Returns:
            tuple[float, float, float, float]: The quaternion in the format
                (w, x, y, z).
        """
        usd_cam: UsdGeom.Camera = self._viewport_cam_state.usd_camera
        local2world: Gf.Matrix4d = usd_cam.ComputeLocalToWorldTransform(
            time=self._time_code
        )
        local2world_quat = local2world.ExtractRotationQuat()
        local2world_quat.Normalize(eps=1e-6)
        quat_im = local2world_quat.GetImaginary()

        return (local2world_quat.GetReal(), quat_im[0], quat_im[1], quat_im[2])

    def capture_viewport_to_buffer(
        self, on_capture_fn: Callable[[torch.Tensor], None]
    ):
        """Capture viewport to buffer.

        Capture the current viewport to a buffer and call the given
        callback with the buffer.

        Args:
            on_capture_fn(np.ndarray): The callback function that takes the
                captured image as a numpy array.
        """
        return capture_viewport_to_buffer(self._viewport_api, on_capture_fn)

    def set_rotation_quat_world(self, quat: tuple[float, float, float, float]):
        """Set the camera rotation quaternion in world coordinates.

        Args:
            quat (tuple[float, float, float, float]): The quaternion in the
                format (w, x, y, z).
        """
        quat = Gf.Quatd(*quat)
        usd_cam: UsdGeom.Camera = self._viewport_cam_state.usd_camera
        local2world = usd_cam.ComputeLocalToWorldTransform(self._time_code)
        parent2world = usd_cam.ComputeParentToWorldTransform(self._time_code)
        world2parent = parent2world.GetInverse()
        local2parent = local2world * world2parent
        new_local2world = Gf.Matrix4d(local2parent)
        new_local2world.SetRotateOnly(quat)

        omni.kit.commands.create(
            "TransformPrimCommand",
            path=self._camera_path,
            new_transform_matrix=new_local2world,
            old_transform_matrix=local2parent,
            time_code=self._time_code,
            usd_context_name=self._viewport_api.usd_context_name,
        ).do()

    def set_position_world(
        self, pos: tuple[float, float, float], rotate: bool
    ):
        """Set the camera position in world coordinates.

        Args:
            pos (tuple[float, float, float]): The position to set.
            rotate (bool): If True, the camera will be rotated to maintain
                its current orientation towards the center of interest.
        """
        self._viewport_cam_state.set_position_world(
            Gf.Vec3d(*pos), rotate=rotate
        )

    def set_look_target_world(
        self, target: tuple[float, float, float], rotate: bool
    ):
        """Set the camera target in world coordinates.

        Args:
            target (tuple[float, float, float]): The target to look at.
            rotate: If True, the camera will rotate to look at the new target.
                If False, the camera will move to maintain its orientation and
                distance relative to the target.
        """
        self._viewport_cam_state.set_target_world(
            Gf.Vec3d(*target), rotate=rotate
        )

    def get_intrinsics_matrix(self) -> np.ndarray:
        """Get the intrinsics matrix of the camera.

        Returns:
            np.ndarray: The intrinsics matrix. The following image convention
                is assumed: x -> right, y -> down, z -> forward.
        """

        prim = self._cam_prim
        (width, height) = self._viewport_api.get_texture_resolution()
        focal_length = prim.GetAttribute("focalLength").Get()
        horizontal_aperture = prim.GetAttribute("horizontalAperture").Get()
        vertical_aperture = prim.GetAttribute("verticalAperture").Get()
        horiz_aperture_offset = prim.GetAttribute(
            "horizontalApertureOffset"
        ).Get()
        vert_aperture_offset = prim.GetAttribute(
            "verticalApertureOffset"
        ).Get()
        # vertical_aperture = horizontal_aperture * (float(height) / width)
        fx = width * focal_length / horizontal_aperture
        fy = height * focal_length / vertical_aperture
        cx = width * 0.5 + horiz_aperture_offset * fx
        cy = height * 0.5 + vert_aperture_offset * fy
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    def set_intrinsics_matrix(
        self,
        intrinsics_matrix: np.ndarray,
        focal_length: Optional[float] = None,
    ):
        if intrinsics_matrix.shape != (3, 3):
            raise ValueError("intrinsics_matrix must be 3x3")

        fx = intrinsics_matrix[0, 0]
        fy = intrinsics_matrix[1, 1]
        cx = intrinsics_matrix[0, 2]
        cy = intrinsics_matrix[1, 2]

        if focal_length is None:
            focal_length = self.focal_length

        prim = self._cam_prim
        (width, height) = self._viewport_api.get_texture_resolution()
        horizontal_aperture = width * focal_length / fx
        vertical_aperture = height * focal_length / fy
        horizontal_aperture_offset = (cx - width / 2) / fx
        vertical_aperture_offset = (cy - height / 2) / fy

        prim.GetAttribute("focalLength").Set(focal_length)
        prim.GetAttribute("horizontalAperture").Set(horizontal_aperture)
        prim.GetAttribute("verticalAperture").Set(vertical_aperture)
        prim.GetAttribute("horizontalApertureOffset").Set(
            horizontal_aperture_offset
        )
        prim.GetAttribute("verticalApertureOffset").Set(
            vertical_aperture_offset
        )
