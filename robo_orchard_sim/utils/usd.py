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
import functools
import logging

from isaacsim.core.utils.extensions import enable_extension
from pxr import Usd, UsdGeom


@functools.lru_cache()
def enable_urdf_exporter():
    try:
        import omni.exporter.urdf as urdf  # noqa
    except ImportError:
        enable_extension("omni.exporter.urdf")


def usd_to_urdf(
    usd_path: str | Usd.Stage,
    urdf_output_path: str,
    kinematics_only: bool = False,
    log_level=logging.INFO,
) -> str:
    """Export a USD file to URDF format.

    Args:
        usd_path (str | Usd.Stage): The path to the USD file or the USD stage.
            If a string is provided, the USD file will be loaded from the path.
            If a USD stage is provided, the stage will be used directly.
        urdf_output_path (str): The path to the output URDF file. If it is a
            file path then it is saved to that file. If it is a directory path,
            then it is a saved into that directory with the file name matching
            the USD name but with the .urdf extension.
        kinematics_only (bool, optional): If True, only the kinematic structure
            of the robot is exported. Defaults to False.
        log_level (int, optional): The log level. Defaults to logging.INFO.

    Returns:
        str: The path to the URDF file.
    """

    enable_urdf_exporter()
    from nvidia.srl.from_usd.to_urdf import UsdToUrdf  # type: ignore

    if isinstance(usd_path, str):
        usd_to_urdf = UsdToUrdf.init_from_file(usd_path, log_level=log_level)
    else:
        usd_to_urdf = UsdToUrdf(usd_path, log_level=log_level)

    return usd_to_urdf.save_to_file(
        urdf_output_path=urdf_output_path,
        mesh_dir=None,
        mesh_path_prefix="",
        kinematics_only=kinematics_only,
    ).as_posix()


def get_prim_aabb(stage: Usd.Stage, prim_path: str):
    """Get the axis-aligned bounding box of an object in the scene.

    Args:
        stage: The USD stage containing the object.
        prim_path: The path to the object in the USD stage.

    Returns:
        The axis-aligned bounding box of the object.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    boundable = UsdGeom.Boundable(prim)

    time = Usd.TimeCode.Default()
    bbox = boundable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bbox_range = bbox.GetRange()

    x_min, y_min, z_min = bbox_range.GetMin()
    x_max, y_max, z_max = bbox_range.GetMax()
    return (x_max, x_min), (y_max, y_min), (z_max, z_min)
