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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd


def is_object_center_in_obb(
    stage: "Usd.Stage",
    prim_path: str,
    obb_pose: np.ndarray,
    point: np.ndarray,
    idx_env: int = 0,
    axes: str = "xy",
) -> bool:
    """Check if a point is inside the oriented bounding box of a prim.

    Uses inverse-transform: the query point is transformed into the OBB's
    local coordinate frame and compared against the untransformed AABB,
    avoiding the precision loss from recomputing an axis-aligned box after
    rotation.

    Args:
        stage: The USD stage containing the object.
        prim_path: The path to the OBB prim in the USD stage.
        obb_pose: Pose array of shape [N, 7] (px, py, pz, qw, qx, qy, qz).
        point: World-space position of shape [3] (or longer, only [:3] used).
        idx_env: Environment index into obb_pose. Default: 0.
        axes: Which axes to check — "xy" or "xyz". Default: "xy".

    Returns:
        True if the point lies inside the OBB (on the specified axes).
    """
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")

    bbox = UsdGeom.Boundable(prim).ComputeUntransformedBound(
        Usd.TimeCode.Default(), UsdGeom.Tokens.default_
    )
    mn, mx = bbox.GetRange().GetMin(), bbox.GetRange().GetMax()

    scale = np.ones(3)
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            s = op.Get()
            scale = np.array([s[0], s[1], s[2]])
            break

    local_min = np.array([mn[0], mn[1], mn[2]]) * scale
    local_max = np.array([mx[0], mx[1], mx[2]]) * scale

    pose = obb_pose[idx_env]
    pos = pose[:3].astype(float)
    qw = float(pose[3])
    qx, qy, qz = float(pose[4]), float(pose[5]), float(pose[6])

    dp = point[:3].astype(float) - pos
    inv_xyz = np.array([-qx, -qy, -qz])
    t = 2.0 * np.cross(inv_xyz, dp)
    local_pt = dp + qw * t + np.cross(inv_xyz, t)

    if axes == "xy":
        return bool(
            local_min[0] <= local_pt[0] <= local_max[0]
            and local_min[1] <= local_pt[1] <= local_max[1]
        )
    return bool(
        np.all(local_pt >= local_min) and np.all(local_pt <= local_max)
    )
