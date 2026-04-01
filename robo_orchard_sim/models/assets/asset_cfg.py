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
#
from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    TypeVar,
)

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR

from robo_orchard_sim.cfg_wrappers.assets_cfg import AssetBaseCfg
from robo_orchard_sim.cfg_wrappers.sensor_cfg import SensorBaseCfg

__all__ = [
    "ASSET_CFG",
    "NV_ASSET_ROOT_DIR",
    "NV_ISAAC_DIR",
    "NV_ISAACLAB_DIR",
    "asset_replace_usd_path",
    "GroupAssetCfg",
    "ORCHARD_ASSET",
]

# Try to get NV_ASSET_ROOT_DIR from environment variables
# If not found, use the default value
NV_ASSET_ROOT_DIR = os.getenv("NV_ASSET_ROOT_DIR", NUCLEUS_ASSET_ROOT_DIR)
"""Path to the root directory on the Nucleus Server.

Isaac lab seems to use fixed root directories for assets in
`NUCLEUS_ASSET_ROOT_DIR`.  This variable is used to replace it with the
environment variable `NV_ASSET_ROOT_DIR` if it is set.
"""

NV_ISAAC_DIR = f"{NV_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

NV_ISAACLAB_DIR = f"{NV_ISAAC_DIR}/IsaacLab"
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""

ORCHARD_NV_ISAAC_ASSET_ROOT = (
    "/horizon-bucket/robot_lab/assets/NVIDIA/Assets/Isaac/4.1/Isaac/"
)

ORCHARD_ASSET = os.getenv("ORCHARD_ASSET", "/horizon-bucket/robot_lab/assets")
"""Path to the ``ORCHARD`` directory."""

SensorCfgType_co = TypeVar(
    "SensorCfgType_co", bound=SensorBaseCfg, covariant=True
)

ASSET_CFG = TypeVar("ASSET_CFG", bound=AssetBaseCfg)
ASSET_CFG_co = TypeVar("ASSET_CFG_co", bound=AssetBaseCfg, covariant=True)
GroupAssetCfgType = dict[str, SensorCfgType_co | AssetBaseCfg]
if TYPE_CHECKING:

    class GroupAssetCfg(GroupAssetCfgType):
        """A dict of asset configurations.

        The asset names in the GroupAssetCfg are concatenated with the group
        name separated by '/' when added to the scene. For example, if the
        group name is 'group1' and the asset name is 'asset1', the asset name
        in the scene will be 'group1/asset1'.
        """

        pass

else:
    GroupAssetCfg = dict


def asset_replace_usd_path(
    asset_cfg: ASSET_CFG,
    old: str = NUCLEUS_ASSET_ROOT_DIR,
    new: str = NV_ASSET_ROOT_DIR,
) -> ASSET_CFG:
    new_cfg = asset_cfg.copy()
    spawn_cfg = new_cfg.spawn
    if isinstance(spawn_cfg, UsdFileCfg) or hasattr(spawn_cfg, "usd_path"):
        spawn_cfg.usd_path = spawn_cfg.usd_path.replace(old, new)  # type: ignore
    return new_cfg
