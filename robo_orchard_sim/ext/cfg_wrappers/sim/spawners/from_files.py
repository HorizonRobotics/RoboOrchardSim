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


import fcntl
import shutil
from pathlib import Path
from typing import Any

from isaaclab.sim import converters
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    FileCfg as _FileCfg,
    GroundPlaneCfg as _GroundPlaneCfg,
    UrdfFileCfg as _UrdfFileCfg,
    UsdFileCfg as _UsdFileCfg,
)
from isaaclab.sim.utils import clone

from robo_orchard_sim.ext.cfg_wrappers.materials.physics_materials_cfg import (
    RigidBodyMaterialCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.converters import UrdfConverterCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.spawner_cfg import (
    DeformableObjectSpawnerCfg,
    RigidObjectSpawnerCfg,
    SpawnerCfg,
)
from robo_orchard_sim.ext.models.prim import USDPrimCreatorType
from robo_orchard_sim.utils.config import isaac_configclass2pydantic

__all__ = [
    "FileCfg",
    "GroundPlaneCfg",
    "UrdfFileCfg",
    "UsdFileCfg",
    "spawn_from_urdf_locked",
]


def convert_urdf_to_usd_locked(cfg: _UrdfFileCfg) -> str:
    """Convert a URDF while holding a lock for its output directory."""
    if not cfg.usd_dir:
        raise ValueError("Locked URDF conversion requires a fixed usd_dir")

    configured_usd_dir = Path(cfg.usd_dir)
    if not configured_usd_dir.is_absolute():
        raise ValueError("Locked URDF conversion requires an absolute usd_dir")

    usd_dir = configured_usd_dir.resolve()
    if usd_dir == Path(usd_dir.anchor):
        raise ValueError("Locked URDF conversion cannot use a root directory")

    lock_path = usd_dir.parent / f".{usd_dir.name}.conversion.lock"
    in_progress_path = usd_dir / ".conversion_in_progress"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        if in_progress_path.exists():
            shutil.rmtree(usd_dir, ignore_errors=True)

        usd_dir.mkdir(parents=True, exist_ok=True)
        in_progress_path.touch()
        urdf_loader = converters.UrdfConverter(cfg)
        in_progress_path.unlink()
        return urdf_loader.usd_path


@clone
def spawn_from_urdf_locked(
    prim_path: str,
    cfg: _UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Any:
    """Convert a URDF under a process lock, then spawn its USD."""
    usd_path = convert_urdf_to_usd_locked(cfg)

    # Conversion is complete and the lock is released before stage loading.
    return from_files._spawn_from_usd_file(
        prim_path,
        usd_path,
        cfg,
        translation,
        orientation,
    )


class FileCfg(
    RigidObjectSpawnerCfg,
    DeformableObjectSpawnerCfg,
    isaac_configclass2pydantic(_FileCfg),
):
    """The pydantic version of isaaclab.sim.spawners.from_files.FileCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.from_files.FileCfg`

    """

    __doc__ = _FileCfg.__doc__


class UsdFileCfg(
    FileCfg,
    isaac_configclass2pydantic(_UsdFileCfg),
):
    """The pydantic version of UsdFileCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.from_files.UsdFileCfg`

    """

    # __doc__ = _UsdFileCfg.__doc__

    func: USDPrimCreatorType = from_files.spawn_from_usd
    # override the default value of func field to be from_files.spawn_from_usd
    # otherwise, the default value is Missing(defines in FileCfg)

    usd_path: str
    # override the default value of the usd_path field to be required


class UrdfFileCfg(
    FileCfg,
    UrdfConverterCfg,
    isaac_configclass2pydantic(_UrdfFileCfg),
):
    """The pydantic version of UrdfFileCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.from_files.UrdfFileCfg`

    """

    __doc__ = _UrdfFileCfg.__doc__

    func: USDPrimCreatorType = spawn_from_urdf_locked
    # override the default value of func field with the process-safe spawner
    # otherwise, the default value is Missing(defines in FileCfg)


class GroundPlaneCfg(
    SpawnerCfg,
    isaac_configclass2pydantic(_GroundPlaneCfg),
):
    """The pydantic version of GroundPlaneCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.sim.spawners.from_files.GroundPlaneCfg`

    """

    __doc__ = _GroundPlaneCfg.__doc__

    func: USDPrimCreatorType = from_files.spawn_ground_plane
    # override the default value of func field to be from_files.spawn_from_usd
    # otherwise, the default value is Missing(defines in SpawnerCfg)

    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
    # override physics_material to pydantic version of RigidBodyMaterialCfg
