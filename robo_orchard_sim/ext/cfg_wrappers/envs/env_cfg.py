# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
#
#
#       http://www.apache.org/licenses/LICENSE-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# You may obtain a copy of the License at
# distributed under the License is distributed on an "AS IS" BASIS,
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# you may not use this file except in compliance with the License.

from typing import Any

from isaaclab.envs.manager_based_env_cfg import (
    BaseEnvWindow,
    DefaultEventManagerCfg as _DefaultEventManagerCfg,
    ManagerBasedEnvCfg as _ManagerBasedEnvCfg,
    ViewerCfg as _ViewerCfg,
)
from isaaclab.envs.mdp import reset_scene_to_default

from robo_orchard_sim.ext.cfg_wrappers.managers.manager_term_cfg import (
    EventTermCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.scenes_cfg import InteractiveSceneCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.simulation_cfg import SimulationCfg
from robo_orchard_sim.utils.config import (
    ClassType,
    isaac_configclass2pydantic,
)

__all__ = [
    "DefaultEventManagerCfg",
    "ManagerBasedEnvCfg",
    "ViewerCfg",
]


class ViewerCfg(isaac_configclass2pydantic(_ViewerCfg)):
    """The pydantic version of ViewerCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.envs.manager_based_env_cfg.ViewerCfg`

    """

    __doc__ = _ViewerCfg.__doc__


class DefaultEventManagerCfg(
    isaac_configclass2pydantic(_DefaultEventManagerCfg)
):
    """The pydantic version of DefaultEventManagerCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.envs.manager_based_env_cfg.DefaultEventManagerCfg`

    """

    __doc__ = _DefaultEventManagerCfg.__doc__

    reset_scene_to_default: EventTermCfg = EventTermCfg(
        func=reset_scene_to_default, mode="reset"
    )
    """The event term for resetting the scene to the default state.

    Override to fit the pydantic schema.
    """


class ManagerBasedEnvCfg(isaac_configclass2pydantic(_ManagerBasedEnvCfg)):
    """The pydantic version of ManagerBasedEnvCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.envs.manager_based_env_cfg.ManagerBasedEnvCfg`

    """

    __doc__ = _ManagerBasedEnvCfg.__doc__

    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg().

    Override to fit the pydantic schema.
    """

    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg().

    Override to fit the pydantic schema.
    """

    ui_window_class_type: ClassType[Any] | None = BaseEnvWindow
    """Override to fit the pydantic schema."""

    scene: InteractiveSceneCfg
    """Override to fit the pydantic schema."""

    events: Any = DefaultEventManagerCfg()
    """Override to fit the pydantic schema."""
