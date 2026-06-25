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

from isaaclab.managers.scene_entity_cfg import (
    InteractiveScene,
    SceneEntityCfg as _SceneEntityCfg,
)
from robo_orchard_core.envs.managers.scene_entity_cfg import (
    SceneEntityCfg as OrchardSceneEntityCfg,
)

from robo_orchard_sim.utils.config import isaac_configclass2pydantic


class SceneEntityCfg(
    OrchardSceneEntityCfg, isaac_configclass2pydantic(_SceneEntityCfg)
):
    """The pydantic version of SceneEntityCfg.

    Please refer to the origin class for more information:
    :py:class:`isaaclab.managers.scene_entity_cfg.SceneEntityCfg`

    """

    __doc__ = _SceneEntityCfg.__doc__

    def resolve(self, scene: InteractiveScene):
        return _SceneEntityCfg.resolve(self, scene)
