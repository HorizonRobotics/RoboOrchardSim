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

from typing import TYPE_CHECKING

from isaaclab.scene import InteractiveSceneCfg as _InteractiveSceneCfg

from robo_orchard_sim.utils.config import Config, isaac_configclass2pydantic

if TYPE_CHECKING:

    class InteractiveSceneCfg(Config, _InteractiveSceneCfg):
        """The pydantic version of isaac lab InteractiveSceneCfg class.

        Please refer to the origin class for more information:
        :py:class:`isaaclab.scene.InteractiveSceneCfg

        """

        __doc__ = _InteractiveSceneCfg.__doc__

else:

    class InteractiveSceneCfg(
        isaac_configclass2pydantic(_InteractiveSceneCfg)
    ):
        """The pydantic version of isaac lab InteractiveSceneCfg class.

        Please refer to the origin class for more information:
        :py:class:`isaaclab.scene.InteractiveSceneCfg`

        """
