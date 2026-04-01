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


from isaaclab.sim.converters.asset_converter_base_cfg import (
    AssetConverterBaseCfg as _AssetConverterBaseCfg,
)

from robo_orchard_sim.utils.config import (
    isaac_configclass2pydantic,
)


class AssetConverterBaseCfg(
    isaac_configclass2pydantic(_AssetConverterBaseCfg)
):
    """The pydantic version of AssetConverterBaseCfg."""

    asset_path: str  # type: ignore
    """The absolute path to the asset file to convert into USD."""
