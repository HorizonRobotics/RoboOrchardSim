# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

"""Asset Labeller - Standalone 3D asset labelling tool.

Generates URDF files with physical and semantic attributes for 3D assets.
Supports OBJ and USD mesh formats with texture.
"""

from .gpt_client import GPTClient, load_client_from_config
from .labeller import AssetLabeller
from .mesh_utils import load_mesh
from .renderer import render_views

__version__ = "0.1.0"
__all__ = [
    "AssetLabeller",
    "GPTClient",
    "load_client_from_config",
    "load_mesh",
    "render_views",
]
