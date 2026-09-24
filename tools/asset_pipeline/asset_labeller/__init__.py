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

from typing import TYPE_CHECKING, Any

from .labeller import (
    ARTICULATION_SPEC_TYPE,
    RIGID_OBJECT_SPEC_TYPE,
    AssetLabeller,
)
from .mesh_utils import load_mesh
from .renderer import render_views

if TYPE_CHECKING:
    from .gpt_client import GPTClient

__version__ = "0.1.0"
__all__ = [
    "AssetLabeller",
    "ARTICULATION_SPEC_TYPE",
    "GPTClient",
    "RIGID_OBJECT_SPEC_TYPE",
    "load_client_from_config",
    "load_mesh",
    "render_views",
]


def __getattr__(name: str) -> Any:
    """Load optional GPT dependencies only when GPT symbols are requested."""
    if name in {"GPTClient", "load_client_from_config"}:
        from .gpt_client import GPTClient, load_client_from_config

        exports = {
            "GPTClient": GPTClient,
            "load_client_from_config": load_client_from_config,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
