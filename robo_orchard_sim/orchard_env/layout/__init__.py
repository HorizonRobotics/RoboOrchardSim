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

"""Layout JSON loader + env-level layout schedule."""

from typing import Any

from robo_orchard_sim.orchard_env.layout.loader import (
    Layout,
    LayoutObject,
    LayoutSequence,
    LayoutValidationError,
    parse_layout,
)

__all__ = [
    "Layout",
    "LayoutBuilder",
    "LayoutObject",
    "LayoutSequence",
    "LayoutValidationError",
    "parse_layout",
]


def __getattr__(name: str) -> Any:
    if name == "LayoutBuilder":
        from robo_orchard_sim.orchard_env.layout.builder import LayoutBuilder

        return LayoutBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
