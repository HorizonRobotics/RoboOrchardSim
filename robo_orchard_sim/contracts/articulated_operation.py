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

"""Shared operation vocabulary for articulated assets and tasks."""

from typing import Final, Literal

ArticulatedOperation = Literal[
    "open",
    "close",
    "pull",
    "push",
    "press",
    "rotate",
]

SUPPORTED_ARTICULATED_OPERATIONS: Final[frozenset[ArticulatedOperation]] = (
    frozenset({"open", "close", "pull", "push", "press", "rotate"})
)
