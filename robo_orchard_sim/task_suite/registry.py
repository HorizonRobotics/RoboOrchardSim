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

"""Task-suite runtime registry helpers."""

from __future__ import annotations

from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
from robo_orchard_sim.task_suite.registration import (
    build_task as _build_task,
)


def _bootstrap_task_definitions() -> None:
    """User should register task definitions in this function."""
    from robo_orchard_sim.task_suite.manipulation import (
        place_a2b as _place_a2b,
    )

    del _place_a2b


def build_task(task_name: str) -> OrchardEnv:
    """Build a fresh orchard task lazily from its registered name."""
    _bootstrap_task_definitions()
    return _build_task(task_name)
