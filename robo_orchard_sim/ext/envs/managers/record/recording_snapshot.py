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

"""CPU-only frame snapshots passed to the MCAP recording worker."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal


@dataclass(frozen=True, slots=True)
class RecordingFrameSnapshot:
    """One coherent robot, action, transform, and camera sample."""

    recording_step: int
    timestamp: datetime
    observation: dict[str, Any] | None
    observation_builder: Callable[[], dict[str, Any]] | None
    ready_events: tuple[Any, ...]
    build_duration_ms: float
    event: Literal["post_reset", "step"] = "step"
    step_user_data: dict[str, Any] | None = None
    episode_user_data: dict[str, Any] | None = None

    def build_observation(self) -> dict[str, Any]:
        """Return the worker-owned RoboOrchard observation dictionary."""
        if self.observation_builder is not None:
            return self.observation_builder()
        if self.observation is None:
            raise RuntimeError("Recording snapshot has no observation data.")
        return self.observation
