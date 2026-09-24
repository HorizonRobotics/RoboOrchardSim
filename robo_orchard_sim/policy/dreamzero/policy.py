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

"""DreamZero remote policy."""

from typing import Any

import gymnasium as gym
from pydantic import Field
from robo_orchard_core.policy.base import PolicyConfig
from robo_orchard_core.utils.config import ClassType

from robo_orchard_sim.policy.dreamzero.client import DreamZeroClient
from robo_orchard_sim.policy.motus.adapter import DEFAULT_CAMERA_MAPPING
from robo_orchard_sim.policy.motus.policy import MotusPolicy


class DreamZeroPolicy(MotusPolicy):
    """DreamZero policy using canonical Motus-compatible adaptation."""

    cfg: "DreamZeroPolicyCfg"

    def __init__(
        self,
        cfg: "DreamZeroPolicyCfg",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,  # type: ignore[arg-type]
            observation_space=observation_space,
            action_space=action_space,
            client=client
            or DreamZeroClient(
                host=cfg.host,
                port=cfg.port,
            ),
        )


class DreamZeroPolicyCfg(PolicyConfig[DreamZeroPolicy]):
    """Configuration for :class:`DreamZeroPolicy`."""

    class_type: ClassType[DreamZeroPolicy] = DreamZeroPolicy

    host: str = "localhost"
    port: int = 8000
    logging_tag: str | None = None
    valid_action_step: int | None = None
    camera_mapping: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_CAMERA_MAPPING)
    )


__all__ = ["DreamZeroPolicy", "DreamZeroPolicyCfg"]
