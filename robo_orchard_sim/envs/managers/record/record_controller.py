# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from robo_orchard_sim.utils.config import ClassConfig, ClassType_co
from robo_orchard_sim.utils.env_utils import SettleTracker


@dataclass
class RecordControlDecision:
    start: bool = False
    stop: bool = False


class RecordController:
    def __init__(self, cfg: "RecordControllerCfg", env: Any):
        self.cfg = cfg
        self.env = env

    def on_post_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision()

    def on_post_step(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision()

    def on_pre_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision(stop=True)

    def on_manual_start(self) -> None:
        return None


class RecordControllerCfg(ClassConfig[RecordController]):
    class_type: ClassType_co[RecordController] = RecordController

    def __call__(self, env: Any, **kwargs) -> RecordController:
        return self.class_type(self, env, **kwargs)


class NoOpRecordController(RecordController):
    pass


class NoOpRecordControllerCfg(RecordControllerCfg):
    class_type: ClassType_co[NoOpRecordController] = NoOpRecordController


class ManualRecordController(RecordController):
    """Controller for externally started episode recording."""

    pass


class ManualRecordControllerCfg(RecordControllerCfg):
    class_type: ClassType_co[ManualRecordController] = ManualRecordController


class EpisodeRecordController(RecordController):
    def on_post_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision(start=True)

    def on_pre_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision(stop=True)


class EpisodeRecordControllerCfg(RecordControllerCfg):
    class_type: ClassType_co[EpisodeRecordController] = EpisodeRecordController


class StationaryEpisodeRecordController(RecordController):
    def __init__(
        self,
        cfg: "StationaryEpisodeRecordControllerCfg",
        env: Any,
    ):
        super().__init__(cfg, env)
        self._started = False
        self._tracker = SettleTracker(
            streak=cfg.streak,
            rot_eps_deg=cfg.rot_eps_deg,
            pos_eps_m=cfg.pos_eps_m,
        )

    def on_post_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        self._started = False
        self._tracker.reset()
        return RecordControlDecision()

    def on_post_step(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        if self._started:
            return RecordControlDecision()
        settled = self._tracker.update(self.env.scene)
        if self.env.step_count < self.cfg.min_wait_step:
            return RecordControlDecision()
        if settled or self.env.step_count >= self.cfg.max_wait_step:
            self._started = True
            return RecordControlDecision(start=True)
        return RecordControlDecision()

    def on_pre_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision(stop=True)


class StationaryEpisodeRecordControllerCfg(RecordControllerCfg):
    class_type: ClassType_co[StationaryEpisodeRecordController] = (
        StationaryEpisodeRecordController
    )
    rot_eps_deg: float = 0.5
    pos_eps_m: float = 0.001
    min_wait_step: int = 50
    max_wait_step: int = 250
    streak: int = 50
