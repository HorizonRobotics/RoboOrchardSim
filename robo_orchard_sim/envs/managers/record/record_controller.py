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

import torch

from robo_orchard_sim.utils.config import ClassConfig, ClassType_co


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


class RecordControllerCfg(ClassConfig[RecordController]):
    class_type: ClassType_co[RecordController] = RecordController

    def __call__(self, env: Any, **kwargs) -> RecordController:
        return self.class_type(self, env, **kwargs)


class NoOpRecordController(RecordController):
    pass


class NoOpRecordControllerCfg(RecordControllerCfg):
    class_type: ClassType_co[NoOpRecordController] = NoOpRecordController


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

    def on_post_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        self._started = False
        return RecordControlDecision()

    def on_post_step(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        if self._started:
            return RecordControlDecision()
        if self.env.step_count < self.cfg.min_wait_step:
            return RecordControlDecision()
        if self._scene_is_stationary() or (
            self.env.step_count >= self.cfg.max_wait_step
        ):
            self._started = True
            return RecordControlDecision(start=True)
        return RecordControlDecision()

    def on_pre_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision(stop=True)

    def _scene_is_stationary(self) -> bool:
        checked_assets = 0
        for asset in self._iter_scene_assets():
            root_state_w = getattr(asset.data, "root_state_w", None)
            if not isinstance(root_state_w, torch.Tensor):
                continue
            if root_state_w.shape[-1] != 13:
                continue

            lin_vel = root_state_w[..., 7:10]
            ang_vel = root_state_w[..., 10:13]
            lin_norm = torch.linalg.vector_norm(lin_vel, dim=-1)
            ang_norm = torch.linalg.vector_norm(ang_vel, dim=-1)
            checked_assets += 1
            if not torch.all(
                lin_norm < self.cfg.linear_velocity_threshold
            ) or not torch.all(ang_norm < self.cfg.angular_velocity_threshold):
                return False

        return checked_assets > 0

    def _iter_scene_assets(self) -> list[Any]:
        return [
            asset
            for key in self.env.scene.keys()
            if (asset := self.env.scene[key]) is not None
        ]


class StationaryEpisodeRecordControllerCfg(RecordControllerCfg):
    class_type: ClassType_co[StationaryEpisodeRecordController] = (
        StationaryEpisodeRecordController
    )
    linear_velocity_threshold: float = 0.02
    angular_velocity_threshold: float = 0.1
    min_wait_step: int = 10
    max_wait_step: int = 100
