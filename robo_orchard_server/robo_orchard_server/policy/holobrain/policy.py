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

from __future__ import annotations
import logging
import os
from typing import Any, TypeAlias

import torch
from pydantic import BaseModel, field_validator

from robo_orchard_server.policy.holobrain.adapter import HolobrainAdapter
from robo_orchard_server.server.policy import BasePolicy
from robo_orchard_server.server.types import JointAction, Observation

logger = logging.getLogger(__name__)
HolobrainAction: TypeAlias = JointAction
_MODEL_DIR_ENV_VAR = "ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR"


class HolobrainPolicy(BasePolicy):
    """Holobrain inference with an independent action cache per connection."""

    cfg: "HolobrainPolicyCfg"

    def __init__(
        self,
        cfg: "HolobrainPolicyCfg",
        *,
        pipeline: Any | None = None,
    ) -> None:
        """Create policy state, optionally using an initialized eval pipeline.

        A supplied pipeline is shared by reference. Its caller must serialize
        inference across sessions and keep episode state outside the pipeline.
        """
        self.cfg = cfg
        self._adapter: HolobrainAdapter | None = None
        self._embodiment_type: str | None = cfg.embodiment_type
        if self._embodiment_type is not None:
            self._adapter = self._build_adapter(
                embodiment_type=self._embodiment_type
            )
        self._pipeline: Any = (
            self._load_pipeline(cfg) if pipeline is None else pipeline
        )
        if pipeline is None:
            self._pipeline.model.eval()
        self._cached_actions: list[HolobrainAction] = []
        self._cached_index = 0

    def new_session(self) -> "HolobrainPolicy":
        """Share the pipeline with fresh cache and embodiment binding."""
        return HolobrainPolicy(
            cfg=self.cfg.model_copy(deep=True), pipeline=self._pipeline
        )

    def reset(self) -> None:
        self._cached_actions = []
        self._cached_index = 0

    def close(self) -> None:
        """Release this session's cached actions, keeping the model alive."""
        self.reset()

    def act(self, obs: Observation) -> HolobrainAction:
        """Return one action, reusing the local cached horizon if possible."""
        self._validate_observation_batch(obs)
        # Reject a changed embodiment even while a previous chunk is cached.
        self._ensure_adapter(obs)
        if self._cached_index >= len(self._cached_actions):
            self._cached_actions = self._get_fresh_action_sequence(obs)
            self._cached_index = 0
        action = self._cached_actions[self._cached_index]
        self._cached_index += 1
        return action

    def act_sequence(self, obs: Observation) -> list[HolobrainAction]:
        """Return a freshly inferred action horizon for remote batch use."""
        sequence = self._get_fresh_action_sequence(obs)
        self._cached_actions = []
        self._cached_index = 0
        return sequence

    def _get_fresh_action_sequence(
        self, obs: Observation
    ) -> list[HolobrainAction]:
        self._validate_observation_batch(obs)
        self._refresh_action_cache(obs)
        return list(self._cached_actions)

    def _refresh_action_cache(self, obs: Observation) -> None:
        self._cached_actions = self._run_inference(obs)
        self._cached_index = 0

    def _run_inference(self, obs: Observation) -> list[HolobrainAction]:
        adapter = self._ensure_adapter(obs)
        model_input = adapter.build_model_input(obs)
        model_output = self._pipeline(model_input)
        return adapter.build_action_sequence(
            model_output,
            obs,
            valid_action_step=self.cfg.valid_action_step,
        )

    @staticmethod
    def _validate_observation_batch(obs: Observation) -> None:
        if not obs["manipulators"]:
            raise ValueError(
                "Holobrain observation must contain at least one manipulator"
            )
        for manipulator_obs in obs["manipulators"].values():
            positions = manipulator_obs["joint_position"]
            if positions.ndim != 2 or positions.shape[0] != 1:
                raise ValueError(
                    "HolobrainPolicy currently supports single environment "
                    "only; joint_position must have shape [1, N]"
                )

    @staticmethod
    def _build_adapter(*, embodiment_type: str) -> HolobrainAdapter:
        return HolobrainAdapter(embodiment_type=embodiment_type)

    def _ensure_adapter(self, obs: Observation) -> HolobrainAdapter:
        runtime_embodiment_type = obs["action_layout"]["embodiment_type"]
        if not isinstance(runtime_embodiment_type, str):
            raise ValueError(
                "Holobrain observation requires action_layout with "
                "embodiment_type"
            )
        if self._embodiment_type is None:
            adapter = self._build_adapter(
                embodiment_type=runtime_embodiment_type
            )
            self._embodiment_type = runtime_embodiment_type
            self._adapter = adapter
            return adapter
        if runtime_embodiment_type != self._embodiment_type:
            raise ValueError(
                "HolobrainPolicy is already bound to embodiment_type "
                f"{self._embodiment_type!r}, got "
                f"{runtime_embodiment_type!r}."
            )
        if self._adapter is None:
            self._adapter = self._build_adapter(
                embodiment_type=self._embodiment_type
            )
        return self._adapter

    @staticmethod
    def _load_pipeline(cfg: "HolobrainPolicyCfg") -> Any:
        model_dir = cfg.model_dir or os.getenv(_MODEL_DIR_ENV_VAR)
        if not model_dir:
            raise ValueError(
                "Holobrain model_dir must be set in the config or via "
                f"{_MODEL_DIR_ENV_VAR}"
            )

        from robo_orchard_lab.models.holobrain.pipeline import (
            HoloBrainInferencePipeline,
        )

        logger.info("Resolved Holobrain model_dir: %s", model_dir)
        device = cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return HoloBrainInferencePipeline.load_pipeline(
            directory=model_dir,
            device=device,
            load_weights=True,
            load_impl="native",
            inference_prefix=cfg.inference_prefix,
        )


class HolobrainPolicyCfg(BaseModel):
    """Config for :class:`HolobrainPolicy`."""

    model_dir: str | None = None
    logging_tag: str | None = None
    inference_prefix: str
    embodiment_type: str | None = None
    device: str | None = None
    valid_action_step: int | None = None

    def __call__(self) -> HolobrainPolicy:
        """Create a policy with an independent copy of this configuration."""
        return HolobrainPolicy(cfg=self.model_copy(deep=True))

    @field_validator("valid_action_step")
    @classmethod
    def _validate_valid_action_step(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("valid_action_step must be greater than 0")
        return value
