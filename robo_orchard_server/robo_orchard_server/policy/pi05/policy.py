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

"""Session-local caches and sampling RNG, sharing an official OpenPI model."""

import copy

from robo_orchard_server.policy.pi05.adapter import Pi05Adapter
from robo_orchard_server.server.policy import BasePolicy
from robo_orchard_server.server.types import JointAction, Observation


class Pi05Policy(BasePolicy):
    """Call official inference with the same initial-noise RNG convention."""

    def __init__(self, pipeline, adapter: Pi05Adapter, *, steps=None):
        if steps is not None and steps < 1:
            raise ValueError("Execution steps must be positive")
        # Official Policy holds immutable JAX RNG keys and reassigns its key
        # on infer(). A shallow session copy shares weights/compiled kernels,
        # but keeps that reassignment local to the session.
        self._layout = None
        self._template = pipeline
        self.pipeline = copy.copy(pipeline)
        self.steps = steps
        self.adapter = adapter
        self._cached_actions = []

    def new_session(self):
        """Share weights, but never cached actions or random state."""
        return Pi05Policy(self._template, self.adapter, steps=self.steps)

    def reset(self) -> None:
        """Discard pending actions; retain RNG progression like OpenPI."""
        self._cached_actions = []

    def close(self) -> None:
        """Release session caches without unloading shared weights."""
        self.reset()

    def act(self, obs: Observation) -> JointAction:
        self._bind_layout(obs)
        if not self._cached_actions:
            self._cached_actions = self.act_sequence(obs)
        return self._cached_actions.pop(0)

    def act_sequence(self, obs: Observation) -> list[JointAction]:
        """Infer one full chunk, returning the requested execution prefix."""
        self._bind_layout(obs)
        model_input = self.adapter.build_model_input(obs)
        output = self.pipeline.infer(model_input)
        self._cached_actions = []
        return self.adapter.build_action_sequence(
            output["actions"], obs, self.steps
        )

    def _bind_layout(self, obs: Observation) -> None:
        self.adapter.layout(obs)
        if self._layout is None:
            self._layout = copy.deepcopy(obs["action_layout"])
        elif self._layout != obs["action_layout"]:
            raise ValueError("Policy session action layout changed")
