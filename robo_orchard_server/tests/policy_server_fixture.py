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

"""Deterministic policies for session isolation and action chunk tests."""

from collections import deque

from robo_orchard_server.policy.dummy.main import DummyPolicy
from robo_orchard_server.server import (
    BasePolicy,
    JointAction,
    Observation,
)


class SharedModel:
    """Predict a deterministic three-step chunk without session state."""

    def __init__(self, scale: float = 1.0, offset: float = 0.0) -> None:
        self.scale = scale
        self.offset = offset

    def predict(self, obs: Observation) -> list[JointAction]:
        """Return named joint targets using the observation's robot layout."""
        return [
            DummyPolicy(self.scale * step + self.offset).act(obs)
            for step in (1, 2, 3)
        ]


class SessionPolicy(BasePolicy):
    """Keep each connection's pending actions separate from shared weights."""

    def __init__(self, model: SharedModel) -> None:
        self.model = model
        self.pending: deque[JointAction] = deque()

    def reset(self) -> None:
        """Discard pending actions without changing the shared model."""
        self.pending.clear()

    def act(self, obs: Observation) -> JointAction:
        """Consume one cached action, predicting a fresh chunk when empty."""
        if not self.pending:
            self.pending.extend(self.model.predict(obs))
        return self.pending.popleft()

    def act_sequence(self, obs: Observation) -> list[JointAction]:
        """Discard old pending actions and return a complete fresh chunk."""
        self.pending.clear()
        return self.model.predict(obs)
