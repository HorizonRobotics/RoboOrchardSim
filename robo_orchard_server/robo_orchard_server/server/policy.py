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

"""Public interfaces for synchronous policies and session factories."""

from abc import ABC, abstractmethod
from typing import Protocol

from robo_orchard_server.server.types import JointAction, Observation


class BasePolicy(ABC):
    """Required base class for synchronous model adapters.

    Implement act() and reset(). Overriding act_sequence() to return multiple
    future steps is recommended: clients can execute the returned chunk before
    requesting another, reducing observation transfers and network round trips.
    The default single-step sequence provides compatibility, not that saving.

    Keep session history separate from shared model weights. Only sessions
    created by a policy factory are closed by the server. Constructors, model
    loading, configuration and device management belong to the adapter.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset this session when requested by the client.

        Clear episode history and pending actions, retaining model weights.
        Stateless adapters must explicitly implement a no-op. Shared policies
        must not change other clients' subsequent results.
        """

    @abstractmethod
    def act(self, obs: Observation) -> JointAction:
        """Return named physical joint targets for the current observation.

        Values must be finite float32 arrays [B, J], covering all layout
        joints exactly once and matching the observation's batch size.
        """

    def act_sequence(self, obs: Observation) -> list[JointAction]:
        """Return actions in time order; override for efficient chunked use.

        Return a nonempty list of valid JointAction objects. Multi-step model
        predictions reduce requests when the client executes the chunk before
        requesting actions from fresh observations. Longer chunks delay
        feedback; choose a horizon appropriate to the model and task.

        The default wraps one act() result and does not reduce request counts.
        """
        return [self.act(obs)]

    def close(self) -> None:
        """Release this session's resources without unloading shared weights.

        The default is a no-op and does not call reset(). Override when the
        session holds resources or caches needing explicit cleanup.
        """
        return None


class PolicyFactory(Protocol):
    """Callable creating a fresh, initialized policy for each Client session.

    Ordinary functions, classes and callable objects can satisfy this
    interface without inheriting it. Bind model/configuration dependencies
    in a closure or functools.partial so the server can call the factory
    without arguments. Shared weights must be separate from session state.
    """

    def __call__(self) -> BasePolicy:
        """Create a ready-to-use session synchronously, without arguments.

        The server does not reset the returned policy on creation.
        On session closure, it calls the policy's close() method.
        """
        ...
