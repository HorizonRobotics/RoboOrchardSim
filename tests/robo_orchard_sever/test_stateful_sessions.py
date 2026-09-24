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

"""Connection-owned policy state exercised through the existing real client."""

from __future__ import annotations
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Queue

import numpy as np
import pytest
import torch
from examples.server.stateful_policy_server_example import (
    SessionPolicy,
    SharedModel,
)
from robo_orchard_server.server import BasePolicy, PolicyWebsocketServer
from test_server_interop import (
    FixedOutputPolicy,
    PolicyClient,
    build_observation,
    cpu_client as cpu_client,
    receive_header,
    running_server as running_server,
)
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand


class CounterPolicy(FixedOutputPolicy):
    def __init__(self):
        super().__init__()
        self.count = 0

    def reset(self):
        self.count = 0

    def act(self, obs):
        self.count += 1
        self.value = self.count
        return super().act(obs)


def position(client, observation, instruction=None):
    return (
        client.request_action(observation, instruction=instruction)
        .values[0, 0]
        .item()
    )


@pytest.mark.parametrize("source", ["missing", "both", "noncallable"])
def test_server_constructor_invalid_policy_source_raises(source):
    options = {
        "missing": {},
        "both": {"policy": CounterPolicy(), "policy_factory": CounterPolicy},
        "noncallable": {"policy_factory": 1},
    }
    source_conflict = source in ("missing", "both")
    error_type = ValueError if source_conflict else TypeError
    message = "exactly one" if source_conflict else "callable"
    with pytest.raises(error_type, match=message):
        PolicyWebsocketServer(**options[source])


@pytest.mark.parametrize(
    "method", ["act", "act_sequence", "fallback", "chunk"]
)
@pytest.mark.parametrize("operation", ["reset", "disconnect", "reconnect"])
def test_server_sessions_interleaved_histories_remain_independent(
    running_server, operation, method
):
    observation, _ = build_observation()
    model = SharedModel(scale=2.0, offset=0.5)

    class SequenceSession(SessionPolicy):
        def act_sequence(self, obs):
            return [self.act(obs), self.act(obs)]

    class SingleStepSession(SessionPolicy):
        act_sequence = BasePolicy.act_sequence

    if method == "act_sequence":
        factory = SequenceSession
        continuous = [[2.5, 4.5], [6.5, 2.5], [4.5, 6.5], [2.5, 4.5]]
    elif method == "chunk":
        factory = SessionPolicy
        continuous = [[2.5, 4.5, 6.5]] * 4
    else:
        factory = SingleStepSession
        continuous = [[2.5], [4.5], [6.5], [2.5]]

    def request(client):
        if method == "act":
            return [position(client, observation)]
        return [
            action.values[0, 0].item()
            for action in client.request_action_sequence(observation)
        ]

    with running_server(policy_factory=lambda: factory(model)) as port:
        with (
            PolicyClient(host="127.0.0.1", port=port) as first,
            PolicyClient(host="127.0.0.1", port=port) as second,
        ):
            first_values = [request(first)]
            second_values = [request(second)]
            first_values.append(request(first))
            second_values.append(request(second))
            if operation == "reset":
                first.reset_remote_policy()
            else:
                first.close()
            second_values.append(request(second))
            if operation != "disconnect":
                # PolicyClient reconnects on its next request after close().
                first_values.append(request(first))
            second_values.append(request(second))
    assert (first_values, second_values) == (
        continuous[:2]
        if operation == "disconnect"
        else [*continuous[:2], continuous[0]],
        continuous,
    )


def test_stateful_example_action_requests_consume_cached_chunk(
    running_server,
):
    observation, _ = build_observation()
    model = SharedModel(scale=2.0, offset=0.5)
    with running_server(policy_factory=lambda: SessionPolicy(model)) as port:
        with PolicyClient(host="127.0.0.1", port=port) as client:
            values = [position(client, observation) for _ in range(4)]
    assert values == [2.5, 4.5, 6.5, 2.5]


def test_stateful_example_sequence_request_returns_entire_fresh_chunk(
    running_server,
):
    observation, _ = build_observation()
    model = SharedModel(scale=2.0, offset=0.5)
    with running_server(policy_factory=lambda: SessionPolicy(model)) as port:
        with PolicyClient(host="127.0.0.1", port=port) as client:
            position(client, observation)
            sequence = client.request_action_sequence(observation)
            following = position(client, observation)
    values = []
    for action in sequence:
        if not isinstance(action, UnifiedJointCommand):
            pytest.fail("Expected a named joint command")
        values.append(action.values[0, 0].item())
    assert (values, following) == (
        [2.5, 4.5, 6.5],
        2.5,
    )


def test_server_factory_initialized_session_is_not_reset(running_server):
    class InitializedPolicy(CounterPolicy):
        def __init__(self):
            super().__init__()
            self.count = 10

    observation, _ = build_observation()
    with running_server(policy_factory=InitializedPolicy) as port:
        with PolicyClient(host="127.0.0.1", port=port) as client:
            actual = position(client, observation)
    assert actual == 11


@pytest.mark.parametrize("method", ["act", "act_sequence"])
@pytest.mark.parametrize("failure", ["raise", "invalid", "encoding", "reset"])
def test_server_session_request_failure_allows_later_actions(
    running_server, method, failure
):
    observation, _ = build_observation()

    class UnencodableArray(np.ndarray):
        def tobytes(self, order="C"):
            raise RuntimeError("encoding failed")

    class FailingPolicy(CounterPolicy):
        def reset(self):
            self.count += 1
            raise RuntimeError("reset failed after state advance")

        def act(self, obs):
            action = super().act(obs)
            if obs["instruction"] == "fail":
                if failure == "raise":
                    raise RuntimeError("model failed after state advance")
                if failure == "invalid":
                    action["values"] = np.zeros((1, 1), np.float32)
                if failure == "encoding":
                    action["values"] = action["values"].view(UnencodableArray)
            return action

        def act_sequence(self, obs):
            return [self.act(obs)]

    def request(client, instruction):
        if method == "act":
            return client.request_action(
                observation, instruction=instruction
            ).values
        return client.request_action_sequence(
            observation, instruction=instruction
        )[0].values

    with running_server(policy_factory=FailingPolicy) as port:
        with (
            PolicyClient(host="127.0.0.1", port=port) as failed,
            PolicyClient(host="127.0.0.1", port=port) as survivor,
        ):
            with pytest.raises(RuntimeError, match="failed|shape"):
                if failure == "reset":
                    failed.reset_remote_policy()
                else:
                    request(failed, "fail")
            continued = request(failed, "continue")
            unaffected = request(survivor, "continue")
    # Failure does not implicitly reset or retry the policy, or block it.
    torch.testing.assert_close(
        [continued, unaffected], [torch.full((2, 9), 2.0), torch.ones(2, 9)]
    )


@pytest.mark.parametrize("transport", ["raw", "policy"])
@pytest.mark.parametrize(
    "failure", ["factory", "interface", "awaitable", "signature"]
)
def test_server_factory_failure_closes_only_failed_connection(
    running_server, failure, transport
):
    closed = Queue()
    created = 0

    class InvalidPolicy:
        def close(self):
            closed.put("invalid")

    async def deferred():
        raise AssertionError("Server must not execute asynchronous factories")

    def invalid_act(self):
        raise AssertionError("Server must not call an incompatible method")

    invalid_signature = type(
        "InvalidSignaturePolicy",
        (CounterPolicy,),
        {"act": invalid_act, "close": InvalidPolicy.close},
    )

    def factory():
        nonlocal created
        created += 1
        if created != 2:
            return CounterPolicy()
        if failure == "factory":
            raise RuntimeError("factory failed")
        if failure == "awaitable":
            return deferred()
        if failure == "signature":
            return invalid_signature()
        return InvalidPolicy()

    observation, _ = build_observation()
    with running_server(policy_factory=factory) as port:
        with PolicyClient(host="127.0.0.1", port=port) as survivor:
            values = [position(survivor, observation)]
            if transport == "raw":
                with connect(f"ws://127.0.0.1:{port}") as failed:
                    assert "error" in receive_header(failed)
                    with pytest.raises(ConnectionClosed):
                        failed.recv(timeout=5)
            else:
                with PolicyClient(host="127.0.0.1", port=port) as rejected:
                    with pytest.raises(
                        RuntimeError, match="factory|policy|InvalidSignature"
                    ):
                        rejected.request_action(observation)
            values.append(position(survivor, observation))
            with PolicyClient(host="127.0.0.1", port=port) as newcomer:
                new = position(newcomer, observation)
    assert (values, new) == ([1, 2], 1)
    if failure in ("interface", "signature"):
        assert closed.get(timeout=5) == "invalid"


def test_server_factory_active_policy_reuse_preserves_original_owner(
    running_server,
):
    closed = Queue()

    class TrackedPolicy(CounterPolicy):
        def close(self):
            closed.put("closed")

    policy = TrackedPolicy()
    observation, _ = build_observation()
    with running_server(policy_factory=lambda: policy) as port:
        with PolicyClient(host="127.0.0.1", port=port) as original:
            position(original, observation)
            with connect(f"ws://127.0.0.1:{port}") as duplicate:
                assert "already in use" in receive_header(duplicate)["error"]
                with pytest.raises(ConnectionClosed):
                    duplicate.recv(timeout=5)
            assert position(original, observation) == 2
        assert closed.get(timeout=5) == "closed"
    assert closed.empty()


@pytest.mark.parametrize("cleanup", ["normal", "raise", "return", "awaitable"])
def test_server_disconnect_closes_session_once_and_preserves_other_history(
    running_server, cleanup, caplog
):
    closed = Queue()
    next_id = 0

    async def deferred():
        raise AssertionError("Server must not execute asynchronous cleanup")

    class TrackedPolicy(CounterPolicy):
        def __init__(self):
            nonlocal next_id
            super().__init__()
            self.session_id = next_id
            next_id += 1

        def close(self):
            closed.put(self.session_id)
            if cleanup == "raise":
                raise RuntimeError("cleanup failed")
            if cleanup == "return":
                return 1
            if cleanup == "awaitable":
                return deferred()

    observation, _ = build_observation()
    with running_server(policy_factory=TrackedPolicy) as port:
        with PolicyClient(host="127.0.0.1", port=port) as survivor:
            values = [position(survivor, observation)]
            with PolicyClient(host="127.0.0.1", port=port) as first:
                position(first, observation)
            closed_ids = [closed.get(timeout=5)]
            values.append(position(survivor, observation))
        closed_ids.append(closed.get(timeout=5))
    assert (closed_ids, values, closed.empty()) == ([1, 0], [1, 2], True)
    if cleanup in ("return", "awaitable"):
        assert "TrackedPolicy.close() must return" in caplog.text


@pytest.mark.parametrize("operation", ["factory", "close"])
def test_server_session_lifecycle_during_inference_runs_serially(
    running_server, operation
):
    active = threading.Lock()
    entered = threading.Event()
    release = threading.Event()
    overlap = threading.Event()
    completed = Queue()
    connected = Queue()

    @contextmanager
    def model_access():
        if not active.acquire(blocking=False):
            overlap.set()
            raise RuntimeError("Overlapping model access")
        try:
            yield
        finally:
            active.release()

    class GuardedPolicy(CounterPolicy):
        def __init__(self):
            with model_access():
                super().__init__()
                completed.put("factory")

        def act(self, obs):
            with model_access():
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("Model was not released")
                return super().act(obs)

        def close(self):
            with model_access():
                completed.put("close")

    observation, _ = build_observation()
    with running_server(
        policy_factory=GuardedPolicy, connected=connected
    ) as port:
        with (
            PolicyClient(host="127.0.0.1", port=port) as first,
            PolicyClient(host="127.0.0.1", port=port) as second,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            first.reset_remote_policy()
            second.reset_remote_policy()
            for _ in range(2):
                connected.get(timeout=5)
                completed.get(timeout=5)
            pending = pool.submit(position, first, observation)
            try:
                assert entered.wait(timeout=5), "Model call never started"
                if operation == "factory":
                    with connect(f"ws://127.0.0.1:{port}"):
                        connected.get(timeout=5)
                        overlap.wait(timeout=0.1)
                else:
                    second.close()
                    overlap.wait(timeout=0.1)
            finally:
                release.set()
            assert pending.result(timeout=5) == 1
            assert completed.get(timeout=5) == operation
    assert not overlap.is_set(), "Lifecycle hook overlapped model inference"
