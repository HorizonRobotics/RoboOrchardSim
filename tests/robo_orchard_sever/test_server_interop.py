# ruff: noqa: E402
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

"""Real transport tests for the lightweight package and the existing client."""

from __future__ import annotations
import importlib.util
import json
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, serve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robo_orchard_server.policy.dummy.main import (
    DummyPolicy as FixedOutputPolicy,
)
from robo_orchard_server.server import BasePolicy, PolicyWebsocketServer

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.policy.canonicalizer import canonicalize_observations
from robo_orchard_sim.policy.server import (
    PolicyClient,
    ServerPolicy,
    ServerPolicyCfg,
)


def load_source(name, path):
    # Schema imports normally initialize Isaac through the package __init__.
    # Load the actual schema file directly; the network tests need no engine.
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cpu_client(monkeypatch):
    # Hardware boundary: exercise protocol correctness without a GPU context.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def running_server():
    @contextmanager
    def start(
        policy=None, received=None, *, policy_factory=None, connected=None
    ):
        class ObservedConnection(ServerConnection):
            def recv(self, *args, **kwargs):
                message = super().recv(*args, **kwargs)
                if received is not None:
                    received.put(message)
                return message

        service = (
            PolicyWebsocketServer(policy_factory=policy_factory)
            if policy_factory is not None
            else PolicyWebsocketServer(policy)
        )

        def handle(websocket):
            if connected is not None:
                connected.put(websocket.id)
            service.handle_client(websocket)

        with serve(
            handle,
            "127.0.0.1",
            0,
            max_size=200 * 1024 * 1024,
            compression=None,
            create_connection=ObservedConnection,
        ) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield server.socket.getsockname()[1]
            finally:
                server.shutdown()
                thread.join(timeout=5)

    return start


def build_observation(embodiment="franka_panda", batch=2):
    schema_path = (
        REPO_ROOT
        / "robo_orchard_sim/orchard_env/embodiments"
        / embodiment
        / "schema.py"
    )
    module = load_source(f"{embodiment}_schema", schema_path)
    if embodiment == "franka_panda":
        schema = module.build_franka_panda_policy_binding_schema(embodiment)
    else:
        schema = module.build_dualarm_piperx_policy_binding_schema(embodiment)
    raw = {"/camera": {}, "/robot": {}}
    expected_cameras = {}
    for index, (slot, binding) in enumerate(schema.camera_slots.items()):
        rgb = (
            torch.arange(batch * 3 * 4 * 3, dtype=torch.uint8)
            .reshape(batch, 3, 4, 3)
            .transpose(1, 2)
        )
        depth = torch.full((batch, 4, 3, 1), index + 0.25)
        intrinsic = torch.eye(3, dtype=torch.float64).repeat(batch, 1, 1)
        xyz = torch.arange(batch * 3, dtype=torch.float32).reshape(batch, 3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1)
        raw["/camera"][binding.obs_term] = {
            "rgb": SimpleNamespace(
                sensor_data=rgb,
                intrinsic_matrices=intrinsic,
                pose=SimpleNamespace(xyz=xyz, quat=quat),
            ),
            "depth": SimpleNamespace(sensor_data=depth),
        }
        expected_cameras[slot] = {
            "rgb": rgb.numpy(),
            "depth": depth.numpy(),
            "intrinsic_matrices": intrinsic.numpy(),
            "pose": {"xyz": xyz.numpy(), "quat": quat.numpy()},
        }
    joint_count = 9 if embodiment == "franka_panda" else 8
    expected_manipulators = {}
    for index, (slot, binding) in enumerate(schema.manipulator_slots.items()):
        position = (
            torch.arange(batch * joint_count, dtype=torch.float32)
            .reshape(batch, joint_count)
            .div(10)
            .add(index)
        )
        raw["/robot"][binding.joint_position_obs_key] = position
        # Exercise optional bindings in addition to the actual robot schema.
        binding.gripper_position_obs_key = f"{slot}_gripper"
        binding.ee_pose_obs_key = f"{slot}_ee"
        binding.base_pose_obs_key = f"{slot}_base"
        gripper = position[:, -2:]
        ee = torch.arange(batch * 7, dtype=torch.float64).reshape(batch, 1, 7)
        base = torch.zeros(batch, 7)
        raw["/robot"].update(
            {
                binding.gripper_position_obs_key: gripper,
                binding.ee_pose_obs_key: ee,
                binding.base_pose_obs_key: base,
            }
        )
        expected_manipulators[slot] = {
            "joint_position": position.numpy(),
            "gripper_position": gripper.numpy(),
            "ee_pose": ee.numpy(),
            "base_pose": base.numpy(),
        }
    canonical = canonicalize_observations(
        observations=raw, instruction="拿起苹果", schema=schema
    )
    if canonical.action_layout is None:
        raise AssertionError("Canonicalizer must supply an action layout")
    return canonical, {
        "instruction": canonical.instruction,
        "cameras": expected_cameras,
        "manipulators": expected_manipulators,
        "action_layout": canonical.action_layout.to_payload(),
    }


def assert_same_tree(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_same_tree(actual[key], value)
    else:
        assert actual == expected


@pytest.mark.parametrize("embodiment", ["franka_panda", "dualarm_piperx"])
@pytest.mark.parametrize("batch", [1, 2])
def test_server_transport_canonical_input_preserves_arrays_and_joint_names(
    running_server, embodiment, batch
):
    observation, expected = build_observation(embodiment, batch)
    expected["instruction"] = "updated instruction"
    names = [
        name
        for spec in expected["action_layout"]["manipulators"].values()
        for name in spec["arm_joint_names"] + spec["gripper_joint_names"]
    ]
    values = np.arange(batch * len(names), dtype=np.float32).reshape(batch, -1)

    class Adapter(BasePolicy):
        def reset(self):
            pass

        def act(self, obs):
            assert_same_tree(obs, expected)
            # Reordered, noncontiguous output must retain column associations.
            return {"joint_names": names[::-1], "values": values[:, ::-1]}

    with running_server(Adapter()) as port:
        with PolicyClient(host="127.0.0.1", port=port) as client:
            action = client.request_action(
                observation, instruction="updated instruction"
            )
    assert isinstance(action, UnifiedJointCommand)
    torch.testing.assert_close(action.select(*names), torch.from_numpy(values))


def test_server_sequence_cached_horizon_and_reset_restart_actions(
    running_server,
):
    observation, _ = build_observation()

    class SequencePolicy(FixedOutputPolicy):
        def __init__(self):
            super().__init__()
            self.offset = 0

        def reset(self):
            self.offset = 0

        def act_sequence(self, obs):
            sequence = []
            for _ in range(3):
                self.offset += 1
                self.value = self.offset
                sequence.append(self.act(obs))
            return sequence

    def first_position(policy, observation):
        action = policy.act(observation)
        if not isinstance(action, UnifiedJointCommand):
            raise AssertionError("Expected a named joint command")
        return action.values[0, 0].item()

    with running_server(SequencePolicy()) as port:
        policy = ServerPolicy(ServerPolicyCfg(host="127.0.0.1", port=port))
        try:
            policy.reset()
            values = [first_position(policy, observation) for _ in range(4)]
            policy.reset()  # Discard the remaining two cached actions.
            values.append(first_position(policy, observation))
        finally:
            policy.close()
    assert values == [1.0, 2.0, 3.0, 4.0, 1.0]


@pytest.mark.parametrize(
    "invalid, error",
    [
        ({}, "joint_names"),
        ({"joint_names": "panda_joint1"}, "joint_names"),
        ({"joint_names": []}, "every layout joint"),
        ({"joint_names": ["panda_joint1"] * 9}, "every layout joint"),
        ({"joint_names": ["unknown"]}, "every layout joint"),
        ({"values": np.zeros((2, 8), np.float32)}, "shape"),
        ({"values": np.zeros((1, 9), np.float32)}, "shape"),
        ({"values": np.zeros((2, 1, 9), np.float32)}, "shape"),
        ({"values": np.zeros((2, 9), np.float64)}, "float32"),
        ({"values": [[0.0] * 9] * 2}, "float32"),
        ({"values": np.full((2, 9), np.nan, np.float32)}, "finite"),
        ({"values": np.full((2, 9), np.inf, np.float32)}, "finite"),
        ([], "nonempty list"),
        (({},), "nonempty list"),
        ([None], "dictionary"),
    ],
    ids=[
        "missing-names",
        "names-string",
        "missing-joints",
        "duplicate-joints",
        "unknown-joint",
        "columns",
        "batch",
        "rank",
        "dtype",
        "list-values",
        "nan",
        "infinity",
        "empty-sequence",
        "tuple-sequence",
        "invalid-sequence-item",
    ],
)
def test_server_action_invalid_output_reports_error_and_recovers(
    running_server, invalid, error
):
    observation, _ = build_observation()

    class InvalidPolicy(FixedOutputPolicy):
        def __init__(self):
            super().__init__()
            self.invalid = True

        def reset(self):
            self.invalid = False

        def act_sequence(self, obs):
            valid = self.act(obs)
            if not self.invalid:
                return [valid]
            if isinstance(invalid, (list, tuple)):
                return invalid
            if invalid == {}:
                return [invalid]
            return [{**valid, **invalid}]

    with running_server(InvalidPolicy()) as port:
        with PolicyClient(host="127.0.0.1", port=port) as client:
            with pytest.raises(RuntimeError, match=error):
                client.request_action_sequence(observation)
            client.reset_remote_policy()
            recovered = client.request_action_sequence(observation)
    torch.testing.assert_close(recovered[0].values, torch.zeros(2, 9))


@pytest.mark.parametrize("method", ["act", "reset"])
def test_server_policy_exception_reports_error_and_recovers(
    running_server, method
):
    observation, _ = build_observation()

    class RaisingPolicy(FixedOutputPolicy):
        def __init__(self):
            super().__init__()
            self.should_raise = True

        def act(self, obs):
            if method == "act" and self.should_raise:
                self.should_raise = False
                raise RuntimeError("model failed")
            return super().act(obs)

        def reset(self):
            if method == "reset" and self.should_raise:
                self.should_raise = False
                raise RuntimeError("reset failed")

    with running_server(RaisingPolicy()) as port:
        with PolicyClient(host="127.0.0.1", port=port) as client:
            with pytest.raises(RuntimeError, match="failed"):
                if method == "act":
                    client.request_action(observation)
                else:
                    client.reset_remote_policy()
            recovered = client.request_action(observation)
    torch.testing.assert_close(recovered.values, torch.zeros(2, 9))


def receive_header(client):
    response = client.recv(timeout=5)
    if not isinstance(response, bytes):
        raise AssertionError("Expected a binary protocol response")
    size = struct.unpack("!Q", response[:8])[0]
    return json.loads(response[8 : 8 + size])


def wire_message(payload):
    header = json.dumps(payload).encode()
    return struct.pack("!Q", len(header)) + header


@pytest.mark.parametrize(
    "message",
    [
        "text frame",
        b"short",
        struct.pack("!Q", 100) + b"{}",
        wire_message({}) + b"short",
        wire_message({"type": "unknown"}),
        wire_message({"type": "act", "obs_data": {"format": "full"}}),
        wire_message({"type": "act", "obs_data": {"format": "canonical"}}),
        wire_message({"__bytes_idx__": 0}),
    ],
)
def test_server_request_malformed_message_reports_error_and_recovers(
    running_server, message
):
    with running_server(FixedOutputPolicy()) as port:
        with connect(f"ws://127.0.0.1:{port}") as client:
            client.send(message)
            assert "error" in receive_header(client)
            client.send(wire_message({"type": "reset"}))
            assert receive_header(client)["ok"] is True


class ObservationPolicy(FixedOutputPolicy):
    def act(self, obs):
        action = super().act(obs)
        action["values"] = obs["manipulators"]["single_arm"]["joint_position"]
        return action


def test_server_sessions_concurrent_observations_return_matching_actions(
    running_server,
):
    observations = [build_observation()[0] for _ in range(2)]
    observations[1].manipulators["single_arm"]["joint_position"] += 5
    start = threading.Barrier(2)

    def request(client, observation):
        start.wait(timeout=5)
        return client.request_action(observation).values

    with running_server(ObservationPolicy()) as port:
        with (
            PolicyClient(host="127.0.0.1", port=port) as first,
            PolicyClient(host="127.0.0.1", port=port) as second,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            futures = [
                pool.submit(request, client, obs)
                for client, obs in zip(
                    (first, second), observations, strict=True
                )
            ]
            values = [future.result(timeout=5) for future in futures]
    torch.testing.assert_close(
        values,
        [
            obs.manipulators["single_arm"]["joint_position"]
            for obs in observations
        ],
    )


def request_policy(client, method, observation):
    if method == "reset":
        return client.reset_remote_policy()
    if method == "act_sequence":
        return client.request_action_sequence(observation)[0].values
    return client.request_action(observation).values


@pytest.mark.parametrize(
    "method", ["act", "act_sequence", "reset", "fallback"]
)
def test_server_policy_awaitable_result_reports_method_and_recovers(
    running_server, method
):
    observation, _ = build_observation()

    async def deferred():
        raise AssertionError("Server must not execute asynchronous results")

    def result(self, *args):
        if self.fail:
            self.fail = False
            return deferred()
        if method == "reset":
            return None
        action = FixedOutputPolicy.act(self, *args)
        return [action] if method == "act_sequence" else action

    target = "act" if method == "fallback" else method
    request = "act_sequence" if method == "fallback" else method
    policy_type = type(
        "DeferredPolicy", (FixedOutputPolicy,), {target: result, "fail": True}
    )
    with running_server(policy_type()) as port:
        with PolicyClient(host="127.0.0.1", port=port) as client:
            with pytest.raises(RuntimeError, match="DeferredPolicy.*sync"):
                request_policy(client, request, observation)
            request_policy(client, request, observation)
            recovered = client.request_action(observation)
    torch.testing.assert_close(recovered.values, torch.zeros(2, 9))


def test_server_reset_non_none_result_reports_error_and_recovers(
    running_server,
):
    observation, _ = build_observation()

    class InvalidResetPolicy(FixedOutputPolicy):
        def reset(self):
            return 1

    with running_server(InvalidResetPolicy()) as port:
        with PolicyClient(host="127.0.0.1", port=port) as client:
            with pytest.raises(RuntimeError, match="reset.*return None"):
                client.reset_remote_policy()
            recovered = client.request_action(observation)
    torch.testing.assert_close(recovered.values, torch.zeros(2, 9))


@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("first_method", ["act", "act_sequence", "reset"])
@pytest.mark.parametrize("second_method", ["act", "act_sequence", "reset"])
@pytest.mark.parametrize(
    "has_sequence", [False, True], ids=["fallback", "sequence"]
)
def test_server_policy_concurrent_methods_run_serially(
    running_server, first_method, second_method, has_sequence, stateful
):
    observation, _ = build_observation()
    entered = threading.Event()
    release = threading.Event()
    overlap = threading.Event()
    active = threading.Lock()
    received = Queue()

    @contextmanager
    def guard():
        if not active.acquire(blocking=False):
            overlap.set()
            raise RuntimeError("Concurrent policy call")
        try:
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Policy was not released")
            yield
        finally:
            active.release()

    class GuardedPolicy(FixedOutputPolicy):
        def act(self, obs):
            with guard():
                return super().act(obs)

        def reset(self):
            with guard():
                pass

    class SequencePolicy(GuardedPolicy):
        def act_sequence(self, obs):
            with guard():
                return [FixedOutputPolicy.act(self, obs)]

    factory = SequencePolicy if has_sequence else GuardedPolicy
    options = (
        {"policy_factory": factory} if stateful else {"policy": factory()}
    )
    with running_server(received=received, **options) as port:
        with (
            PolicyClient(host="127.0.0.1", port=port) as first,
            PolicyClient(host="127.0.0.1", port=port) as second,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            release.set()
            first.reset_remote_policy()
            second.reset_remote_policy()
            release.clear()
            entered.clear()
            received.get(timeout=5)
            received.get(timeout=5)
            pending = pool.submit(
                request_policy, first, first_method, observation
            )
            try:
                assert entered.wait(timeout=5), (
                    "First call never entered policy"
                )
                received.get(timeout=5)
                competing = pool.submit(
                    request_policy, second, second_method, observation
                )
                received.get(timeout=5)
                # Both messages reached the transport. Give a competing call
                # a bounded opportunity to expose reentry before releasing.
                overlap.wait(timeout=0.1)
            finally:
                release.set()
            outputs = [pending.result(timeout=5), competing.result(timeout=5)]
    torch.testing.assert_close(
        outputs,
        [
            None if method == "reset" else torch.zeros(2, 9)
            for method in (first_method, second_method)
        ],
    )


@pytest.mark.parametrize("failure", ["disconnect", "exception"])
def test_server_sessions_client_failure_preserves_other_and_new_clients(
    running_server, failure
):
    observation, _ = build_observation()

    reset_called = threading.Event()

    class IsolatedPolicy(ObservationPolicy):
        def reset(self):
            reset_called.set()

        def close(self):
            reset_called.set()

        def act(self, obs):
            if reset_called.is_set():
                raise RuntimeError(
                    "Disconnect reset or closed the shared policy"
                )
            if obs["instruction"] == "fail":
                raise RuntimeError("Requested model failure")
            return super().act(obs)

    with running_server(IsolatedPolicy()) as port:
        with PolicyClient(host="127.0.0.1", port=port) as survivor:
            survivor.request_action(observation)
            with PolicyClient(host="127.0.0.1", port=port) as first:
                if failure == "exception":
                    with pytest.raises(
                        RuntimeError, match="Requested model failure"
                    ):
                        first.request_action(observation, instruction="fail")
                else:
                    first.request_action(observation)
            continued = survivor.request_action(observation).values
            with PolicyClient(host="127.0.0.1", port=port) as newcomer:
                new = newcomer.request_action(observation).values
    expected = observation.manipulators["single_arm"]["joint_position"]
    torch.testing.assert_close([continued, new], [expected, expected])


@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("method", ["act", "act_sequence"])
def test_server_encoding_reused_output_preserves_each_response(
    running_server, method, stateful
):
    observations = [build_observation()[0] for _ in range(2)]
    observations[1].manipulators["single_arm"]["joint_position"] += 5
    encoding = threading.Event()
    release = threading.Event()
    overwritten = threading.Event()
    received = Queue()

    class GatedArray(np.ndarray):
        def tobytes(self, order="C"):
            encoding.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Encoding was not released")
            return super().tobytes(order=order)

    buffer = np.zeros((2, 9), np.float32).view(GatedArray)

    class ReusingPolicy(ObservationPolicy):
        def __init__(self):
            super().__init__()
            self.buffer = buffer

        def act(self, obs):
            action = super().act(obs)
            if encoding.is_set():
                overwritten.set()
            self.buffer[:] = action["values"]
            return {**action, "values": self.buffer}

        def act_sequence(self, obs):
            return [self.act(obs)]

    options = (
        {"policy_factory": ReusingPolicy}
        if stateful
        else {"policy": ReusingPolicy()}
    )
    with running_server(received=received, **options) as port:
        with (
            PolicyClient(host="127.0.0.1", port=port) as first,
            PolicyClient(host="127.0.0.1", port=port) as second,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            first.reset_remote_policy()
            second.reset_remote_policy()
            received.get(timeout=5)
            received.get(timeout=5)
            pending = pool.submit(
                request_policy, first, method, observations[0]
            )
            try:
                assert encoding.wait(timeout=5), "First action never encoded"
                received.get(timeout=5)
                competing = pool.submit(
                    request_policy, second, method, observations[1]
                )
                received.get(timeout=5)
                overwritten.wait(timeout=0.1)
            finally:
                release.set()
            values = [pending.result(timeout=5), competing.result(timeout=5)]
    torch.testing.assert_close(
        values,
        [
            obs.manipulators["single_arm"]["joint_position"]
            for obs in observations
        ],
    )


def test_server_policy_variable_horizons_execute_all_actions_in_order(
    running_server,
):
    observation, _ = build_observation()

    class VariablePolicy(FixedOutputPolicy):
        def __init__(self):
            super().__init__()
            self.chunk = 0
            self.next_value = 0

        def act_sequence(self, obs):
            length = (1, 3, 2)[self.chunk % 3]
            self.chunk += 1
            actions = []
            for _ in range(length):
                self.next_value += 1
                self.value = self.next_value
                actions.append(self.act(obs))
            return actions

    with running_server(VariablePolicy()) as port:
        policy = ServerPolicy(ServerPolicyCfg(host="127.0.0.1", port=port))
        try:
            values = [
                policy.act(observation).values[0, 0].item() for _ in range(9)
            ]
        finally:
            policy.close()
    assert values == list(range(1, 10))
