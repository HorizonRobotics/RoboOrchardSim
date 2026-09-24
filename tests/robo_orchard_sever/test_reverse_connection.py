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

"""Network tests for model-initiated connections; no Isaac runtime needed."""

import json
import os
import struct
import subprocess
import sys
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path

import pytest
from robo_orchard_server.policy.dummy.main import DummyPolicy
from robo_orchard_server.server import PolicyWebsocketServer
from robo_orchard_server.server.multiplex import PROTOCOL, unwrap, wrap
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.server import serve


def frame(header):
    data = json.dumps(header).encode()
    return struct.pack("!Q", len(data)) + data


def header(payload):
    size = struct.unpack_from("!Q", payload)[0]
    return json.loads(payload[8 : 8 + size])


def observation():
    request = frame(
        {
            "type": "act_sequence",
            "obs_data": {
                "format": "canonical",
                "cameras": {},
                "manipulators": {
                    "arm": {
                        "joint_position": {
                            "__tensor__": {
                                "dtype": "float32",
                                "shape": [1, 1],
                            },
                            "__tensor_idx__": 0,
                        }
                    }
                },
                "action_layout": {
                    "manipulator_order": ["arm"],
                    "manipulators": {
                        "arm": {
                            "arm_joint_names": ["joint"],
                            "gripper_joint_names": [],
                        }
                    },
                },
            },
        }
    )
    return request + struct.pack("!Qf", 4, 0.0)


@contextmanager
def supervisor_endpoint(handler):
    result = Future()

    def handle(ws):
        try:
            result.set_result(handler(ws))
        except BaseException as exc:
            result.set_exception(exc)

    with serve(handle, "127.0.0.1", 0, close_timeout=1) as endpoint:
        thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{endpoint.socket.getsockname()[1]}", result
        finally:
            endpoint.shutdown()
            thread.join(timeout=2)


def negotiate(ws):
    ws.send(json.dumps({"type": "policy_protocol", "protocol": PROTOCOL}))
    assert json.loads(ws.recv(timeout=2)) == {
        "type": "policy_protocol_ready",
        "protocol": PROTOCOL,
    }


def exchange(ws, routed=False):
    if routed:
        negotiate(ws)
    replies = []
    for index, request in enumerate((frame({"type": "reset"}), observation())):
        ws.send(wrap(request, "client", str(index)) if routed else request)
        response = ws.recv(timeout=2)
        replies.append(unwrap(response)[2] if routed else response)
    return replies


@pytest.mark.parametrize("reverse", [False, True])
def test_connection_modes_reset_and_inference_preserve_binary_protocol(
    reverse,
):
    closed = threading.Event()

    class Policy(DummyPolicy):
        def close(self):
            closed.set()

    server = PolicyWebsocketServer(policy_factory=lambda: Policy(0.25))
    if reverse:

        def supervisor(ws):
            ws.send(json.dumps({"type": "authenticate"}))
            identity = json.loads(ws.recv(timeout=2))
            if identity != {
                "type": "authenticate",
                "team_id": "t",
                "evaluation_id": "e",
            }:
                raise ValueError("Wrong identity")
            ws.send(json.dumps({"type": "authenticated"}))
            return exchange(ws, routed=True)

        with supervisor_endpoint(supervisor) as (uri, result):
            server.run_reverse(uri, team_id="t", evaluation_id="e")
            replies = result.result(timeout=2)
    else:
        with supervisor_endpoint(server.handle_client) as (uri, result):
            with connect(uri) as client:
                replies = exchange(client)
            result.result(timeout=2)
    action = header(replies[1])["actions"][0]["__joint_command__"]
    assert (
        header(replies[0])["ok"],
        action["joint_names"],
        struct.unpack("=f", replies[1][-4:]),
    ) == (True, ["joint"], (0.25,))
    assert closed.wait(2)


@pytest.mark.parametrize(
    "challenge", [b"binary", "[]", '{"type":"authenticated"}']
)
def test_reverse_invalid_challenge_rejects_before_policy_creation(
    tmp_path, challenge
):
    marker = tmp_path / "session"

    def factory():
        marker.touch()
        return DummyPolicy()

    def supervisor(ws):
        ws.send(challenge)
        with pytest.raises(ConnectionClosed):
            ws.recv(timeout=2)

    with supervisor_endpoint(supervisor) as (uri, result):
        with pytest.raises(ValueError):
            PolicyWebsocketServer(policy_factory=factory).run_reverse(
                uri, team_id="t", evaluation_id="e"
            )
        result.result(timeout=2)
    assert not marker.exists()


def test_reverse_rejected_identity_never_creates_policy(tmp_path):
    marker = tmp_path / "session"

    def factory():
        marker.touch()
        return DummyPolicy()

    def supervisor(ws):
        ws.send(json.dumps({"type": "authenticate"}))
        ws.recv(timeout=2)
        ws.close(code=1008, reason="Invalid evaluation identity")

    with supervisor_endpoint(supervisor) as (uri, result):
        with pytest.raises(ConnectionClosed):
            PolicyWebsocketServer(policy_factory=factory).run_reverse(
                uri, team_id="t", evaluation_id="e"
            )
        result.result(timeout=2)
    assert not marker.exists()


def test_dummy_reverse_cli_environment_identity_serves_requests_and_exits():
    def supervisor(ws):
        ws.send(json.dumps({"type": "authenticate"}))
        identity = json.loads(ws.recv(timeout=2))
        ws.send(json.dumps({"type": "authenticated"}))
        return identity, exchange(ws, routed=True)

    env = dict(os.environ, TEAM_ID="t", SESSION_ID="e")
    env["PYTHONPATH"] = str(
        Path(__file__).resolve().parents[2] / "robo_orchard_server"
    )
    with supervisor_endpoint(supervisor) as (uri, result):
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "robo_orchard_server.policy.dummy.main",
                "--supervisor-uri",
                uri,
                "--value",
                "0.5",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        identity, replies = result.result(timeout=2)
    assert process.returncode == 0, process.stderr
    assert (identity, struct.unpack("=f", replies[1][-4:])) == (
        {"type": "authenticate", "team_id": "t", "evaluation_id": "e"},
        (0.5,),
    )


def test_reverse_sessions_one_connection_preserves_order_and_isolation():
    closed = []

    class CounterPolicy(DummyPolicy):
        def reset(self):
            self.value = 0.0

        def act(self, obs):
            self.value += 1.0
            return super().act(obs)

        def close(self):
            closed.append(True)

    requests = [
        ("a", frame({"type": "reset"})),
        ("b", frame({"type": "reset"})),
        ("a", observation()),
        ("b", observation()),
        ("a", frame({"type": "reset"})),
        ("b", observation()),
        ("a", frame({"type": "close_session"})),
        ("b", frame({"type": "close_session"})),
    ]

    def supervisor(ws):
        ws.send(json.dumps({"type": "authenticate"}))
        ws.recv(timeout=2)
        ws.send(json.dumps({"type": "authenticated"}))
        negotiate(ws)
        # Enqueue requests from different Clients without waiting for replies.
        for index, (session, request) in enumerate(requests):
            ws.send(wrap(request, session, str(index)))
        return [unwrap(ws.recv(timeout=3)) for _ in requests]

    with supervisor_endpoint(supervisor) as (uri, result):
        PolicyWebsocketServer(policy_factory=CounterPolicy).run_reverse(
            uri,
            team_id="t",
            evaluation_id="e",
        )
        replies = result.result(timeout=2)
    assert [(s, r) for s, r, _ in replies] == [
        (s, str(i)) for i, (s, _) in enumerate(requests)
    ]
    assert [struct.unpack("=f", replies[i][2][-4:])[0] for i in (2, 3, 5)] == [
        1.0,
        1.0,
        2.0,
    ]
    assert closed == [True, True]


def test_reverse_disconnect_releases_all_logical_sessions():
    closed = []

    class Policy(DummyPolicy):
        def close(self):
            closed.append(True)

    def supervisor(ws):
        ws.send(json.dumps({"type": "authenticate"}))
        ws.recv(timeout=2)
        ws.send(json.dumps({"type": "authenticated"}))
        negotiate(ws)
        for session in ("a", "b", "c", "d"):
            ws.send(wrap(frame({"type": "reset"}), session, "reset"))
            ws.recv(timeout=2)

    with supervisor_endpoint(supervisor) as (uri, result):
        PolicyWebsocketServer(policy_factory=Policy).run_reverse(
            uri,
            team_id="t",
            evaluation_id="e",
        )
        result.result(timeout=2)
    assert closed == [True] * 4


def test_reverse_unknown_session_inference_returns_routed_error():
    def supervisor(ws):
        ws.send(json.dumps({"type": "authenticate"}))
        ws.recv(timeout=2)
        ws.send(json.dumps({"type": "authenticated"}))
        negotiate(ws)
        ws.send(wrap(observation(), "unknown", "request"))
        return unwrap(ws.recv(timeout=2))

    with supervisor_endpoint(supervisor) as (uri, result):
        PolicyWebsocketServer(policy_factory=DummyPolicy).run_reverse(
            uri,
            team_id="t",
            evaluation_id="e",
        )
        session, request, payload = result.result(timeout=2)
    assert (session, request, "error" in header(payload)) == (
        "unknown",
        "request",
        True,
    )
