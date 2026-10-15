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

"""SIGINT closes sockets and drains only already-started session work."""

from __future__ import annotations
import json
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from queue import Empty, Queue

import pytest
from test_server_interop import wire_message
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect


class LifecycleProcess:
    def __init__(self, mode):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.events = []
        self.incoming = Queue()
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("session_lifecycle_server.py")),
                str(self.port),
                mode,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"event": "output", "message": line}
            self.events.append(event)
            self.incoming.put(event)

    def wait_for(self, event, **fields):
        deadline = time.monotonic() + 5
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(f"Missing {event}: {self.events}")
            try:
                item = self.incoming.get(timeout=remaining)
            except Empty:
                pytest.fail(f"Missing {event}: {self.events}")
            if item["event"] == event and all(
                item.get(key) == value for key, value in fields.items()
            ):
                return item

    def release(self):
        assert self.process.stdin is not None
        self.process.stdin.write("release\n")
        self.process.stdin.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


def action_message(instruction):
    # This request is never allowed to produce an action after SIGINT. The
    # active request has a valid single-joint observation to enter the model.
    values = struct.pack("f", 0.0)
    return (
        wire_message(
            {
                "type": "act",
                "obs_data": {
                    "format": "canonical",
                    "instruction": instruction,
                    "action_layout": {
                        "manipulator_order": ["arm"],
                        "manipulators": {
                            "arm": {
                                "arm_joint_names": ["joint"],
                                "gripper_joint_names": [],
                            }
                        },
                    },
                    "manipulators": {
                        "arm": {
                            "joint_position": {
                                "dtype": "float32",
                                "shape": [1, 1],
                                "data": {"__bytes_idx__": 0},
                            }
                        }
                    },
                },
            }
        )
        + struct.pack("!Q", len(values))
        + values
    )


@pytest.mark.parametrize("blocked", ["act", "factory", "idle"])
def test_server_shutdown_pending_work_is_skipped_and_sessions_close_once(
    blocked,
):
    with LifecycleProcess(blocked) as probe, ExitStack() as stack:
        # Listening is a public server log, not private synchronization state.
        probe.wait_for(
            "log", message=f"Listening on ws://127.0.0.1:{probe.port}"
        )
        url = f"ws://127.0.0.1:{probe.port}"
        sockets = [stack.enter_context(connect(url))]
        expected_sessions = [0]
        if blocked == "factory":
            probe.wait_for("blocked", operation="factory")
        else:
            probe.wait_for("created", session=0)
            sockets.append(stack.enter_context(connect(url)))
            probe.wait_for("created", session=1)
            expected_sessions.append(1)
            if blocked == "act":
                sockets[0].send(action_message("block"))
                probe.wait_for("blocked", operation="act")
                sockets[1].send(action_message("queued"))
        if blocked != "idle":
            # Handshake succeeds while inference is blocked; factory waits.
            sockets.append(stack.enter_context(connect(url)))
        probe.process.send_signal(signal.SIGINT)
        for client in sockets:
            with pytest.raises(ConnectionClosed):
                client.recv(timeout=5)
        if blocked != "idle":
            assert probe.process.poll() is None, "Running call was interrupted"
            probe.release()
        assert probe.process.wait(timeout=5) == 0
        probe.reader.join(timeout=5)
        created = [
            item["session"]
            for item in probe.events
            if item["event"] == "created"
        ]
        closed = [
            item["session"]
            for item in probe.events
            if item["event"] == "close"
        ]
        actions = [
            item["instruction"]
            for item in probe.events
            if item["event"] == "act"
        ]
        lifecycle = [item for item in probe.events if item["event"] != "log"]
        assert (created, sorted(closed), actions, lifecycle[-1]) == (
            expected_sessions,
            expected_sessions,
            ["block"] if blocked == "act" else [],
            {"event": "stopped"},
        ), probe.events
