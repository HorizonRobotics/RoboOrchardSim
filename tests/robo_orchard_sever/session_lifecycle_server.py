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

"""Subprocess probe for real shutdown of blocked policy sessions."""

from __future__ import annotations
import json
import logging
import sys
import threading

import numpy as np
from robo_orchard_server.server import (
    BasePolicy,
    JointAction,
    Observation,
    PolicyWebsocketServer,
)

OUTPUT_LOCK = threading.Lock()


def emit(event, **fields):
    # print writes the JSON and newline separately; keep worker and listener
    # events on distinct lines so the probe cannot lose concurrent events.
    with OUTPUT_LOCK:
        print(json.dumps({"event": event, **fields}), flush=True)


class LogEvents(logging.Handler):
    def emit(self, record):
        emit("log", message=record.getMessage())


class Session(BasePolicy):
    def __init__(self, session_id):
        self.session_id = session_id

    def reset(self):
        emit("reset", session=self.session_id)

    def act(self, obs: Observation) -> JointAction:
        emit("act", session=self.session_id, instruction=obs["instruction"])
        if obs["instruction"] == "block":
            emit("blocked", operation="act")
            if sys.stdin.readline().strip() != "release":
                raise RuntimeError("Missing release")
        names = [
            name
            for slot in obs["action_layout"]["manipulator_order"]
            for name in (
                obs["action_layout"]["manipulators"][slot]["arm_joint_names"]
                + obs["action_layout"]["manipulators"][slot][
                    "gripper_joint_names"
                ]
            )
        ]
        return {
            "joint_names": names,
            "values": np.zeros((1, len(names)), np.float32),
        }

    def close(self):
        emit("close", session=self.session_id)


def main():
    port = int(sys.argv[1])
    block = sys.argv[2]
    next_id = 0

    def factory():
        nonlocal next_id
        session_id = next_id
        next_id += 1
        emit("factory", session=session_id)
        if block == "factory" and session_id == 0:
            emit("blocked", operation="factory")
            if sys.stdin.readline().strip() != "release":
                raise RuntimeError("Missing release")
        emit("created", session=session_id)
        return Session(session_id)

    logger = logging.getLogger("robo_orchard_server.server")
    logger.setLevel(logging.INFO)
    logger.addHandler(LogEvents())
    try:
        PolicyWebsocketServer(
            policy_factory=factory, host="127.0.0.1", port=port
        ).run()
    except KeyboardInterrupt:
        pass
    emit("stopped")


if __name__ == "__main__":
    main()
