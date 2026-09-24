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

"""Minimal dummy policy with one independent instance per Client session."""

import argparse
import logging

import numpy as np
from robo_orchard_server.server import (
    BasePolicy,
    JointAction,
    Observation,
    PolicyWebsocketServer,
)
from robo_orchard_server.server.cli import (
    add_connection_arguments,
    run_connection,
    validate_connection_arguments,
)


class DummyPolicy(BasePolicy):
    """Return a fixed joint target to test communication, not solve tasks."""

    def __init__(self, value: float = 0.0) -> None:
        # Store per-session state here when adapting this example.
        self.value = value

    def reset(self) -> None:
        """Clear episode history; this stateless example has none."""

    def act(self, obs: Observation) -> JointAction:
        """Return all joint targets as a float32 array of shape [B, J]."""
        layout = obs["action_layout"]
        joint_names = []
        for slot in layout["manipulator_order"]:
            manipulator = layout["manipulators"][slot]
            joint_names.extend(manipulator["arm_joint_names"])
            joint_names.extend(manipulator["gripper_joint_names"])

        first_slot = layout["manipulator_order"][0]
        batch_size = obs["manipulators"][first_slot]["joint_position"].shape[0]

        # Replace the constant values with your model's physical joint targets.
        # Column j must correspond to joint_names[j], including gripper joints.
        return {
            "joint_names": joint_names,
            "values": np.full(
                (batch_size, len(joint_names)), self.value, dtype=np.float32
            ),
        }


def main() -> None:
    """Start a server with an independent dummy policy per Client session."""
    parser = argparse.ArgumentParser(description="Dummy test policy server")
    add_connection_arguments(parser)
    parser.add_argument("--value", type=float, default=0.0)
    args = parser.parse_args()
    validate_connection_arguments(parser, args)

    def create_policy() -> DummyPolicy:
        # Called once per Client session; return a fresh policy instance.
        return DummyPolicy(value=args.value)

    logging.basicConfig(level=logging.INFO)
    try:
        server = PolicyWebsocketServer(
            policy_factory=create_policy, host=args.host, port=args.port
        )
        run_connection(server, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
