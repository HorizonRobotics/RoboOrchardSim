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

"""Serve OpenPI pi0.5 through the canonical joint-action protocol."""

import argparse
import logging
from pathlib import Path

import yaml

from robo_orchard_server.server import PolicyWebsocketServer
from robo_orchard_server.server.cli import (
    add_connection_arguments,
    run_connection,
    validate_connection_arguments,
)


def main(argv: list[str] | None = None) -> None:
    """Load model once, then create independent policies for each client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-yaml", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--config-module", default="openpi.training.config")
    parser.add_argument("--action-steps", type=int)
    add_connection_arguments(parser)
    args = parser.parse_args(argv)
    validate_connection_arguments(parser, args)
    from robo_orchard_server.policy.pi05.loader import load_policy
    from robo_orchard_server.policy.pi05.policy import Pi05Policy

    logging.basicConfig(level=logging.INFO)
    from robo_orchard_server.policy.pi05.adapter import Pi05Adapter

    cfg = yaml.safe_load(args.model_yaml.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Model input mapping must be a YAML mapping")
    adapter = Pi05Adapter(
        cfg["camera_paths"],
        cfg["state_path"],
        cfg.get("prompt_path", ["prompt"]),
    )
    pipeline = load_policy(
        args.checkpoint, args.config_name, args.config_module
    )
    shared = Pi05Policy(pipeline, adapter, steps=args.action_steps)
    server = PolicyWebsocketServer(
        policy_factory=shared.new_session,
        host=args.host,
        port=args.port,
        logging_tag="pi05",
    )
    run_connection(server, args)


if __name__ == "__main__":
    main()
