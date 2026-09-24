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

"""Load one Holobrain model and serve independent policy sessions."""

from __future__ import annotations
import argparse
import logging
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from robo_orchard_server.server import PolicyFactory, PolicyWebsocketServer
from robo_orchard_server.server.cli import (
    add_connection_arguments,
    run_connection,
    validate_connection_arguments,
)

if TYPE_CHECKING:
    from robo_orchard_server.policy.holobrain.policy import HolobrainPolicyCfg

LOGGER = logging.getLogger(__name__)


def load_policy_config(
    model_yaml: Path | None = None,
    *,
    model_dir: str | None = None,
    device: str | None = None,
) -> "HolobrainPolicyCfg":
    """Load the supplied or packaged YAML and apply explicit CLI overrides."""
    import yaml

    from robo_orchard_server.policy.holobrain.policy import HolobrainPolicyCfg

    source = (
        model_yaml
        if model_yaml is not None
        else files("robo_orchard_server.policy.holobrain").joinpath(
            "config.yaml"
        )
    )
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Holobrain config must be a mapping: {source}")
    declared_policy = config.pop("policy", "holobrain")
    if declared_policy != "holobrain":
        raise ValueError(
            f"Expected policy 'holobrain', got {declared_policy!r}"
        )
    if model_dir is not None:
        config["model_dir"] = model_dir
    if device is not None:
        config["device"] = device
    return HolobrainPolicyCfg(**config)


def create_policy_factory(cfg: "HolobrainPolicyCfg") -> PolicyFactory:
    """Load weights once and create independent policies for new clients."""
    from robo_orchard_server.policy.holobrain.policy import HolobrainPolicy

    shared = HolobrainPolicy(cfg=cfg.model_copy(deep=True))
    return shared.new_session


def main(argv: list[str] | None = None) -> None:
    """Start a Holobrain WebSocket server in the model environment."""
    parser = argparse.ArgumentParser(
        description="Serve Holobrain with independent client sessions"
    )
    parser.add_argument(
        "--model-yaml",
        type=Path,
        help="Holobrain policy YAML (default: the installed package config)",
    )
    parser.add_argument(
        "--model-dir", help="Override the exported model directory"
    )
    parser.add_argument(
        "--device", help="Override the model device, e.g. cuda:0"
    )
    add_connection_arguments(parser)
    args = parser.parse_args(argv)
    validate_connection_arguments(parser, args)
    cfg = load_policy_config(
        args.model_yaml, model_dir=args.model_dir, device=args.device
    )
    logging_tag = f"{args.host}:{args.port}"
    if cfg.logging_tag:
        logging_tag += f":{cfg.logging_tag}"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [{logging_tag}] %(message)s",
    )
    policy_factory = create_policy_factory(cfg)
    LOGGER.info("Holobrain loaded; sharing model across client sessions")
    server = PolicyWebsocketServer(
        policy_factory=policy_factory,
        host=args.host,
        port=args.port,
        logging_tag=logging_tag,
    )
    try:
        run_connection(server, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
