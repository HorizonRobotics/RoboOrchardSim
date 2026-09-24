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

"""Shared connection options for model-side entry points."""

import argparse
import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from robo_orchard_server.server.policy_server import PolicyWebsocketServer


def add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add passive listening and opt-in reverse connection arguments."""
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--supervisor-uri",
        help="Connect to this supervisor instead of listening",
    )
    parser.add_argument("--team-id", default=os.environ.get("TEAM_ID"))
    parser.add_argument(
        "--evaluation-id", default=os.environ.get("SESSION_ID")
    )


def validate_connection_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject missing reverse identity before loading model weights."""
    if args.supervisor_uri is not None and (
        urlsplit(args.supervisor_uri).scheme not in {"ws", "wss"}
        or not urlsplit(args.supervisor_uri).hostname
    ):
        parser.error("--supervisor-uri must be a ws:// or wss:// URL")
    if args.supervisor_uri and any(
        value is None or not value.strip()
        for value in (args.team_id, args.evaluation_id)
    ):
        parser.error(
            "--supervisor-uri requires --team-id and --evaluation-id "
            "(or TEAM_ID and SESSION_ID)"
        )


def run_connection(
    server: "PolicyWebsocketServer", args: argparse.Namespace
) -> None:
    """Run the explicitly selected connection mode."""
    if args.supervisor_uri:
        server.run_reverse(
            args.supervisor_uri,
            team_id=args.team_id,
            evaluation_id=args.evaluation_id,
        )
    else:
        server.run()
