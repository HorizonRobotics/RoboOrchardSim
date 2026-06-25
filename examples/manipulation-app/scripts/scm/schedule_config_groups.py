#!/usr/bin/env python3
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

"""Schedule explicit task YAML configs into fixed-size batch groups."""

from __future__ import annotations
import argparse
from datetime import datetime

from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
    schedule_batch_plan,
    write_batch_plan,
)


def default_batch_id(task: str) -> str:
    """Return a timestamped default batch id."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task}_{timestamp}"


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the scheduler CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--config-list", required=True)
    parser.add_argument("--configs-per-group", type=int, required=True)
    parser.add_argument("--episodes-per-config", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-id")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    """Run the scheduler CLI."""
    args = build_arg_parser().parse_args()
    plan = schedule_batch_plan(
        task=args.task,
        config_list_path=args.config_list,
        configs_per_group=args.configs_per_group,
        episodes_per_config=args.episodes_per_config,
        base_seed=args.seed,
        batch_id=args.batch_id or default_batch_id(args.task),
    )
    write_batch_plan(plan, args.output)
    print(
        f"wrote {args.output}: groups={len(plan.groups)}, "
        f"configs={sum(len(group.configs) for group in plan.groups)}"
    )


if __name__ == "__main__":
    main()
