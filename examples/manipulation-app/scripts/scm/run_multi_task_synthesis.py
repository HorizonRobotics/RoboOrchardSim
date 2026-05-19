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

"""Run one configured batch synthesis group."""

from __future__ import annotations
import argparse

from robo_orchard_sim.runner.data_synthesis.batch_synthesis import (
    load_batch_plan,
    run_group_data_synthesis,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the group runner CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-plan", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output-root-dir", required=True)
    return parser


def main() -> None:
    """Run the group runner CLI."""
    args = build_arg_parser().parse_args()
    result = run_group_data_synthesis(
        plan=load_batch_plan(args.batch_plan),
        group_id=args.group_id,
        asset_root=args.asset_root,
        task_root_dir=args.output_root_dir,
    )
    print(
        f"group run finished: tasks={result.total_tasks}, "
        f"episodes={result.total_episodes}, "
        f"successes={result.success_episodes}"
    )


if __name__ == "__main__":
    main()
