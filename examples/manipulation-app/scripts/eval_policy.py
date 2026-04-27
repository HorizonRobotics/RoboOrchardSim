# ruff: noqa: E402, I001
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

"""Run first-pass Holobrain evaluation on the place-a2b orchard task."""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Registered task name to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for evaluation episodes.",
    )
    parser.add_argument(
        "--asset-root",
        type=str,
        required=True,
        help="Asset root directory used to construct AssetRegistry.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional task YAML config path.",
    )
    parser.add_argument(
        "--episode-num",
        type=int,
        default=3,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of policy steps per episode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_result/isaac_eval/eval_result.json",
        help="Output path for serialized evaluation result.",
    )
    parser.add_argument(
        "--enable-recording",
        action="store_true",
        help="Enable MCAP recording during evaluation.",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="logs/records",
        help="Output directory for MCAP recording files.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    from robo_orchard_core.utils.logging import LoggerManager
    from robo_orchard_sim.evaluator import EvaluatorCfg, LaunchConfig
    from robo_orchard_sim.policy.DummyPolicy import DummyPolicyCfg

    logger = LoggerManager().get_child(__name__)
    policy = DummyPolicyCfg()

    evaluator_cfg = EvaluatorCfg(
        task_name=args.task_name,
        asset_root=args.asset_root,
        task_config_path=args.config,
        enable_recording=args.enable_recording,
        record_dir=args.record_dir,
        launch=LaunchConfig(
            headless=True,
            enable_cameras=True,
            virtual_display=False,
        ),
        seed=args.seed,
        episode_num=args.episode_num,
        max_steps=args.max_steps,
    )
    with evaluator_cfg() as evaluator:
        result = evaluator.evaluate(policy)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=4)
    logger.info("Evaluation done. Output: %s", args.output)
    logger.info("Evaluation result: %s", result)


if __name__ == "__main__":
    main()
