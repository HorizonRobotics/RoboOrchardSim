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

"""CLI entry point for single-task data synthesis."""

from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path

from robo_orchard_sim.pipeline.data_synthesis import single_task

_ASSET_ROOT_ENV = "ORCHARD_ASSET_LIBRARY"


def _task_run_dir_name(task: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{task}_{timestamp}"


def _append_task_run_dir(path: str | None, task_run_dir: str) -> str | None:
    if path is None:
        return None
    return os.path.join(path, task_run_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for single-task data synthesis."""
    env_default = os.environ.get(_ASSET_ROOT_ENV)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=(
            "Registered task namespace to build (e.g. place_a2b_easy, "
            "pick_category, pick_attribute, pick_disambiguation)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML path overriding the selected task config_path.",
    )
    parser.add_argument(
        "--asset-root",
        type=str,
        default=env_default,
        required=env_default is None,
        help=(
            f"Asset library root. Defaults to the ${_ASSET_ROOT_ENV} env "
            "var and is required if that env var is not set."
        ),
    )
    parser.add_argument(
        "--snapshot",
        dest="snapshot_path",
        type=Path,
        default=None,
        help=(
            "Optional snapshot YAML; restricts asset sampling to its "
            "uuid set for reproducibility."
        ),
    )
    parser.add_argument(
        "--splits",
        dest="splits_path",
        type=Path,
        default=None,
        help=(
            "Optional benchmark splits YAML; binds seen / unseen_category / "
            "unseen_instance for a task config's `split:` field."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="First RNG seed for reproducible asset sampling.",
    )
    parser.add_argument(
        "--episodes",
        dest="episode_num",
        type=int,
        default=1,
        help="How many episodes to synthesize.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum manager-driven env steps per episode.",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="logs/data_synthesis",
        help=(
            "Base output directory for synthesized MCAP recordings. "
            "The CLI appends <task>_<timestamp_ms> before invoking the "
            "runner."
        ),
    )
    parser.add_argument(
        "--disable-recording",
        dest="enable_recording",
        action="store_false",
        help="Run the synthesis loop without enabling MCAP recording.",
    )
    parser.add_argument(
        "--output-config-dir",
        type=str,
        default="configs/data_synthesis",
        help=(
            "Base directory for per-episode serialized env configs. "
            "The CLI appends <task>_<timestamp_ms> before invoking the "
            "runner."
        ),
    )
    parser.add_argument(
        "--task-save-root",
        type=str,
        default=None,
        help=(
            "Optional task output root. When set, config artifacts are "
            "written below <task-save-root>/<task>_<timestamp_ms>/config "
            "and recordings below "
            "<task-save-root>/<task>_<timestamp_ms>/data."
        ),
    )
    parser.set_defaults(
        enable_recording=True,
        enable_cameras=True,
        headless=True,
    )
    return parser


def main() -> None:
    """Run single-task data synthesis from CLI arguments."""
    args = build_arg_parser().parse_args()
    cfg_kwargs = vars(args)
    task_run_dir = _task_run_dir_name(cfg_kwargs["task"])
    for key in ("record_dir", "output_config_dir", "task_save_root"):
        cfg_kwargs[key] = _append_task_run_dir(
            cfg_kwargs[key],
            task_run_dir,
        )
    launch = single_task.LaunchConfig(
        headless=cfg_kwargs.pop("headless"),
        enable_cameras=cfg_kwargs.pop("enable_cameras"),
    )
    cfg = single_task.TaskDataSynthesisCfg(**cfg_kwargs, launch=launch)
    result = single_task.TaskDataSynthesisRunner(cfg).run()

    print("Data synthesis finished successfully.")
    for summary in result.episodes:
        print(
            f"episode={summary.episode_index}, seed={summary.seed}, "
            f"steps={summary.steps}, stop_reason={summary.stop_reason}, "
            f"success={summary.success}"
        )
    print(
        f"Overall success rate: {result.success_count}/{result.total} "
        f"({result.success_rate:.2%})"
    )


if __name__ == "__main__":
    main()
