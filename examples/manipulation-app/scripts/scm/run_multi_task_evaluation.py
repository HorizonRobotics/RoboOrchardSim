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

"""Run one configured batch evaluation group."""

from __future__ import annotations
import argparse
from pathlib import Path

import yaml

from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
    load_batch_plan,
)
from robo_orchard_sim.pipeline.evaluator.batch_evaluation import (
    run_group_evaluation,
)
from robo_orchard_sim.policy.factory import create_policy_from_model_cfg

_REPO_ROOT = Path(__file__).resolve().parents[4]
_POLICY_CONFIG_DIR = _REPO_ROOT / "robo_orchard_sim" / "policy" / "configs"


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the group evaluation CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-plan", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output-root-dir", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument(
        "--model-yaml",
        default=None,
        help=(
            "Optional policy config yaml. If omitted, load default config "
            "from robo_orchard_sim/policy/configs/<model-type>.yaml."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--enable-recording", action="store_true")
    parser.add_argument(
        "--fail-fast",
        dest="continue_on_task_error",
        action="store_false",
        help="Stop the group run when one task config fails.",
    )
    parser.set_defaults(continue_on_task_error=True)
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
    return parser


def _resolve_model_cfg_yaml(
    model_type: str,
    model_yaml: str | None,
) -> Path:
    if model_yaml is not None:
        return Path(model_yaml)
    return _POLICY_CONFIG_DIR / f"{model_type}.yaml"


def _load_model_cfg(args: argparse.Namespace) -> dict:
    config_path = _resolve_model_cfg_yaml(
        model_type=args.model_type,
        model_yaml=args.model_yaml,
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config yaml not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Model yaml must contain a mapping: {config_path}")

    yaml_policy = loaded.get("policy")
    if yaml_policy is not None and yaml_policy != args.model_type:
        raise ValueError(
            "Policy type mismatch: "
            f"--model-type={args.model_type}, "
            f"but yaml declares policy={yaml_policy}"
        )

    model_cfg = dict(loaded)
    model_cfg["policy"] = args.model_type
    return model_cfg


def _normalize_policy_instance(policy_or_cfg):
    from robo_orchard_core.policy.base import PolicyConfig

    if isinstance(policy_or_cfg, PolicyConfig):
        return policy_or_cfg()
    return policy_or_cfg


def _close_policy_if_needed(policy) -> None:
    close = getattr(policy, "close", None)
    if callable(close):
        close()


def main() -> None:
    """Run the group evaluation CLI."""
    args = build_arg_parser().parse_args()
    policy = _normalize_policy_instance(
        create_policy_from_model_cfg(_load_model_cfg(args))
    )
    try:
        result = run_group_evaluation(
            plan=load_batch_plan(args.batch_plan),
            group_id=args.group_id,
            asset_root=args.asset_root,
            output_root_dir=args.output_root_dir,
            policy_or_cfg=policy,
            max_steps=args.max_steps,
            enable_recording=args.enable_recording,
            snapshot_path=args.snapshot_path,
            splits_path=args.splits_path,
            continue_on_task_error=args.continue_on_task_error,
        )
        print(
            f"group evaluation finished: tasks={result.total_tasks}, "
            f"episodes={result.total_episodes}, "
            f"successes={result.success_episodes}"
        )
    finally:
        _close_policy_if_needed(policy)


if __name__ == "__main__":
    main()
