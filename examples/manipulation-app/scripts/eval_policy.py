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

"""Run orchard evaluation with a local or remote policy.

Example:
    PYTHONPATH=$PWD python3 examples/manipulation-app/scripts/eval_policy.py \
        --task-name place-a2b \
        --asset-root /path/to/assets \
        --model-type holobrain \
        --model-yaml /path/to/holobrain.yaml \
        --episode-num 3 \
        --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_POLICY_CONFIG_DIR = _REPO_ROOT / "robo_orchard_sim" / "policy" / "configs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Policy type to evaluate.",
    )
    parser.add_argument(
        "--model-yaml",
        type=str,
        default=None,
        help=(
            "Optional policy config yaml. If omitted, load default config "
            "from robo_orchard_sim/policy/configs/<model-type>.yaml."
        ),
    )
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
        "--task-config-yaml",
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
        "--eval-result-json",
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


def _log_remote_policy_logging_tag(logger, policy) -> None:
    client = getattr(policy, "_client", None)
    logging_tag = getattr(client, "logging_tag", None)
    if logging_tag is None:
        return
    logger.info("Remote policy logging tag: %s", logging_tag)


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
    args = _build_parser().parse_args()

    from robo_orchard_core.utils.logging import LoggerManager
    from robo_orchard_sim.evaluator import EvaluatorCfg, LaunchConfig
    from robo_orchard_sim.policy.factory import create_policy_from_model_cfg

    logger = LoggerManager().get_child(__name__)
    model_cfg = _load_model_cfg(args)
    policy = _normalize_policy_instance(
        create_policy_from_model_cfg(model_cfg)
    )
    try:
        evaluator_cfg = EvaluatorCfg(
            task_name=args.task_name,
            asset_root=args.asset_root,
            task_config_path=args.task_config_yaml,
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
            snapshot_path=args.snapshot_path,
        )
        with evaluator_cfg() as evaluator:
            result = evaluator.evaluate(policy)
        _log_remote_policy_logging_tag(logger, policy)

        output_dir = os.path.dirname(args.eval_result_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.eval_result_json, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=4)
        logger.info("Evaluation done. Output: %s", args.eval_result_json)
        logger.info("Evaluation result: %s", result)
    finally:
        _close_policy_if_needed(policy)


if __name__ == "__main__":
    main()
