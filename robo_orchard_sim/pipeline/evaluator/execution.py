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

"""Worker lifecycle and persistence for one evaluation shard.

The user entry point is examples/manipulation-app/scripts/eval_policy.py.
This module is invoked only inside a dispatched worker process.
"""

from __future__ import annotations
import json
import os
import shutil
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robo_orchard_sim.pipeline.evaluator.distributed import (
        EvaluationManifest,
        EvaluationShard,
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Persist JSON without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_shard(
    *,
    manifest: EvaluationManifest,
    shard: EvaluationShard,
    task_settings: dict[str, Any],
    policy_config: dict[str, Any],
    output_dir: Path,
    enable_recording: bool,
) -> int:
    """Own Isaac, the policy client, evaluation and output for one shard."""
    from robo_orchard_core.policy.base import PolicyConfig

    from robo_orchard_sim.pipeline.evaluator.base import ShardSummary
    from robo_orchard_sim.pipeline.evaluator.evaluator import (
        EvaluatorCfg,
        SwapConfig,
        evaluation_runtime,
    )
    from robo_orchard_sim.policy.factory import create_policy_from_model_cfg

    swap = SwapConfig(**task_settings["swap"])
    episodes_per_scene = swap.swap_per_scene if swap.enabled else 1
    if shard.episodes_per_scene != episodes_per_scene:
        raise ValueError("shard swap configuration differs from task config")

    shard_out = output_dir / shard.task_name / "shards" / shard.shard_id
    shard_out.mkdir(parents=True, exist_ok=True)
    result_path = shard_out / "eval_result.json"
    summary_path = shard_out / "shard_summary.json"
    result_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)
    config_out = shard_out / "config"
    config_out.mkdir(exist_ok=True)
    shutil.copy2(shard.config_path, config_out / Path(shard.config_path).name)

    evaluator_cfg = EvaluatorCfg(
        task_name=shard.task_type,
        asset_root=task_settings["asset_root"],
        task_config_path=shard.config_path,
        enable_recording=enable_recording,
        record_dir=str(shard_out / "records"),
        seed=shard.seed_start,
        episode_num=shard.episode_count,
        swap=swap,
        scene_seed_candidates=shard.candidate_scene_seeds(),
        max_steps=task_settings["max_steps"],
        snapshot_path=task_settings["snapshot"],
        splits_path=task_settings["splits"],
    )
    outcome = ShardSummary(shard=shard, result_json_path=str(result_path))

    def save_summary() -> None:
        write_json_atomic(
            summary_path,
            {
                "manifest_version": manifest.version,
                "run_id": manifest.run_id,
                "policy_id": manifest.policy_id,
                **outcome.to_payload(),
            },
        )

    # Persist before closing Isaac: its shutdown watchdog can force-exit.
    with evaluation_runtime() as runtime:
        policy = None
        try:
            configured = create_policy_from_model_cfg(policy_config)
            policy = (
                configured()
                if isinstance(configured, PolicyConfig)
                else configured
            )
            with evaluator_cfg() as evaluator:
                result = evaluator.run_with_runtime(policy, runtime=runtime)
            write_json_atomic(result_path, asdict(result))
            outcome.result = result
            save_summary()
        except (Exception, SystemExit) as exc:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            outcome.error = f"{type(exc).__name__}: {exc}"
            save_summary()
            return 1
        finally:
            # Existing policy implementations expose optional cleanup.
            close = getattr(policy, "close", None)
            if callable(close):
                close()
    return 0
