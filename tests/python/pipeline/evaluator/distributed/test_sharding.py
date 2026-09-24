#
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

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import pytest

from robo_orchard_sim.pipeline.data_synthesis.batch_synthesis import (
    BatchGroup,
    BatchPlan,
)
from robo_orchard_sim.pipeline.evaluator.distributed import (
    EvaluationManifest,
    build_evaluation_shards,
    read_evaluation_manifest,
    validate_shard_result,
    write_evaluation_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_EVAL_POLICY_PATH = (
    _REPO_ROOT / "examples/manipulation-app/scripts/eval_policy.py"
)
_EVAL_POLICY_SPEC = importlib.util.spec_from_file_location(
    "robo_orchard_eval_policy_script",
    _EVAL_POLICY_PATH,
)
assert _EVAL_POLICY_SPEC is not None
assert _EVAL_POLICY_SPEC.loader is not None
_EVAL_POLICY = importlib.util.module_from_spec(_EVAL_POLICY_SPEC)
sys.modules[_EVAL_POLICY_SPEC.name] = _EVAL_POLICY
_EVAL_POLICY_SPEC.loader.exec_module(_EVAL_POLICY)


def _plan(*, episodes: int = 10) -> BatchPlan:
    return BatchPlan(
        batch_id="test",
        task="pick_category",
        episodes_per_config=episodes,
        base_seed=100,
        groups=[BatchGroup("group_0000", 100, ["first.yaml", "second.yaml"])],
    )


@pytest.mark.parametrize(
    ("episodes", "shard_count", "per_scene"),
    [(10, 3, 1), (5, 2, 2), (2, 4, 1)],
)
def test_build_evaluation_shards_partition_preserves_each_config_episodes(
    episodes: int,
    shard_count: int,
    per_scene: int,
) -> None:
    shards = build_evaluation_shards(
        plan=_plan(episodes=episodes),
        task_name="pick",
        shards_per_config=shard_count,
        episodes_per_scene=per_scene,
    )
    for config_index in (0, 1):
        parts = [s for s in shards if s.config_index == config_index]
        assert [key for part in parts for key in part.episode_keys()] == [
            (100 + config_index * episodes + i // per_scene, i % per_scene)
            for i in range(episodes)
        ]
        assert all(part.episode_offset % per_scene == 0 for part in parts)


def test_build_evaluation_shards_fallback_seed_pools_do_not_overlap():
    shards = build_evaluation_shards(
        plan=_plan(episodes=4),
        task_name="pick",
        shards_per_config=2,
    )

    candidates = [
        seed for shard in shards for seed in shard.candidate_scene_seeds()
    ]

    assert (len(candidates), len(set(candidates))) == (24, 24)


def test_validate_shard_result_resampled_seed_from_fallback_pool_succeeds():
    shard = build_evaluation_shards(
        plan=_plan(episodes=4),
        task_name="pick",
        shards_per_config=2,
    )[0]
    candidates = shard.candidate_scene_seeds()
    result_payload = {
        "episode_results": [
            {"seed": candidates[1]},
            {"seed": candidates[2]},
        ],
        "skipped_episodes": [{"seed": candidates[0]}],
        "attempted_scene_seeds": list(candidates[:3]),
    }

    validate_shard_result(shard, result_payload)


def test_validate_shard_result_missing_episodes_raises_value_error():
    shard = build_evaluation_shards(
        plan=_plan(episodes=4),
        task_name="pick",
        shards_per_config=2,
    )[0]

    with pytest.raises(ValueError, match="episode count mismatch"):
        validate_shard_result(
            shard,
            {
                "episode_results": [],
                "skipped_episodes": [],
                "attempted_scene_seeds": [],
            },
        )


def test_validate_shard_result_swap_seed_reuse_succeeds():
    shard = build_evaluation_shards(
        plan=_plan(episodes=5),
        task_name="pick_category",
        shards_per_config=2,
        episodes_per_scene=2,
    )[0]
    primary_seeds = shard.candidate_scene_seeds()[:2]

    validate_shard_result(
        shard,
        {
            "episode_results": [
                {"seed": primary_seeds[0]},
                {"seed": primary_seeds[0]},
                {"seed": primary_seeds[1]},
                {"seed": primary_seeds[1]},
            ],
            "skipped_episodes": [],
            "attempted_scene_seeds": list(primary_seeds),
        },
    )


def test_load_eval_config_execution_and_task_shards_are_resolved(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        """
policy:
  model_type: server
execution:
  gpus: [2, 3]
defaults:
  asset_root: /tmp/assets
tasks:
  category:
    task_type: pick_category
  color:
    task_type: pick_attribute
    shards: 1
  place:
    task_type: place_a2b_hard
    shards: 2
""",
        encoding="utf-8",
    )

    config = _EVAL_POLICY.load_eval_config(config_path)

    assert (
        config.execution.gpus,
        [task.shards for task in config.tasks],
    ) == (("2", "3"), [None, 1, 2])


@pytest.mark.parametrize(
    ("shards_per_config", "episodes_per_scene"),
    [(0, 1), (1, 0)],
)
def test_build_evaluation_shards_invalid_parallelism_raises_value_error(
    shards_per_config: int,
    episodes_per_scene: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_evaluation_shards(
            plan=_plan(),
            task_name="pick",
            shards_per_config=shards_per_config,
            episodes_per_scene=episodes_per_scene,
        )


def test_evaluation_manifest_round_trip_preserves_shards(
    tmp_path: Path,
) -> None:
    manifest = EvaluationManifest(
        run_id="run-001",
        policy_id="pi05",
        shards=tuple(
            build_evaluation_shards(
                plan=_plan(episodes=3),
                task_name="pick",
                shards_per_config=2,
            )
        ),
    )
    output_path = tmp_path / "manifest.json"

    written_path = write_evaluation_manifest(manifest, output_path)
    loaded = read_evaluation_manifest(output_path)

    assert written_path == output_path.resolve()
    assert loaded == manifest


@pytest.mark.parametrize(
    ("layout", "failure"),
    [
        ("shards", "missing"),
        ("shards", "invalid"),
        ("shards", "invalid_episode"),
        ("shards", "exit"),
        ("shards", "all"),
        ("configs", "missing"),
    ],
)
def test_eval_policy_failed_shard_preserves_completed_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    layout: str,
) -> None:
    import json
    import subprocess

    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text("{}\n")
    task_source = f"    yaml: {task_yaml}\n"
    if layout == "configs":
        plan = BatchPlan(
            batch_id="test",
            task="pick_category",
            episodes_per_config=1,
            base_seed=0,
            groups=[BatchGroup("group", 0, [str(task_yaml)] * 2)],
        )
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan.to_payload()))
        task_source = f"    batch_plan: {plan_path}\n"
    config = tmp_path / "eval.yaml"
    config.write_text(
        "policy:\n  model_type: dummy\n"
        "defaults:\n  episode_num: 2\n"
        f"  asset_root: {tmp_path}\n"
        "tasks:\n  pick:\n    task_type: pick_category\n    shards: 2\n"
        + task_source
    )
    output = tmp_path / "results"

    def run_worker(cmd, **kwargs):
        manifest_path = cmd[cmd.index("--_shard-manifest") + 1]
        shard_id = cmd[cmd.index("--_shard-id") + 1]
        manifest = read_evaluation_manifest(manifest_path)
        shard = next(s for s in manifest.shards if s.shard_id == shard_id)
        index = (
            shard.config_index if layout == "configs" else shard.shard_index
        )
        shard_dir = output / "pick" / "shards" / shard_id
        if failure == "all" or (index == 1 and failure == "missing"):
            return subprocess.CompletedProcess(cmd, 1)
        result_path = shard_dir / "eval_result.json"
        result_path.write_text(
            json.dumps(
                {
                    "episode_results": [
                        {
                            "seed": shard.seed_start,
                            "success": True,
                            "progress": 1.0,
                            "metrics": {},
                        }
                    ],
                    "attempted_scene_seeds": [shard.seed_start],
                    "skipped_episodes": [],
                }
            )
        )
        (shard_dir / "shard_summary.json").write_text(
            json.dumps(
                {
                    "shard": shard.to_payload(),
                    "configs": [{"eval_result_json": str(result_path)}],
                }
            )
        )
        if index == 1:
            if failure == "invalid":
                result_path.write_text("invalid json")
            elif failure == "invalid_episode":
                result_path.write_text(
                    json.dumps(
                        {
                            "episode_results": [
                                {
                                    "seed": shard.seed_start,
                                    "success": True,
                                    "progress": 2.0,
                                    "metrics": {},
                                }
                            ],
                            "attempted_scene_seeds": [shard.seed_start],
                            "skipped_episodes": [],
                        }
                    )
                )
            elif failure == "exit":
                return subprocess.CompletedProcess(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)

    class WorkerProcess:
        def __init__(self, cmd, **kwargs):
            self.returncode = run_worker(cmd, **kwargs).returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", WorkerProcess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_EVAL_POLICY_PATH),
            "--eval-config",
            str(config),
            "--output-dir",
            str(output),
            "--gpus",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        _EVAL_POLICY.main()

    summary = json.loads((output / "distributed_summary.json").read_text())
    task = summary["tasks"]["pick"]
    leaderboard = json.loads((output / "summary.json").read_text())
    leaderboard_task = leaderboard["task_results"]["pick"]
    assert exc.value.code == 1
    assert (
        task["status"],
        task["episodes"],
        task["configs_failed"],
        task["configs_total"],
        summary["overall"]["total_episodes"],
        summary["overall"]["success_rate"],
    ) == (
        "error" if failure == "all" else "partial_error",
        0 if failure == "all" else 1,
        2 if failure == "all" and layout == "configs" else 1,
        2 if layout == "configs" else 1,
        0 if failure == "all" else 1,
        0.0 if failure == "all" else 1.0,
    )
    assert (
        leaderboard["status"],
        leaderboard_task["episode_count"],
        leaderboard_task["failed_episode_count"],
        leaderboard_task["score"],
    ) == (
        "failed",
        2,
        2 if failure == "all" else 1,
        0.0 if failure == "all" else 0.5,
    )


def test_eval_policy_distributed_success_preserves_leaderboard_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import subprocess

    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text("{}\n")
    config = tmp_path / "eval.yaml"
    config.write_text(
        "evaluation_id: run-42\n"
        "team_id: team-a\n"
        "policy:\n  model_type: dummy\n"
        f"defaults:\n  asset_root: {tmp_path}\n  episode_num: 2\n"
        "tasks:\n  pick:\n    task_type: pick_category\n"
        f"    yaml: {task_yaml}\n    shards: 2\n"
    )
    output = tmp_path / "results"

    class WorkerProcess:
        def __init__(self, cmd, **kwargs):
            manifest = read_evaluation_manifest(
                cmd[cmd.index("--_shard-manifest") + 1]
            )
            shard_id = cmd[cmd.index("--_shard-id") + 1]
            shard = next(s for s in manifest.shards if s.shard_id == shard_id)
            shard_dir = output / "pick" / "shards" / shard_id
            result_path = shard_dir / "eval_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "episode_results": [
                            {
                                "seed": shard.seed_start,
                                "success": True,
                                "progress": 0.5 + 0.5 * shard.shard_index,
                                "steps": 5,
                                "stop_reason": "done",
                                "instruction": "pick the mug",
                                "stage_scores": {"pick": 1.0},
                                "metrics": {
                                    "criteria_reached": {"pick": True}
                                },
                            }
                        ],
                        "attempted_scene_seeds": [shard.seed_start],
                        "skipped_episodes": [],
                    }
                )
            )
            (shard_dir / "shard_summary.json").write_text(
                json.dumps(
                    {
                        "shard": shard.to_payload(),
                        "configs": [{"eval_result_json": str(result_path)}],
                    }
                )
            )

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", WorkerProcess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_EVAL_POLICY_PATH),
            "--eval-config",
            str(config),
            "--output-dir",
            str(output),
            "--gpus",
            "0,1",
        ],
    )
    with pytest.raises(SystemExit, match="0"):
        _EVAL_POLICY.main()

    summary = json.loads((output / "summary.json").read_text())
    assert (
        summary["evaluation_id"],
        summary["team_id"],
        summary["status"],
        summary["overall_score"],
        summary["task_results"]["pick"]["stage_success_rate"],
        [
            episode["instruction"]
            for episode in summary["task_results"]["pick"]["preview_episodes"]
        ],
    ) == (
        "run-42",
        "team-a",
        "succeeded",
        0.75,
        {"pick": 1.0},
        [
            "pick the mug",
            "pick the mug",
        ],
    )


@pytest.mark.parametrize(
    ("gpus", "expected"),
    [
        ("0", ["heavy", "light"]),
        ("0,1,2,3", ["heavy", "heavy", "heavy", "light"]),
    ],
)
def test_eval_policy_weighted_tasks_dispatches_expected_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gpus: str,
    expected: list[str],
) -> None:
    import subprocess

    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text("{}\n")
    config = tmp_path / "eval.yaml"
    config.write_text(
        "policy:\n  model_type: dummy\n"
        f"defaults:\n  asset_root: {tmp_path}\n  episode_num: 100\n"
        "tasks:\n  light:\n    task_type: pick_category\n"
        f"    yaml: {task_yaml}\n    max_steps: 100\n"
        "  heavy:\n    task_type: pick_category\n"
        f"    yaml: {task_yaml}\n    max_steps: 1000\n"
    )
    dispatched = []

    class WorkerProcess:
        def __init__(self, cmd, **kwargs):
            manifest = read_evaluation_manifest(
                cmd[cmd.index("--_shard-manifest") + 1]
            )
            dispatched.append(manifest.shards[0].task_name)
            self.returncode = 1

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", WorkerProcess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_EVAL_POLICY_PATH),
            "--eval-config",
            str(config),
            "--output-dir",
            str(tmp_path / "output"),
            "--gpus",
            gpus,
        ],
    )
    with pytest.raises(SystemExit, match="1"):
        _EVAL_POLICY.main()

    # One GPU exposes queue order; concurrent starts may interleave.
    assert (dispatched if gpus == "0" else sorted(dispatched)) == expected
