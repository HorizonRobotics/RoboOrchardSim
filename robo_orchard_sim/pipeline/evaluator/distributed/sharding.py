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

"""Deterministic episode sharding for multi-client policy evaluation."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from robo_orchard_sim.pipeline.evaluator.base import (
    RESAMPLE_ATTEMPT_MULTIPLIER,
)


class BatchGroupLike(Protocol):
    """Minimal batch-group fields required for evaluation sharding."""

    @property
    def group_id(self) -> str: ...

    @property
    def seed(self) -> int: ...

    @property
    def configs(self) -> Sequence[str]: ...


class BatchPlanLike(Protocol):
    """Minimal batch-plan fields required for evaluation sharding."""

    @property
    def task(self) -> str: ...

    @property
    def episodes_per_config(self) -> int: ...

    @property
    def groups(self) -> Sequence[BatchGroupLike]: ...


@dataclass(frozen=True)
class EvaluationShard:
    """A contiguous, independently executable portion of one task config."""

    shard_id: str
    task_name: str
    task_type: str
    group_id: str
    config_index: int
    config_path: str
    shard_index: int
    shard_count: int
    episode_offset: int
    episode_count: int
    seed_start: int
    fallback_seed_start: int
    fallback_seed_count: int
    episodes_per_scene: int

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable shard representation."""
        return asdict(self)

    def episode_keys(self) -> tuple[tuple[int, int], ...]:
        """Return stable `(scene_seed, episode_in_scene)` work identities."""
        keys = []
        for episode_index in range(
            self.episode_offset,
            self.episode_offset + self.episode_count,
        ):
            scene_index = episode_index // self.episodes_per_scene
            episode_in_scene = episode_index % self.episodes_per_scene
            scene_seed = (
                self.seed_start
                - self.episode_offset // self.episodes_per_scene
                + scene_index
            )
            keys.append((scene_seed, episode_in_scene))
        return tuple(keys)

    def candidate_scene_seeds(self) -> tuple[int, ...]:
        """Return this shard's primary seeds followed by fallback seeds."""
        primary_scene_count = -(-self.episode_count // self.episodes_per_scene)
        return (
            *range(self.seed_start, self.seed_start + primary_scene_count),
            *range(
                self.fallback_seed_start,
                self.fallback_seed_start + self.fallback_seed_count,
            ),
        )


def build_evaluation_shards(
    *,
    plan: BatchPlanLike,
    task_name: str,
    shards_per_config: int,
    episodes_per_scene: int = 1,
) -> list[EvaluationShard]:
    """Split every config's episode range into deterministic scene shards."""
    if shards_per_config < 1:
        raise ValueError("shards_per_config must be positive")
    if episodes_per_scene < 1:
        raise ValueError("episodes_per_scene must be positive")
    if plan.episodes_per_config < 1:
        raise ValueError("plan episodes_per_config must be positive")

    shards: list[EvaluationShard] = []
    for group in plan.groups:
        fallback_seed_cursor = (
            group.seed + len(group.configs) * plan.episodes_per_config
        )
        for config_index, config_path in enumerate(group.configs):
            config_seed = group.seed + config_index * plan.episodes_per_config
            config_shards = _build_config_shards(
                task_name=task_name,
                task_type=plan.task,
                group_id=group.group_id,
                config_index=config_index,
                config_path=config_path,
                episode_count=plan.episodes_per_config,
                seed_start=config_seed,
                fallback_seed_start=fallback_seed_cursor,
                shards_per_config=shards_per_config,
                episodes_per_scene=episodes_per_scene,
            )
            shards.extend(config_shards)
            fallback_seed_cursor += sum(
                shard.fallback_seed_count for shard in config_shards
            )
    _validate_unique_work(shards)
    return shards


def _build_config_shards(
    *,
    task_name: str,
    task_type: str,
    group_id: str,
    config_index: int,
    config_path: str,
    episode_count: int,
    seed_start: int,
    fallback_seed_start: int,
    shards_per_config: int,
    episodes_per_scene: int,
) -> list[EvaluationShard]:
    scene_count = -(-episode_count // episodes_per_scene)
    shard_count = min(shards_per_config, scene_count)
    base_scenes, extra_scenes = divmod(scene_count, shard_count)

    shards = []
    scene_offset = 0
    fallback_offset = 0
    for shard_index in range(shard_count):
        assigned_scenes = base_scenes + int(shard_index < extra_scenes)
        episode_offset = scene_offset * episodes_per_scene
        assigned_episodes = min(
            assigned_scenes * episodes_per_scene,
            episode_count - episode_offset,
        )
        fallback_seed_count = assigned_scenes * (
            RESAMPLE_ATTEMPT_MULTIPLIER - 1
        )
        shard_id = (
            f"{task_name}__{group_id}__config_{config_index:04d}"
            f"__shard_{shard_index:04d}"
        )
        shards.append(
            EvaluationShard(
                shard_id=shard_id,
                task_name=task_name,
                task_type=task_type,
                group_id=group_id,
                config_index=config_index,
                config_path=config_path,
                shard_index=shard_index,
                shard_count=shard_count,
                episode_offset=episode_offset,
                episode_count=assigned_episodes,
                seed_start=seed_start + scene_offset,
                fallback_seed_start=fallback_seed_start + fallback_offset,
                fallback_seed_count=fallback_seed_count,
                episodes_per_scene=episodes_per_scene,
            )
        )
        scene_offset += assigned_scenes
        fallback_offset += fallback_seed_count
    return shards


def _validate_unique_work(shards: list[EvaluationShard]) -> None:
    shard_ids: set[str] = set()
    work_keys: set[tuple[str, str, int, int, int]] = set()
    candidate_seed_owners: dict[tuple[str, int], str] = {}
    for shard in shards:
        if shard.shard_id in shard_ids:
            raise ValueError(f"duplicate evaluation shard: {shard.shard_id}")
        shard_ids.add(shard.shard_id)
        for scene_seed, episode_in_scene in shard.episode_keys():
            work_key = (
                shard.group_id,
                shard.config_path,
                shard.config_index,
                scene_seed,
                episode_in_scene,
            )
            if work_key in work_keys:
                raise ValueError(
                    f"duplicate evaluation episode assignment: {work_key}"
                )
            work_keys.add(work_key)
        for seed in shard.candidate_scene_seeds():
            candidate_key = (shard.group_id, seed)
            previous_owner = candidate_seed_owners.get(candidate_key)
            if previous_owner is not None:
                raise ValueError(
                    "duplicate evaluation candidate seed "
                    f"{seed} in shards {previous_owner!r} and "
                    f"{shard.shard_id!r}"
                )
            candidate_seed_owners[candidate_key] = shard.shard_id


def validate_shard_result(
    shard: EvaluationShard,
    result_payload: Mapping[str, Any],
) -> None:
    """Validate completed episodes against one shard's candidate schedule."""
    episode_results = result_payload.get("episode_results")
    skipped_episodes = result_payload.get("skipped_episodes")
    attempted_scene_seeds = result_payload.get("attempted_scene_seeds")
    if not isinstance(episode_results, list):
        raise ValueError("result is missing episode_results")
    if not isinstance(skipped_episodes, list):
        raise ValueError("result is missing skipped_episodes")
    if not isinstance(attempted_scene_seeds, list):
        raise ValueError("result is missing attempted_scene_seeds")

    actual_seeds = [int(episode["seed"]) for episode in episode_results]
    skipped_seeds = [int(episode["seed"]) for episode in skipped_episodes]
    attempted_seeds = [int(seed) for seed in attempted_scene_seeds]
    candidates = shard.candidate_scene_seeds()

    if len(actual_seeds) != shard.episode_count:
        raise ValueError(
            f"episode count mismatch: expected={shard.episode_count}, "
            f"actual={len(actual_seeds)}"
        )
    if attempted_seeds != list(candidates[: len(attempted_seeds)]):
        raise ValueError(
            "attempted scene seeds do not match the shard candidate schedule"
        )

    actual_scene_seeds = set(actual_seeds)
    expected_skipped_seeds = [
        seed for seed in attempted_seeds if seed not in actual_scene_seeds
    ]
    if skipped_seeds != expected_skipped_seeds:
        raise ValueError(
            "skipped scene seeds do not match attempted scene seeds: "
            f"expected={expected_skipped_seeds}, actual={skipped_seeds}"
        )

    skipped_seed_set = set(skipped_seeds)
    expected_actual_seeds = []
    remaining_episodes = shard.episode_count
    for seed in attempted_seeds:
        if remaining_episodes == 0:
            raise ValueError(
                "attempted scene seeds continue after the shard completed"
            )
        if seed in skipped_seed_set:
            continue
        scene_episodes = min(
            shard.episodes_per_scene,
            remaining_episodes,
        )
        expected_actual_seeds.extend([seed] * scene_episodes)
        remaining_episodes -= scene_episodes
    if actual_seeds != expected_actual_seeds:
        raise ValueError(
            "episode seeds do not match successful attempted scenes: "
            f"expected={expected_actual_seeds}, actual={actual_seeds}"
        )
