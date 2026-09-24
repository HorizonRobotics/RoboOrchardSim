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

"""Persistent run manifest for distributed policy evaluation."""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robo_orchard_sim.pipeline.evaluator.distributed.sharding import (
    EvaluationShard,
)

MANIFEST_VERSION = 2


@dataclass(frozen=True)
class EvaluationManifest:
    """Complete immutable work definition for one evaluation run."""

    run_id: str
    policy_id: str
    shards: tuple[EvaluationShard, ...]
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.policy_id:
            raise ValueError("policy_id must not be empty")
        shard_ids = [shard.shard_id for shard in self.shards]
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("manifest contains duplicate shard ids")

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""
        return {
            "version": self.version,
            "run_id": self.run_id,
            "policy_id": self.policy_id,
            "shards": [shard.to_payload() for shard in self.shards],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EvaluationManifest":
        """Build and validate a manifest from decoded JSON."""
        version = int(payload["version"])
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"unsupported evaluation manifest version: {version}"
            )
        return cls(
            version=version,
            run_id=str(payload["run_id"]),
            policy_id=str(payload["policy_id"]),
            shards=tuple(
                EvaluationShard(**shard) for shard in payload["shards"]
            ),
        )


def write_evaluation_manifest(
    manifest: EvaluationManifest,
    output_path: str | Path,
) -> Path:
    """Atomically persist a manifest and return its resolved path."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temp_path.write_text(
            json.dumps(
                manifest.to_payload(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def read_evaluation_manifest(
    manifest_path: str | Path,
) -> EvaluationManifest:
    """Read and validate an evaluation manifest."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation manifest must contain a JSON object")
    return EvaluationManifest.from_payload(payload)
