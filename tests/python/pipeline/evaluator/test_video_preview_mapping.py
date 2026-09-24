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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video preview mapping for completed episodes in partial configurations."""

import json
from pathlib import Path

import pytest

from tools.mcap_to_video import collect_preview_recordings


@pytest.mark.parametrize("with_record_dir", [True, False])
def test_collect_preview_recordings_partial_config_maps_completed_episode(
    tmp_path: Path,
    with_record_dir: bool,
) -> None:
    task_dir = tmp_path / "pick"
    config_dir = task_dir / "group_0000" / "config_0000"
    config_dir.mkdir(parents=True)
    recording = (
        task_dir
        / "shards"
        / "shard_0000"
        / "records"
        / "episode_0000_seed_42"
        / "record.mcap"
    )
    recording.parent.mkdir(parents=True)
    recording.touch()
    episode = {
        "seed": 42,
        "steps": 5,
        "stop_reason": "success",
        "instruction": "pick the mug",
        "metrics": (
            {"record_dir": str(recording.parent)} if with_record_dir else {}
        ),
    }
    (config_dir / "eval_result.json").write_text(
        json.dumps({"episode_results": [episode]}), encoding="utf-8"
    )
    (task_dir / "task_eval_summary.json").write_text(
        json.dumps(
            {
                "configs": [
                    {
                        "group_id": "group_0000",
                        "config_index": 0,
                        "error": "another shard failed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    preview = {
        "episode_index": 0,
        "seed": 42,
        "steps": 5,
        "stop_reason": "success",
        "instruction": "pick the mug",
    }

    selected, unresolved = collect_preview_recordings(
        tmp_path, {"task_results": {"pick": {"preview_episodes": [preview]}}}
    )

    assert (selected, unresolved) == ([(preview, recording)], 0)
