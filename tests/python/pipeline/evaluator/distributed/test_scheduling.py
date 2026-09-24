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

import pytest

from robo_orchard_sim.pipeline.evaluator.distributed import (
    EvaluationExecutionConfig,
)
from robo_orchard_sim.pipeline.evaluator.distributed.scheduling import (
    EvaluationWorkload,
    allocate_task_shards,
)


@pytest.mark.parametrize(
    ("gpus", "work_items", "expected"),
    [
        (("0",), 6, 1),
        (("0", "1", "2", "3"), 6, 4),
        (("0", "1", "2", "3"), 1, 1),
        (("0",), 0, 0),
    ],
)
def test_execution_config_resource_layout_returns_expected_concurrency(
    gpus: tuple[str, ...],
    work_items: int,
    expected: int,
) -> None:
    assert EvaluationExecutionConfig(gpus).concurrency(work_items) == expected


def test_execution_config_duplicate_gpus_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unique"):
        EvaluationExecutionConfig(("0", "0"))


@pytest.mark.parametrize(
    ("workloads", "gpus", "expected"),
    [
        ([EvaluationWorkload(100, 100)] * 6, 4, [1] * 6),
        ([EvaluationWorkload(100, 100)] * 2, 4, [2, 2]),
        (
            [EvaluationWorkload(100, 1000), EvaluationWorkload(100, 100)],
            4,
            [3, 1],
        ),
        (
            [EvaluationWorkload(1000, 100), EvaluationWorkload(100, 100)],
            4,
            [3, 1],
        ),
        (
            [
                EvaluationWorkload(100, 1000, shards=1),
                EvaluationWorkload(100, 100),
            ],
            4,
            [1, 3],
        ),
        ([EvaluationWorkload(3, 100, episodes_per_scene=2)], 4, [2]),
        ([EvaluationWorkload(100, 100, config_count=4)], 4, [1]),
        ([EvaluationWorkload(100, 100)] * 2, 1, [1, 1]),
        ([EvaluationWorkload(1, 100, shards=4)], 4, [1]),
    ],
)
def test_allocate_task_shards_resource_layout_returns_expected_counts(
    workloads: list[EvaluationWorkload],
    gpus: int,
    expected: list[int],
) -> None:
    assert allocate_task_shards(workloads, gpus) == expected
