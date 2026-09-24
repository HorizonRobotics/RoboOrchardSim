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

"""Close-only articulated manipulation task."""

from typing import ClassVar

from robo_orchard_sim.contracts.articulated_operation import (
    ArticulatedOperation,
)
from robo_orchard_sim.orchard_env.task_templates.articulated.joint_direction_task import (  # noqa: E501
    JointOperationTask,
)


class JointCloseTask(JointOperationTask):
    """Grade only the closing of an articulation joint."""

    ALLOWED_OPERATIONS: ClassVar[frozenset[ArticulatedOperation]] = frozenset(
        {"close"}
    )


__all__ = ["JointCloseTask"]
