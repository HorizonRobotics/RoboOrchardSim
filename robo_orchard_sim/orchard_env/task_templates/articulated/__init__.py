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

"""Task templates for articulated manipulation."""

from robo_orchard_sim.orchard_env.task_templates.articulated.affordance_task import (  # noqa: E501, F401
    AffordanceTask,
    AffordanceTaskParams,
    AffordanceValidatorConfig,
)
from robo_orchard_sim.orchard_env.task_templates.articulated.base import (  # noqa: F401
    PRIMARY_ROLE,
    ArticulatedTaskBase,
    ArticulatedTaskParams,
)
from robo_orchard_sim.orchard_env.task_templates.articulated.close_task import (  # noqa: E501, F401
    JointCloseTask,
)
from robo_orchard_sim.orchard_env.task_templates.articulated.joint_direction_task import (  # noqa: E501, F401
    JointDirectionTask,
    JointDirectionTaskParams,
    JointOperationTask,
    JointOperationTaskParams,
    JointOperationValidatorConfig,
)
from robo_orchard_sim.orchard_env.task_templates.articulated.open_task import (  # noqa: E501, F401
    JointOpenTask,
)

__all__ = [
    "PRIMARY_ROLE",
    "AffordanceTask",
    "AffordanceTaskParams",
    "AffordanceValidatorConfig",
    "ArticulatedTaskBase",
    "ArticulatedTaskParams",
    "JointCloseTask",
    "JointDirectionTask",
    "JointDirectionTaskParams",
    "JointOpenTask",
    "JointOperationTask",
    "JointOperationTaskParams",
    "JointOperationValidatorConfig",
]
