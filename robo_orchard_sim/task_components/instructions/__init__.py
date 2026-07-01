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

"""Instruction helpers for task definitions."""

from robo_orchard_sim.task_components.instructions.base import (  # noqa: F401
    InstructionActor,
    InstructionRenderError,
    InstructionWrapper,
    render_instruction_from_registry,
)
from robo_orchard_sim.task_components.instructions.mcap_render import (  # noqa: F401
    extract_instruction_actor_uuids_from_mcap,
    render_instruction_from_mcap,
    render_instructions_from_mcaps,
)
from robo_orchard_sim.task_components.instructions.registry import (  # noqa: F401
    INSTRUCTION_TEMPLATE_REGISTRY,
    build_instruction_wrapper,
    get_instruction_template,
    register_instruction_template,
)

__all__ = [
    "InstructionWrapper",
    "InstructionActor",
    "InstructionRenderError",
    "render_instruction_from_registry",
    "extract_instruction_actor_uuids_from_mcap",
    "render_instruction_from_mcap",
    "render_instructions_from_mcaps",
    "INSTRUCTION_TEMPLATE_REGISTRY",
    "build_instruction_wrapper",
    "get_instruction_template",
    "register_instruction_template",
]
