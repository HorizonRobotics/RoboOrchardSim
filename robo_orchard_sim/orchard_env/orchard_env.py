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

"""Top-level orchard env description object."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_orchard_sim.envs.manager_based_env import (
        IsaacManagerBasedEnvCfg,
    )
    from robo_orchard_sim.envs.managers.record import RecordControllerCfg
    from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
        EmbodimentBase,
    )
    from robo_orchard_sim.orchard_env.scene.scene_base import SceneBase
    from robo_orchard_sim.orchard_env.tasks.task_base import TaskBase


class OrchardEnv:
    """Top-level orchard env description used by downstream callers."""

    DEFAULT_RECORD_FILE_PATH = "logs/records"

    def __init__(
        self,
        embodiment: EmbodimentBase,
        task: TaskBase,
        scene: SceneBase | None = None,
    ):
        if scene is None:
            from robo_orchard_sim.orchard_env.scene.plane_table_scene import (  # noqa: E501
                PlaneTableScene,
            )

            scene = PlaneTableScene()
        self.scene = scene
        self.embodiment = embodiment
        self.task = task
        from robo_orchard_sim.envs.managers.record import (
            NoOpRecordControllerCfg,
        )

        self._record_file_path = self.DEFAULT_RECORD_FILE_PATH
        self._record_controller: RecordControllerCfg = (
            NoOpRecordControllerCfg()
        )

    def configure_recording(
        self,
        file_path: str | None = None,
        controller: RecordControllerCfg | None = None,
    ) -> "OrchardEnv":
        """Enable recording with optional file path and controller."""
        from robo_orchard_sim.envs.managers.record import (
            EpisodeRecordControllerCfg,
        )

        if file_path is not None:
            self._record_file_path = file_path
        self._record_controller = controller or EpisodeRecordControllerCfg()
        return self

    def disable_recording(self) -> "OrchardEnv":
        """Disable recording while keeping the current file path."""
        from robo_orchard_sim.envs.managers.record import (
            NoOpRecordControllerCfg,
        )

        self._record_controller = NoOpRecordControllerCfg()
        return self

    def to_isaac_env_cfg(self) -> IsaacManagerBasedEnvCfg:
        """Build an Isaac-compatible env cfg from the orchard env."""
        from robo_orchard_sim.orchard_env.env_builder.builder import (
            EnvBuilder,
        )

        return EnvBuilder(
            scene=self.scene,
            embodiment=self.embodiment,
            task=self.task,
            record_file_path=self._record_file_path,
            record_controller=self._record_controller,
        ).build()
