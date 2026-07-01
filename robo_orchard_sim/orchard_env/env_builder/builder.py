# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

"""Env builder that composes scene + embodiment + task configs."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any

from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationManagerCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.envs.env_cfg import ViewerCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.simulation_cfg import SimulationCfg
from robo_orchard_sim.ext.envs.manager_based_env import IsaacManagerBasedEnvCfg
from robo_orchard_sim.ext.envs.managers.record import (
    NoOpRecordControllerCfg,
    RecordControllerCfg,
    RecordManagerCfg,
    RecordTermBaseCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.ext.models.scenes.asset_scene import AssetSceneCfg
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)
from robo_orchard_sim.orchard_env.layout.builder import LayoutBuilder
from robo_orchard_sim.orchard_env.scene.scene_base import SceneBase
from robo_orchard_sim.orchard_env.task_templates.task_base import TaskBase


class EnvBuilder:
    """Compose scene, embodiment, and task into one env cfg."""

    def __init__(
        self,
        scene: SceneBase,
        embodiment: EmbodimentBase,
        task: TaskBase,
        layout_builder: LayoutBuilder | None = None,
        record_file_path: str = "logs/records",
        record_controller: RecordControllerCfg | None = None,
    ):
        self.scene = scene
        self.embodiment = embodiment
        self.task = task
        self.layout_builder = layout_builder
        self.record_file_path = record_file_path
        self.record_controller = record_controller or NoOpRecordControllerCfg()

    def build(self) -> IsaacManagerBasedEnvCfg:
        """Build the final ``IsaacManagerBasedEnvCfg``."""
        scene_cfg = AssetSceneCfg(
            num_envs=self._get_num_envs(),
            env_spacing=self._get_env_spacing(),
            assets=self._merge_assets(
                self.scene.get_assets_cfg(),
                self.embodiment.get_assets_cfg(),
                self.task.get_assets_cfg(),
            ),
        )

        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=self._get_decimation(),
            viewer=self._get_viewer_cfg(),
            sim=self._get_sim_cfg(),
            scene=scene_cfg,
        )

        env_cfg.observations = self._merge_observation_cfg(
            self.scene.get_observation_cfg(),
            self.embodiment.get_observation_cfg(),
            self.task.get_observation_cfg(),
        )
        env_cfg.actions = self._merge_action_cfg(
            self.scene.get_action_cfg(),
            self.embodiment.get_action_cfg(),
            self.task.get_action_cfg(),
        )
        task_event_cfg = self.task.get_event_cfg()
        if self.layout_builder is not None:
            task_event_cfg = self.layout_builder.apply_to(task_event_cfg)
        env_cfg.events = self._merge_event_cfg(
            self.scene.get_event_cfg(),
            self.embodiment.get_event_cfg(),
            task_event_cfg,
        )
        env_cfg.records = self._build_record_cfg(
            self.scene.get_record_terms(),
            self.embodiment.get_record_terms(),
            self.task.get_record_terms(),
        )
        return env_cfg

    def _get_decimation(self) -> int:
        return self.scene.get_decimation()

    def _get_viewer_cfg(self) -> ViewerCfg:
        return self.scene.get_viewer_cfg()

    def _get_sim_cfg(self) -> SimulationCfg:
        return self.scene.get_sim_cfg()

    def _get_num_envs(self) -> int:
        return self.scene.get_num_envs()

    def _get_env_spacing(self) -> float:
        return self.scene.get_env_spacing()

    @staticmethod
    def _to_dict(fragment: Any) -> dict[str, Any]:
        if fragment is None:
            return {}
        if isinstance(fragment, dict):
            return dict(fragment)
        return dict(fragment)

    def _merge_assets(
        self,
        *asset_groups: dict[str, GroupAssetCfg],
    ) -> dict[str, GroupAssetCfg]:
        merged: dict[str, dict[str, Any]] = {}
        for assets in asset_groups:
            for namespace, group_cfg in assets.items():
                merged.setdefault(namespace, {})
                group_assets = self._to_dict(group_cfg)
                for asset_name, asset_cfg in group_assets.items():
                    if asset_name in merged[namespace]:
                        raise ValueError(
                            "Duplicate asset "
                            f"'{namespace}/{asset_name}' when merging scene "
                            "assets."
                        )
                    merged[namespace][asset_name] = asset_cfg
        merged = self._maybe_inject_inactive_pool_storage(merged)
        return {
            namespace: GroupAssetCfg(**group_assets)
            for namespace, group_assets in merged.items()
        }

    @staticmethod
    def _maybe_inject_inactive_pool_storage(
        merged: dict[str, dict],
    ) -> dict[str, dict]:
        """Insert the inactive-pool storage shelf when any role is pooled."""
        any_pool = any(
            "_pool_" in name for group in merged.values() for name in group
        )
        if not any_pool:
            return merged
        from robo_orchard_sim.orchard_env.env_builder.inactive_pool_storage import (  # noqa: E501
            INACTIVE_POOL_STORAGE_NAME,
            make_inactive_pool_storage_cfg,
        )

        merged.setdefault("objects", {})
        if INACTIVE_POOL_STORAGE_NAME not in merged["objects"]:
            merged["objects"][INACTIVE_POOL_STORAGE_NAME] = (
                make_inactive_pool_storage_cfg()
            )
        return merged

    def _merge_observation_cfg(
        self,
        *fragments: Any,
    ) -> ObservationManagerCfg:
        groups: dict[str, Any] = {}
        for fragment in fragments:
            if fragment is None:
                continue
            fragment_groups = (
                fragment.groups
                if isinstance(fragment, ObservationManagerCfg)
                else self._to_dict(fragment)
            )
            for key, value in fragment_groups.items():
                if key in groups:
                    raise ValueError(f"Duplicate observation group '{key}'.")
                groups[key] = value
        return ObservationManagerCfg(
            concatenate_terms=False,
            groups=groups,
        )

    def _merge_action_cfg(self, *fragments: Any) -> ActionManagerCfg:
        terms: dict[str, Any] = {}
        for fragment in fragments:
            if fragment is None:
                continue
            fragment_terms = (
                fragment.terms
                if isinstance(fragment, ActionManagerCfg)
                else self._to_dict(fragment)
            )
            for key, value in fragment_terms.items():
                if key in terms:
                    raise ValueError(f"Duplicate action term '{key}'.")
                terms[key] = value
        return ActionManagerCfg(terms=terms)

    def _merge_event_cfg(self, *fragments: Any) -> EventManagerCfg:
        terms: dict[str, Any] = {}
        for fragment in fragments:
            if fragment is None:
                continue
            fragment_terms = (
                fragment.terms
                if isinstance(fragment, EventManagerCfg)
                else self._to_dict(fragment)
            )
            for key, value in fragment_terms.items():
                if key in terms:
                    raise ValueError(f"Duplicate event term '{key}'.")
                terms[key] = value
        return EventManagerCfg(terms=terms)

    def _build_record_cfg(
        self,
        *fragments: Mapping[str, RecordTermBaseCfg],
    ) -> RecordManagerCfg:
        terms: dict[str, RecordTermBaseCfg] = {}

        for fragment in fragments:
            for key, value in fragment.items():
                if key in terms:
                    raise ValueError(f"Duplicate record term '{key}'.")
                terms[key] = value

        return RecordManagerCfg(
            file_path=self.record_file_path,
            controller=self.record_controller,
            terms=terms,
        )
