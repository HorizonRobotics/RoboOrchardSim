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

"""Base abstractions for task-suite task definitions."""

from __future__ import annotations
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import yaml
from pydantic import Field
from robo_orchard_core.utils.config import Config
from typing_extensions import Literal

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
        EmbodimentBase,
    )
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
    from robo_orchard_sim.orchard_env.scene.scene_base import SceneBase
    from robo_orchard_sim.task_components.instructions.base import (
        InstructionWrapper,
    )
    from robo_orchard_sim.task_components.trajs_gen.base_executor import (
        BaseExecutorCfg,
    )
from robo_orchard_sim.task_components.instructions.registry import (
    build_instruction_wrapper,
)

SCENE_REGISTRY: dict[str, type["SceneBase"]] = {}
EMBODIMENT_REGISTRY: dict[str, type["EmbodimentBase"]] = {}


class SceneConfig(Config):
    """Strongly typed scene configuration loaded from task YAML."""

    type: str
    num_envs: int = 1
    env_spacing: float = 2.5
    physics_fps: int = 600
    render_fps: int = 30
    step_fps: int = 30
    params: dict[str, Any] = Field(default_factory=dict)


class EmbodimentConfig(Config):
    """Strongly typed embodiment configuration loaded from task YAML."""

    type: str
    initial_pos: tuple[float, float, float] | None = None
    init_joint_noise_std: float | None = None
    init_joint_pos: dict[str, float] | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class InstructionConfig(Config):
    """Strongly typed instruction configuration loaded from task YAML."""

    template: str
    template_mode: Literal["fixed", "variants"] = "fixed"
    actor_description_mode: Literal["raw", "seen", "unseen"] = "raw"
    attribute_name: Literal["color", "shape", "material"] | None = None


class TaskConfig(Config):
    """Strongly typed task-level configuration loaded from task YAML."""

    params: dict[str, Any] = Field(default_factory=dict)


class TaskDefinitionConfig(Config):
    """Typed task-definition YAML payload."""

    scene: SceneConfig | None = None
    embodiment: EmbodimentConfig | None = None
    instruction: InstructionConfig | None = None
    asset_configs: dict[str, dict[str, Any]] | None = None
    layout: str | None = None
    task: TaskConfig | None = None


def register_scene(
    name: str,
    scene_cls: type["SceneBase"],
) -> type["SceneBase"]:
    """Register a scene class under its string type name."""
    registered = SCENE_REGISTRY.get(name)
    if registered is scene_cls:
        return scene_cls
    if registered is not None:
        raise ValueError(f"Duplicate scene registered: {name!r}.")
    SCENE_REGISTRY[name] = scene_cls
    return scene_cls


def register_embodiment(
    name: str,
    embodiment_cls: type["EmbodimentBase"],
) -> type["EmbodimentBase"]:
    """Register an embodiment class under its string type name."""
    registered = EMBODIMENT_REGISTRY.get(name)
    if registered is embodiment_cls:
        return embodiment_cls
    if registered is not None:
        raise ValueError(f"Duplicate embodiment registered: {name!r}.")
    EMBODIMENT_REGISTRY[name] = embodiment_cls
    return embodiment_cls


def _bootstrap_scene_registry() -> None:
    if "plane_table" not in SCENE_REGISTRY:
        from robo_orchard_sim.orchard_env.scene.plane_table_scene import (
            PlaneTableScene,
        )

        register_scene("plane_table", PlaneTableScene)
    if "room_table" not in SCENE_REGISTRY:
        from robo_orchard_sim.orchard_env.scene.room_table_scene import (
            RoomTableScene,
        )

        register_scene("room_table", RoomTableScene)


def _bootstrap_embodiment_registry() -> None:
    if "dualarm_piper" not in EMBODIMENT_REGISTRY:
        from robo_orchard_sim.orchard_env.embodiments.dualarm_piper import (
            DualArmPiperEmbodiment,
        )

        register_embodiment("dualarm_piper", DualArmPiperEmbodiment)
    if "dualarm_piperx" not in EMBODIMENT_REGISTRY:
        from robo_orchard_sim.orchard_env.embodiments.dualarm_piperx import (
            DualArmPiperXEmbodiment,
        )

        register_embodiment("dualarm_piperx", DualArmPiperXEmbodiment)
    if "franka_panda" not in EMBODIMENT_REGISTRY:
        from robo_orchard_sim.orchard_env.embodiments.franka_panda import (
            FrankaPandaEmbodiment,
        )

        register_embodiment("franka_panda", FrankaPandaEmbodiment)
    if "panda_droid" not in EMBODIMENT_REGISTRY:
        from robo_orchard_sim.orchard_env.embodiments.panda_droid import (
            PandaDroidEmbodiment,
        )

        register_embodiment("panda_droid", PandaDroidEmbodiment)


def build_scene(cfg: SceneConfig) -> "SceneBase":
    """Construct a scene from registry-backed scene config."""
    _bootstrap_scene_registry()
    try:
        scene_cls = SCENE_REGISTRY[cfg.type]
    except KeyError as exc:
        known_scenes = ", ".join(sorted(SCENE_REGISTRY))
        raise ValueError(
            f"Unknown scene {cfg.type!r}. Known scenes: {known_scenes}."
        ) from exc
    return scene_cls(
        num_envs=cfg.num_envs,
        env_spacing=cfg.env_spacing,
        physics_fps=cfg.physics_fps,
        render_fps=cfg.render_fps,
        step_fps=cfg.step_fps,
        **cfg.params,
    )


def build_embodiment(cfg: EmbodimentConfig) -> "EmbodimentBase":
    """Construct an embodiment from registry-backed embodiment config."""
    _bootstrap_embodiment_registry()
    try:
        embodiment_cls = EMBODIMENT_REGISTRY[cfg.type]
    except KeyError as exc:
        known_embodiments = ", ".join(sorted(EMBODIMENT_REGISTRY))
        raise ValueError(
            "Unknown embodiment "
            f"{cfg.type!r}. Known embodiments: {known_embodiments}."
        ) from exc
    kwargs = dict(cfg.params)
    if cfg.initial_pos is not None:
        kwargs["initial_pos"] = cfg.initial_pos
    if cfg.init_joint_noise_std is not None:
        kwargs["init_joint_noise_std"] = cfg.init_joint_noise_std
    if cfg.init_joint_pos is not None:
        kwargs["init_joint_pos"] = cfg.init_joint_pos
    return embodiment_cls(**kwargs)


class TaskDefinition(ABC):
    """Base class for a task-suite default task entry.

    Subclasses must define ``namespace`` and implement ``build()``.
    ``scene``, ``embodiment`` and ``instruction`` can be configured
    via a YAML file pointed to by ``config_path``, or set as class
    attributes directly.

    YAML layout::

        scene:
          type: plane_table
          num_envs: 1
          env_spacing: 2.5
          physics_fps: 600
          render_fps: 30
          step_fps: 30

        embodiment:
          type: dualarm_piper
          initial_pos: [0.0, 0.3, 0.0]
          init_joint_noise_std: 0.05

        instruction:
          template: place_a2b_default
          template_mode: fixed
          actor_description_mode: raw
    """

    namespace: ClassVar[str]
    config_path: ClassVar[str | None] = None

    scene: ClassVar[str | SceneBase] = "plane_table"
    embodiment: ClassVar[str | EmbodimentBase] = "dualarm_piper"
    instruction: ClassVar[str | InstructionWrapper | None] = None
    asset: None  # TODO

    @classmethod
    def _load_config(
        cls,
        config_path: str | None = None,
    ) -> TaskDefinitionConfig:
        """Load and validate task-definition YAML config."""
        path_str = cls.config_path if config_path is None else config_path
        if path_str is None:
            return TaskDefinitionConfig()
        path = cls._resolve_config_path(path_str)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return TaskDefinitionConfig(**raw)

    @classmethod
    def _config_dir(cls, config_path: str | None = None) -> Path | None:
        path_str = cls.config_path if config_path is None else config_path
        if path_str is None:
            return None
        return cls._resolve_config_path(path_str).parent

    @classmethod
    def _resolve_config_path(cls, config_path: str | None = None) -> Path:
        path_str = cls.config_path if config_path is None else config_path
        if path_str is None:
            raise ValueError("config_path is not set.")
        path = Path(path_str)
        if path.is_absolute():
            return path
        module = sys.modules.get(cls.__module__)
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            return Path(module_file).resolve().parent / path
        return path.resolve()

    @classmethod
    def resolve_scene(cls, config_path: str | None = None) -> SceneBase:
        """Resolve scene from YAML config, falling back to class default."""
        cfg = cls._load_config(config_path=config_path).scene
        if cfg is not None:
            return build_scene(cfg)
        if isinstance(cls.scene, str):
            return build_scene(SceneConfig(type=cls.scene))
        return cls.scene

    @classmethod
    def resolve_embodiment(
        cls,
        config_path: str | None = None,
    ) -> EmbodimentBase:
        """Resolve embodiment from YAML config, falling back to default."""
        cfg = cls._load_config(config_path=config_path).embodiment
        if cfg is not None:
            return build_embodiment(cfg)
        if isinstance(cls.embodiment, str):
            return build_embodiment(EmbodimentConfig(type=cls.embodiment))
        return cls.embodiment

    @classmethod
    def resolve_instruction(
        cls,
        config_path: str | None = None,
    ) -> InstructionWrapper | None:
        """Resolve instruction from YAML config (or class attribute)."""
        cfg = cls._load_config(config_path=config_path)
        if cfg.instruction is not None:
            return build_instruction_wrapper(
                cfg.instruction.template,
                template_mode=cfg.instruction.template_mode,
                actor_description_mode=cfg.instruction.actor_description_mode,
                attribute_name=cfg.instruction.attribute_name,
            )
        if cls.instruction is None:
            return None
        if isinstance(cls.instruction, str):
            return build_instruction_wrapper(cls.instruction)
        return cls.instruction

    @classmethod
    def resolve_asset_configs(
        cls,
        config_path: str | None = None,
    ) -> dict[str, dict[str, Any]] | None:
        """Resolve per-role asset configs from YAML, or None if unset.

        When the task YAML contains an ``asset_configs:`` block, return
        it as the payload to pass to ``AssetResolver.resolve()``. The
        dict shape is: ``{role: {filter: ..., name: ..., ...}}``.

        Returning None means the caller must either supply
        ``asset_configs`` explicitly or fall through to whatever default
        asset path the concrete ``build()`` defines.
        """
        return cls._load_config(config_path=config_path).asset_configs

    @classmethod
    def resolve_task_params(
        cls,
        config_path: str | None = None,
    ) -> dict[str, Any]:
        """Resolve task-level params from YAML config."""
        cfg = cls._load_config(config_path=config_path).task
        if cfg is None:
            return {}
        return dict(cfg.params)

    @classmethod
    @abstractmethod
    def build(
        cls,
        resolver: "AssetResolver | None" = None,
        config_path: str | None = None,
    ) -> OrchardEnv:
        """Build a fresh orchard env for this task.

        Args:
            resolver: Optional ``AssetResolver`` instance used to sample
                task assets from the task configuration.
            config_path: Optional YAML path overriding ``cls.config_path``
                for this build only.
        """

    @classmethod
    def build_atomic_action_plan(
        cls,
        orchard_env: "OrchardEnv",
    ) -> list["BaseExecutorCfg"]:
        """Build the default atomic action plan for this task."""
        del cls, orchard_env
        return []
