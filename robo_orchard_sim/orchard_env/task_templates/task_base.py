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

"""Base abstractions for composable task configuration."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationGroupCfg,
    ObservationManagerCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.managers.scene_entity_cfg import (
    SceneEntityCfg,
)
from robo_orchard_sim.ext.envs.managers.observations.transform_frame import (
    FrameTransformTermCfg,
)
from robo_orchard_sim.ext.envs.managers.record import RecordTermBaseCfg
from robo_orchard_sim.ext.envs.managers.record.mcap import (
    McapDictTermCfg,
    McapMultiTFTermCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import ObjectSpec

if TYPE_CHECKING:
    from robo_orchard_sim.ext.envs.manager_based_env import (
        IsaacManagerBasedEnv,
    )
    from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
    from robo_orchard_sim.orchard_env.task_spec import RoleSpec
    from robo_orchard_sim.task_components.instructions.base import (
        InstructionActor,
        InstructionWrapper,
    )
    from robo_orchard_sim.task_components.role_registry import TargetRef
    from robo_orchard_sim.task_components.validators.base import (
        Validator,
    )
    from robo_orchard_sim.task_components.validators.context import (
        ValidatorContext,
    )
    from robo_orchard_sim.task_components.validators.physical_entity import (
        SceneEntityKey,
    )


class TaskBase(ABC):
    """Abstract base for composable task configuration.

    A task knows which objects it needs, how to observe them, and
    how to reset them.  Concrete subclasses return ready-to-merge
    cfg fragments via the ``get_*`` methods.

    Subclasses declare in ``roles`` the operable objects the task
    cannot do without. Anything else the YAML provides — clutter,
    layout anchors — is spawned and reset like the rest but is not
    part of the contract.
    """

    EPISODE_META_RECORD_KEY = "episode/meta_dict"

    roles: ClassVar[dict[str, "RoleSpec"]] = {}

    def __init__(
        self,
        assets: "TaskAssets",
        instruction: "InstructionWrapper | None" = None,
    ) -> None:
        self._validate_roles(assets)
        # Scene entities are namespaced once, here, so every scene name
        # the task hands out is the one that keys ``env.scene``.
        self.assets = assets.with_default_namespace("objects")
        self.instruction = instruction

    @classmethod
    def _validate_roles(cls, assets: "TaskAssets") -> None:
        """Fail the build when a declared role cannot be operated on.

        ``asset_configs`` keys in the task YAML are role names, so a
        typo there would otherwise leave a role empty and only surface
        much later, far from the cause. A role filled by an asset with
        no pose of its own — a light, a camera — fails just as late,
        inside a checker or a reset term.
        """
        missing = [r for r in sorted(cls.roles) if not assets.by_role(r)]
        if missing:
            raise ValueError(
                f"{cls.__name__} needs assets for role(s) {missing}, but "
                f"the task YAML provides {sorted(assets.role_candidates)}."
            )
        for role_id in sorted(cls.roles):
            for spec in assets.by_role(role_id):
                if not isinstance(spec, ObjectSpec):
                    raise TypeError(
                        f"{cls.__name__} role {role_id!r} must be filled by "
                        f"an object, got {type(spec).__name__}."
                    )

    # ---------------------------------------------------------
    # Scene assets
    # ---------------------------------------------------------
    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return task-owned assets grouped by namespace."""
        grouped: dict[str, dict[str, object]] = {}
        for spec in self.assets.flatten().values():
            ns = spec.namespace
            assert ns is not None, (
                "namespace must be set after __init__'s "
                "with_default_namespace; got None for "
                f"{type(spec).__name__}({spec.name!r})"
            )
            grouped.setdefault(ns, {})
            if spec.name in grouped[ns]:
                raise ValueError(
                    f"Duplicate task asset '{spec.scene_name}' in "
                    f"{type(self).__name__}."
                )
            grouped[ns][spec.name] = spec.to_isaac_cfg()
        return {
            namespace: GroupAssetCfg(**group_assets)
            for namespace, group_assets in grouped.items()
        }

    # ---------------------------------------------------------
    # Observations
    # ---------------------------------------------------------

    def get_observation_cfg(self) -> ObservationManagerCfg:
        """Return task-specific observation groups.

        Default: an ``/object`` group with a pose term per object asset.
        """
        scene_names = self.get_operable_scene_names()
        if not scene_names:
            return ObservationManagerCfg(groups={})
        return ObservationManagerCfg(
            groups={
                "/object": ObservationGroupCfg(
                    terms={
                        f"{name}_tf": FrameTransformTermCfg(
                            child_asset_cfg=SceneEntityCfg(name=name),
                            world_parent=True,
                            bidirectional=False,
                        )
                        for name in scene_names
                    }
                )
            }
        )

    def get_action_cfg(self) -> ActionManagerCfg:
        """Return task-specific action cfg fragment (default empty)."""
        return ActionManagerCfg(terms={})

    def get_record_terms(self) -> Mapping[str, RecordTermBaseCfg]:
        """Return task-specific record term fragments.

        Every task records episode metadata the same way, so that term
        lives here rather than in each subclass. Per-frame object poses
        are only recorded when the task actually owns objects.
        """
        terms: dict[str, RecordTermBaseCfg] = {
            "meta_dict_term": McapDictTermCfg(
                topic="/meta_data",
                fps=1.0,
                # Use the task-level metadata record key contract.
                key=TaskBase.EPISODE_META_RECORD_KEY,
                record_mode="once",
            ),
        }
        if self.get_operable_scene_names():
            terms["object_tf_term"] = McapMultiTFTermCfg(
                topic="/observation/objects/tf",
                # Advisory: the builder overrides this with the scene's
                # step rate, since the scene owns the simulation clock.
                fps=30.0,
                key="/object",
            )
        return terms

    # ---------------------------------------------------------
    # Events
    # ---------------------------------------------------------

    @abstractmethod
    def get_event_cfg(self) -> EventManagerCfg:
        """Return task-specific event terms.

        Typically pose-reset events for each task object.
        """

    def get_operable_scene_names(self) -> list[str]:
        """Return every task-owned rigid scene entity in stable asset order."""
        return list(dict.fromkeys(self.assets.all_scene_names()))

    @abstractmethod
    def get_role_candidates(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list["TargetRef"]:
        """Return everything a role could be bound to this episode.

        Left to each task rather than defaulting here: which objects a
        role may draw from is a task-level decision. A pick task may
        rotate through its clutter as well as its declared target, while
        a layout-driven one must stay with what the layout named.

        ``swap`` asks for the widest candidate set the task considers valid
        for this role. A task with nothing to rotate through returns the
        same single candidate either way, which keeps swap a no-op for tasks
        that cannot support it.
        """

    def get_role_scene_entities(
        self,
        role_id: str,
        *,
        swap: bool = False,
    ) -> list["SceneEntityKey"]:
        """Map task candidates to physical entities observed by checkers."""
        return [
            target.scene_name
            for target in self.get_role_candidates(role_id, swap=swap)
        ]

    @abstractmethod
    def build_validator(
        self,
        context: "ValidatorContext | None" = None,
    ) -> "Validator":
        """Build the task validator used for evaluation.

        Returns:
            Validator: Task-specific success/progress validator.
        """

    def build_instruction_context(
        self,
        env: "IsaacManagerBasedEnv",
        *,
        actor_description_seed: int,
        context: "ValidatorContext | None" = None,
    ) -> Mapping[str, "InstructionActor"]:
        """Build named instruction actors from the runtime task context.

        ``context`` answers which object a role currently points at, for
        tasks whose target changes between episodes. Tasks that fix their
        target at build time read it off ``self`` and ignore it.
        """
        del env, actor_description_seed, context
        return {}
