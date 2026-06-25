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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ConfigDict, field_validator, model_validator
from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationManagerCfg,
)
from robo_orchard_core.utils.config import Config

from robo_orchard_sim.ext.envs.managers.record import RecordTermBaseCfg
from robo_orchard_sim.ext.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import (
    AssetSpec,
    ObjectSpec,
    PoolSpec,
)

if TYPE_CHECKING:
    from robo_orchard_sim.ext.envs.manager_based_env import (
        IsaacManagerBasedEnv,
    )
    from robo_orchard_sim.task_components.instructions.base import (
        InstructionActor,
        InstructionWrapper,
    )
    from robo_orchard_sim.task_components.validators.base import (
        Validator,
        ValidatorActor,
    )
    from robo_orchard_sim.task_components.validators.context import (
        ValidatorContext,
    )


class TaskAssetsBase(Config):
    """Base schema for tasks with required object assets and distractors.

    Subclasses declare per-role ObjectSpec fields and list the required
    ones in ``required_object_fields``. Pydantic enforces presence of
    required fields and rejects unknown role names (``extra="forbid"``);
    the upstream resolver only produces ``dict[role, AssetSpec]`` and
    is unaware of which roles a given task expects.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
        extra="forbid",
    )

    required_object_fields: ClassVar[tuple[str, ...]] = ()

    distractors: Any = None

    @classmethod
    def from_resolved(cls, resolved: Mapping[str, Any]) -> "TaskAssetsBase":
        """Build assets, folding every non-target role into distractors."""
        targets = {
            role: resolved[role]
            for role in cls.required_object_fields
            if role in resolved
        }
        rest = [
            value
            for role, value in resolved.items()
            if role not in cls.required_object_fields
        ]
        if not rest:
            distractors: Any = None
        elif len(rest) == 1 and not isinstance(rest[0], list):
            distractors = rest[0]
        elif any(not isinstance(value, list) for value in rest):
            raise TypeError(
                f"{cls.__name__} does not support multiple distractor "
                "groups when one is a pool."
            )
        else:
            distractors = [spec for group in rest for spec in group]
        return cls(**targets, distractors=distractors)

    @model_validator(mode="before")
    @classmethod
    def validate_required_objects(cls, value: Any) -> Any:
        """Reject non-object required assets with a task-specific error."""
        if not isinstance(value, dict):
            return value

        invalid_fields = [
            field_name
            for field_name in cls.required_object_fields
            if field_name in value
            and not isinstance(value[field_name], (ObjectSpec, PoolSpec))
        ]
        if invalid_fields:
            field_list = ", ".join(invalid_fields)
            raise TypeError(
                f"{cls.__name__} {field_list} must be ObjectSpec or PoolSpec "
                "instances."
            )
        return value

    @field_validator("distractors", mode="plain")
    @classmethod
    def validate_distractors(cls, value: Any) -> Any:
        """Accept zero, one, or many distractor objects, or a PoolSpec."""
        if value is None:
            return value
        if isinstance(value, (ObjectSpec, PoolSpec)):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for spec in value:
                if not isinstance(spec, ObjectSpec):
                    raise TypeError(
                        f"{cls.__name__} distractors must contain ObjectSpec "
                        "instances."
                    )
            return value
        raise TypeError(
            f"{cls.__name__} distractors must be an ObjectSpec, PoolSpec, "
            "a sequence of ObjectSpec instances, or None."
        )

    def flatten_distractors(self) -> "dict[str, ObjectSpec | PoolSpec]":
        """Return distractors in the flattened shape expected by TaskBase."""
        flattened: dict = {}
        distractors = self.distractors
        if distractors is None:
            return flattened
        if isinstance(distractors, PoolSpec):
            flattened["distractors_pool"] = distractors
            return flattened
        if isinstance(distractors, ObjectSpec):
            flattened["distractor_0"] = distractors
            return flattened
        for index, spec in enumerate(distractors):
            flattened[f"distractor_{index}"] = spec
        return flattened


class TaskBase(ABC):
    """Abstract base for composable task configuration.

    A task knows which objects it needs, how to observe them, and
    how to reset them.  Concrete subclasses return ready-to-merge
    cfg fragments via the ``get_*`` methods.
    """

    EPISODE_META_RECORD_KEY = "episode/meta_dict"

    def __init__(
        self,
        assets: Mapping[str, "AssetSpec | PoolSpec"],
        instruction: "InstructionWrapper | None" = None,
    ) -> None:
        self._assets: dict[str, AssetSpec | PoolSpec] = {}
        for role, spec in assets.items():
            if isinstance(spec, PoolSpec):
                self._assets[role] = PoolSpec(
                    role_id=spec.role_id,
                    members=[
                        m.with_default_namespace("objects")
                        for m in spec.members
                    ],
                    active_count=spec.active_count,
                )
            else:
                self._assets[role] = spec.with_default_namespace("objects")
        self.instruction = instruction

    # ---------------------------------------------------------
    # Scene assets
    # ---------------------------------------------------------
    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return task-owned assets grouped by namespace."""
        grouped: dict[str, dict[str, object]] = {}
        for spec in self._assets.values():
            specs_to_emit = (
                spec.members if isinstance(spec, PoolSpec) else [spec]
            )
            for s in specs_to_emit:
                ns = s.namespace
                assert ns is not None, (
                    "namespace must be set after __init__'s "
                    "with_default_namespace; got None for "
                    f"{type(s).__name__}({s.name!r})"
                )
                grouped.setdefault(ns, {})
                if s.name in grouped[ns]:
                    raise ValueError(
                        f"Duplicate task asset '{s.scene_name}' in "
                        f"{type(self).__name__}."
                    )
                grouped[ns][s.name] = s.to_isaac_cfg()
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
        return ObservationManagerCfg(groups={})

    def get_action_cfg(self) -> ActionManagerCfg:
        """Return task-specific action cfg fragment (default empty)."""
        return ActionManagerCfg(terms={})

    def get_record_terms(self) -> Mapping[str, RecordTermBaseCfg]:
        """Return task-specific record term fragments."""
        return {}

    # ---------------------------------------------------------
    # Events
    # ---------------------------------------------------------

    @abstractmethod
    def get_event_cfg(self) -> EventManagerCfg:
        """Return task-specific event terms.

        Typically pose-reset events for each task object.
        """

    @abstractmethod
    def get_validator_actor_names(self) -> list[str]:
        """Return scene names of actors that feed validator metadata."""

    @abstractmethod
    def build_validator(
        self,
        actors: list["ValidatorActor"],
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
    ) -> Mapping[str, "InstructionActor"]:
        """Build named instruction actors from the runtime task context."""
        del env, actor_description_seed
        return {}
