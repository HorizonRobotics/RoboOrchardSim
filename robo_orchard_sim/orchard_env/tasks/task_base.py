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

from robo_orchard_sim.envs.managers.record import RecordTermBaseCfg
from robo_orchard_sim.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import AssetSpec, ObjectSpec

if TYPE_CHECKING:
    from robo_orchard_sim.envs.manager_based_env import IsaacManagerBasedEnv
    from robo_orchard_sim.tasks.instructions.base import (
        InstructionActor,
        InstructionWrapper,
    )
    from robo_orchard_sim.tasks.validators.base import (
        Validator,
        ValidatorActor,
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

    distractors: ObjectSpec | Sequence[ObjectSpec] | None = None

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
            and not isinstance(value[field_name], ObjectSpec)
        ]
        if invalid_fields:
            field_list = ", ".join(invalid_fields)
            raise TypeError(
                f"{cls.__name__} {field_list} must be ObjectSpec instances."
            )
        return value

    @field_validator("distractors")
    @classmethod
    def validate_distractors(
        cls, value: ObjectSpec | Sequence[ObjectSpec] | None
    ) -> ObjectSpec | Sequence[ObjectSpec] | None:
        """Accept zero, one, or many distractor objects."""
        if value is None:
            return value
        if isinstance(value, ObjectSpec):
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
            f"{cls.__name__} distractors must be an ObjectSpec, a sequence "
            "of ObjectSpec instances, or None."
        )

    def flatten_distractors(self) -> dict[str, ObjectSpec]:
        """Return distractors in the flattened shape expected by TaskBase."""
        flattened: dict[str, ObjectSpec] = {}
        distractors = self.distractors
        if distractors is None:
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
        assets: Mapping[str, AssetSpec],
        instruction: "InstructionWrapper | None" = None,
    ):
        self._assets = {
            role: spec.with_default_namespace("objects")
            for role, spec in assets.items()
        }
        self.instruction = instruction

    # ---------------------------------------------------------
    # Scene assets
    # ---------------------------------------------------------
    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return task-owned assets grouped by namespace."""
        grouped: dict[str, dict[str, object]] = {}
        for spec in self._assets.values():
            grouped.setdefault(spec.namespace, {})
            if spec.name in grouped[spec.namespace]:
                raise ValueError(
                    "Duplicate task asset "
                    f"'{spec.scene_name}' in PlaceA2BTask."
                )
            grouped[spec.namespace][spec.name] = spec.to_isaac_cfg()
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
