## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import dataclasses
import json
import random
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional

__all__ = [
    "Actor",
    "ActorDescription",
    "ActorStoreAdapter",
    "InstructionWrapper",
    "InstructionRenderError",
]
_CUR_DIR = Path(__file__).resolve().parent
_DEFAULT_OBJECT_ROOT = _CUR_DIR / "object_descriptions"


class ActorStoreAdapter:
    """UUID -> JSON file adapter for actor payload lookup."""

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        fallback_to_stem: bool = True,
    ) -> None:
        self._root = Path(root) if root is not None else _DEFAULT_OBJECT_ROOT
        self._fallback_to_stem = fallback_to_stem
        self._uuid_to_path: dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Recursively scan *.json and build the UUID (or stem) index."""
        self._uuid_to_path.clear()
        if not self._root.is_dir():
            return
        for path in self._root.rglob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            uuid_val = data.get("uuid")
            if uuid_val is None and self._fallback_to_stem:
                uuid_val = path.stem
            if uuid_val is None:
                continue
            if uuid_val in self._uuid_to_path:
                raise ValueError(
                    f"Duplicate uuid {uuid_val!r}: "
                    f"{self._uuid_to_path[uuid_val]} vs {path}"
                )
            self._uuid_to_path[uuid_val] = path.resolve()

    def get_path(self, uuid: str) -> Optional[Path]:
        """Return the absolute JSON path for uuid, or None if missing."""
        return self._uuid_to_path.get(uuid)

    def get_path_or_raise(self, uuid: str) -> Path:
        """Return the JSON path for uuid, or raise KeyError if missing."""
        if uuid not in self._uuid_to_path:
            raise KeyError(f"No description file for uuid: {uuid!r}")
        return self._uuid_to_path[uuid]

    def get_actor_json(self, uuid: str) -> Mapping[str, Any] | None:
        """Load and return actor JSON content for uuid."""
        path = self.get_path(uuid)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, Mapping):
            raise InstructionRenderError(
                f"Actor payload for uuid '{uuid}' must be a mapping"
            )
        return payload

    def get_actor_json_or_raise(self, uuid: str) -> Mapping[str, Any]:
        """Load and return actor JSON content, or raise on failure."""
        path = self.get_path_or_raise(uuid)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise InstructionRenderError(
                f"Actor payload for uuid '{uuid}' must be a mapping"
            )
        return payload

    def all_uuids(self) -> list[str]:
        """Return all indexed UUID keys."""
        return list(self._uuid_to_path.keys())

    def __contains__(self, uuid: str) -> bool:
        return uuid in self._uuid_to_path

    def __len__(self) -> int:
        return len(self._uuid_to_path)


@dataclasses.dataclass(slots=True)
class ActorDescription:
    """Actor description payload."""

    seen: list[Any] = dataclasses.field(default_factory=list)
    unseen: list[Any] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class Actor:
    """Instruction rendering actor data."""

    category: str
    uuid: str
    description: str = ""
    description_pool: ActorDescription = dataclasses.field(
        default_factory=ActorDescription
    )

    def __post_init__(self) -> None:
        if self.description == "":
            self.description = self.category

    @classmethod
    def from_uuid(
        cls, uuid: str, store: ActorStoreAdapter | None = None
    ) -> "Actor":
        """Load actor data from an injected adapter and construct an Actor.

        Args:
            uuid (str): Actor uuid to query.
            store (ActorStoreAdapter | None, optional): Actor store adapter.
                If None, uses the default description root.

        Returns:
            Actor: Actor populated from database payload.

        Raises:
            InstructionRenderError: If query result is missing or invalid.
        """
        resolved_store = store or ActorStoreAdapter()
        payload = resolved_store.get_actor_json(uuid)
        if payload is None:
            raise InstructionRenderError(f"Actor not found for uuid '{uuid}'")
        if not isinstance(payload, Mapping):
            raise InstructionRenderError(
                f"Actor payload for uuid '{uuid}' must be a mapping"
            )

        missing = [
            key
            for key in ("uuid", "category", "seen", "unseen")
            if key not in payload
        ]
        if missing:
            raise InstructionRenderError(
                "Actor payload missing required fields: " + ", ".join(missing)
            )
        if payload["uuid"] != uuid:
            raise InstructionRenderError(
                "Actor payload uuid mismatch: "
                f"expected '{uuid}', got '{payload['uuid']}'"
            )

        return cls(
            category=str(payload.get("category")),
            uuid=str(payload["uuid"]),
            description_pool=ActorDescription(
                seen=list(payload["seen"]),
                unseen=list(payload["unseen"]),
            ),
        )


ActorsInput = Mapping[str, Actor | Any] | Sequence[Actor | Any]


class InstructionRenderError(ValueError):
    """Raised when strict rendering cannot resolve one or more placeholders."""


class InstructionWrapper:
    """Wrap an instruction template and render text from actor/context values.

    Args:
        template_source (Path | str | Mapping[str, Any]): Template source.
            It can be a json file path or an in-memory template mapping with
            keys ``raw``, ``seen`` and ``unseen``.
        template_mode (Literal["raw", "seen", "unseen"], optional): Template
            mode to use. Default is "raw".
        strict (bool, optional): Whether unresolved fields raise an error in
            ``render``. Default is True.
    """

    def __init__(
        self,
        template_source: Path | str | Mapping[str, Any],
        *,
        template_mode: Literal["raw", "seen", "unseen"] = "raw",
        strict: bool = True,
    ) -> None:
        self.template_mode = template_mode
        self.strict = strict
        self._formatter = string.Formatter()
        self._template_payload = self._load_template_payload(template_source)

    def _load_template_payload(
        self, template_source: Path | str | Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if isinstance(template_source, Mapping):
            payload = template_source
        else:
            source_path = Path(template_source)
            try:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise InstructionRenderError(
                    f"Invalid template json source: {template_source!r}"
                ) from exc
        if not isinstance(payload, Mapping):
            raise InstructionRenderError(
                "Template payload must be a mapping with raw/seen/unseen"
            )
        return payload

    def _select_template(self, seed: int | None = None) -> str:
        if self.template_mode == "raw":
            raw_template = self._template_payload.get("raw")
            if not isinstance(raw_template, str):
                raise InstructionRenderError(
                    "Template payload field 'raw' must be a string"
                )
            return raw_template

        mode_templates = self._template_payload.get(self.template_mode)
        if not isinstance(mode_templates, Sequence) or isinstance(
            mode_templates, (str, bytes)
        ):
            raise InstructionRenderError(
                f"Template payload field '{self.template_mode}' "
                "must be a string list"
            )
        if not mode_templates:
            raise InstructionRenderError(
                f"Template payload field '{self.template_mode}' is empty"
            )
        if not all(isinstance(item, str) for item in mode_templates):
            raise InstructionRenderError(
                f"Template payload field '{self.template_mode}' "
                "must contain only strings"
            )
        rng = random.Random(seed)
        return str(rng.choice(list(mode_templates)))

    def placeholders(self, seed: int | None = None) -> set[str]:
        """Return all placeholder field names declared in the template.

        Returns:
            set[str]: Field names parsed from the template.
        """
        fields: set[str] = set()
        template = self._select_template(seed=seed)
        for _, field_name, _, _ in self._formatter.parse(template):
            if field_name:
                fields.add(field_name)
        return fields

    def build_context(
        self,
        actors: Actor | Any | ActorsInput | None = None,
        extra: Mapping[str, Any] | None = None,
        obj_desc_mode: Literal["raw", "seen", "unseen"] = "raw",
        obj_desc_seed: int | None = None,
    ) -> dict[str, Any]:
        """Build a render context from actor and extra mappings.

        Args:
            actors (Any | Mapping[str, Any] | Sequence[Any] | None, optional):
                Actor input. Supports:
                - single actor object/mapping
                - actor mapping (for example ``{"actor1": ...}``)
                - actor sequence (auto keys ``actor1``, ``actor2``...)
                Default is None.
            extra (Mapping[str, Any] | None, optional): Extra context values.
                Default is None.
            obj_desc_mode (Literal["raw", "seen", "unseen"], optional):
                Actor description mode for placeholder replacement.
                Default is "raw".
            obj_desc_seed (int | None, optional): Sampling seed used when
                ``obj_desc_mode`` is ``seen`` or ``unseen``. Default is None.

        Returns:
            dict[str, Any]: Context used by template rendering.
        """
        context: dict[str, Any] = {}
        rng = self._resolve_obj_desc_rng(
            obj_desc_mode=obj_desc_mode,
            obj_desc_seed=obj_desc_seed,
        )
        normalized_actors = self._normalize_actors(actors=actors)
        normalized_actors = {
            key: self._resolve_actor_for_obj_desc_mode(
                actor=value,
                obj_desc_mode=obj_desc_mode,
                rng=rng,
            )
            for key, value in normalized_actors.items()
        }
        primary_actor = None
        if normalized_actors:
            primary_actor = next(iter(normalized_actors.values()))

        if primary_actor is not None:
            context.update(self._to_context_dict(primary_actor))
            context["actor"] = primary_actor

        if normalized_actors:
            context["actors"] = normalized_actors
            context.update(normalized_actors)

        if extra:
            context.update(dict(extra))
        return context

    def validate(
        self,
        actors: Actor | Any | ActorsInput | None = None,
        extra: Mapping[str, Any] | None = None,
        seed: int | None = None,
        obj_desc_mode: Literal["raw", "seen", "unseen"] = "raw",
        obj_desc_seed: int | None = None,
    ) -> None:
        """Validate that all placeholders can be resolved from actor/context.

        Args:
            actors (Actor | Any | ActorsInput | None): Input actor(s), actor
                mapping, or None.
            extra (Mapping[str, Any] | None, optional): Extra context values.
                Default is None.

        Raises:
            InstructionRenderError: If one or more placeholders are missing.
        """
        context = self.build_context(
            actors=actors,
            extra=extra,
            obj_desc_mode=obj_desc_mode,
            obj_desc_seed=obj_desc_seed,
        )
        template = self._select_template(seed=seed)
        missing = sorted(
            field
            for field in self._placeholders_from_template(template)
            if not self._field_exists(field_name=field, context=context)
        )
        if missing:
            raise InstructionRenderError(
                f"Unresolved placeholders: {', '.join(missing)}"
            )

    def render(
        self,
        actors: Actor | Any | ActorsInput | None = None,
        extra: Mapping[str, Any] | None = None,
        seed: int | None = None,
        obj_desc_mode: Literal["raw", "seen", "unseen"] = "raw",
        obj_desc_seed: int | None = None,
    ) -> str:
        """Render the instruction text from actor/context values.

        Args:
            actors (Actor | Any | ActorsInput | None): Input actor(s), actor
                mapping, or None.
            extra (Mapping[str, Any] | None, optional): Extra context values.
                Default is None.
            seed (int | None, optional): Sampling seed used for template
                selection. Default is None.
            obj_desc_mode (Literal["raw", "seen", "unseen"], optional):
                Object description mode for placeholder replacement.
                Default is "raw".
            obj_desc_seed (int | None, optional): Sampling seed used when
                ``obj_desc_mode`` is ``seen`` or ``unseen``. Default is None.

        Returns:
            str: Rendered instruction text.

        Raises:
            InstructionRenderError: If strict mode is enabled and placeholders
                cannot be resolved.
        """
        context = self.build_context(
            actors=actors,
            extra=extra,
            obj_desc_mode=obj_desc_mode,
            obj_desc_seed=obj_desc_seed,
        )
        template = self._select_template(seed=seed)
        if self.strict:
            self.validate(
                actors=actors,
                extra=extra,
                seed=seed,
                obj_desc_mode=obj_desc_mode,
                obj_desc_seed=obj_desc_seed,
            )
        return self._render_with_context(
            template=template,
            context=context,
            allow_partial=False,
        )

    def render_partial(
        self,
        actors: Actor | Any | ActorsInput | None = None,
        extra: Mapping[str, Any] | None = None,
        seed: int | None = None,
        obj_desc_mode: Literal["raw", "seen", "unseen"] = "raw",
        obj_desc_seed: int | None = None,
    ) -> str:
        """Render text while preserving unresolved placeholders in output.

        Args:
            actors (Actor | Any | ActorsInput | None): Input actor(s), actor
                mapping, or None.
            extra (Mapping[str, Any] | None, optional): Extra context values.
                Default is None.

        Returns:
            str: Partially rendered instruction text.
        """
        context = self.build_context(
            actors=actors,
            extra=extra,
            obj_desc_mode=obj_desc_mode,
            obj_desc_seed=obj_desc_seed,
        )
        template = self._select_template(seed=seed)
        return self._render_with_context(
            template=template,
            context=context,
            allow_partial=True,
        )

    def _placeholders_from_template(self, template: str) -> set[str]:
        fields: set[str] = set()
        for _, field_name, _, _ in self._formatter.parse(template):
            if field_name:
                fields.add(field_name)
        return fields

    def _normalize_actors(
        self,
        actors: Actor | Any | ActorsInput | None,
    ) -> dict[str, Actor | Any]:
        if actors is None:
            return {}
        if isinstance(actors, Mapping):
            if self._is_actor_mapping(actors):
                return {str(key): value for key, value in actors.items()}
            return {"actor": actors}
        if isinstance(actors, Sequence) and not isinstance(
            actors, (str, bytes)
        ):
            return {
                f"actor{index + 1}": value
                for index, value in enumerate(actors)
            }
        return {"actor": actors}

    def _is_actor_mapping(self, value: Mapping[Any, Any]) -> bool:
        if not value:
            return True
        first_val = next(iter(value.values()))
        is_mapping_actor = isinstance(first_val, (Mapping, Actor))
        is_dataclass_actor = dataclasses.is_dataclass(first_val)
        return is_mapping_actor or is_dataclass_actor

    def _to_context_dict(self, actor: Actor | Any) -> dict[str, Any]:
        if isinstance(actor, Mapping):
            return dict(actor)
        if dataclasses.is_dataclass(actor) and not isinstance(actor, type):
            return dataclasses.asdict(actor)
        try:
            return vars(actor)
        except TypeError as exc:
            raise InstructionRenderError(
                f"Unsupported actor type: {type(actor)!r}"
            ) from exc

    def _resolve_obj_desc_rng(
        self,
        obj_desc_mode: Literal["raw", "seen", "unseen"],
        obj_desc_seed: int | None,
    ) -> random.Random | None:
        if obj_desc_mode == "raw":
            return None
        if obj_desc_seed is None:
            raise InstructionRenderError(
                "obj_desc_seed is required when obj_desc_mode is seen/unseen"
            )
        return random.Random(obj_desc_seed)

    def _resolve_actor_for_obj_desc_mode(
        self,
        actor: Actor | Any,
        obj_desc_mode: Literal["raw", "seen", "unseen"],
        rng: random.Random | None,
    ) -> Actor | Any:
        if not isinstance(actor, Actor):
            return actor
        if obj_desc_mode == "raw":
            return dataclasses.replace(actor, description=actor.category)
        candidates = getattr(actor.description_pool, obj_desc_mode)
        if not candidates:
            raise InstructionRenderError(
                "Actor description "
                f"'{obj_desc_mode}' is empty for uuid '{actor.uuid}'"
            )
        assert rng is not None
        sampled_desc = str(rng.choice(list(candidates)))
        return dataclasses.replace(actor, description=sampled_desc)

    def _field_exists(self, field_name: str, context: dict[str, Any]) -> bool:
        try:
            self._formatter.get_field(field_name, (), context)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return False
        return True

    def _render_with_context(
        self,
        template: str,
        context: dict[str, Any],
        *,
        allow_partial: bool,
    ) -> str:
        chunks: list[str] = []
        parsed_template = self._formatter.parse(template)
        for literal, field_name, format_spec, conversion in parsed_template:
            chunks.append(literal)
            if field_name is None:
                continue
            try:
                value, _ = self._formatter.get_field(field_name, (), context)
                if conversion:
                    value = self._formatter.convert_field(value, conversion)
                chunks.append(self._formatter.format_field(value, format_spec))
            except (
                AttributeError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                if not allow_partial:
                    raise InstructionRenderError(
                        f"Failed to render placeholder '{field_name}'"
                    ) from exc
                chunks.append(
                    self._rebuild_field(field_name, format_spec, conversion)
                )
        return "".join(chunks)

    def _rebuild_field(
        self,
        field_name: str,
        format_spec: str,
        conversion: str | None,
    ) -> str:
        conversion_part = ""
        if conversion:
            conversion_part = f"!{conversion}"
        format_part = ""
        if format_spec:
            format_part = f":{format_spec}"
        return f"{{{field_name}{conversion_part}{format_part}}}"
