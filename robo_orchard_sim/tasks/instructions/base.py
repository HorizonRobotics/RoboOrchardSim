## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import dataclasses
import json
import random
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.registry.registry import (
        AssetRegistry,
    )
    from robo_orchard_sim.models.assets.rigid_object import RigidObject

__all__ = [
    "InstructionActor",
    "InstructionWrapper",
    "InstructionRenderError",
]


def _load_caption_payload(caption_path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(caption_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise InstructionRenderError(
            f"Invalid caption payload at '{caption_path}'"
        ) from exc
    if not isinstance(payload, Mapping):
        raise InstructionRenderError(
            f"Caption payload at '{caption_path}' must be a mapping"
        )
    return payload


def _parse_caption_payload(
    *,
    uuid: str,
    category: str,
    payload: Mapping[str, Any],
    fallback_description: str,
) -> tuple[str, list[str], list[str]]:
    payload_uuid = payload.get("uuid")
    if payload_uuid is not None and str(payload_uuid) != uuid:
        raise InstructionRenderError(
            "Caption payload uuid mismatch: "
            f"expected '{uuid}', got '{payload_uuid}'"
        )
    missing = [
        field for field in ("raw", "seen", "unseen") if field not in payload
    ]
    if missing:
        raise InstructionRenderError(
            "Caption payload missing required fields: " + ", ".join(missing)
        )
    raw_description_val = payload.get("raw")
    if not isinstance(raw_description_val, str):
        raise InstructionRenderError(
            "Caption payload field 'raw' must be a string"
        )
    raw_description = raw_description_val or fallback_description or category
    seen_descriptions: list[str] = []
    unseen_descriptions: list[str] = []
    for field_name, target in (
        ("seen", seen_descriptions),
        ("unseen", unseen_descriptions),
    ):
        values = payload.get(field_name)
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes)
        ):
            raise InstructionRenderError(
                f"Caption payload field '{field_name}' must be a string list"
            )
        if not all(isinstance(item, str) for item in values):
            raise InstructionRenderError(
                f"Caption payload field '{field_name}' must "
                "contain only strings"
            )
        target.extend(values)
    return raw_description, seen_descriptions, unseen_descriptions


def _select_description(
    *,
    uuid: str,
    raw_description: str,
    seen_descriptions: list[str],
    unseen_descriptions: list[str],
    actor_description_mode: Literal["raw", "seen", "unseen"],
    actor_description_seed: int | None,
) -> str:
    if actor_description_mode == "raw":
        return raw_description
    if actor_description_seed is None:
        raise InstructionRenderError(
            "actor_description_seed is required when "
            "actor_description_mode is seen/unseen"
        )
    candidates = (
        seen_descriptions
        if actor_description_mode == "seen"
        else unseen_descriptions
    )
    if not candidates:
        raise InstructionRenderError(
            "Actor description "
            f"'{actor_description_mode}' is empty for uuid '{uuid}'"
        )
    rng = random.Random(actor_description_seed)
    return str(rng.choice(candidates))


def _build_instruction_actor(
    *,
    uuid: str,
    category: str | None,
    payload: Mapping[str, Any],
    fallback_description: str,
    actor_description_mode: Literal["raw", "seen", "unseen"],
    actor_description_seed: int | None,
) -> "InstructionActor":
    resolved_category = category or str(payload.get("category") or "unknown")
    raw_description, seen_descriptions, unseen_descriptions = (
        _parse_caption_payload(
            uuid=uuid,
            category=resolved_category,
            payload=payload,
            fallback_description=fallback_description,
        )
    )
    description = _select_description(
        uuid=uuid,
        raw_description=raw_description,
        seen_descriptions=seen_descriptions,
        unseen_descriptions=unseen_descriptions,
        actor_description_mode=actor_description_mode,
        actor_description_seed=actor_description_seed,
    )
    return InstructionActor(
        uuid=uuid,
        description=description,
        raw_description=raw_description,
        category=resolved_category,
        seen_descriptions=seen_descriptions,
        unseen_descriptions=unseen_descriptions,
    )


@dataclasses.dataclass(slots=True)
class InstructionActor:
    """Instruction-focused actor data resolved from asset captions."""

    uuid: str
    description: str
    raw_description: str
    category: str = "unknown"
    seen_descriptions: list[str] = dataclasses.field(default_factory=list)
    unseen_descriptions: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.category == "":
            self.category = "unknown"

    @classmethod
    def from_rigid_object(
        cls,
        rigid_object: "RigidObject",
        *,
        actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
        actor_description_seed: int | None = None,
    ) -> "InstructionActor":
        """Build an instruction actor from a runtime RigidObject.

        Reads ``caption_path`` / ``uuid`` / ``category`` from
        ``rigid_object.cfg`` (a ``RigidObjectSpec``) and delegates to the
        shared caption payload pipeline.

        Args:
            rigid_object (RigidObject): Runtime rigid object whose ``cfg``
                carries the asset-library identity.
            actor_description_mode (
                Literal["raw", "seen", "unseen"], optional
            ):
                Description selection mode. Default is ``"raw"``.
            actor_description_seed (int | None, optional): Sampling seed
                used when ``actor_description_mode`` is ``"seen"`` or
                ``"unseen"``. Default is None.

        Raises:
            InstructionRenderError: If ``cfg.caption_path`` is unset or
                cannot be parsed.
        """
        cfg = rigid_object.cfg
        if cfg.caption_path is None:
            raise InstructionRenderError(
                "RigidObjectSpec.caption_path is required to build an "
                "InstructionActor from a RigidObject"
            )
        uuid = cfg.uuid or "unknown"
        category = cfg.category or "unknown"
        payload = _load_caption_payload(Path(str(cfg.caption_path)))
        return _build_instruction_actor(
            uuid=uuid,
            category=category,
            payload=payload,
            fallback_description=category,
            actor_description_mode=actor_description_mode,
            actor_description_seed=actor_description_seed,
        )

    @classmethod
    def from_registry(
        cls,
        uuid: str,
        registry: "AssetRegistry",
        *,
        actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
        actor_description_seed: int | None = None,
    ) -> "InstructionActor":
        """Build an instruction actor by uuid via AssetRegistry lookup.

        Resolves the ``AssetMeta`` through ``registry.get_meta(uuid)`` and
        loads its caption payload. Errors from registry lookup
        (``UnknownAssetError``) propagate to the caller.

        Args:
            uuid (str): Asset uuid to resolve.
            registry (AssetRegistry): Registry that owns the asset
                metadata.
            actor_description_mode (
                Literal["raw", "seen", "unseen"], optional
            ):
                Description selection mode. Default is ``"raw"``.
            actor_description_seed (int | None, optional): Sampling seed
                used when ``actor_description_mode`` is ``"seen"`` or
                ``"unseen"``. Default is None.

        Raises:
            InstructionRenderError: If the caption payload pointed to by
                the registered ``AssetMeta`` is missing or invalid.
        """
        meta = registry.get_meta(uuid)
        category = str(meta.category)
        fallback_description = str(meta.description or category)
        payload = _load_caption_payload(Path(str(meta.caption_path)))
        return _build_instruction_actor(
            uuid=str(meta.uuid),
            category=category,
            payload=payload,
            fallback_description=fallback_description,
            actor_description_mode=actor_description_mode,
            actor_description_seed=actor_description_seed,
        )


class InstructionRenderError(ValueError):
    """Raised when strict rendering cannot resolve one or more placeholders."""


class InstructionWrapper:
    """Wrap an instruction template and render text from actor/context values.

    Args:
        template (str): Registered instruction template name from YAML.
        template_mode (Literal["fixed", "variants"], optional): Template
            mode to use. Default is "fixed".
        actor_description_mode
            (Literal["raw", "seen", "unseen"], optional): Actor
            description selection mode. Default is "raw".
        strict (bool, optional): Whether unresolved fields raise an error in
            ``render``. Default is True.
    """

    def __init__(
        self,
        template: str,
        *,
        template_mode: Literal["fixed", "variants"] = "fixed",
        actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
        strict: bool = True,
    ) -> None:
        self.template = template
        self.template_mode = template_mode
        self.actor_description_mode = actor_description_mode
        self.strict = strict
        self._formatter = string.Formatter()
        self._template_payload = self._load_template_payload(template)

    def _load_template_payload(self, template: str) -> Mapping[str, Any]:
        from robo_orchard_sim.tasks.instructions.registry import (
            get_instruction_template,
        )

        try:
            payload = get_instruction_template(template)
        except ValueError as exc:
            raise InstructionRenderError(str(exc)) from exc
        if not isinstance(payload, Mapping):
            raise InstructionRenderError(
                "Template payload must be a mapping with fixed/variants"
            )
        return payload

    def _select_template(self, seed: int | None = None) -> str:
        if self.template_mode == "fixed":
            fixed_template = self._template_payload.get("fixed")
            if not isinstance(fixed_template, str):
                raise InstructionRenderError(
                    "Template payload field 'fixed' must be a string"
                )
            return fixed_template

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

    def _build_context(
        self,
        actors: Mapping[str, Any] | None = None,
        actor_description_seed: int | None = None,
    ) -> dict[str, Any]:
        """Build a render context from a named actor mapping.

        Args:
            actors (Mapping[str, Any] | None, optional): Named context values
                used during rendering. ``actor1`` is treated as the primary
                actor and also populates ``actor`` plus top-level actor
                fields. Default is None.
            actor_description_seed (int | None, optional): Sampling seed
                used when ``self.actor_description_mode`` is ``seen`` or
                ``unseen``. Default is None.

        Returns:
            dict[str, Any]: Context used by template rendering.
        """
        context: dict[str, Any] = {}
        if not actors:
            return context

        resolved_actors = self._resolve_value_for_actor_description_mode(
            value=actors,
            actor_description_mode=self.actor_description_mode,
            actor_description_seed=actor_description_seed,
        )
        resolved_actors_dict = dict(resolved_actors)
        context.update(resolved_actors_dict)
        context["actors"] = resolved_actors_dict

        primary_actor = resolved_actors.get("actor1")
        if primary_actor is not None:
            context.update(self._to_context_dict(primary_actor))
            context["actor"] = primary_actor

        return context

    def render(
        self,
        actors: Mapping[str, Any] | None = None,
        template_seed: int | None = None,
        actor_description_seed: int | None = None,
    ) -> str:
        """Render the instruction text from named actor/context values.

        Args:
            actors (Mapping[str, Any] | None, optional): Named render inputs.
                Default is None.
            template_seed (int | None, optional): Sampling seed used for
                template selection. Default is None.
            actor_description_seed (int | None, optional): Sampling seed
                used when ``self.actor_description_mode`` is ``seen`` or
                ``unseen``. Default is None.

        Returns:
            str: Rendered instruction text.

        Raises:
            InstructionRenderError: If strict mode is enabled and placeholders
                cannot be resolved.
        """
        context = self._build_context(
            actors=actors,
            actor_description_seed=actor_description_seed,
        )
        template = self._select_template(seed=template_seed)
        if self.strict:
            missing = sorted(
                field
                for field in self._placeholders_from_template(template)
                if not self._field_exists(field_name=field, context=context)
            )
            if missing:
                raise InstructionRenderError(
                    f"Unresolved placeholders: {', '.join(missing)}"
                )
        return self._render_with_context(
            template=template,
            context=context,
            allow_partial=False,
        )

    def _placeholders_from_template(self, template: str) -> set[str]:
        fields: set[str] = set()
        for _, field_name, _, _ in self._formatter.parse(template):
            if field_name:
                fields.add(field_name)
        return fields

    def _to_context_dict(
        self,
        actor: InstructionActor | Mapping[str, Any],
    ) -> dict[str, Any]:
        if isinstance(actor, Mapping):
            return dict(actor)
        if isinstance(actor, InstructionActor):
            return dataclasses.asdict(actor)
        raise InstructionRenderError(
            f"Unsupported actor type: {type(actor)!r}"
        )

    def _resolve_value_for_actor_description_mode(
        self,
        value: Any,
        actor_description_mode: Literal["raw", "seen", "unseen"],
        actor_description_seed: int | None,
    ) -> Any:
        if value is None:
            return value
        if isinstance(value, InstructionActor):
            description = _select_description(
                uuid=value.uuid,
                raw_description=value.raw_description,
                seen_descriptions=value.seen_descriptions,
                unseen_descriptions=value.unseen_descriptions,
                actor_description_mode=actor_description_mode,
                actor_description_seed=actor_description_seed,
            )
            return dataclasses.replace(value, description=description)
        if isinstance(value, Mapping):
            return {
                key: self._resolve_value_for_actor_description_mode(
                    value=item,
                    actor_description_mode=actor_description_mode,
                    actor_description_seed=actor_description_seed,
                )
                for key, item in value.items()
            }
        return value

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
