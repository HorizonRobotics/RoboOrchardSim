## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import dataclasses
import json
import random
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.registry.registry import (
        AssetRegistry,
    )
    from robo_orchard_sim.ext.models.assets.rigid_object import RigidObject

__all__ = [
    "InstructionActor",
    "InstructionAttributeName",
    "InstructionWrapper",
    "InstructionRenderError",
    "render_instruction_from_registry",
]

InstructionAttributeName = Literal["color", "shape", "material"]


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
    raw_description_val = payload.get("raw")
    if raw_description_val is not None and not isinstance(
        raw_description_val, str
    ):
        raise InstructionRenderError(
            "Caption payload field 'raw' must be a string"
        )
    raw_description = raw_description_val or fallback_description or category
    seen_descriptions: list[str] = []
    unseen_descriptions: list[str] = []

    def _extend_string_list(
        source_field_name: str,
        error_field_name: str,
        target: list[str],
    ) -> None:
        values = payload.get(source_field_name)
        if values is None:
            return
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes)
        ):
            raise InstructionRenderError(
                "Caption payload field "
                f"'{error_field_name}' must be a string list"
            )
        if not all(isinstance(item, str) for item in values):
            raise InstructionRenderError(
                f"Caption payload field '{error_field_name}' must "
                "contain only strings"
            )
        target.extend(values)

    _extend_string_list("unseen", "unseen", unseen_descriptions)
    _extend_string_list("seen", "seen", seen_descriptions)
    if not seen_descriptions:
        # Temporary compatibility: keep default `seen` mode working while
        # caption payloads are still being unified on the `seen` field.
        _extend_string_list("candidates", "candidates", seen_descriptions)
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
    if actor_description_mode not in {"seen", "unseen"}:
        raise InstructionRenderError(
            f"Unsupported actor_description_mode: '{actor_description_mode}'"
        )
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
    attribute_name: str | None = None,
    attribute_value: str | None = None,
) -> "InstructionActor":
    resolved_category = category or str(payload.get("category") or "unknown")
    resolved_category = resolved_category.replace("_", " ")
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
        attribute_name=attribute_name,
        attribute_value=attribute_value,
    )


def _resolve_single_asset_attribute(
    *,
    uuid: str,
    attributes: Mapping[
        str,
        tuple[str, ...] | list[str] | frozenset[str] | None,
    ],
    attribute_name: str,
) -> str:
    if attribute_name not in {"color", "shape", "material"}:
        raise InstructionRenderError(
            f"Unsupported asset attribute: {attribute_name!r}"
        )
    values = _normalize_runtime_attribute_values(
        attributes.get(attribute_name)
    )
    if not values:
        raise InstructionRenderError(
            f"Asset {uuid!r} has no {attribute_name!r} value"
        )
    if attribute_name == "color":
        return _format_attribute_values(values)
    if len(values) != 1:
        raise InstructionRenderError(
            f"Asset {uuid!r} has multiple {attribute_name!r} values: "
            f"{sorted(values)}"
        )
    return next(iter(values))


def _normalize_runtime_attribute_values(
    values: tuple[str, ...] | list[str] | frozenset[str] | None,
) -> frozenset[str] | None:
    if values is None:
        return None
    normalized = frozenset(str(value).strip().lower() for value in values)
    normalized = frozenset(value for value in normalized if value)
    if not normalized:
        return None
    return normalized


def _format_attribute_values(values: frozenset[str]) -> str:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return f"{ordered[0]} and {ordered[1]}"
    return ", ".join(ordered[:-1]) + f", and {ordered[-1]}"


@dataclasses.dataclass(slots=True)
class InstructionActor:
    """Instruction-focused actor data resolved from asset captions."""

    uuid: str
    description: str
    raw_description: str
    category: str = "unknown"
    seen_descriptions: list[str] = dataclasses.field(default_factory=list)
    unseen_descriptions: list[str] = dataclasses.field(default_factory=list)
    attribute_name: str | None = None
    attribute_value: str | None = None

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
    def from_rigid_object_with_attribute(
        cls,
        rigid_object: "RigidObject",
        *,
        attribute_name: InstructionAttributeName,
        actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
        actor_description_seed: int | None = None,
    ) -> "InstructionActor":
        """Build an instruction actor with one runtime object attribute."""
        actor = cls.from_rigid_object(
            rigid_object,
            actor_description_mode=actor_description_mode,
            actor_description_seed=actor_description_seed,
        )
        cfg = rigid_object.cfg
        attribute_value = _resolve_single_asset_attribute(
            uuid=actor.uuid,
            attributes=cfg.attributes,
            attribute_name=attribute_name,
        )
        return dataclasses.replace(
            actor,
            attribute_name=attribute_name,
            attribute_value=attribute_value,
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
        payload = _load_caption_payload(
            Path(
                str(meta.caption_path).replace(
                    "caption_candidates_updated.json",
                    "caption_candidates.json",
                )
            )
        )
        return _build_instruction_actor(
            uuid=str(meta.uuid),
            category=category,
            payload=payload,
            fallback_description=fallback_description,
            actor_description_mode=actor_description_mode,
            actor_description_seed=actor_description_seed,
        )

    @classmethod
    def from_registry_with_attribute(
        cls,
        uuid: str,
        registry: "AssetRegistry",
        *,
        attribute_name: str,
        actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
        actor_description_seed: int | None = None,
    ) -> "InstructionActor":
        """Build an instruction actor with one structured asset attribute."""
        meta = registry.get_meta(uuid)
        category = str(meta.category)
        fallback_description = str(meta.description or category)
        payload = _load_caption_payload(
            Path(
                str(meta.caption_path).replace(
                    "caption_candidates_updated.json",
                    "caption_candidates.json",
                )
            )
        )
        attribute_value = _resolve_single_asset_attribute(
            uuid=str(meta.uuid),
            attributes={
                "color": meta.color,
                "shape": meta.shape,
                "material": meta.material,
            },
            attribute_name=attribute_name,
        )
        return _build_instruction_actor(
            uuid=str(meta.uuid),
            category=category,
            payload=payload,
            fallback_description=fallback_description,
            actor_description_mode=actor_description_mode,
            actor_description_seed=actor_description_seed,
            attribute_name=attribute_name,
            attribute_value=attribute_value,
        )


class InstructionRenderError(ValueError):
    """Raised when strict rendering cannot resolve one or more placeholders."""


def _resolve_registry(
    *,
    registry: Optional["AssetRegistry"],
    asset_root: str | None,
) -> "AssetRegistry":
    if registry is not None:
        return registry
    if asset_root is None:
        raise ValueError("Either registry or asset_root is required")
    from robo_orchard_sim.asset_manager.registry import AssetRegistry

    return AssetRegistry(asset_root)


def render_instruction_from_registry(
    *,
    template_name: str,
    # Mapping from template actor names to asset uuids, e.g.
    # {"actor1": "uuid-pick", "actor2": "uuid-place"}.
    actor_uuids: Mapping[str, str],
    registry: Optional["AssetRegistry"] = None,
    asset_root: str | None = None,
    template_mode: Literal["fixed", "variants"] | None = None,
    actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
    template_seed: int | None = None,
    actor_description_seed: int | None = None,
) -> str:
    """Render one instruction from a template name and actor-uuid mapping."""
    resolved_registry = _resolve_registry(
        registry=registry,
        asset_root=asset_root,
    )

    actors = {
        actor_name: InstructionActor.from_registry(
            actor_uuid,
            resolved_registry,
            actor_description_mode=actor_description_mode,
            actor_description_seed=actor_description_seed,
        )
        for actor_name, actor_uuid in actor_uuids.items()
    }
    wrapper = InstructionWrapper(
        template_name,
        template_mode=template_mode,
        actor_description_mode=actor_description_mode,
    )
    return wrapper.render(
        actors=actors,
        template_seed=template_seed,
        actor_description_seed=actor_description_seed,
    )


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
        template_mode: Literal["fixed", "variants"] | None = None,
        actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
        attribute_name: InstructionAttributeName | None = None,
        strict: bool = True,
    ) -> None:
        self.template = template
        self.actor_description_mode = actor_description_mode
        self.attribute_name = attribute_name
        self.strict = strict
        self._formatter = string.Formatter()
        self._template_payload = self._load_template_payload(template)
        self.template_mode = self._resolve_template_mode(template_mode)

    def _load_template_payload(self, template: str) -> Mapping[str, Any]:
        from robo_orchard_sim.task_components.instructions.registry import (
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

    def _resolve_template_mode(
        self,
        template_mode: Literal["fixed", "variants"] | None,
    ) -> Literal["fixed", "variants"]:
        if template_mode is not None:
            return template_mode
        variants = self._template_payload.get("variants")
        if isinstance(variants, Sequence) and not isinstance(
            variants, (str, bytes)
        ):
            if variants:
                return "variants"
        return "fixed"

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
