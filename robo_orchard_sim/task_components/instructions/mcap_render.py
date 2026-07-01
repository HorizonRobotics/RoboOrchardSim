## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

from __future__ import annotations
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional

from google.protobuf.json_format import MessageToDict
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.task_components.instructions.base import (
    InstructionRenderError,
    _resolve_registry,
    render_instruction_from_registry,
)
from robo_orchard_sim.task_components.instructions.registry import (
    get_instruction_template,
)

_META_DATA_TOPIC = "/meta_data"
_ACTORS_KEY = "actors"
_PICK_ACTOR_TYPE = "pick"
_PLACE_ACTOR_TYPE = "place"
_ACTOR1_KEY = "actor1"
_ACTOR2_KEY = "actor2"
_ACTOR_TYPE_BY_TEMPLATE_KEY = {
    _ACTOR1_KEY: _PICK_ACTOR_TYPE,
    _ACTOR2_KEY: _PLACE_ACTOR_TYPE,
}


def _extract_actor_keys_from_template_text(template_text: str) -> set[str]:
    formatter = string.Formatter()
    actor_keys: set[str] = set()
    for _, field_name, _, _ in formatter.parse(template_text):
        if not field_name:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root.startswith("actor") and root != "actors":
            actor_keys.add(root)
    return actor_keys


def _load_meta_dict_from_mcap(mcap_path: str) -> Mapping[str, Any]:
    """Return the first decodable /meta_data payload from one MCAP file."""
    with Path(mcap_path).open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, _, _, proto_msg in reader.iter_decoded_messages(
            topics=[_META_DATA_TOPIC]
        ):
            try:
                return MessageToDict(
                    proto_msg,
                    preserving_proto_field_name=True,
                )
            except (AttributeError, TypeError, ValueError):
                continue
    raise InstructionRenderError(
        f"No {_META_DATA_TOPIC} Struct message found in '{mcap_path}'"
    )


def _map_required_actor_uuids(
    actors: Mapping[str, Any],
    *,
    required_actor_keys: set[str],
    mcap_path: str,
) -> dict[str, str]:
    """Map required template actor keys to uuids from actor_type metadata."""
    picked: dict[str, list[str]] = {
        _PICK_ACTOR_TYPE: [],
        _PLACE_ACTOR_TYPE: [],
    }
    for actor_payload in actors.values():
        if not isinstance(actor_payload, Mapping):
            continue
        actor_type = actor_payload.get("actor_type")
        actor_uuid = actor_payload.get("actor_uuid")
        if actor_type in picked and isinstance(actor_uuid, str):
            picked[str(actor_type)].append(actor_uuid)

    actor_uuids: dict[str, str] = {}
    for actor_key in sorted(required_actor_keys):
        try:
            actor_type = _ACTOR_TYPE_BY_TEMPLATE_KEY[actor_key]
        except KeyError as exc:
            raise InstructionRenderError(
                f"Unsupported template actor key '{actor_key}'"
            ) from exc
        count = len(picked[actor_type])
        if count != 1:
            raise InstructionRenderError(
                f"Expected exactly one {actor_type} actor_uuid for "
                f"template key '{actor_key}' in '{mcap_path}', found {count}"
            )
        actor_uuids[actor_key] = picked[actor_type][0]
    return actor_uuids


def _extract_required_actor_keys_from_template(
    template_name: str,
    *,
    template_mode: Literal["fixed", "variants"] | None = None,
) -> set[str]:
    """Return actor keys referenced by the effective template mode."""
    payload = get_instruction_template(template_name)
    resolved_mode = template_mode
    if resolved_mode is None:
        variants = payload.get("variants")
        if isinstance(variants, Sequence) and not isinstance(
            variants, (str, bytes)
        ):
            if variants:
                resolved_mode = "variants"
        if resolved_mode is None:
            resolved_mode = "fixed"

    actor_keys: set[str] = set()
    if resolved_mode == "fixed":
        fixed_template = payload.get("fixed")
        if not isinstance(fixed_template, str):
            raise InstructionRenderError(
                "Template payload field 'fixed' must be a string"
            )
        return _extract_actor_keys_from_template_text(fixed_template)

    mode_templates = payload.get(resolved_mode)
    if not isinstance(mode_templates, Sequence) or isinstance(
        mode_templates, (str, bytes)
    ):
        raise InstructionRenderError(
            f"Template payload field '{resolved_mode}' must be a string list"
        )
    if not mode_templates:
        raise InstructionRenderError(
            f"Template payload field '{resolved_mode}' is empty"
        )
    if not all(isinstance(item, str) for item in mode_templates):
        raise InstructionRenderError(
            f"Template payload field '{resolved_mode}' "
            "must contain only strings"
        )
    for template_text in mode_templates:
        actor_keys.update(
            _extract_actor_keys_from_template_text(template_text)
        )
    return actor_keys


def _map_pick_place_actor_uuids(
    actors: Mapping[str, Any],
    *,
    required_actor_keys: set[str],
    mcap_path: str,
) -> dict[str, str]:
    """Map pick/place actor metadata to the required template actor keys."""
    return _map_required_actor_uuids(
        actors,
        required_actor_keys=required_actor_keys,
        mcap_path=mcap_path,
    )


def extract_instruction_actor_uuids_from_mcap(
    mcap_path: str,
    *,
    template_name: str,
    template_mode: Literal["fixed", "variants"] | None = None,
) -> dict[str, str]:
    """Extract required template actor uuids from one MCAP metadata stream."""
    payload = _load_meta_dict_from_mcap(mcap_path)
    actors = payload.get(_ACTORS_KEY)
    if not isinstance(actors, Mapping):
        raise InstructionRenderError(
            f"MCAP metadata in '{mcap_path}' missing '{_ACTORS_KEY}' mapping"
        )
    required_actor_keys = _extract_required_actor_keys_from_template(
        template_name,
        template_mode=template_mode,
    )
    return _map_pick_place_actor_uuids(
        actors,
        required_actor_keys=required_actor_keys,
        mcap_path=mcap_path,
    )


def render_instruction_from_mcap(
    *,
    mcap_path: str,
    template_name: str,
    registry: Optional["AssetRegistry"] = None,
    asset_root: str | None = None,
    template_mode: Literal["fixed", "variants"] | None = None,
    actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
    template_seed: int | None = None,
    actor_description_seed: int | None = None,
) -> str:
    """Render one instruction from one MCAP file."""
    actor_uuids = extract_instruction_actor_uuids_from_mcap(
        mcap_path,
        template_name=template_name,
        template_mode=template_mode,
    )
    return render_instruction_from_registry(
        template_name=template_name,
        actor_uuids=actor_uuids,
        registry=registry,
        asset_root=asset_root,
        template_mode=template_mode,
        template_seed=template_seed,
        actor_description_mode=actor_description_mode,
        actor_description_seed=actor_description_seed,
    )


def render_instructions_from_mcaps(
    *,
    mcap_paths: Sequence[str],
    template_name: str,
    registry: Optional["AssetRegistry"] = None,
    asset_root: str | None = None,
    template_mode: Literal["fixed", "variants"] | None = None,
    actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
    template_seed: int | None = None,
    actor_description_seed: int | None = None,
) -> list[dict[str, str]]:
    """Render one instruction per MCAP file."""
    resolved_registry = _resolve_registry(
        registry=registry,
        asset_root=asset_root,
    )
    return [
        {
            "mcap_path": mcap_path,
            "instruction": render_instruction_from_mcap(
                mcap_path=mcap_path,
                template_name=template_name,
                registry=resolved_registry,
                template_mode=template_mode,
                actor_description_mode=actor_description_mode,
                template_seed=template_seed,
                actor_description_seed=actor_description_seed,
            ),
        }
        for mcap_path in mcap_paths
    ]
