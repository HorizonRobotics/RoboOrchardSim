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

"""Generic asset resolver: config + registry + splits -> AssetSpec dict."""

from __future__ import annotations
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np

from robo_orchard_sim.asset_manager.registry.errors import (
    AssetRegistryError,
    EmptyPoolError,
    InsufficientPoolError,
    UnknownAssetError,
)
from robo_orchard_sim.asset_manager.registry.registry import (
    AssetRegistry,
    AssetSampler,
)
from robo_orchard_sim.asset_manager.registry.types import (
    RIGID_OBJECT_SPEC_TYPE,
    AssetFilter,
    AssetMeta,
    DistractorSpec,
)
from robo_orchard_sim.asset_manager.splits.splits import AssetSplits

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.assets import ObjectSpec

    class SemanticReferentRule(Protocol):
        """Fields required to select an absent semantic referent."""

        @property
        def match(self) -> tuple[str, ...]: ...

        @property
        def differ(self) -> tuple[str, ...]: ...

        @property
        def referent_fields(self) -> tuple[str, ...]: ...

        @property
        def required_tags(self) -> frozenset[str]: ...


logger = logging.getLogger(__name__)
T = TypeVar("T")

# -----------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------


class AssetResolverError(Exception):
    """Base class for resolver errors."""


class AssetResolutionError(AssetResolverError):
    """Sampling failed for a specific role."""

    def __init__(self, role: str, filter_repr: str, cause: Exception) -> None:
        self.role = role
        self.filter_repr = filter_repr
        self.cause = cause
        super().__init__(f"role '{role}' — {cause}")


def _normalize_filter(entry: dict, role: str) -> dict:
    """Return filter dict; missing key or null -> empty (match-all)."""
    raw = entry.get("filter")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise AssetResolutionError(
            role=role,
            filter_repr=str(raw),
            cause=TypeError(
                f"filter must be a dict (or omitted/null for match-all); "
                f"got {type(raw).__name__}"
            ),
        )
    return dict(raw)


def _synth_meta_from_path(usd_path: str, entry: dict) -> AssetMeta:
    """Build a minimal in-memory AssetMeta for a path-pinned target.

    Only fields consumed by a no-distractor, no-instruction showcase scene
    are meaningful; the rest are deterministic placeholders.
    """
    import hashlib
    import os

    stem = os.path.splitext(os.path.basename(usd_path))[0] or "asset"
    metadata_path = Path(usd_path).parent / "metadata.json"
    caption_candidate_path = Path(usd_path).parent / "caption_candidates.json"
    uuid = hashlib.sha1(usd_path.encode("utf-8")).hexdigest()
    if metadata_path.is_file() and caption_candidate_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if isinstance(metadata, dict) and metadata.get("uuid"):
            uuid = str(metadata["uuid"])
    mass = float(entry.get("mass", 0.05))
    category = str(entry.get("category", stem))
    interaction_path = str(entry.get("interaction_path", ""))
    caption_path = str(
        entry.get(
            "caption_path",
            caption_candidate_path if caption_candidate_path.is_file() else "",
        )
    )
    return AssetMeta(
        uuid=uuid,
        asset_id=stem,
        relative_path=usd_path,
        domain="showcase",
        super_category="showcase",
        category=category,
        name=stem,
        description=category,
        color=None,
        shape=None,
        material=None,
        real_height=0.0,
        real_mass=mass,
        min_height=0.0,
        max_height=0.0,
        min_mass=mass,
        max_mass=mass,
        usd_path=usd_path,
        urdf_path="",
        interaction_path=interaction_path,
        caption_path=caption_path,
        metadata_path=str(metadata_path),
        spec_type=str(entry.get("spec_type", RIGID_OBJECT_SPEC_TYPE)),
    )


# -----------------------------------------------------------------------
# Split name -> AssetSplits field mapping
# -----------------------------------------------------------------------

_SPLIT_FIELDS = frozenset({"seen", "unseen_category", "unseen_instance"})
_SET_VALUED_SEMANTIC_FIELDS = frozenset({"color", "shape", "material"})

# Allowed keys per entry kind. Typos (e.g. ``macth`` instead of ``match``)
# would otherwise silently no-op — validate up-front so authoring errors
# surface on the first resolve call.
_TARGET_ENTRY_KEYS = frozenset(
    {
        "filter",
        "prim_name",
        "same_as",
        "sample_count",
        "split",
        "uuid",
        "usd_path",
        "interaction_path",
        "mass",
        "category",
        "spec_type",
    }
)

_SAMPLING_ENTRY_KEYS = frozenset(
    {"filter", "sample_count", "split", "uuid", "usd_path"}
)
"""Keys that choose an asset, and so have nothing to say for a clone."""
_DISTRACTOR_ENTRY_KEYS = frozenset(
    {
        "anchor",
        "match",
        "differ",
        "filter",
        "min_count",
        "max_count",
        "prim_name_prefix",
        "split",
    }
)


def _matches_referent(
    candidate: AssetMeta,
    present: AssetMeta,
    fields: tuple[str, ...],
) -> bool:
    """Return whether a present asset satisfies a candidate phrase."""
    for field in fields:
        candidate_value = getattr(candidate, field)
        present_value = getattr(present, field)
        if field in _SET_VALUED_SEMANTIC_FIELDS:
            if candidate_value is None or present_value is None:
                return False
            if not candidate_value.issubset(present_value):
                return False
        elif candidate_value != present_value:
            return False
    return True


def _has_renderable_referent_fields(
    candidate: AssetMeta,
    fields: tuple[str, ...],
) -> bool:
    """Return whether instruction rendering can express every field."""
    for field in fields:
        value = getattr(candidate, field)
        if value is None or value == "":
            return False
        if field in {"shape", "material"} and len(value) != 1:
            return False
    return True


# -----------------------------------------------------------------------
# AssetResolver
# -----------------------------------------------------------------------


class AssetResolver:
    """Resolve per-role asset configs into concrete AssetSpec instances.

    Constructed once per evaluator session with a shared registry,
    optional asset splits, and an rng. Called once per task with the
    task's asset configs.

    The resolver is task-agnostic: it transforms a ``dict[role, config]``
    into a ``dict[role, AssetSpec | list[AssetSpec]]`` and never inspects
    the calling task's schema. Role membership and required/optional
    semantics are owned by the task's declared ``roles`` and enforced
    when the task is constructed from this resolver's output.
    """

    def __init__(
        self,
        registry: AssetRegistry,
        splits: AssetSplits | None = None,
        rng: np.random.Generator | None = None,
        *,
        active_snapshot: frozenset[str] | None = None,
    ) -> None:
        self._registry = registry
        self._splits = splits
        self._sampler = AssetSampler(registry)
        self._rng = rng or np.random.default_rng()
        self._active_snapshot = active_snapshot

    def sample_without_replacement(
        self,
        candidates: Sequence[T],
        *,
        count: int,
    ) -> list[T]:
        """Draw an ordered sample from this resolver's seeded RNG."""
        if not 0 <= count <= len(candidates):
            raise ValueError(
                f"Cannot draw {count} candidates from a pool of "
                f"{len(candidates)}."
            )
        indices = self._rng.choice(
            len(candidates),
            size=count,
            replace=False,
        )
        return [candidates[int(index)] for index in indices]

    def resolve(
        self,
        asset_configs: dict[str, dict],
    ) -> dict[str, Any]:
        """Resolve asset_configs into AssetSpec instances per role."""
        for role, entry in asset_configs.items():
            allowed = (
                _DISTRACTOR_ENTRY_KEYS
                if "anchor" in entry
                else _TARGET_ENTRY_KEYS
            )
            unknown = frozenset(entry.keys()) - allowed
            if unknown:
                raise AssetResolutionError(
                    role=role,
                    filter_repr=str(entry),
                    cause=ValueError(
                        f"Unknown entry key(s): {sorted(unknown)}. "
                        f"Allowed: {sorted(allowed)}"
                    ),
                )

        committed_uuids: set[str] = set()
        result: dict[str, Any] = {}
        target_metas: dict[str, AssetMeta] = {}

        for role, entry in asset_configs.items():
            if "anchor" not in entry and "same_as" not in entry:
                metas, spec_or_specs = self._resolve_target(
                    role,
                    entry,
                    committed_uuids,
                )
                target_metas[role] = metas[0]
                committed_uuids.update(m.uuid for m in metas)
                result[role] = spec_or_specs

        # Clones run once every sampling slot has drawn, so a clone may
        # name its source regardless of the order the two were written
        # in. They add nothing to ``committed_uuids``: the asset was
        # already claimed by the slot they copy, and claiming it twice
        # would say a second one had been used up.
        for role, entry in asset_configs.items():
            if "same_as" in entry:
                result[role] = self._resolve_clone(
                    role,
                    entry,
                    asset_configs,
                    target_metas,
                )

        for role, entry in asset_configs.items():
            if "anchor" in entry:
                metas, specs = self._resolve_distractors(
                    role,
                    entry,
                    target_metas,
                    committed_uuids,
                )
                committed_uuids.update(m.uuid for m in metas)
                result[role] = specs

        self._log_resolution_report(asset_configs, result)
        return result

    def resolve_absent_referent(
        self,
        *,
        present_uuids: list[str],
        rule: "SemanticReferentRule",
    ) -> "ObjectSpec":
        """Resolve a non-spawned rigid object absent from the whole scene."""
        role = "instruction.actor1"
        filter_repr = repr(rule)
        unique_present_uuids = list(dict.fromkeys(present_uuids))
        if not unique_present_uuids:
            raise AssetResolutionError(
                role=role,
                filter_repr=filter_repr,
                cause=ValueError(
                    "absent-object resolution requires at least one "
                    "present asset"
                ),
            )

        try:
            present_metas = [
                self._registry.get_meta(uuid) for uuid in unique_present_uuids
            ]
            only_in = self._scene_asset_scope(unique_present_uuids)
            anchor_order = self._rng.permutation(len(present_metas))
            candidate = None
            for anchor_idx in anchor_order:
                anchor = present_metas[int(anchor_idx)]
                candidates = self._sampler.sample_distractors(
                    anchor,
                    DistractorSpec(
                        min_count=0,
                        max_count=len(self._registry),
                        match=rule.match,
                        differ=rule.differ,
                        absolute_filter=AssetFilter(
                            tags=rule.required_tags,
                            spec_type=RIGID_OBJECT_SPEC_TYPE,
                        ),
                        only_in=only_in,
                        exclude=frozenset(unique_present_uuids),
                    ),
                    self._rng,
                )
                for possible in candidates:
                    if not _has_renderable_referent_fields(
                        possible,
                        rule.referent_fields,
                    ):
                        continue
                    if any(
                        _matches_referent(
                            possible,
                            present,
                            rule.referent_fields,
                        )
                        for present in present_metas
                    ):
                        continue
                    candidate = possible
                    break
                if candidate is not None:
                    break
        except (AssetRegistryError, ValueError) as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=filter_repr,
                cause=exc,
            ) from exc

        if candidate is None:
            raise AssetResolutionError(
                role=role,
                filter_repr=filter_repr,
                cause=ValueError(
                    "no renderable absent referent satisfies "
                    f"match={list(rule.match)}, "
                    f"differ={list(rule.differ)}, "
                    f"referent_fields={list(rule.referent_fields)}"
                ),
            )

        try:
            return self._registry.build_spec(
                candidate,
                name="absent_object",
                role="instruction",
            )
        except AssetRegistryError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=filter_repr,
                cause=exc,
            ) from exc

    def _scene_asset_scope(
        self,
        present_uuids: list[str],
    ) -> frozenset[str] | None:
        """Return the snapshot/split scope represented by a scene."""
        only_in: frozenset[str] | None = None
        if self._splits is not None:
            present = frozenset(present_uuids)
            represented_scopes = [
                split_scope
                for split_scope in (
                    self._splits.seen,
                    self._splits.unseen_category,
                    self._splits.unseen_instance,
                )
                if present & split_scope
            ]
            if represented_scopes:
                only_in = frozenset().union(*represented_scopes)
        if self._active_snapshot is not None:
            only_in = (
                self._active_snapshot
                if only_in is None
                else only_in & self._active_snapshot
            )
        return only_in

    def _log_resolution_report(
        self,
        asset_configs: dict[str, dict],
        resolved: dict[str, Any],
    ) -> None:
        """Log the number of concrete assets resolved for each role."""
        lines = ["asset resolver — resolved assets:"]
        for role, value in resolved.items():
            count = len(value) if isinstance(value, list) else 1
            entry = asset_configs[role]
            requested = entry.get("sample_count", count if count > 1 else 1)
            empty_filter = (
                "uuid" not in entry
                and "same_as" not in entry
                and not entry.get("filter")
            )
            hint = " [filter=<empty: full registry>]" if empty_filter else ""
            lines.append(
                f"  {role}: requested={requested}, resolved={count}{hint}"
            )
        logger.info("\n".join(lines))

    def _resolve_target(
        self,
        role: str,
        entry: dict,
        committed_uuids: set[str],
    ) -> tuple[
        list[AssetMeta],
        "ObjectSpec | list[ObjectSpec]",
    ]:
        """Resolve a target-kind role.

        ``sample_count`` returns ordinary, simultaneously active object
        specs.
        """
        sample_count = int(entry.get("sample_count", 1))
        if sample_count < 1:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    f"sample_count must be >= 1, got {sample_count}"
                ),
            )
        if sample_count != 1 and ("usd_path" in entry or "uuid" in entry):
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    "sample_count > 1 requires registry filter sampling; "
                    "it cannot be combined with uuid or usd_path."
                ),
            )
        if "spec_type" in entry and "usd_path" not in entry:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    "spec_type is only valid for a direct usd_path entry; "
                    "indexed assets obtain it from AssetMeta."
                ),
            )

        if "usd_path" in entry:
            meta, spec = self._resolve_target_by_path(role, entry)
            return [meta], spec

        if "uuid" in entry:
            meta, spec = self._resolve_target_by_uuid(role, entry)
            return [meta], spec

        filter_dict = _normalize_filter(entry, role)
        filter_dict.setdefault("spec_type", RIGID_OBJECT_SPEC_TYPE)

        only_in = self._resolve_split_only_in(
            role, entry, err_repr=str(filter_dict)
        )
        if only_in is not None:
            filter_dict["only_in"] = only_in
        if committed_uuids:
            filter_dict["exclude"] = frozenset(committed_uuids)

        try:
            asset_filter = AssetFilter(**filter_dict)
        except TypeError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(filter_dict),
                cause=exc,
            ) from exc

        try:
            prim_name = entry["prim_name"]
        except KeyError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=repr(asset_filter),
                cause=exc,
            ) from exc

        try:
            if sample_count == 1:
                meta = self._sampler.sample_target(
                    asset_filter,
                    rng=self._rng,
                )
                spec = self._registry.build_spec(
                    meta, name=prim_name, role=role
                )
                return [meta], spec
            metas = self._sampler.sample_target_pool(
                asset_filter,
                k=sample_count,
                rng=self._rng,
            )
            specs = [
                self._registry.build_spec(
                    meta,
                    name=f"{prim_name}_{index}",
                    role=role,
                )
                for index, meta in enumerate(metas)
            ]
            return metas, specs
        except (EmptyPoolError, InsufficientPoolError) as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=repr(asset_filter),
                cause=exc,
            ) from exc

    def _resolve_clone(
        self,
        role: str,
        entry: dict,
        asset_configs: dict[str, dict],
        target_metas: dict[str, AssetMeta],
    ) -> "ObjectSpec":
        """Give this role the asset another role already drew.

        Sharing one asset is how a scene puts several identical objects
        in front of a policy: when they cannot be told apart by looking,
        the only thing left to go on is the instruction.
        """
        source = entry["same_as"]
        if source not in target_metas:
            known = sorted(target_metas)
            hint = (
                f" '{source}' is itself a clone; point at the slot it "
                "copies instead."
                if source in asset_configs
                and "same_as" in asset_configs[source]
                else ""
            )
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    f"same_as names {source!r}, which is not a slot that "
                    f"draws its own asset. Slots that do: {known}.{hint}"
                ),
            )

        conflicting = sorted(_SAMPLING_ENTRY_KEYS & set(entry))
        if conflicting:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    f"same_as takes the asset from {source!r}, so "
                    f"{conflicting} could not be honoured; drop them, or "
                    "drop same_as and let this slot draw its own."
                ),
            )

        source_entry = asset_configs[source]
        if int(source_entry.get("sample_count", 1)) != 1:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    f"same_as names {source!r}, which draws "
                    f"{source_entry['sample_count']} assets; there is no "
                    "one asset to share. Point at a slot that draws a "
                    "single asset."
                ),
            )

        try:
            prim_name = entry["prim_name"]
        except KeyError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        return self._registry.build_spec(
            target_metas[source], name=prim_name, role=role
        )

    def _resolve_target_by_uuid(
        self, role: str, entry: dict
    ) -> tuple[AssetMeta, "ObjectSpec"]:
        """Resolve a target pinned by uuid.

        uuid is authoritative: the asset is fetched directly from the
        registry. ``filter`` and ``split`` are still parsed for
        consistency checks but only emit a warning on mismatch — they do
        not override the explicit uuid.
        """
        uuid = entry["uuid"]
        try:
            meta = self._registry.get_meta(uuid)
        except UnknownAssetError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        try:
            prim_name = entry["prim_name"]
        except KeyError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        filter_dict = _normalize_filter(entry, role)
        if filter_dict:
            try:
                filter_check = AssetFilter(**filter_dict)
            except TypeError as exc:
                raise AssetResolutionError(
                    role=role,
                    filter_repr=str(filter_dict),
                    cause=exc,
                ) from exc
            if not filter_check.matches(meta):
                logger.warning(
                    "role %r: pinned uuid %r does not satisfy filter %r; "
                    "uuid takes precedence.",
                    role,
                    uuid,
                    filter_check,
                )

        only_in = self._resolve_split_only_in(role, entry, err_repr=str(entry))
        if only_in is not None and uuid not in only_in:
            split_name = entry.get("split")
            if split_name is not None and self._active_snapshot is not None:
                scope_desc = f"split {split_name!r} ∩ active_snapshot"
            elif split_name is not None:
                scope_desc = f"split {split_name!r}"
            else:
                scope_desc = "active_snapshot"
            logger.warning(
                "role %r: pinned uuid %r is not in %s; uuid takes precedence.",
                role,
                uuid,
                scope_desc,
            )

        return meta, self._registry.build_spec(meta, name=prim_name, role=role)

    def _resolve_target_by_path(
        self, role: str, entry: dict
    ) -> tuple[AssetMeta, "ObjectSpec"]:
        """Resolve a target pinned by a direct USD path (no registry lookup).

        For showcase scenes whose assets live as loose directories outside
        any registered library. ``usd_path`` is authoritative and may not be
        combined with registry-driven keys.
        """
        conflicting = sorted({"uuid", "filter", "split"} & set(entry))
        if conflicting:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    f"usd_path is mutually exclusive with {conflicting}."
                ),
            )
        try:
            prim_name = entry["prim_name"]
        except KeyError as exc:
            raise AssetResolutionError(
                role=role, filter_repr=str(entry), cause=exc
            ) from exc

        meta = _synth_meta_from_path(str(entry["usd_path"]), entry)
        spec = self._registry.build_spec(meta, name=prim_name, role=role)
        return meta, spec

    def _resolve_split_only_in(
        self,
        role: str,
        entry: dict,
        err_repr: str,
    ) -> frozenset[str] | None:
        """Resolve the `split` field to an `only_in` uuid set.

        Returns None when neither split nor active_snapshot applies.
        Raises AssetResolutionError for unknown split names or when
        the resulting (split ∩ active_snapshot) is empty.
        """
        only_in: frozenset[str] | None = None
        split_name = entry.get("split")
        if split_name is not None and self._splits is not None:
            if split_name not in _SPLIT_FIELDS:
                raise AssetResolutionError(
                    role=role,
                    filter_repr=err_repr,
                    cause=ValueError(
                        f"Unknown split '{split_name}'. "
                        f"Must be one of: {sorted(_SPLIT_FIELDS)}"
                    ),
                )
            only_in = getattr(self._splits, split_name)

        if self._active_snapshot is not None:
            only_in = (
                (only_in & self._active_snapshot)
                if only_in is not None
                else self._active_snapshot
            )
            if not only_in:
                scope_desc = (
                    f"split {split_name!r} ∩ active_snapshot"
                    if split_name is not None
                    else "active_snapshot"
                )
                raise AssetResolutionError(
                    role=role,
                    filter_repr=err_repr,
                    cause=ValueError(
                        f"{scope_desc} is empty — snapshot does not cover "
                        "any assets for this role"
                    ),
                )
        return only_in

    def _resolve_distractors(
        self,
        role: str,
        entry: dict,
        target_metas: dict[str, AssetMeta],
        committed_uuids: set[str],
    ) -> tuple[list[AssetMeta], list["ObjectSpec"]]:
        """Resolve a distractor entry into concrete object specs."""
        anchor = entry.get("anchor")
        if anchor not in target_metas:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=ValueError(
                    f"anchor '{anchor}' does not refer to an "
                    f"already-resolved target role. Known target roles: "
                    f"{sorted(target_metas.keys())}"
                ),
            )
        anchor_meta = target_metas[anchor]

        try:
            min_count = int(entry["min_count"])
            max_count = int(entry["max_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        only_in = self._resolve_split_only_in(role, entry, err_repr=str(entry))

        filter_dict = _normalize_filter(entry, role)
        filter_dict.setdefault("spec_type", anchor_meta.spec_type)
        try:
            absolute_filter = AssetFilter(**filter_dict)
        except TypeError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        try:
            spec = DistractorSpec(
                min_count=min_count,
                max_count=max_count,
                match=tuple(entry.get("match", ())),
                differ=tuple(entry.get("differ", ())),
                absolute_filter=absolute_filter,
                only_in=only_in,
                exclude=frozenset(committed_uuids),
            )
        except ValueError as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(entry),
                cause=exc,
            ) from exc

        try:
            metas = self._sampler.sample_distractors(
                anchor_meta,
                spec,
                rng=self._rng,
            )
        except (EmptyPoolError, InsufficientPoolError, ValueError) as exc:
            raise AssetResolutionError(
                role=role,
                filter_repr=str(spec),
                cause=exc,
            ) from exc

        prim_name_prefix = entry.get("prim_name_prefix", role)
        specs = [
            self._registry.build_spec(
                meta, name=f"{prim_name_prefix}_{idx}", role=role
            )
            for idx, meta in enumerate(metas)
        ]
        return metas, specs
