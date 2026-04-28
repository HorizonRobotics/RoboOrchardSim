# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tag vocabulary loader for the tag labeller skill.

Loads tag definitions from a YAML file with this schema:

    tags:
      is_container:
        description: <one-line summary, max 200 chars>
        criteria: |
          <multi-line judgement criteria for GPT>
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterator

import yaml

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_DESCRIPTION_LEN = 200


class TagVocabError(ValueError):
    """Raised when the tag vocabulary YAML is invalid."""


@dataclass(frozen=True)
class TagSpec:
    """Specification for a single tag.

    Attributes:
        name: Snake-case tag identifier (e.g. ``is_container``).
        description: One-line human-readable summary (max 200 chars).
        criteria: Judgement criteria text shown to GPT.
    """

    name: str
    description: str
    criteria: str


class TagVocab:
    """Ordered collection of TagSpec loaded from YAML.

    Order of insertion in the YAML file is preserved.
    """

    def __init__(self, specs: list[TagSpec]):
        self._specs = list(specs)
        self._by_name = {s.name: s for s in self._specs}
        if len(self._by_name) != len(self._specs):
            raise TagVocabError("duplicate tag names in vocab")

    @classmethod
    def from_yaml(cls, path: str) -> "TagVocab":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "tags" not in data or not isinstance(data["tags"], dict):
            raise TagVocabError(f"{path}: missing top-level 'tags' mapping")
        raw_tags = data["tags"]
        if not raw_tags:
            raise TagVocabError(f"{path}: vocab must have at least one tag")

        specs: list[TagSpec] = []
        for name, raw in raw_tags.items():
            specs.append(_validate_and_build(name, raw, path))
        return cls(specs)

    def names(self) -> Iterator[str]:
        return iter(s.name for s in self._specs)

    def get(self, name: str) -> TagSpec:
        return self._by_name[name]

    def is_known(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self) -> Iterator[TagSpec]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


def _validate_and_build(name: str, raw: object, src: str) -> TagSpec:
    if not isinstance(name, str) or not _NAME_PATTERN.match(name):
        raise TagVocabError(
            f"{src}: tag name {name!r} must be snake_case"
            f" (matching {_NAME_PATTERN.pattern})"
        )
    if not isinstance(raw, dict):
        raise TagVocabError(
            f"{src}: tag {name!r} must be a mapping, got {type(raw).__name__}"
        )
    description = raw.get("description")
    if not isinstance(description, str) or not description:
        raise TagVocabError(
            f"{src}: tag {name!r} missing string field 'description'"
        )
    if len(description) > _MAX_DESCRIPTION_LEN:
        raise TagVocabError(
            f"{src}: tag {name!r} description exceeds"
            f" {_MAX_DESCRIPTION_LEN} chars"
        )
    criteria = raw.get("criteria")
    if not isinstance(criteria, str) or not criteria.strip():
        raise TagVocabError(
            f"{src}: tag {name!r} missing string field 'criteria'"
        )
    return TagSpec(name=name, description=description, criteria=criteria)
