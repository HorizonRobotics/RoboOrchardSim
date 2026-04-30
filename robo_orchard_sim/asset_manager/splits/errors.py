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

"""Exception types for the asset splits package."""

from __future__ import annotations


class AssetSplitsError(Exception):
    """Base class for all asset-splits errors."""


class InvalidSplitsYamlError(AssetSplitsError):
    """Raised for file, parse, or structural problems in a splits YAML."""


class UnsupportedSchemaVersionError(AssetSplitsError):
    """Raised when schema_version is not recognized."""

    def __init__(self, found: int, expected: int) -> None:
        self.found = found
        self.expected = expected
        super().__init__(
            f"Unsupported schema_version {found} (expected {expected})"
        )


class DuplicateAssetIdInSplitError(AssetSplitsError):
    """Raised when the same asset_id appears more than once in a split list."""

    def __init__(self, split_name: str, asset_id: str) -> None:
        self.split_name = split_name
        self.asset_id = asset_id
        super().__init__(
            f"Duplicate asset_id '{asset_id}' in split '{split_name}'"
        )


class UnknownAssetIdError(AssetSplitsError):
    """Raised when asset_ids in the YAML are not found in the registry."""

    def __init__(self, unknown_ids: tuple[str, ...]) -> None:
        self.unknown_ids = unknown_ids
        joined = ", ".join(unknown_ids)
        super().__init__(f"Unknown asset_id(s) not in registry: {joined}")


class EmptySeenSplitError(AssetSplitsError):
    """Raised when the 'seen' split is empty."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Benchmark '{name}' has an empty 'seen' split")


class OverlappingSplitsError(AssetSplitsError):
    """Raised when the three splits are not pairwise disjoint."""

    def __init__(
        self,
        name: str,
        seen_vs_unseen_category: frozenset[str],
        seen_vs_unseen_instance: frozenset[str],
        unseen_category_vs_unseen_instance: frozenset[str],
    ) -> None:
        self.name = name
        self.seen_vs_unseen_category = seen_vs_unseen_category
        self.seen_vs_unseen_instance = seen_vs_unseen_instance
        self.unseen_category_vs_unseen_instance = (
            unseen_category_vs_unseen_instance
        )
        parts: list[str] = []
        if seen_vs_unseen_category:
            parts.append(
                f"seen ∩ unseen_category: {sorted(seen_vs_unseen_category)}"
            )
        if seen_vs_unseen_instance:
            parts.append(
                f"seen ∩ unseen_instance: {sorted(seen_vs_unseen_instance)}"
            )
        if unseen_category_vs_unseen_instance:
            parts.append(
                f"unseen_category ∩ unseen_instance: "
                f"{sorted(unseen_category_vs_unseen_instance)}"
            )
        super().__init__(
            f"Benchmark '{name}' has overlapping splits: " + "; ".join(parts)
        )
