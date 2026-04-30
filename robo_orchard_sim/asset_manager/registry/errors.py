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

"""Exception types for the asset_registry package."""

from __future__ import annotations


class AssetRegistryError(Exception):
    """Base class for all asset_registry errors."""


class AssetIndexNotFoundError(AssetRegistryError):
    """Raised when the index parquet file is missing."""


class AssetIndexVersionError(AssetRegistryError):
    """Raised when the index parquet schema_version is incompatible."""


class DuplicateAssetIdError(AssetRegistryError):
    """Raised during build when two dirs share the same asset_id."""

    def __init__(self, asset_id: str, paths: list[str]) -> None:
        self.asset_id = asset_id
        self.paths = list(paths)
        joined = ", ".join(self.paths)
        super().__init__(f"Duplicate asset_id '{asset_id}' found at: {joined}")


class UnknownAssetError(AssetRegistryError):
    """Raised when a uuid or asset_id is not in the registry."""

    def __init__(
        self,
        key: str,
        closest_matches: list[str] | None = None,
    ) -> None:
        self.key = key
        self.closest_matches = list(closest_matches or [])
        suffix = ""
        if self.closest_matches:
            suffix = f" (did you mean: {', '.join(self.closest_matches)}?)"
        super().__init__(f"Unknown asset '{key}'{suffix}")


class EmptyPoolError(AssetRegistryError):
    """Raised when no asset in the library matches the filter."""

    def __init__(self, filter_repr: str) -> None:
        self.filter_repr = filter_repr
        super().__init__(f"no assets in library match filter {filter_repr}")


class InsufficientPoolError(AssetRegistryError):
    """Raised when the filter matched too few assets for the request."""

    def __init__(self, mode: str, available: int, requested: int) -> None:
        self.mode = mode
        self.available = available
        self.requested = requested
        super().__init__(
            f"only {available} asset(s) match (requested {requested}, "
            f"mode={mode})"
        )


class CollisionExhaustedError(EmptyPoolError):
    """Raised when sample_compatible_pair could not find a disjoint pair.

    Subclass of EmptyPoolError so callers catching EmptyPoolError still
    catch this, but allowing precise handling: `except CollisionExhaustedError`
    distinguishes "unlucky draw, try a different seed" from "filter yields
    empty pool, your constraint is impossible".
    """

    def __init__(
        self,
        pick_filter_repr: str,
        place_filter_repr: str,
        attempts: int,
    ) -> None:
        self.pick_filter_repr = pick_filter_repr
        self.place_filter_repr = place_filter_repr
        self.attempts = attempts
        super().__init__(
            filter_repr=(
                f"pick={pick_filter_repr} place={place_filter_repr} "
                f"(no disjoint pair within {attempts} attempts)"
            )
        )
