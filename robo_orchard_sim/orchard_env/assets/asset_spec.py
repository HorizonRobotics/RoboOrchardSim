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

"""User-facing asset description types for orchard env modules."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from pydantic import field_validator
from robo_orchard_core.utils.config import Config


class AssetSpec(Config, ABC):
    """Stable user-facing description of an orchard scene asset."""

    name: str
    namespace: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject path-like asset names."""
        if "/" in value:
            raise ValueError("Asset name must not contain '/'.")
        return value

    @property
    def scene_name(self) -> str:
        """Return the scene-unique asset name."""
        if self.namespace:
            return f"{self.namespace}/{self.name}"
        return self.name

    def with_default_namespace(self, namespace: str) -> AssetSpec:
        """Return a copy with namespace filled if it is currently unset."""
        if self.namespace is not None:
            return self
        return self.model_copy(update={"namespace": namespace})

    @abstractmethod
    def to_isaac_cfg(self) -> Any:
        """Convert this asset spec into the corresponding Isaac cfg."""


class CustomAssetSpec(AssetSpec):
    """Asset spec that directly wraps a caller-provided Isaac cfg."""

    cfg: Any

    def to_isaac_cfg(self) -> Any:
        """Return the provided Isaac cfg unchanged."""
        return self.cfg
