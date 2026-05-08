# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Owns role-alias state for per-episode asset swap."""

from __future__ import annotations


class PoolAliasState:
    """Owns role-alias state for per-episode asset swap."""

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}
        self._registered_roles: set[str] = set()

    def register_pool(self, role_id: str) -> None:
        if role_id in self._registered_roles:
            raise ValueError(f"role_id {role_id!r} already registered")
        self._registered_roles.add(role_id)

    def set_active(self, role_id: str, scene_name: str | None) -> None:
        if role_id not in self._registered_roles:
            raise KeyError(f"role_id {role_id!r} not registered as a pool")
        if scene_name is None:
            self._aliases.pop(role_id, None)
        else:
            self._aliases[role_id] = scene_name

    def resolve(self, key: str) -> str:
        return self._aliases.get(key, key)

    def has_pool(self, role_id: str) -> bool:
        return role_id in self._registered_roles

    @property
    def aliases(self) -> dict[str, str]:
        """Snapshot of currently bound role_id → scene_name mappings."""
        return dict(self._aliases)
