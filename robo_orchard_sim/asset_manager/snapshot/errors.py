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

"""Exception types for the asset snapshot package."""

from __future__ import annotations


class SnapshotError(Exception):
    """Base class for all snapshot errors."""


class InvalidSnapshotYamlError(SnapshotError):
    """Raised for file, parse, or structural problems in a snapshot YAML."""


class UnsupportedSchemaVersionError(SnapshotError):
    """Raised when schema_version is not recognized."""

    def __init__(self, found: int, expected: int) -> None:
        self.found = found
        self.expected = expected
        super().__init__(
            f"Unsupported schema_version {found} (expected {expected})"
        )


class SnapshotNameMismatchError(SnapshotError):
    """Raised when the file stem does not match the YAML 'name' field."""

    def __init__(self, file_stem: str, yaml_name: str) -> None:
        self.file_stem = file_stem
        self.yaml_name = yaml_name
        super().__init__(
            f"File stem '{file_stem}' does not match YAML name '{yaml_name}'"
        )


class DuplicateUuidInSnapshotError(SnapshotError):
    """Raised when the same uuid appears more than once in a snapshot."""

    def __init__(self, uuid: str) -> None:
        self.uuid = uuid
        super().__init__(f"Duplicate uuid '{uuid}' in snapshot assets")


class UnknownUuidInSnapshotError(SnapshotError):
    """Raised when snapshot uuids are not present in the registry."""

    def __init__(self, unknown_uuids: tuple[str, ...]) -> None:
        self.unknown_uuids = unknown_uuids
        joined = ", ".join(unknown_uuids[:5])
        suffix = (
            ""
            if len(unknown_uuids) <= 5
            else f" (+{len(unknown_uuids) - 5} more)"
        )
        super().__init__(
            f"Snapshot contains {len(unknown_uuids)} uuid(s) not in registry: "
            f"{joined}{suffix}"
        )


class ChecksumMismatchError(SnapshotError):
    """Raised when a parent snapshot's sha256 does not match recorded value."""

    def __init__(self, parent_path: str, expected: str, actual: str) -> None:
        self.parent_path = parent_path
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Checksum mismatch for parent '{parent_path}': "
            f"recorded={expected}, actual={actual}"
        )


class EmptyComposeResultError(SnapshotError):
    """Raised when a compose op (union/intersect/diff) yields empty result."""

    def __init__(self, op: str, input_names: tuple[str, ...]) -> None:
        self.op = op
        self.input_names = input_names
        super().__init__(
            f"compose op '{op}' on {list(input_names)} produced empty result"
        )
