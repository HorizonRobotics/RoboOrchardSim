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

"""Asset snapshots: immutable, uuid-keyed asset set artifacts."""

from robo_orchard_sim.asset_manager.snapshot.compose import compose_snapshots
from robo_orchard_sim.asset_manager.snapshot.errors import (
    ChecksumMismatchError,
    DuplicateUuidInSnapshotError,
    EmptyComposeResultError,
    InvalidSnapshotYamlError,
    SnapshotError,
    SnapshotNameMismatchError,
    UnknownUuidInSnapshotError,
    UnsupportedSchemaVersionError,
)
from robo_orchard_sim.asset_manager.snapshot.snapshot import (
    SCHEMA_VERSION,
    Snapshot,
    from_registry,
    load_snapshot,
    save_snapshot,
)

__all__ = (
    "ChecksumMismatchError",
    "compose_snapshots",
    "DuplicateUuidInSnapshotError",
    "EmptyComposeResultError",
    "from_registry",
    "InvalidSnapshotYamlError",
    "load_snapshot",
    "save_snapshot",
    "SCHEMA_VERSION",
    "Snapshot",
    "SnapshotError",
    "SnapshotNameMismatchError",
    "UnknownUuidInSnapshotError",
    "UnsupportedSchemaVersionError",
)
