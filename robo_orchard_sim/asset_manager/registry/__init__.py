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

"""Asset registry: queryable metadata layer for the asset library."""

from robo_orchard_sim.asset_manager.registry.build_index import (  # noqa: F401
    DEFAULT_CACHE_ROOT,
    SCHEMA_VERSION,
    BuildReport,
    SkippedAsset,
    build_asset_index,
    default_cache_index_path,
)
from robo_orchard_sim.asset_manager.registry.errors import (  # noqa: F401
    AssetIndexNotFoundError,
    AssetIndexVersionError,
    AssetRegistryError,
    CollisionExhaustedError,
    DuplicateAssetIdError,
    EmptyPoolError,
    InsufficientPoolError,
    UnknownAssetError,
)
from robo_orchard_sim.asset_manager.registry.registry import (  # noqa: F401
    AssetRegistry,
    AssetSampler,
)
from robo_orchard_sim.asset_manager.registry.types import (  # noqa: F401
    AssetFilter,
    AssetMeta,
    DistractorSpec,
)
