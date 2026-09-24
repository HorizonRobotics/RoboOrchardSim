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

"""Direct-path articulation metadata-location contract."""

import numpy as np
import pytest

from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
    AssetResolutionError,
    AssetResolver,
)


class _CapturingRegistry:
    def build_spec(self, meta, **kwargs):
        del kwargs
        return meta


def _resolver() -> AssetResolver:
    return AssetResolver(
        registry=_CapturingRegistry(),  # type: ignore[arg-type]
        rng=np.random.default_rng(0),
    )


def test_direct_articulation_path_derives_adjacent_metadata_json() -> None:
    resolved = _resolver().resolve(
        {
            "primary": {
                "usd_path": "/assets/laptop/object.usdz",
                "spec_type": "usd.articulation",
                "prim_name": "laptop",
            }
        }
    )

    assert resolved["primary"].metadata_path == "/assets/laptop/metadata.json"


def test_direct_articulation_path_metadata_override_is_rejected() -> None:
    with pytest.raises(AssetResolutionError, match="metadata_path"):
        _resolver().resolve(
            {
                "primary": {
                    "usd_path": "/assets/laptop/object.usdz",
                    "metadata_path": "/tmp/metadata.json",
                    "spec_type": "usd.articulation",
                    "prim_name": "laptop",
                }
            }
        )
