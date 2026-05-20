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

"""Shared fixtures for asset_manager.snapshot tests.

Asset tree builder is duplicated from sibling splits/conftest.py because
tests/python/ has no __init__.py and cross-directory pytest fixture
imports via pytest_plugins are unavailable in this layout.
"""

from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

import pytest

from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry


def _urdf(
    name: str,
    *,
    uuid: str,
    domain: str = "food",
    super_category: str = "fruits",
    category: str = "apple",
    color: str = "red",
    shape: str = "sphere",
    material: str = "organic",
    real_height: float = 0.08,
    mass: float = 0.15,
) -> str:
    return dedent(f"""\
        <?xml version='1.0' encoding='utf-8'?>
        <robot name="{name}">
          <link name="{name}">
            <inertial>
              <mass value="{mass}"/>
              <origin xyz="0 0 0"/>
              <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0"
                       iyz="0.0" izz="1.0"/>
            </inertial>
            <extra_info>
              <uuid>{uuid}</uuid>
              <domain>{domain}</domain>
              <super_category>{super_category}</super_category>
              <category>{category}</category>
              <name>{color} {category}</name>
              <color>{color}</color>
              <shape>{shape}</shape>
              <material>{material}</material>
              <description>test {category}</description>
              <min_height>0.05</min_height>
              <max_height>0.10</max_height>
              <real_height>{real_height}</real_height>
              <min_mass>0.10</min_mass>
              <max_mass>0.20</max_mass>
              <version>v0.1.0</version>
              <generate_time>20260518000000</generate_time>
              <tags></tags>
            </extra_info>
          </link>
        </robot>""")


def _write_asset(root: Path, rel_path: str, urdf_text: str) -> None:
    asset_dir = root / rel_path
    asset_dir.mkdir(parents=True, exist_ok=True)
    name = Path(rel_path).name
    (asset_dir / f"{name}.urdf").write_text(urdf_text)
    (asset_dir / f"{name}.usd").write_text("fake-usd-placeholder")
    (asset_dir / "interaction.json").write_text(
        json.dumps({"interaction": {}})
    )


@pytest.fixture
def mini_asset_root(tmp_path: Path) -> Path:
    """Build a small on-disk asset tree with 4 assets."""
    root = tmp_path / "mini_assets"
    _write_asset(
        root,
        "food/fruits/apple_001",
        _urdf("apple_001", uuid="u-apple-001"),
    )
    _write_asset(
        root,
        "food/fruits/apple_002",
        _urdf("apple_002", uuid="u-apple-002", color="green"),
    )
    _write_asset(
        root,
        "food/fruits/orange_001",
        _urdf(
            "orange_001",
            uuid="u-orange-001",
            category="orange",
            color="orange",
        ),
    )
    _write_asset(
        root,
        "misc/boxes/box_001",
        _urdf(
            "box_001",
            uuid="u-box-001",
            domain="misc",
            super_category="boxes",
            category="box",
            color="brown",
            real_height=0.10,
        ),
    )
    return root


@pytest.fixture
def mini_registry(mini_asset_root: Path) -> AssetRegistry:
    """Build an AssetRegistry from the mini_asset_root fixture."""
    return AssetRegistry(str(mini_asset_root))
