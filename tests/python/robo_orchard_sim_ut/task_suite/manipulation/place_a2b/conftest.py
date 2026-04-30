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

"""Fixtures for PlaceA2B resolver-integration smoke tests.

Duplicates mini_asset_root from task_suite/resolvers/conftest.py
because pytest collects conftest.py per directory and we cannot
share fixtures across sibling test trees.
"""

from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from robo_orchard_sim.asset_manager.registry.registry import AssetRegistry
from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
    AssetResolver,
)


def _urdf(
    name: str,
    *,
    uuid: str,
    domain: str,
    super_category: str,
    category: str,
    color: str = "red",
    shape: str = "sphere",
    material: str = "organic",
    real_height: float = 0.08,
    mass: float = 0.15,
    tags: tuple[str, ...] = (),
) -> str:
    tags_csv = ", ".join(tags)
    return dedent(
        f"""\
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
              <generate_time>20260414000000</generate_time>
              <tags>{tags_csv}</tags>
            </extra_info>
          </link>
        </robot>"""
    )


def _interaction() -> dict:
    """Minimal interaction.json payload for the existence-gate check."""
    return {"interaction": {}}


def _write_asset(
    root: Path,
    rel_path: str,
    urdf_text: str,
    interaction_data: dict | None,
) -> None:
    asset_dir = root / rel_path
    asset_dir.mkdir(parents=True, exist_ok=True)
    name = Path(rel_path).name
    (asset_dir / f"{name}.urdf").write_text(urdf_text)
    (asset_dir / f"{name}.usd").write_text("fake-usd-placeholder")
    if interaction_data is not None:
        (asset_dir / "interaction.json").write_text(
            json.dumps(interaction_data)
        )


@pytest.fixture
def mini_asset_root(tmp_path: Path) -> Path:
    root = tmp_path / "mini_assets"
    _write_asset(
        root,
        "food/fruits/apple_001",
        _urdf(
            "apple_001",
            uuid="u-apple-001",
            domain="food",
            super_category="fruits",
            category="apple",
            color="red",
            tags=("graspable",),
        ),
        _interaction(),
    )
    _write_asset(
        root,
        "food/fruits/orange_001",
        _urdf(
            "orange_001",
            uuid="u-orange-001",
            domain="food",
            super_category="fruits",
            category="orange",
            color="orange",
            tags=("graspable",),
        ),
        _interaction(),
    )
    _write_asset(
        root,
        "food/fruits/pear_001",
        _urdf(
            "pear_001",
            uuid="u-pear-001",
            domain="food",
            super_category="fruits",
            category="pear",
            color="green",
        ),
        _interaction(),
    )
    _write_asset(
        root,
        "containers/plate_001",
        _urdf(
            "plate_001",
            uuid="u-plate-001",
            domain="containers",
            super_category="dishware",
            category="plate",
            color="white",
            real_height=0.015,
            mass=0.3,
            tags=("container",),
        ),
        _interaction(),
    )
    return root


@pytest.fixture
def mini_registry(mini_asset_root: Path) -> AssetRegistry:
    return AssetRegistry(str(mini_asset_root))


class FakeSpec:
    """Lightweight stand-in for RigidObjectSpec (no Isaac imports)."""

    def __init__(
        self,
        *,
        name: str,
        mass: float | None = None,
        initial_pos: Any = None,
        initial_rot: Any = None,
        usd_path: str = "",
    ) -> None:
        self.name = name
        self.mass = mass
        self.initial_pos = initial_pos
        self.initial_rot = initial_rot
        self.usd_path = usd_path


def _fake_build_spec(
    meta,
    *,
    name=None,
    role=None,
    mass=None,
    initial_pos=None,
    initial_rot=None,
    **_kw,
):
    return FakeSpec(
        name=name if name is not None else meta.asset_id,
        mass=mass if mass is not None else meta.real_mass,
        initial_pos=initial_pos,
        initial_rot=initial_rot,
        usd_path=meta.usd_path,
    )


@pytest.fixture
def mini_resolver(mini_registry):
    with patch.object(
        mini_registry, "build_spec", side_effect=_fake_build_spec
    ):
        yield AssetResolver(
            registry=mini_registry,
            rng=np.random.default_rng(42),
        )
