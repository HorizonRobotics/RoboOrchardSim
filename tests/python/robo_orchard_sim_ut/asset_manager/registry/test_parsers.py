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

"""Tests for the URDF ``<extra_info>`` parser used by the index builder."""

import pytest

from robo_orchard_sim.asset_manager.registry.urdf_parser import (
    ParsedUrdf,
    parse_urdf_extra_info,
)

# ---------------------------------------------------------------------------
# URDF extra_info parser
# ---------------------------------------------------------------------------

FULL_URDF = """<?xml version='1.0' encoding='utf-8'?>
<robot name="lemon_002">
  <link name="lemon_002">
    <inertial>
      <mass value="0.1500"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0"
      izz="1.0"/>
    </inertial>
    <extra_info>
      <uuid>482032ab81585b19af445c0dfd1f170f</uuid>
      <domain>food</domain>
      <super_category>fruits</super_category>
      <category>lemon</category>
      <name>yellow lemon</name>
      <color>yellow</color>
      <shape>ellipsoid</shape>
      <material>organic</material>
      <description>realistic yellow lemon</description>
      <min_height>0.05</min_height>
      <max_height>0.08</max_height>
      <real_height>0.0846</real_height>
      <min_mass>0.1</min_mass>
      <max_mass>0.2</max_mass>
      <version>v0.1.0</version>
      <generate_time>20260403174917</generate_time>
      <tags>graspable, container</tags>
    </extra_info>
  </link>
</robot>
"""


def test_parse_full_urdf():
    p = parse_urdf_extra_info(FULL_URDF)
    assert isinstance(p, ParsedUrdf)
    assert p.uuid == "482032ab81585b19af445c0dfd1f170f"
    assert p.domain == "food"
    assert p.super_category == "fruits"
    assert p.category == "lemon"
    assert p.name == "yellow lemon"
    assert p.color == "yellow"
    assert p.shape == "ellipsoid"
    assert p.material == "organic"
    assert p.description == "realistic yellow lemon"
    assert p.real_height == pytest.approx(0.0846)
    assert p.min_height == pytest.approx(0.05)
    assert p.max_height == pytest.approx(0.08)
    assert p.min_mass == pytest.approx(0.1)
    assert p.max_mass == pytest.approx(0.2)
    assert p.real_mass == pytest.approx(0.15)
    assert p.version == "v0.1.0"
    assert p.generate_time == "20260403174917"
    assert p.tags == frozenset({"graspable", "container"})
    assert p.warnings == []


def test_parse_missing_optional_attribute_fields_logs_warning():
    urdf = FULL_URDF.replace("<color>yellow</color>", "")
    p = parse_urdf_extra_info(urdf)
    assert p.color is None
    assert any("color" in w for w in p.warnings)


def test_parse_missing_required_field_raises_in_strict_mode():
    urdf = FULL_URDF.replace(
        "<uuid>482032ab81585b19af445c0dfd1f170f</uuid>", ""
    )
    with pytest.raises(ValueError, match="uuid"):
        parse_urdf_extra_info(urdf, strict=True)


def test_parse_missing_required_field_returns_none_in_loose_mode():
    urdf = FULL_URDF.replace(
        "<uuid>482032ab81585b19af445c0dfd1f170f</uuid>", ""
    )
    result = parse_urdf_extra_info(urdf, strict=False)
    assert result is None


def test_parse_missing_inertial_mass_fills_zero_with_warning():
    urdf = FULL_URDF.replace('<mass value="0.1500"/>', "")
    p = parse_urdf_extra_info(urdf)
    assert p.real_mass == 0.0
    assert any("mass" in w for w in p.warnings)


def test_parse_no_extra_info_loose_returns_none():
    urdf = FULL_URDF.replace("<extra_info>", "<extra_info_x>").replace(
        "</extra_info>", "</extra_info_x>"
    )
    assert parse_urdf_extra_info(urdf, strict=False) is None


def test_parse_no_extra_info_strict_raises():
    urdf = FULL_URDF.replace("<extra_info>", "<extra_info_x>").replace(
        "</extra_info>", "</extra_info_x>"
    )
    with pytest.raises(ValueError, match="extra_info"):
        parse_urdf_extra_info(urdf, strict=True)


# ---------------------------------------------------------------------------
# <tags> extra_info parsing
# ---------------------------------------------------------------------------


def test_parse_tags_csv_with_spaces():
    urdf = FULL_URDF.replace(
        "<tags>graspable, container</tags>",
        "<tags>  graspable,container ,  stackable  </tags>",
    )
    p = parse_urdf_extra_info(urdf)
    assert p.tags == frozenset({"graspable", "container", "stackable"})


def test_parse_tags_single():
    urdf = FULL_URDF.replace(
        "<tags>graspable, container</tags>", "<tags>graspable</tags>"
    )
    p = parse_urdf_extra_info(urdf)
    assert p.tags == frozenset({"graspable"})


def test_parse_tags_empty_element():
    urdf = FULL_URDF.replace(
        "<tags>graspable, container</tags>", "<tags></tags>"
    )
    p = parse_urdf_extra_info(urdf)
    assert p.tags == frozenset()


def test_parse_tags_missing_warns():
    urdf = FULL_URDF.replace("<tags>graspable, container</tags>", "")
    p = parse_urdf_extra_info(urdf)
    assert p.tags == frozenset()
    assert any("tags" in w for w in p.warnings)
