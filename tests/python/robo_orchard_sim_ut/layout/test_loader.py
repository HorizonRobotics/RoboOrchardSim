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

"""Schema-only tests for parse_layout (no Isaac stack needed)."""

from __future__ import annotations
import json
from pathlib import Path

import pytest

from robo_orchard_sim.orchard_env.layout.loader import (
    LayoutSequence,
    LayoutValidationError,
    parse_layout,
)

FIXTURE = Path(__file__).parent / "data" / "layout_test.json"
FIXTURE_SEQ = Path(__file__).parent / "data" / "layout_test_sequence.json"
FIXTURE_MISMATCH = (
    Path(__file__).parent / "data" / "layout_test_role_mismatch.json"
)


def test_parse_dict_shape_wraps_into_length_one_sequence():
    """Top-level dict yields a length-1 LayoutSequence."""
    seq = parse_layout(FIXTURE)
    assert isinstance(seq, LayoutSequence)
    assert len(seq.entries) == 1
    src = seq.entries[0].objects["src"]
    assert src.category == "apple"
    assert src.position == pytest.approx((0.0451, 0.1068, 0.8324))
    assert src.rotation == pytest.approx((-0.666, 0.0, 0.0, 0.746))
    assert "natural_language" in seq.entries[0].raw


def test_parse_list_shape_yields_one_layout_per_entry():
    seq = parse_layout(FIXTURE_SEQ)
    assert isinstance(seq, LayoutSequence)
    assert [e.objects["src"].category for e in seq.entries] == [
        "apple",
        "orange",
    ]
    assert seq.entries[1].objects["src"].position == pytest.approx(
        (0.08, 0.12, 0.83)
    )


def test_parse_layout_rejects_role_mismatch():
    with pytest.raises(LayoutValidationError, match="role keys differ"):
        parse_layout(FIXTURE_MISMATCH)


@pytest.mark.parametrize(
    "mutate,err_match",
    [
        (lambda p: p.pop("position"), "position"),
        (lambda p: p["position"]["src"].pop("position"), "position"),
        (
            lambda p: p["position"]["src"].__setitem__("position", [0.0, 0.0]),
            "position",
        ),
        (
            lambda p: p["position"]["src"].__setitem__(
                "rotation", [1.0, 0.0, 0.0]
            ),
            "rotation",
        ),
    ],
)
def test_parse_layout_with_malformed_field_raises_validation_error(
    mutate, err_match, tmp_path
):
    """Each malformed input produces a targeted LayoutValidationError."""
    payload = json.loads(FIXTURE.read_text())
    mutate(payload)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(LayoutValidationError, match=err_match):
        parse_layout(bad)
