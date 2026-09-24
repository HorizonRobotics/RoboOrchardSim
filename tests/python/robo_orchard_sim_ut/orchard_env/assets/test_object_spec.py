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

from typing import Any

import pytest
from pydantic import ValidationError

from robo_orchard_sim.asset_manager.metadata import (
    ArticulationOperationMeta,
    JointOperationMeta,
)
from robo_orchard_sim.contracts.articulated_operation import (
    SUPPORTED_ARTICULATED_OPERATIONS,
)
from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import (
    ArticulationCfg as WrapperArticulationCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.ext.models.assets.articulation import ArticulationCfg
from robo_orchard_sim.orchard_env.assets import (
    ArticulatedObjectSpec,
    ArticulationSpec,
    RigidObjectSpec,
)
from robo_orchard_sim.utils.env_utils import bbox_of


def _articulation_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/cabinet",
        spawn=UsdFileCfg(usd_path="/tmp/cabinet.usd"),
        actuators={},
    )


def _joint(
    joint_name: str = "DrawerJoint",
    semantic_name: str = "drawer",
) -> JointOperationMeta:
    operation = ArticulationOperationMeta(
        interaction_link=f"{joint_name}_link",
        initial_joint_position=0.0,
        target_joint_fraction=1.0,
    )
    return JointOperationMeta(
        joint_name=joint_name,
        semantic_name=semantic_name,
        outcome_link=f"{joint_name}_link",
        operations={"pull": operation, "push": operation},
    )


def _articulated_object_values() -> dict[str, Any]:
    return {
        "name": "cabinet",
        "template_cfg": _articulation_cfg(),
        "uuid": "modern-two-drawer-cabinet",
        "description": "modern two-drawer cabinet",
        "tags": frozenset({"is_openable"}),
        "joint_operations": (_joint(),),
    }


def test_rigid_object_spec_to_isaac_cfg_uses_object_spec_shared_fields() -> (
    None
):
    spec = RigidObjectSpec(
        name="pick_object",
        usd_path="/tmp/pick.usd",
        interaction_path="/tmp/pick_interaction.json",
        caption_path="/tmp/pick_caption.json",
        scale=(2.0, 3.0, 4.0),
        mass=0.7,
        uuid="pick-uuid",
        category="apple",
        actor_type="pick_target",
        attributes={
            "color": ("red",),
            "shape": ("round",),
            "material": ("organic",),
        },
    )

    cfg = spec.to_isaac_cfg()

    assert cfg.spawn.usd_path == "/tmp/pick.usd"
    assert cfg.spawn.scale == (2.0, 3.0, 4.0)
    assert cfg.spawn.mass_props.mass == 0.7
    assert cfg.interaction_path == "/tmp/pick_interaction.json"
    assert cfg.caption_path == "/tmp/pick_caption.json"
    assert cfg.uuid == "pick-uuid"
    assert cfg.category == "apple"
    assert cfg.actor_type == "pick_target"
    assert cfg.attributes == {
        "color": ("red",),
        "shape": ("round",),
        "material": ("organic",),
    }
    assert spec.caption_path == "/tmp/pick_caption.json"


def test_rigid_object_spec_visual_color_creates_preview_material() -> None:
    spec = RigidObjectSpec(
        name="cube",
        usd_path="/tmp/cube.usd",
        visual_color=(0.9, 0.1, 0.1),
    )

    cfg = spec.to_isaac_cfg()

    assert cfg.spawn.visual_material is not None
    assert cfg.spawn.visual_material.diffuse_color == pytest.approx(
        (0.9, 0.1, 0.1)
    )


@pytest.mark.parametrize(
    "visual_color",
    [(-0.1, 0.5, 0.5), (0.5, 1.1, 0.5), (float("nan"), 0.5, 0.5)],
)
def test_rigid_object_spec_invalid_visual_color_raises_validation_error(
    visual_color,
) -> None:
    with pytest.raises(ValidationError, match="visual_color"):
        RigidObjectSpec(
            name="cube",
            usd_path="/tmp/cube.usd",
            visual_color=visual_color,
        )


def test_rigid_object_spec_scales_bounding_box_with_the_prim() -> None:
    spec = RigidObjectSpec(
        name="pick_object",
        usd_path="/tmp/pick.usd",
        scale=(2.0, 3.0, 4.0),
        aabb_min=(-0.1, -0.2, 0.0),
        aabb_max=(0.1, 0.2, 0.5),
    )

    cfg = spec.to_isaac_cfg()

    # The box has to follow the same scale the spawner applies, or it
    # would describe an object that is not the one in the scene.
    assert cfg.aabb_min == pytest.approx((-0.2, -0.6, 0.0))
    assert cfg.aabb_max == pytest.approx((0.2, 0.6, 2.0))


def test_rigid_object_spec_keeps_bounding_box_without_scale() -> None:
    spec = RigidObjectSpec(
        name="pick_object",
        usd_path="/tmp/pick.usd",
        aabb_min=(-0.1, -0.2, 0.0),
        aabb_max=(0.1, 0.2, 0.5),
    )

    cfg = spec.to_isaac_cfg()

    assert cfg.aabb_min == (-0.1, -0.2, 0.0)
    assert cfg.aabb_max == (0.1, 0.2, 0.5)


def test_rigid_object_spec_defaults_bounding_box_to_none() -> None:
    spec = RigidObjectSpec(
        name="pick_object", usd_path="/tmp/pick.usd", scale=(2.0, 2.0, 2.0)
    )

    cfg = spec.to_isaac_cfg()

    assert cfg.aabb_min is None
    assert cfg.aabb_max is None


@pytest.mark.parametrize(
    "scale", [(-1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, -2.0)]
)
def test_object_spec_rejects_non_positive_scale(scale) -> None:
    # A mirrored asset would swap the box corners, so the whole case is
    # refused rather than handled in every consumer of the box.
    with pytest.raises(ValidationError):
        RigidObjectSpec(
            name="pick_object", usd_path="/tmp/pick.usd", scale=scale
        )


def test_bbox_of_restates_corners_as_center_and_size() -> None:
    spec = RigidObjectSpec(
        name="pick_object",
        usd_path="/tmp/pick.usd",
        aabb_min=(-0.0345, -0.0347, 0.0),
        aabb_max=(0.0345, 0.0347, 0.0675),
    )

    bbox = bbox_of(spec.to_isaac_cfg())

    assert bbox is not None
    assert bbox["center"] == pytest.approx([0.0, 0.0, 0.03375])
    assert bbox["size"] == pytest.approx([0.069, 0.0694, 0.0675])


def test_bbox_of_returns_none_without_a_box() -> None:
    spec = RigidObjectSpec(name="pick_object", usd_path="/tmp/pick.usd")

    assert bbox_of(spec.to_isaac_cfg()) is None


def test_articulated_object_spec_to_isaac_cfg_carries_asset_metadata() -> None:
    values = _articulated_object_values()
    values["caption_path"] = "/tmp/cabinet_caption.json"
    values["category"] = "cabinet"
    values["actor_type"] = "primary"
    values["aabb_min"] = (-0.2, -0.15, -0.015)
    values["aabb_max"] = (0.2, 0.15, 0.9)
    spec = ArticulatedObjectSpec(**values)

    cfg = spec.to_isaac_cfg()

    assert cfg.caption_path == "/tmp/cabinet_caption.json"
    assert cfg.uuid == "modern-two-drawer-cabinet"
    assert cfg.category == "cabinet"
    assert cfg.actor_type == "primary"
    assert cfg.aabb_min == (-0.2, -0.15, -0.015)
    assert cfg.aabb_max == (0.2, 0.15, 0.9)
    assert cfg.joint_operations == spec.joint_operations
    assert cfg.joint_operations[0].semantic_name == "drawer"


def test_articulated_object_spec_to_isaac_cfg_defaults_optional_metadata() -> (
    None
):
    spec = ArticulatedObjectSpec(**_articulated_object_values())

    cfg = spec.to_isaac_cfg()

    assert cfg.category is None
    assert cfg.actor_type == "object"
    assert cfg.aabb_min is None
    assert cfg.aabb_max is None


def test_articulation_spec_to_isaac_cfg_omits_asset_metadata() -> None:
    """Embodiments pass the plain wrapper cfg, with no asset metadata."""
    spec = ArticulationSpec(
        name="robot",
        template_cfg=WrapperArticulationCfg(
            prim_path="{ENV_REGEX_NS}/robot",
            spawn=UsdFileCfg(usd_path="/tmp/robot.usd"),
            actuators={},
        ),
    )

    cfg = spec.to_isaac_cfg()

    assert cfg.prim_path == "{ENV_REGEX_NS}/robot"
    assert not hasattr(cfg, "joint_operations")


def test_articulated_object_spec_converts_wrapper_template_cfg() -> None:
    """A plain wrapper template still yields a metadata-carrying cfg."""
    values = _articulated_object_values()
    values["template_cfg"] = WrapperArticulationCfg(
        prim_path="{ENV_REGEX_NS}/cabinet",
        spawn=UsdFileCfg(usd_path="/tmp/cabinet.usd"),
        actuators={},
    )
    spec = ArticulatedObjectSpec(**values)

    cfg = spec.to_isaac_cfg()

    assert isinstance(cfg, ArticulationCfg)
    assert cfg.uuid == "modern-two-drawer-cabinet"
    assert cfg.joint_operations[0].semantic_name == "drawer"
    # Nested cfgs must survive the conversion as objects, not dicts.
    assert cfg.spawn.usd_path == "/tmp/cabinet.usd"
    assert cfg.init_state is not None


def test_articulated_object_spec_to_isaac_cfg_keeps_physical_patching() -> (
    None
):
    """Metadata patching must not regress the inherited pose/joint patch."""
    values = _articulated_object_values()
    values["initial_pos"] = (1.0, 2.0, 3.0)
    values["joint_pos"] = {"UpperDrawerJoint": 0.25}
    spec = ArticulatedObjectSpec(**values)

    cfg = spec.to_isaac_cfg()

    assert cfg.prim_path == "{ENV_REGEX_NS}/cabinet"
    assert tuple(cfg.init_state.pos) == (1.0, 2.0, 3.0)
    assert cfg.init_state.joint_pos == {"UpperDrawerJoint": 0.25}
    assert cfg.uuid == "modern-two-drawer-cabinet"


def test_articulated_object_spec_to_isaac_cfg_leaves_template_untouched() -> (
    None
):
    """to_isaac_cfg copies; the shared template must not be mutated."""
    template = _articulation_cfg()
    values = _articulated_object_values()
    values["template_cfg"] = template
    spec = ArticulatedObjectSpec(**values)

    spec.to_isaac_cfg()

    assert template.uuid is None
    assert template.joint_operations == ()


def test_joint_operation_meta_shared_supported_operations_are_accepted() -> (
    None
):
    metadata = JointOperationMeta(
        joint_name="joint",
        semantic_name="part",
        outcome_link="part_link",
        operations={
            operation: {
                "interaction_link": "part_link",
                "initial_joint_position": 0.0,
                "target_joint_fraction": 0.5,
            }
            for operation in SUPPORTED_ARTICULATED_OPERATIONS
        },
    )

    assert frozenset(metadata.operations) == SUPPORTED_ARTICULATED_OPERATIONS


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {
                "joint_name": "",
                "semantic_name": "drawer",
                "outcome_link": "drawer_link",
                "operations": {
                    "open": {
                        "interaction_link": "drawer_link",
                        "initial_joint_position": 0.0,
                        "target_joint_fraction": 1.0,
                    }
                },
            },
            "joint_name",
        ),
        (
            {
                "joint_name": "drawer_joint",
                "semantic_name": " ",
                "outcome_link": "drawer_link",
                "operations": {
                    "open": {
                        "interaction_link": "drawer_link",
                        "initial_joint_position": 0.0,
                        "target_joint_fraction": 1.0,
                    }
                },
            },
            "semantic_name",
        ),
        (
            {
                "joint_name": "drawer_joint",
                "semantic_name": "drawer",
                "outcome_link": "drawer_link",
                "operations": {},
            },
            "operations must be non-empty",
        ),
        (
            {
                "joint_name": "drawer_joint",
                "semantic_name": "drawer",
                "outcome_link": "drawer_link",
                "operations": {
                    "lock": {
                        "interaction_link": "drawer_link",
                        "initial_joint_position": 0.0,
                        "target_joint_fraction": 1.0,
                    }
                },
            },
            "Input should be",
        ),
    ],
)
def test_joint_operation_meta_invalid_metadata_raises_value_error(
    values: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        JointOperationMeta(**values)


@pytest.mark.parametrize("field", ["uuid", "description"])
def test_articulated_object_spec_blank_identity_raises_value_error(
    field: str,
) -> None:
    values = _articulated_object_values()
    values[field] = " "

    with pytest.raises(ValueError, match="must be non-empty"):
        ArticulatedObjectSpec(**values)


@pytest.mark.parametrize(
    ("joint_operations", "message"),
    [
        (
            (
                _joint("DrawerJoint", "upper drawer"),
                _joint("DrawerJoint", "lower drawer"),
            ),
            "duplicate mechanical joint name",
        ),
        (
            (
                _joint("UpperDrawerJoint", "drawer"),
                _joint("LowerDrawerJoint", "drawer"),
            ),
            "duplicate semantic joint description",
        ),
    ],
)
def test_articulated_object_spec_ambiguous_joint_metadata_raises_value_error(
    joint_operations: tuple[JointOperationMeta, ...],
    message: str,
) -> None:
    values = _articulated_object_values()
    values["joint_operations"] = joint_operations

    with pytest.raises(ValueError, match=message):
        ArticulatedObjectSpec(**values)
