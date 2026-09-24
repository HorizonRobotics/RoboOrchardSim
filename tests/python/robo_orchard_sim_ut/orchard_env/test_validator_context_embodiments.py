## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

"""Integration: ValidatorContext.from_embodiment over real embodiments."""

import pytest

from robo_orchard_sim.orchard_env.embodiments.dualarm_piper import (
    DualArmPiperEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.dualarm_piperx import (
    DualArmPiperXEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.franka_panda import (
    FRANKA_PANDA_ROBOT_INFO_CFGS,
    FrankaPandaEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.panda_droid import (
    PANDA_DROID_ROBOT_INFO_CFGS,
    PandaDroidEmbodiment,
)
from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
)


def _specs_by_name(embodiment):
    context = ValidatorContext.from_embodiment(embodiment, RoleRegistry())
    assert context.robot is not None
    return {spec.name: spec for spec in context.robot.gripper_joints}


def test_from_embodiment_dualarm_piperx_returns_four_gripper_ranges():
    specs = _specs_by_name(DualArmPiperXEmbodiment(enable_cameras=False))

    assert set(specs) == {
        "left_joint7",
        "left_joint8",
        "right_joint7",
        "right_joint8",
    }
    # joint7 opens positive, joint8 opens negative (mirrored finger).
    assert (specs["left_joint7"].open_val, specs["left_joint7"].close_val) == (
        0.05,
        0.0,
    )
    assert (specs["left_joint8"].open_val, specs["left_joint8"].close_val) == (
        -0.05,
        0.0,
    )
    assert (
        specs["right_joint7"].open_val,
        specs["right_joint7"].close_val,
    ) == (0.05, 0.0)
    assert (
        specs["right_joint8"].open_val,
        specs["right_joint8"].close_val,
    ) == (-0.05, 0.0)


def test_from_embodiment_franka_panda_returns_two_gripper_ranges():
    specs = _specs_by_name(FrankaPandaEmbodiment(enable_cameras=False))

    assert set(specs) == {"panda_finger_joint1", "panda_finger_joint2"}
    assert (
        specs["panda_finger_joint1"].open_val,
        specs["panda_finger_joint1"].close_val,
    ) == (0.04, 0.0)
    assert (
        specs["panda_finger_joint2"].open_val,
        specs["panda_finger_joint2"].close_val,
    ) == (0.04, 0.0)


@pytest.mark.parametrize(
    ("embodiment_type", "expected_groups"),
    [
        (
            DualArmPiperEmbodiment,
            (
                ("left_link7", "left_link8"),
                ("right_link7", "right_link8"),
            ),
        ),
        (
            DualArmPiperXEmbodiment,
            (
                ("left_link7", "left_link8"),
                ("right_link7", "right_link8"),
            ),
        ),
        (
            FrankaPandaEmbodiment,
            (("panda_leftfinger", "panda_rightfinger"),),
        ),
        (
            PandaDroidEmbodiment,
            (("left_inner_finger", "right_inner_finger"),),
        ),
    ],
)
def test_from_embodiment_gripper_bodies_configured_returns_groups(
    embodiment_type,
    expected_groups,
) -> None:
    context = ValidatorContext.from_embodiment(
        embodiment_type(enable_cameras=False),
        RoleRegistry(),
    )

    assert context.robot is not None
    assert context.robot.gripper_body_groups == expected_groups


@pytest.mark.parametrize(
    ("embodiment_type", "expected_count"),
    [
        (DualArmPiperEmbodiment, 4),
        (DualArmPiperXEmbodiment, 4),
        (FrankaPandaEmbodiment, 2),
        (PandaDroidEmbodiment, 2),
    ],
)
def test_embodiment_contact_assets_configured_returns_expected_sensor_count(
    embodiment_type,
    expected_count,
) -> None:
    embodiment = embodiment_type(enable_cameras=False)

    sensors = embodiment.get_assets_cfg()["sensors"]

    assert len(sensors) == expected_count


@pytest.mark.parametrize(
    "robot_info_cfgs",
    [
        FRANKA_PANDA_ROBOT_INFO_CFGS,
        PANDA_DROID_ROBOT_INFO_CFGS,
    ],
    ids=["franka_panda", "panda_droid"],
)
def test_embodiment_collision_spheres_configured_uses_curobo_relative_path(
    robot_info_cfgs,
) -> None:
    collision_spheres = robot_info_cfgs[
        "main_arm"
    ].planner.robot.kinematics.collision_spheres

    assert collision_spheres == "spheres/franka_mesh.yml"
