## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

"""Integration: build_validator_context over real embodiments."""

from robo_orchard_sim.orchard_env.embodiments.dualarm_piperx import (
    DualArmPiperXEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.franka_panda import (
    FrankaPandaEmbodiment,
)
from robo_orchard_sim.task_components.validators.context import (
    build_validator_context,
)


def _specs_by_name(embodiment):
    context = build_validator_context(embodiment)
    assert context.robot is not None
    return {spec.name: spec for spec in context.robot.gripper_joints}


def test_build_validator_context_dualarm_piperx_returns_four_gripper_ranges():
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


def test_build_validator_context_franka_panda_returns_two_gripper_ranges():
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
