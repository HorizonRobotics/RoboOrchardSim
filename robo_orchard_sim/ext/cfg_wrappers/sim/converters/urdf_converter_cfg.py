# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

from isaaclab.sim.converters.urdf_converter_cfg import (
    UrdfConverterCfg as _UrdfConverterCfg,
)

from robo_orchard_sim.ext.cfg_wrappers.sim.converters.asset_converter_base_cfg import (  # noqa: E501
    AssetConverterBaseCfg,
)
from robo_orchard_sim.utils.config import isaac_configclass2pydantic


class UrdfConverterCfg(
    AssetConverterBaseCfg, isaac_configclass2pydantic(_UrdfConverterCfg)
):
    """The pydantic version of isaaclab.sim.converters.UrdfConverterCfg.

    Only the attributes that need to manually be converted are included here.
    """

    class JointDriveCfg(_UrdfConverterCfg.JointDriveCfg):
        """Pydantic version of JointDriveCfg."""

        class PDGainsCfg(_UrdfConverterCfg.JointDriveCfg.PDGainsCfg):
            """Pydantic version of PDGainsCfg."""

            stiffness: dict[str, float] | float
            """The stiffness of the joint drive in Nm/rad or N/rad.

            If None, the stiffness is set to the value parsed from the URDF
            file. If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set
            to ``"velocity"``, this value determines the drive strength in
            joint velocity space.
            """

        class NaturalFrequencyGainsCfg(
            _UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg
        ):
            """Pydantic version of NaturalFrequencyGainsCfg."""

            natural_frequency: dict[str, float] | float
            """The natural frequency of the joint drive.

            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set to
            ``"velocity"``, this value determines the drive's natural frequency
            in joint velocity space.
            """

        gains: PDGainsCfg | NaturalFrequencyGainsCfg = PDGainsCfg()
        """The drive gains configuration."""

    fix_base: bool  # type: ignore
    """Create a fix joint to the root/base link."""

    collision_from_visuals: bool = False
    """Create collision geometry from visual geometry."""

    joint_drive: JointDriveCfg | None = JointDriveCfg()
    """The joint drive settings.

    None can be used for URDFs without joints.
    """
