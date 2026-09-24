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

from typing import Any, Sequence

from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.timestamp_pb2 import Timestamp
from robo_orchard_core.datatypes.geometry import BatchFrameTransform

from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.ext.envs.managers.record import (
    RecordTermBase,
    RecordTermBaseCfg,
)
from robo_orchard_sim.ext.envs.managers.record.mcap.message import Message
from robo_orchard_sim.utils.config import ClassType_co

ReturnType = dict[str, list[Message]]


class McapMultiTFTerm(
    RecordTermBase[IsaacEnvType_co, "McapMultiTFTermCfg", ReturnType]
):
    """Record all transforms in an observation group onto one topic.

    Unlike :class:`McapTFTerm` (one topic per parent/child pair), this
    term reads an entire observation *group* -- e.g. one term per scene
    object -- and flattens every ``BatchFrameTransformGraph`` in it into
    a single ``foxglove.FrameTransforms`` message per env. The topic
    name stays fixed no matter how many objects a task has, or what
    they are named; identity lives inside the message
    (``child_frame_id``), not in the topic string.
    """

    def __init__(self, cfg: "McapMultiTFTermCfg", env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg

    def __call__(
        self, data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> ReturnType:
        """The implementation of the record term.

        All subclasses should implement this method to return the observation.

        """
        group = self._parse_data_from_dict(data, self._cfg.key)[self._cfg.key]

        flat: list[BatchFrameTransform] = []
        seen: set[tuple[str, str]] = set()
        for term_name in sorted(group):
            for tf in group[term_name].as_state().tf_list:
                pair = (tf.parent_frame_id, tf.child_frame_id)
                if self._cfg.dedup and pair in seen:
                    # bidirectional=True on the source term emits a
                    # mirrored inverse edge for every pair; drop it.
                    continue
                seen.add(pair)
                flat.append(tf)

        if not flat:
            return {}

        env_nums = flat[0].xyz.shape[0]
        msgs: list[Message] = []
        for env_id in range(env_nums):
            transforms = [
                FrameTransform(
                    timestamp=ts,
                    parent_frame_id=tf.parent_frame_id,
                    child_frame_id=tf.child_frame_id,
                    translation=Vector3(
                        x=float(tf.xyz[env_id, 0]),
                        y=float(tf.xyz[env_id, 1]),
                        z=float(tf.xyz[env_id, 2]),
                    ),
                    rotation=Quaternion(
                        w=float(tf.quat[env_id, 0]),
                        x=float(tf.quat[env_id, 1]),
                        y=float(tf.quat[env_id, 2]),
                        z=float(tf.quat[env_id, 3]),
                    ),
                )
                for tf in flat
            ]
            msgs.append(
                Message(
                    data=FrameTransforms(transforms=transforms),
                    log_time=ts,
                    pub_time=ts,
                )
            )

        return {self._cfg.topic: msgs}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the observation term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        """
        pass


class McapMultiTFTermCfg(RecordTermBaseCfg):
    """Configuration class for the MCAP multi-object TF record term."""

    class_type: ClassType_co[McapMultiTFTerm] = McapMultiTFTerm

    key: str
    """Observation key to read. Pointing at a group name (e.g. ``/object``,
    with no term suffix) pulls the entire group -- one entry per term --
    so this single record term can cover any number of objects.
    Pointing at a single term name (e.g. ``/object/apple_001_tf``) yields
    only that one transform."""

    dedup: bool = True
    """Drop mirrored inverse edges produced when the source
    ``FrameTransformTermCfg`` has ``bidirectional=True``."""
