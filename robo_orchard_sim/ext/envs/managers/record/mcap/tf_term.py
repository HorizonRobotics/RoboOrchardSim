# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.timestamp_pb2 import Timestamp

from robo_orchard_sim.ext.envs.env_base import IsaacEnvType_co
from robo_orchard_sim.ext.envs.managers.record import (
    RecordTermBase,
    RecordTermBaseCfg,
)
from robo_orchard_sim.ext.envs.managers.record.mcap.message import Message
from robo_orchard_sim.utils.config import ClassType_co

ReturnType = dict[str, list[Message]]


class McapTFTerm(RecordTermBase[IsaacEnvType_co, "McapTFTermCfg", ReturnType]):
    def __init__(self, cfg: "McapTFTermCfg", env: IsaacEnvType_co):
        super().__init__(cfg, env)
        self._cfg = cfg

        self._validate_frame_configuration()

        if (
            self._cfg.parent_frame is not None
            and self._cfg.child_frame is not None
        ):
            self._parent_frames = self._normalize_frame_list(
                self._cfg.parent_frame
            )
            self._child_frames = self._normalize_frame_list(
                self._cfg.child_frame
            )
            self._validate_and_build_frame_pairs()
            self._use_runtime_frames = False
        else:
            self._parent_frames = []
            self._child_frames = []
            self._frame_pairs = None
            self._use_runtime_frames = True
            print(
                "No parent_frame and child_frame specified. "
                "Will use runtime frame data."
            )

    def __call__(
        self, data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> ReturnType:
        """The implementation of the record term.

        All subclasses should implement this method to return the observation.

        """
        transform_data = self._parse_data_from_dict(data, self._cfg.key)[
            self._cfg.key
        ]

        msgs: ReturnType = {}
        tf_list = transform_data.as_state().tf_list
        if not tf_list:
            return msgs
        env_nums = tf_list[0].xyz.shape[0]
        has_multiple_transforms = len(tf_list) > 1

        for id, tf in enumerate(tf_list):
            env_msgs = []
            parent_frame_id, child_frame_id = self._resolve_frame_ids(
                tf_index=id, runtime_tf=tf
            )
            for env_id in range(env_nums):
                pos = tf.xyz[env_id, :]
                quat = tf.quat[env_id, :]

                trans_vec3 = Vector3(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                )
                rot_q = Quaternion(
                    w=float(quat[0]),
                    x=float(quat[1]),
                    y=float(quat[2]),
                    z=float(quat[3]),
                )

                msg = Message(
                    data=FrameTransform(
                        timestamp=ts,
                        parent_frame_id=parent_frame_id,
                        child_frame_id=child_frame_id,
                        translation=trans_vec3,
                        rotation=rot_q,
                    ),
                    log_time=ts,
                    pub_time=ts,
                )
                env_msgs.append(msg)

            topic_name = self._resolve_topic_name(
                tf_index=id,
                has_multiple_transforms=has_multiple_transforms,
            )
            msgs[topic_name] = env_msgs

        return msgs

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the observation term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        """
        pass

    def _validate_frame_configuration(self):
        parent_configured = self._cfg.parent_frame is not None
        child_configured = self._cfg.child_frame is not None

        if parent_configured != child_configured:
            raise ValueError(
                "parent_frame and child_frame must be both configured or "
                "both None. "
                f"Current: parent_frame="
                f"{'configured' if parent_configured else 'None'}, "
                f"child_frame={'configured' if child_configured else 'None'}"
            )

    def _normalize_frame_list(self, frames) -> list[str]:
        if isinstance(frames, str):
            return [frames]
        elif isinstance(frames, list):
            return frames
        else:
            raise ValueError(
                f"Invalid frame type: {type(frames)}. Expected str or list"
                "[str]."
            )

    def _validate_and_build_frame_pairs(self):
        parent_count = len(self._parent_frames)
        child_count = len(self._child_frames)

        if parent_count == 0 or child_count == 0:
            raise ValueError(
                "parent_frame and child_frame cannot be empty when configured."
            )

        if parent_count != child_count:
            if parent_count == 1:
                self._frame_pairs = [
                    (self._parent_frames[0], child_frame)
                    for child_frame in self._child_frames
                ]
            else:
                raise ValueError(
                    f"Incompatible frame count between parent and child. "
                    f"Parent frames: {parent_count} ({self._parent_frames}), "
                    f"Child frames: {child_count} ({self._child_frames}). "
                    f"Parent frame count must be either 1 or equal to child "
                    f"frame count ({child_count})."
                )
        else:
            self._frame_pairs = list(
                zip(self._parent_frames, self._child_frames, strict=False)
            )

        print(f"TF frame pairs: {self._frame_pairs}")

    def _resolve_frame_ids(self, tf_index: int, runtime_tf) -> tuple[str, str]:
        if self._use_runtime_frames:
            return runtime_tf.parent_frame_id, runtime_tf.child_frame_id

        assert self._frame_pairs is not None
        if tf_index >= len(self._frame_pairs):
            raise ValueError(
                "Transform count does not match configured frame pairs. "
                f"Got tf_index={tf_index} with only "
                f"{len(self._frame_pairs)} configured pair(s)."
            )
        return self._frame_pairs[tf_index]

    def _resolve_topic_name(
        self, tf_index: int, has_multiple_transforms: bool
    ) -> str:
        if "{id}" in self._cfg.topic:
            return self._cfg.topic.format(id=tf_index + 1)

        if not has_multiple_transforms:
            return self._cfg.topic

        if self._cfg.strict_topic_id:
            raise ValueError(
                "Multiple transforms require a topic containing '{id}' "
                "when strict_topic_id is enabled."
            )

        topic_suffix = str(tf_index + 1)
        if self._cfg.topic.endswith("/"):
            return self._cfg.topic + topic_suffix
        return f"{self._cfg.topic}/{topic_suffix}"


class McapTFTermCfg(RecordTermBaseCfg):
    """Configuration class for the MCAP TF record term."""

    class_type: ClassType_co[McapTFTerm] = McapTFTerm

    key: str

    parent_frame: list[str] | str | None = None
    """The parent frame ID(s) for the transform. Can be:
    - None: Use runtime frame data from observations
    - str: Single parent frame for all transforms
    - list[str]: Multiple parent frames paired with child frames

    If None, the system will use frame names from the runtime observation
    data."""

    child_frame: list[str] | str | None = None
    """The child frame ID(s) for the transform. Can be:
    - None: Use runtime frame data from observations
    - str: Single child frame (parent_frame must also be str)
    - list[str]: Multiple child frames paired with parent frames

    Must be configured together with parent_frame
    (both None or both specified)."""

    strict_topic_id: bool = False
    """Whether to require '{id}' in topic for multiple transforms."""


TFTerm = McapTFTerm
