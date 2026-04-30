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
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

from google.protobuf.timestamp import Timestamp
from mcap_protobuf.writer import Writer

MessageType = TypeVar("MessageType")


@dataclass
class Message(Generic[MessageType]):
    """Message class for storing message with log time and pub time."""

    data: MessageType

    log_time: Timestamp
    pub_time: Timestamp

    def write_message(self, mcap_writer: Writer, topic: str) -> None:
        """Write message to the mcap writer."""
        mcap_writer.write_message(
            topic=topic,
            message=self.data,
            log_time=self.log_time.ToNanoseconds(),
            publish_time=self.pub_time.ToNanoseconds(),
        )

    @staticmethod
    def sort_key(item: Message[MessageType]):
        """Sort key for sorting messages."""
        return item.log_time.ToDatetime()


class TopicMessages(list[Message[MessageType]]):
    def __init__(self, *args, topic: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._topic = topic

    @property
    def topic(self):
        """Return the topic of the messages."""
        return self._topic

    @topic.setter
    def topic(self, topic: str):
        """Set the topic of the messages."""
        self._topic = topic

    def extend(self, iterable: Iterable[Message] | TopicMessages) -> None:
        if isinstance(iterable, TopicMessages):
            if iterable.topic != self.topic:
                raise ValueError(
                    f"Cannot extend messages with different topic: {iterable.topic}"  # noqa: E501
                )

        return super().extend(iterable)

    def sort(self, reverse=False) -> None:
        """Sort the messages in place."""
        return super().sort(key=Message.sort_key, reverse=reverse)

    def write_messages(self, mcap_writer: Writer) -> None:
        """Write messages to the mcap writer."""
        self.sort()
        for msg in self:
            msg.write_message(mcap_writer, self.topic)
