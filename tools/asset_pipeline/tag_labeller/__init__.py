# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tag Labeller - Re-tags URDFs with semantic capability tags."""

from .tag_labeller import (
    PROMPT_FIELDS,
    ProcessResult,
    TagLabeller,
    TagLabellerParseError,
    parse_response,
    render_prompt,
)
from .tag_vocab import TagSpec, TagVocab, TagVocabError

__version__ = "0.1.0"
__all__ = [
    "PROMPT_FIELDS",
    "ProcessResult",
    "TagLabeller",
    "TagLabellerParseError",
    "TagSpec",
    "TagVocab",
    "TagVocabError",
    "parse_response",
    "render_prompt",
]
