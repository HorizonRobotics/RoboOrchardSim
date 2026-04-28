# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TagLabeller — re-tags already-labelled URDFs with semantic capability tags.

Reads the existing <extra_info> fields, asks GPT to judge each tag in the
vocabulary, and writes the resulting tag set back into <extra_info><tags>.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Mapping

from tag_labeller.tag_vocab import TagVocab
from tag_labeller.urdf_tags import (
    has_tags_element,
    read_extra_info,
    write_tags,
)

PROMPT_FIELDS: tuple[str, ...] = (
    "name",
    "super_category",
    "category",
    "description",
    "shape",
    "material",
    "real_height",
)


class TagLabellerParseError(ValueError):
    """Raised when the GPT response cannot be parsed into a tag verdict."""


def render_prompt(vocab: TagVocab, fields: Mapping[str, str]) -> str:
    """Render the GPT prompt for a single asset.

    Only fields listed in PROMPT_FIELDS are used. Missing fields render
    as the literal string ``unknown``.
    """
    asset_lines = []
    for key in PROMPT_FIELDS:
        value = fields.get(key) or "unknown"
        asset_lines.append(f"- {key}: {value}")
    asset_block = "\n".join(asset_lines)

    tag_lines = []
    for spec in vocab:
        tag_lines.append(f"- {spec.name}: {spec.criteria}")
    tag_block = "\n".join(tag_lines)

    json_keys = ", ".join(f'"{s.name}": true | false' for s in vocab)

    intro = (
        "You are evaluating a 3D asset to decide which"
        " capability tags apply.\n"
    )
    instruct = (
        "Respond with a single JSON object, no commentary,"
        " no markdown fences:\n"
    )
    return (
        intro
        + "\n"
        + "Asset attributes:\n"
        + f"{asset_block}\n"
        + "\n"
        + "Decide yes/no for each of the following tags."
        + " Use common sense.\n"
        + f"{tag_block}\n"
        + "\n"
        + instruct
        + f'{{ {json_keys}, "_reasoning": "<one short sentence>" }}\n'
    )


_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL
)


def parse_response(raw: str | None, vocab: TagVocab) -> dict[str, bool]:
    """Parse a GPT response into ``{tag_name: bool}``.

    Strips surrounding markdown code fences if present. Validates that
    every key (except ``_reasoning``) is in the vocab, every value is a
    bool, and every vocab tag is covered.

    Raises:
        TagLabellerParseError: If parsing or validation fails.
    """
    if raw is None:
        raise TagLabellerParseError("empty response (None)")
    text = raw.strip()
    fence_match = _FENCE_PATTERN.match(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise TagLabellerParseError(
            f"response is not valid JSON: {e.msg}"
        ) from e

    if not isinstance(data, dict):
        raise TagLabellerParseError(
            f"response must be a JSON object, got {type(data).__name__}"
        )

    verdict: dict[str, bool] = {}
    for key, value in data.items():
        if key == "_reasoning":
            continue
        if not vocab.is_known(key):
            raise TagLabellerParseError(
                f"response contains unknown tag {key!r}"
            )
        if not isinstance(value, bool):
            raise TagLabellerParseError(
                f"response value for {key!r} must be bool, got"
                f" {type(value).__name__}"
            )
        verdict[key] = value

    missing = [s.name for s in vocab if s.name not in verdict]
    if missing:
        raise TagLabellerParseError(
            f"response missing tags: {', '.join(missing)}"
        )
    return verdict


@dataclass
class ProcessResult:
    """Outcome of processing one URDF.

    Attributes:
        urdf_path: Path to the URDF.
        status: One of ``"ok"``, ``"skipped"``, ``"error"``.
        tags_written: List of tag names written (only for status=='ok').
        error_msg: Error description (only for status=='error').
    """

    urdf_path: str
    status: str
    tags_written: list[str] | None = None
    error_msg: str | None = None


class TagLabeller:
    """Re-tag URDFs by querying GPT for each asset.

    Args:
        gpt_client: An object exposing ``query(text_prompt) -> str | None``,
            typically the ``GPTClient`` from ``asset_labeller.gpt_client``.
        vocab: The TagVocab to evaluate.
    """

    def __init__(self, gpt_client, vocab: TagVocab):
        self.gpt_client = gpt_client
        self.vocab = vocab

    def process(
        self,
        urdf_path: str,
        force: bool = False,
        merge: bool = False,
    ) -> ProcessResult:
        """Process one URDF and update its <tags> element.

        Args:
            urdf_path: Path to the URDF.
            force: If False (default), skip URDFs that already have a
                <tags> element. If True, re-evaluate.
            merge: If True, union new tags with existing ones (used by
                ``--only-tags`` mode). If False, overwrite.
        """
        try:
            already_tagged = has_tags_element(urdf_path)
        except (ValueError, FileNotFoundError) as e:
            return ProcessResult(
                urdf_path=urdf_path,
                status="error",
                error_msg=f"cannot read URDF: {e}",
            )

        if already_tagged and not force:
            return ProcessResult(urdf_path=urdf_path, status="skipped")

        try:
            fields = read_extra_info(urdf_path)
        except (ValueError, FileNotFoundError) as e:
            return ProcessResult(
                urdf_path=urdf_path,
                status="error",
                error_msg=f"cannot read extra_info: {e}",
            )

        prompt = render_prompt(self.vocab, fields)
        raw = self.gpt_client.query(prompt)

        try:
            verdict = parse_response(raw, self.vocab)
        except TagLabellerParseError as e:
            return ProcessResult(
                urdf_path=urdf_path,
                status="error",
                error_msg=f"cannot parse GPT response: {e}",
            )

        true_tags = sorted(name for name, value in verdict.items() if value)
        try:
            write_tags(urdf_path, tags=true_tags, merge_with_existing=merge)
        except (OSError, ValueError) as e:
            return ProcessResult(
                urdf_path=urdf_path,
                status="error",
                error_msg=f"cannot write URDF: {e}",
            )

        return ProcessResult(
            urdf_path=urdf_path,
            status="ok",
            tags_written=true_tags,
        )
