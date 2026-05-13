# Project RoboOrchard
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Caption candidates labeller.

Orchestrates per-asset caption generation: GPT vision call, response
parsing, normalization, deduplication, optional one-shot top-up, and
writes the ``caption_candidates.json`` file plus the URDF
``<caption_candidates>`` link.
"""

from __future__ import annotations
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

from caption_labeller.urdf_desc import (
    find_renders,
    has_caption_candidates,
    read_asset_fields,
    write_caption_candidates_link,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CAPTION_JSON_NAME",
    "CaptionLabeller",
    "CaptionParseError",
    "ProcessResult",
    "dedup",
    "iter_urdfs",
    "normalize",
    "parse_response",
    "render_initial_prompt",
    "render_topup_prompt",
]

CAPTION_JSON_NAME = "caption_candidates.json"

_MIN_WORDS = 4
_MAX_WORDS = 6

# Retry configuration for caption generation. The initial GPT call is
# retried because GPT is non-deterministic and a second call often clears
# transient failures (drop-rate-too-high, malformed JSON, no response).
# Top-up is retried independently because it runs only after the initial
# call already produced some valid phrases.
MAX_INITIAL_ATTEMPTS = 3
MAX_TOPUP_ATTEMPTS = 4
_DROP_TOLERANCE = 0.30


class CaptionParseError(RuntimeError):
    """Raised when a GPT response cannot be parsed into valid candidates."""


def _strip_code_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _valid_phrase(phrase) -> bool:
    if not isinstance(phrase, str):
        return False
    s = phrase.strip()
    if not s:
        return False
    if not s.isascii():
        return False
    words = s.split()
    return _MIN_WORDS <= len(words) <= _MAX_WORDS


_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _category_in_phrase(phrase: str, category: str) -> bool:
    """True iff the literal `category` token sequence appears in `phrase`.

    `category` is lowercased and underscores normalised to spaces (matching
    how the prompt presents the category to the model). `phrase` is split
    on any run of non-alphanumeric characters. The category tokens must
    appear as a contiguous unbroken run inside the phrase tokens.
    """
    cat_tokens = category.lower().replace("_", " ").split()
    if not cat_tokens:
        return False
    phrase_tokens = [t for t in _TOKEN_SPLIT_RE.split(phrase.lower()) if t]
    n = len(cat_tokens)
    for i in range(len(phrase_tokens) - n + 1):
        if phrase_tokens[i : i + n] == cat_tokens:
            return True
    return False


def parse_response(raw: str, category: str) -> list[str]:
    """Parse a GPT response into a validated list of caption phrases.

    Args:
        raw: raw text returned by the GPT client (may include a markdown
            code fence; the response body must be a JSON object with a
            ``"candidates"`` list of strings).
        category: asset category string from the URDF, e.g. ``"apple"`` or
            ``"facial_cleanser"``. Underscores are normalised to spaces by
            ``_category_in_phrase`` before matching.

    Two-stage filter:
      1. Structural validity (`_valid_phrase`): non-empty ASCII string of
         4-6 words.
      2. Category presence (`_category_in_phrase`): literal token
         sequence of `category` (after `_`-to-space norm) must appear
         contiguously in the phrase.

    Raises ``CaptionParseError`` if the response is malformed, the
    ``candidates`` key is missing or wrong type, the candidates list is
    empty, or more than ``_DROP_TOLERANCE`` of phrases are dropped across
    the two stages combined.
    """
    text = _strip_code_fence(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise CaptionParseError(f"response is not valid JSON: {e}") from e
    if not isinstance(obj, dict) or "candidates" not in obj:
        raise CaptionParseError("response missing 'candidates' key")
    raw_list = obj["candidates"]
    if not isinstance(raw_list, list):
        raise CaptionParseError("'candidates' is not a list")
    if not raw_list:
        raise CaptionParseError("candidates list is empty")

    structure_kept = [p for p in raw_list if _valid_phrase(p)]
    kept = [p for p in structure_kept if _category_in_phrase(p, category)]

    total = len(raw_list)
    dropped = total - len(kept)
    if dropped / total > _DROP_TOLERANCE:
        struct_dropped = total - len(structure_kept)
        cat_dropped = len(structure_kept) - len(kept)
        raise CaptionParseError(
            f"drop rate too high: {dropped}/{total} dropped "
            f"({struct_dropped} bad structure, {cat_dropped} missing category)"
        )
    return [p.strip() for p in kept]


_WS_RE = re.compile(r"\s+")
_JACCARD_THRESHOLD = 0.85


def normalize(phrase: str) -> str:
    """Lowercase, strip, and collapse internal whitespace to single spaces."""
    return _WS_RE.sub(" ", phrase.strip().lower())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def dedup(phrases: list[str]) -> list[str]:
    """Remove exact and near-duplicate phrases (token Jaccard >= 0.85).

    Keeps the first occurrence. Comparison happens on normalized tokens,
    but phrases are returned in their original form.
    """
    kept: list[str] = []
    kept_tokens: list[set[str]] = []
    seen: set[str] = set()
    for p in phrases:
        norm = normalize(p)
        if norm in seen:
            continue
        tokens = set(norm.split())
        if any(_jaccard(tokens, t) >= _JACCARD_THRESHOLD for t in kept_tokens):
            continue
        kept.append(p)
        kept_tokens.append(tokens)
        seen.add(norm)
    return kept


_INITIAL_PROMPT_TEMPLATE = (
    "You are generating short visual description phrases for a 3D "
    "object asset. These phrases will be used as REFERRING EXPRESSIONS "
    "(noun phrases that point out which object is meant) inside "
    "natural-language instructions for a robot arm.\n\n"
    "The robot supports many atomic manipulation skills, including "
    'but not limited to: "pick up", "place", "put down", "grasp", '
    '"push", "pull", "press", "rotate", "move", "open", "close", '
    '"pour from". The same phrase will be slotted in after any of '
    'these verbs (e.g. "pick up the {{phrase}}", '
    '"push the {{phrase}}", "rotate the {{phrase}}"), so each phrase '
    "must be a clean object reference that does NOT assume a "
    "specific action.\n\n"
    "A vision-language-action (VLA) model must be able to look at a "
    "scene image, match the phrase to THIS object among other "
    "objects on the table, and execute the requested skill on it.\n\n"
    "Asset category: {category}\n"
    "Number of phrases to generate: {num_candidates}\n\n"
    "You are given {num_views} rendered views of this asset from "
    "different angles, captured at normal viewing distance.\n\n"
    "Generate {num_candidates} short noun phrases that a person would "
    "naturally use to refer to THIS specific object on a table. Each "
    "phrase should highlight a distinctive visual feature that helps "
    "tell this object apart from other objects of the same category.\n\n"
    "**TOP PRIORITY — these rules override everything else:**\n"
    "**1. EACH PHRASE MUST BE 4 TO 6 WORDS, INCLUSIVE.** "
    "Count words before emitting. Phrases outside this range will be "
    "rejected.\n"
    "**2. EACH PHRASE MUST BE GENUINELY DIFFERENT FROM THE OTHERS.** "
    "No trivial rewordings, no swapping one synonym, no shuffling word "
    "order. Each phrase must add new visual information (a different "
    "color word, a different distinctive part, a different shape "
    "descriptor) the reader did not already see in earlier phrases.\n"
    "**3. EACH PHRASE MUST BE A PURE NOUN PHRASE, ACTION-AGNOSTIC.** "
    "It must read naturally after ANY of the action verbs above "
    '(e.g. "pick up the ___", "push the ___", "rotate the ___"). '
    "Use the object category ({category}) as the head noun "
    '(e.g. "the red apple with a stem", '
    'not just "the small object with white dots"). Do NOT include '
    "any verb, action, or grasping/manipulation language inside the "
    'phrase itself (no "to grasp", "for picking", "easy to hold", '
    "etc.).\n\n"
    "**4. EVERY PHRASE MUST CONTAIN THE EXACT CATEGORY TOKEN "
    "({category}).** The literal category word(s) must appear, in "
    "order, with no synonyms, paraphrases, or plural variants. "
    'Phrases like "cup" for category "mug", "vessel" for "container", '
    'or "apples" for "apple" will be rejected.\n'
    "Other constraints on EACH phrase:\n"
    "- Focus on features that visually DISTINGUISH this object from "
    "other objects of the same category — its specific color, "
    "size, shape, or distinctive parts.\n"
    "- Use simple, everyday words a child or non-expert would use. "
    "Prefer concrete words for color (red, blue, golden, white, ...), "
    "shape (round, flat, long, curved, ...), size "
    "(small, tall, short, ...), and parts "
    "(stem, handle, lid, button, clip, ...).\n"
    "- AVOID subjective or aesthetic adjectives that don't describe a "
    'visible feature: no "elegant", "sleek", "modern", '
    '"compact", "stylish", "pretty", "abstract" (as a feeling), etc.\n'
    "- AVOID features that need extreme close-up to see "
    '(no "tiny white specks", "small ridges inside", '
    '"thin engraved lines"). Favor features visible at normal '
    "table-viewing distance.\n"
    "- AVOID technical or fancy vocabulary "
    '(no "silhouette", "tapered", "asymmetrical", "ellipsoidal", '
    '"tonal", "imperfections", etc.).\n'
    "- AVOID viewpoint or rendering words "
    '(no "top view", "side view", "viewed from above", '
    '"close-up", "front-facing").\n'
    "- Noun phrase, NOT a full sentence "
    "(no subject/verb, no trailing period).\n"
    "- Lowercase, ASCII only.\n"
    "- Describe what is visually present; do not invent unseen parts.\n\n"
    "Output style example for a different object (a golden cup with "
    "floral design). Each phrase below is action-agnostic — it works "
    'after "pick up the ___", "push the ___", "rotate the ___", '
    "or any other manipulation verb:\n"
    '{{"candidates": [\n'
    '  "shiny golden cup with flowers",\n'
    '  "golden cup with red pattern",\n'
    '  "tall golden cup with handle",\n'
    '  "small golden cup with rim",\n'
    '  "wide golden cup with curves"\n'
    "]}}\n\n"
    "Now produce {num_candidates} phrases for THIS asset.\n"
    "**Before you respond, mentally test each phrase by inserting it "
    'after several different verbs ("pick up the ___", '
    '"push the ___", "rotate the ___") and confirm it still sounds '
    "natural. Re-check rules 1, 2, 3 against every phrase.**\n"
    "Respond with a single JSON object, no commentary, "
    "no markdown fence:\n"
    '{{"candidates": ["phrase 1", "phrase 2", '
    '..., "phrase {num_candidates}"]}}'
)


def render_initial_prompt(
    category: str, num_candidates: int, num_views: int
) -> str:
    """Render the first-round caption generation prompt."""
    return _INITIAL_PROMPT_TEMPLATE.format(
        category=category.replace("_", " "),
        num_candidates=num_candidates,
        num_views=num_views,
    )


def render_topup_prompt(
    category: str,
    num_needed: int,
    num_views: int,
    existing: list[str],
) -> str:
    """Render the one-shot top-up prompt with phrases to avoid."""
    existing_block = "\n".join(f"- {p}" for p in existing)
    return (
        render_initial_prompt(
            category=category,
            num_candidates=num_needed,
            num_views=num_views,
        )
        + "\n\nAvoid generating phrases similar to any of the following:\n"
        + existing_block
    )


@dataclass
class ProcessResult:
    status: str
    reason: Optional[str] = None
    count: int = 0


class CaptionLabeller:
    """Generate ``caption_candidates.json`` + URDF link for labelled URDFs.

    Args:
        gpt_client: an object exposing ``query(text_prompt, images=...)``
            and returning a string (matches
            ``asset_labeller.gpt_client.GPTClient``).
        seen_count: number of phrases to write into the ``seen`` list.
        unseen_count: number of phrases to write into the ``unseen`` list.
        force: regenerate even if ``<caption_candidates>`` already present.
    """

    def __init__(
        self,
        gpt_client,
        seen_count: int = 15,
        unseen_count: int = 5,
        force: bool = False,
    ) -> None:
        if seen_count < 0 or unseen_count < 0:
            raise ValueError(
                "seen_count and unseen_count must be non-negative"
            )
        if seen_count + unseen_count == 0:
            raise ValueError("seen_count + unseen_count must be > 0")
        self.gpt_client = gpt_client
        self.seen_count = seen_count
        self.unseen_count = unseen_count
        self.num_candidates = seen_count + unseen_count
        self.force = force

    def process(self, urdf_path: str) -> ProcessResult:
        """Process a single URDF. Returns a ProcessResult; never raises."""
        try:
            return self._process_inner(urdf_path)
        except Exception as e:
            logger.exception("failed processing %s", urdf_path)
            return ProcessResult(
                status="failed", reason=f"{type(e).__name__}: {e}"
            )

    def _process_inner(self, urdf_path: str) -> ProcessResult:
        if has_caption_candidates(urdf_path) and not self.force:
            return ProcessResult(status="skipped", reason="already_labelled")

        fields = read_asset_fields(urdf_path)
        renders = find_renders(urdf_path)
        if not renders:
            return ProcessResult(status="failed", reason="no_renders")

        # Guardrail A: over-ask 1.5x on the initial round so the typical
        # combined drop (structure + category) is absorbed without entering
        # top-up. Truncation later in this function caps the final pool at
        # the target.
        ask_n = math.ceil(self.num_candidates * 1.5)
        prompt = render_initial_prompt(
            category=fields["category"],
            num_candidates=ask_n,
            num_views=len(renders),
        )

        # Retry the initial GPT call + parse up to MAX_INITIAL_ATTEMPTS
        # times. GPT is non-deterministic, so a fresh call often clears
        # transient failures (drop-rate-too-high, no response).
        phrases: list[str] = []
        last_err = "no attempts made"
        for attempt in range(1, MAX_INITIAL_ATTEMPTS + 1):
            raw = self.gpt_client.query(text_prompt=prompt, images=renders)
            if raw is None:
                last_err = "gpt_no_response"
                logger.warning(
                    "initial attempt %d/%d returned None for %s",
                    attempt,
                    MAX_INITIAL_ATTEMPTS,
                    urdf_path,
                )
                continue
            try:
                parsed = parse_response(raw, category=fields["category"])
                phrases = dedup([normalize(p) for p in parsed])
                break
            except CaptionParseError as e:
                last_err = str(e)
                logger.warning(
                    "initial attempt %d/%d parse failed for %s: %s",
                    attempt,
                    MAX_INITIAL_ATTEMPTS,
                    urdf_path,
                    e,
                )
                continue

        if not phrases:
            return ProcessResult(
                status="failed",
                reason=f"after {MAX_INITIAL_ATTEMPTS} attempts: {last_err}",
            )

        # Top-up if short. Retry top-up up to MAX_TOPUP_ATTEMPTS times.
        # Top-up failure is non-fatal (we keep what we have).
        for topup_attempt in range(1, MAX_TOPUP_ATTEMPTS + 1):
            if len(phrases) >= self.num_candidates:
                break
            deficit = self.num_candidates - len(phrases)
            topup_prompt = render_topup_prompt(
                category=fields["category"],
                num_needed=deficit,
                num_views=len(renders),
                existing=phrases,
            )
            topup_raw = self.gpt_client.query(
                text_prompt=topup_prompt, images=renders
            )
            if topup_raw is None:
                logger.warning(
                    "top-up attempt %d/%d returned None for %s",
                    topup_attempt,
                    MAX_TOPUP_ATTEMPTS,
                    urdf_path,
                )
                continue
            try:
                extra = parse_response(topup_raw, category=fields["category"])
                phrases = dedup(phrases + [normalize(p) for p in extra])
            except CaptionParseError as e:
                logger.warning(
                    "top-up attempt %d/%d parse failed for %s: %s",
                    topup_attempt,
                    MAX_TOPUP_ATTEMPTS,
                    urdf_path,
                    e,
                )
                continue

        phrases = phrases[: self.num_candidates]
        if len(phrases) < self.num_candidates:
            logger.warning(
                "short of target: %s got %d/%d",
                urdf_path,
                len(phrases),
                self.num_candidates,
            )

        # Slicing below assumes the pool fits inside seen + unseen so the
        # two lists never overlap; line 412 above caps the pool length.
        assert len(phrases) <= self.num_candidates
        # Sort first so the split depends only on (uuid, phrase set), not
        # on GPT's response order. Seed by uuid so re-runs reproduce the
        # same split.
        pool = sorted(phrases)
        random.Random(fields["uuid"]).shuffle(pool)
        seen = pool[: self.seen_count]
        unseen = pool[self.seen_count : self.seen_count + self.unseen_count]
        # Shortage policy: seen fills first, unseen takes the remainder.
        # Warn separately so operators can tell which list degraded.
        if len(unseen) < self.unseen_count:
            logger.warning(
                "unseen short: %s got %d/%d (seen=%d/%d)",
                urdf_path,
                len(unseen),
                self.unseen_count,
                len(seen),
                self.seen_count,
            )

        payload = {
            "raw": fields["category"],
            "uuid": fields["uuid"],
            "seen": seen,
            "unseen": unseen,
        }
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(urdf_path)), CAPTION_JSON_NAME
        )
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        write_caption_candidates_link(urdf_path, f"./{CAPTION_JSON_NAME}")
        return ProcessResult(status="written", count=len(phrases))


def iter_urdfs(root: str) -> list[str]:
    """Recursively list ``*.urdf`` files under ``root`` (sorted)."""
    found: list[str] = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".urdf"):
                found.append(os.path.join(dirpath, f))
    found.sort()
    return found
