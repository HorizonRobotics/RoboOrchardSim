---
description: Load these instructions when creating, editing, or reorganizing AGENTS.md files, .agents/instructions/*.md files, or .agents/skills/* guidance in this workspace.
---

# Guidance Authoring Instructions

## Guidance Model

- Decide the layer before editing text: repository-wide root guidance, non-submodule subtree supplement, or submodule-local independent guidance.
- For non-submodule subtrees, keep local `AGENTS.md` and local instructions as supplements to the root guidance unless the subtree explicitly needs a tighter rule.
- For submodules, make the local `AGENTS.md`, instructions, and skills self-contained; do not rely on parent-repository guidance to make the wording complete.

## AGENTS.md Responsibilities

- Use `AGENTS.md` for scope, precedence, routing, and discovery entrypoints.
- Keep detailed behavior rules in `.agents/instructions/` or `.agents/skills/`, not in `AGENTS.md`.
- Keep `Quick Routing` limited to topics owned by that layer; do not route child-only topics from a parent layer.
- If a file says a local `.agents/instructions/` or `.agents/skills/` tree exists, make sure the referenced paths actually exist.
- Keep `source of truth`, `supplement`, and `independent repository` wording consistent with the real inheritance model.

## Instruction Files

- Use `.agents/instructions/*.md` for detailed rules, constraints, and workflow expectations.
- In non-submodule subtrees, keep local instructions focused on package-specific additions or tighter constraints; do not duplicate root-general rules without narrowing their scope.
- Keep the front-matter `description` aligned with the actual scope of the file after edits.
- Prefer updating or deleting stale guidance over leaving broader text that no longer matches the body.

## Skill Files

- Use `.agents/skills/*` for independent workflows or task playbooks, not for restating general instruction text.
- Reference package-local skills from the applicable `AGENTS.md` layer instead of treating them as implicit discoveries.
- If a package-local skill duplicates a root skill name, make the intended precedence explicit in the relevant `AGENTS.md`.

## Consistency Checks

- After editing guidance, read the affected root and subtree files together to check for duplicate rules, contradictory precedence, and mismatched scope.
- Verify that every referenced file path exists and that every listed local tree actually exists.
- Check that `Quick Routing`, `Read First`, and `Repository Notes` do not describe different ownership models for the same subtree.
- For submodules, confirm the local guidance still reads coherently when the parent repository guidance is ignored.
