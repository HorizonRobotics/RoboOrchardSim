---
name: analyzing-pick-failure-cases
description: Use when analyzing robot specified-object pick evaluations from MCAP recordings and eval_result.json, including continuous multi-view evidence selection, grasp-failure attribution, per-seed visualization, or aggregate failure statistics.
---

# Analyzing Pick Failure Cases

Use deterministic scripts for extraction and signals; use visual reasoning only
for claims supported by continuous images. Attributions are model-generated,
not human-labelled ground truth.

Set:

```bash
SKILL_DIR=.agents/skills/analyzing-pick-failure-cases
```

## Workflow

1. Prepare selected seeds or all failures:

```bash
python3 "$SKILL_DIR/scripts/prepare_cases.py" <eval-root> \
  --seeds 100000,100001 --output-dir <output>

python3 "$SKILL_DIR/scripts/prepare_cases.py" <eval-root> \
  --all-failures --output-dir <output>
```

Add `--include-success` only when success controls are requested.

2. Read `references/label-schema.md` and
   `references/evidence-rules.md`. Inspect every fixed-view overview page before
   selecting an interaction. Close-event sheets are search hints, not proof.

   Before treating a seed as a failure, inspect `/meta_data.task_progress` in
   the MCAP. If `task_progress=1.0`, treat the episode as successful even when
   `eval_result.json` or an incomplete stdout snapshot says failure. Set
   `success=true`, `progress=1.0`, `reach_pick=true`, `lift_pick=true`,
   `stop_reason=success`, and use `cause=NONE`.

3. Select **exactly four** chronological frames from one interaction:
   approach, pre-interaction, interaction, outcome. Use external views for
   target identity and movement; use wrist for alignment. If continuous visual
   evidence is insufficient, use `UNCERTAIN` or `UNKNOWN`.

4. Assign the four observed dimensions first, then choose one cause using the
   decision boundaries in `references/label-schema.md`. In particular, use
   actual gripper state to distinguish closed-finger slip
   (`UNSTABLE_GRASP`) from reopening-caused loss
   (`PREMATURE_GRIPPER_OPENING`).

5. Write `cases/seed_<seed>/analysis.json` conforming to
   `references/analysis-schema.json`. Copy evaluation values from
   `extracted_signals.json`; do not reinterpret them as attribution evidence.
   Add normalized annotations only after visually checking the target marker.

6. Validate before rendering:

```bash
python3 "$SKILL_DIR/scripts/validate_results.py" \
  <output>/cases/seed_<seed>
```

Fix every error. Do not weaken a label to bypass validation; change it to an
unknown state when evidence is missing.

7. Render the standardized evidence image:

```bash
python3 "$SKILL_DIR/scripts/render_case.py" \
  <output>/cases/seed_<seed>
```

8. After validating all completed cases, aggregate:

```bash
python3 "$SKILL_DIR/scripts/aggregate_results.py" \
  <output> <eval-root>/eval_result.json
```

## Integrity Rules

- The `/meta_data` actor with `actor_type=pick` is authoritative.
- MCAP `/meta_data.task_progress=1.0` is authoritative for successful task
  completion and overrides a conflicting external evaluation summary.
- Use actual `/robot/joint_states` `panda_joint8` for gripper action.
- `CLOSE_ATTEMPTED` does not imply contact or grasp.
- Closing too early, too late, or around only part of the target is
  `MISALIGNED_GRASP`, not a separate cause.
- Object loss with closed fingers is `UNSTABLE_GRASP`; object loss caused by
  verified reopening is `PREMATURE_GRIPPER_OPENING`.
- `reach_pick` does not prove correct selection or contact.
- `PUSHED_AWAY` requires visible interaction-caused movement, not only
  initial/final displacement.
- `WRONG_OBJECT` requires both target-location and distractor-interaction
  evidence.
- Never infer contact from one isolated frame.
- Preserve per-seed extraction errors; one corrupt episode must not abort a
  batch.
