# Continuous Evidence Rules

## Required Window

Select exactly four chronological frames from one interaction:

1. `APPROACH`
2. `PRE_INTERACTION`
3. `INTERACTION`
4. `OUTCOME`

Scan the complete fixed external-view overview before choosing the window.
Gripper-close candidates and minimum-distance events are search hints, not
proof of the actual interaction.

## View Roles

- `ext1` and `ext2`: establish target identity, distractors, contact-caused
  motion, and outcome.
- `wrist`: establish alignment and whether the object enters the finger region.
- Use synchronized or nearest-timestamp images across views.
- Do not infer selection from one isolated wrist image.

## Target Evidence

The actor with `actor_type=pick` in `/meta_data` is authoritative. Values under
`initial_target_projection_proposals` are explicitly `UNVERIFIED`: camera and
object coordinate conventions may differ. Visually verify a proposal before
drawing any marker, and omit uncertain markers. `WRONG_OBJECT` requires
evidence for both the target's location and the distractor interaction.

## Signal Evidence

Use actual `/robot/joint_states` `panda_joint8` for gripper action. The action
topic may be shown as supplemental command evidence. Missing actual state means
`UNKNOWN`.

## Evaluation Reconciliation

Read `/meta_data.task_progress` from the MCAP before assigning failure labels.
`task_progress=1.0` is authoritative success evidence. It overrides a
conflicting or incomplete `eval_result.json`/stdout result and requires:

- `evaluation.success=true`
- `evaluation.progress=1.0`
- `reach_pick=true` and `lift_pick=true`
- `stop_reason=success`
- `cause=NONE`

Retain the external values only as provenance. A value below `1.0` does not by
itself override the external evaluation result. Visual frames are still needed
to describe the successful grasp interaction and object outcome.

## Cause Evidence

Choose a cause only after fixing the four observed dimensions:

1. Successful episode, including MCAP `task_progress=1.0`: `NONE`.
2. Verified distractor manipulation: `SELECTED_DISTRACTOR`.
3. Correct target with no contact: `OFF_TARGET_APPROACH`.
4. Target contact while the actual gripper stays open: `NO_CLOSE_ATTEMPT`.
5. Close attempt and contact without a formed grasp: `MISALIGNED_GRASP`.
6. Formed grasp followed by loss: compare actual gripper state at loss.
   - It remains closed: `UNSTABLE_GRASP`.
   - It reopens before or at loss: `PREMATURE_GRIPPER_OPENING`.
   - State is missing or timing is ambiguous: `UNKNOWN`.
7. Stable grasp remains closed but lift height is insufficient:
   `INSUFFICIENT_LIFT_MOTION`.

Early close, late close, side contact, and partial enclosure are evidence
descriptions under `MISALIGNED_GRASP`, not separate causes. Object displacement
such as `PUSHED_AWAY` is an outcome and does not replace the cause.

## Prohibited Inferences

- `reach_pick=true` does not prove correct target selection or contact.
- `lift_pick=false` does not identify the grasp failure mechanism.
- Initial/final displacement alone does not prove `PUSHED_AWAY`.
- One frame cannot prove contact, transient grasp, stable grasp, or pushing.
- Do not replace unavailable evidence with a confident label.
- Do not infer `PREMATURE_GRIPPER_OPENING` from object loss alone; actual
  gripper reopening must precede or coincide with the loss.
- Do not infer `INSUFFICIENT_LIFT_MOTION` after a drop; the stable grasp and
  closed gripper must persist through the final evidence frame.
