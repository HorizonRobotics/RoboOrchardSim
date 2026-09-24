# Pick Failure Attribution Labels

Use the four dimensions independently. Do not encode a mechanism into an
observed-state label.

## Target Selection

- `CORRECT`: continuous views show the gripper approaching or manipulating the
  authoritative target from `/meta_data`.
- `WRONG_OBJECT`: continuous views show interaction with a distractor while the
  target remains separately identifiable.
- `UNCERTAIN`: object identity is occluded, ambiguous, or unsupported.

## Gripper Action

- `CLOSE_ATTEMPTED`: the actual `panda_joint8` trace contains a sustained close.
- `NO_CLOSE_ATTEMPT`: an available trace stays open through the interaction.
- `UNKNOWN`: the actual gripper trace is absent or unusable.

Closing is an action, not proof of contact or grasp.

## Target Interaction

- `NO_CONTACT`: adequate multi-view evidence shows no target contact.
- `CONTACT_NO_GRASP`: contact changes target pose without finger constraint.
- `TRANSIENT_GRASP`: temporary finger constraint is followed by loss.
- `STABLE_GRASP`: the target remains constrained and moves with the gripper.
- `UNKNOWN`: contact or constraint cannot be determined.

## Object Outcome

- `STATIONARY`: no visible target displacement caused by the interaction.
- `PUSHED_AWAY`: continuous frames show contact-caused displacement on the
  support surface.
- `LIFTED_SUCCESSFULLY`: stable grasp raises the target beyond the task
  threshold.
- `LIFTED_THEN_DROPPED`: the target rises and subsequently leaves the grasp.
- `LIFTED_INSUFFICIENTLY`: visible lift remains below the task threshold.
- `UNKNOWN`: the final target outcome is not visible.

## Cause

Assign `cause` only after the four observed dimensions. It identifies the most
specific supported failure mechanism; it is not a fifth observation dimension.
If two causes remain plausible, use `UNKNOWN`.

### `NONE`

Use when: `evaluation.success=true`; the pick succeeded and no failure cause
exists. MCAP `/meta_data.task_progress=1.0` forces this state even if an
external summary reports failure.

Do not use when: the episode failed, even if the available evidence does not
show why. Use `UNKNOWN` for an unexplained failure.

### `SELECTED_DISTRACTOR`

Use when: continuous external views show the gripper manipulating a non-target
object while the authoritative target remains separately identifiable. The
required target-selection label is `WRONG_OBJECT`.

Do not use when: target identity is uncertain, the target is fully occluded, or
the gripper only passes near a distractor without interacting with it.

### `OFF_TARGET_APPROACH`

Use when: the correct target is identified, but the approach trajectory never
contacts it. The required combination is `CORRECT + NO_CONTACT`.

Do not use when: any continuous view verifies target contact, including a push
or open-gripper collision. Those cases are interaction failures, not an
off-target approach.

### `NO_CLOSE_ATTEMPT`

Use when: the correct target is contacted while the actual Panda gripper stays
open through the interaction. This includes an open-gripper collision that
pushes the target. The required combination is `CORRECT + NO_CLOSE_ATTEMPT +
CONTACT_NO_GRASP`.

Do not use when: `/robot/joint_states` shows a close attempt during the relevant
interaction, or when actual gripper state is unavailable. Use
`MISALIGNED_GRASP` for a failed close with contact and `UNKNOWN` for missing
state.

### `MISALIGNED_GRASP`

Use when: the correct target is contacted and closing is attempted, but the
target never becomes constrained by both fingers. It includes early closing,
late closing, side contact, and insufficient or partial enclosure. The required
combination is `CORRECT + CLOSE_ATTEMPTED + CONTACT_NO_GRASP`.

Do not use when: there is no target contact, no close attempt, or a transient or
stable grasp actually forms.

### `UNSTABLE_GRASP`

Use when: a grasp forms, but the target later slips from the fingers while the
actual gripper remains closed through object loss. The required interaction is
`TRANSIENT_GRASP`; the loss must be visible in a continuous sequence.

Do not use when: the gripper reopens before or at object loss, or when the target
was never constrained. Use `PREMATURE_GRIPPER_OPENING` for verified reopening
and `MISALIGNED_GRASP` when no grasp formed.

### `PREMATURE_GRIPPER_OPENING`

Use when: a grasp forms and the actual gripper reopens before or at object loss,
with reopening temporally causing the drop or failed retention. The required
interaction is `TRANSIENT_GRASP`; `LIFTED_THEN_DROPPED` is common but not
mandatory.

Do not use when: the gripper remains closed through object loss, when no grasp
formed, or when reopening happens only after the failure is already complete.

### `INSUFFICIENT_LIFT_MOTION`

Use when: a stable grasp remains formed, the actual gripper remains closed, and
the target moves upward but does not reach the task's lift threshold. The
required combination is `STABLE_GRASP + LIFTED_INSUFFICIENTLY`.

Do not use when: the object leaves the grasp, the gripper opens prematurely, or
the target was never stably constrained.

### `UNKNOWN`

Use when: evidence is missing, contradictory, or insufficient to select one
unique cause. In particular, use it when a formed grasp is lost but actual
gripper state cannot distinguish slipping from premature reopening.

Do not use when: one cause is directly supported by continuous multi-view
evidence and the actual gripper trace.

## Cause Consistency

| Cause | Required observed dimensions |
| --- | --- |
| `SELECTED_DISTRACTOR` | `target_selection=WRONG_OBJECT` |
| `OFF_TARGET_APPROACH` | `CORRECT + NO_CONTACT` |
| `NO_CLOSE_ATTEMPT` | `CORRECT + NO_CLOSE_ATTEMPT + CONTACT_NO_GRASP` |
| `MISALIGNED_GRASP` | `CORRECT + CLOSE_ATTEMPTED + CONTACT_NO_GRASP` |
| `UNSTABLE_GRASP` | `CORRECT + CLOSE_ATTEMPTED + TRANSIENT_GRASP`; gripper stays closed |
| `PREMATURE_GRIPPER_OPENING` | `CORRECT + CLOSE_ATTEMPTED + TRANSIENT_GRASP`; gripper reopens |
| `INSUFFICIENT_LIFT_MOTION` | `CORRECT + CLOSE_ATTEMPTED + STABLE_GRASP + LIFTED_INSUFFICIENTLY` |
