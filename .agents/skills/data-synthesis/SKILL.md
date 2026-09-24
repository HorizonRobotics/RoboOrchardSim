---
name: data-synthesis
description: Guide an interactive RoboOrchard SCM batch data-synthesis workflow from task and embodiment selection through user-defined task YAML generation, per-scene data-volume planning, output-path selection, local or AIDI cluster execution, and successful-record selection. Use when a user wants to start, configure, plan, debug, render, submit, inspect, or post-process manipulation data synthesis based on robo_orchard_sim benchmark tasks.
---

# Data Synthesis

Guide the user through a stateful, stage-gated synthesis workflow. Discover
options from the current repository, preserve the user's choices, and produce a
portable batch bundle for local or cluster execution.

## Interaction Rules

- Run repository commands from the repository root unless a step explicitly
  changes into the batch bundle.
- Ask one blocking question at a time. Use a choice control when available;
  otherwise show a numbered menu and accept either a number or a name.
- Ask questions in the user's language.
- Reuse choices already supplied by the user. Do not ask for them again.
- Show the recommended choice first and explain material compatibility limits.
- Keep a working state containing the task, base config, embodiment, YAML
  requirements, asset or layout inputs, scene count, data samples per scene,
  group size, data output root, execution mode, cluster machine name, and
  cluster queue. Map data samples per scene to the scheduler's
  `episodes_per_config` only internally.
- Treat user requirements as authoritative for task YAML generation. Existing
  generators are examples and reusable implementations, not a closed set of
  allowed generation modes.
- Use the bundled preparation and validation scripts for deterministic batch
  orchestration. Keep the repository SCM scripts as the source of truth for
  config generation, scheduling, synthesis, submission, and record selection.
- Never modify benchmark base YAMLs. Write generated files only inside the
  fixed batch bundle.
- Never submit cluster jobs without a final, explicit confirmation. Rendering
  submit JSONs is not submission.
- Never run the host-side AIDI submission command with HTTP, HTTPS, or ALL
  proxy variables enabled. Clear them only from the submission process
  environment; do not alter the cluster job command for this rule.
- For AIDI submission, derive the concrete registry root from the selected
  template's `ORCHARD_ASSET` value by appending `/OBJECTS`, then pass that
  concrete path to the submit renderer's existing `--asset-root` argument.
  Never pass the `ORCHARD_ASSET` parent directory itself.

## Stage 1: Select the Task

Discover registered benchmark tasks and their base YAMLs from the current
checkout. Read [task-yaml-generation.md](references/task-yaml-generation.md)
before presenting task choices.

List synthesis-ready tasks by default. Mark registered tasks whose task
definition still returns the empty base atomic action plan as unavailable for
data synthesis instead of silently offering them.

Ask:

> Which benchmark task do you want to synthesize data for?

Show the task namespace, base YAML, generation style when known, and default
embodiment. Validate the selected task before advancing.

## Stage 2: Select the Embodiment

Discover embodiments from the benchmark embodiment registry. Do not treat
every directory under `orchard_env/embodiments` as runnable.

Ask:

> Which embodiment should this task use?

Recommend the embodiment in the selected base YAML. Check that the task action
plan supports the selected embodiment. When changing embodiments, replace the
complete `embodiment:` block; do not retain robot-specific joint names or
initial joint positions from the previous robot.

## Stage 3: Define and Generate Task YAMLs

Ask:

> How should the task YAMLs be generated or modified?

Offer examples without limiting the answer:

1. Generate one YAML per graspable asset.
2. Generate one YAML per layout or layout range.
3. Generate by category, attribute, split, snapshot, or UUID.
4. Modify scene, asset filters, instruction, or randomization settings.
5. Apply a custom generation rule.

Collect only inputs required by the user's rule, such as asset root, layout
root, target role, tags, success threshold, split, snapshot, attributes, or
specific YAML overrides. Then follow the generation decision process in
[task-yaml-generation.md](references/task-yaml-generation.md).

Before writing all configs, show a representative YAML or focused diff. After
confirmation, generate deterministic, uniquely named configs and a sorted
`yaml_list.txt`. Validate every generated YAML and report the config count.

## Stage 4: Plan Data per Scene

Ask exactly:

> How many data samples should be generated for each scene?

When speaking Chinese, ask:

> 每个场景希望生成多少条数据？

Treat one generated task YAML as one scene. Do not ask for a total data count.
Let `D` be the positive number of data samples requested per scene and `N` the
generated scene count. Map `D` to the scheduler's `episodes_per_config` and
report `N * D` as the derived total planned data count. Each planned sample is
one attempted episode, so successful records may be fewer; explain this only
when the distinction matters.

Enforce this hard invariant for every local process and cluster job:

```text
len(group.configs) * data_samples_per_scene <= 500
```

If `D <= 500`, compute the maximum group size as `floor(500 / D)`. Use that
size unless the user requests a smaller one. If `D > 500`, stop planning and
ask the user to reduce it or explicitly split the work into separate batches;
the current batch-plan schema cannot safely shard one config across different
episode counts.

Report the scene count, data samples per scene, derived total, group size,
per-GPU planned data maximum, and resulting group count before advancing.
Then add a non-blocking reminder: for AIDI execution, reducing the group size
increases the number of independent groups/jobs and may reduce wall-clock time
through parallelism, subject to available GPU quota and queue capacity. Local
groups run sequentially by default, so increasing the group count does not by
itself speed up local synthesis. Keep the maximum safe group size as the
default unless the user requests more groups or a smaller group size.

## Stage 5: Select the Data Output Path

Ask only for the final data output root. Do not ask the user to choose a
control directory.

Create the internal portable bundle at:

```text
logs/scm_batch_data_synthesis/<batch-id>/
├── batch_plan.json
├── configs/
│   ├── *.yaml
│   └── yaml_list.txt
├── submit_template.json
└── submit_jsons/
```

Store config entries in `batch_plan.json` relative to the plan directory, for
example `configs/pick_example.yaml`. Reject local absolute config paths for
cluster execution.

For local execution, verify that the data output path is writable. For cluster
execution, verify that it is a cluster-visible shared path.

## Stage 6: Select Local or Cluster Execution

Ask:

> Run locally or on an AIDI cluster?

For local execution, run one group per process so Isaac exits and releases GPU
resources at group boundaries. Prefer the repository's built-in virtual
display launcher path.

For cluster execution, ask for the cluster machine name and map it to the
submit template's `queue_name`. Preserve the user-facing cluster machine name
and the concrete `queue_name` as separate working-state values. If the user
provides only a full `queue_name`, reuse it as the displayed cluster machine
name. Read
[batch-execution.md](references/batch-execution.md), render submit JSONs first,
and upload the complete portable bundle. Treat the batch plan plus every YAML
it references as one indivisible upload unit. Disable proxy variables in the
host-side submission process before calling AIDI. Cluster runtime proxy state
is outside this submission rule and may remain unchanged.
Prepare the batch-specific submit template with
`scripts/prepare_submit_template.py`, then validate the bundle with
`scripts/validate_batch_bundle.py` before rendering jobs. Do not replace these
checks with manual inspection alone.
Read `ORCHARD_ASSET` from the selected submit template, append `/OBJECTS`,
validate the resulting registry and required UUIDs, and pass the concrete
result as `--asset-root`. Do not confuse `ORCHARD_ASSET` with the separate
`ORCHARD_ASSET_LIBRARY` default used by some generation or selection scripts.
Track the host batch-plan path and the uploaded workspace batch-plan path as
separate values: AIDI upload entries are copied by basename, so parent
directories such as `logs/scm_batch_data_synthesis/` may not survive upload.
Never submit until an `execute=false` workspace preparation proves that the
exact path used by the cluster command exists and all referenced YAMLs resolve.

## Stage 7: Review and Execute

Before starting local synthesis, summarize the planned workload and data output
root, then ask for confirmation.

Before submitting AIDI jobs, complete all internal validations and the
`execute=false` workspace preflight. Then show the user a compact confirmation
table containing only:

- AIDI job name;
- cluster machine name;
- Docker image;
- GPU allocation;
- data output directory;
- concrete asset registry root passed to `--asset-root`.

Show one row per rendered job when values differ by group. Do not add other
cluster submission details to this user-facing confirmation unless the user
asks for them. Ask for explicit confirmation before submission. If any of
these six values changes, render and preflight again, then request a new
confirmation.

After execution, report group summaries, successful episode counts, record
directories, and MCAP path-list locations. Stop the downstream workflow when
any required artifact or validation is missing.

## Stage 8: Offer Successful Record Selection

Enter this stage only after every required group has completed and its
`batch_run_summary_<task>.json` is accessible. Tell the user:

> Data synthesis is complete. You can use
> `examples/manipulation-app/scripts/scm/select_successful_records.py` to
> select the records you need.

When speaking Chinese, say:

> 数据合成已完成。可以使用
> `examples/manipulation-app/scripts/scm/select_successful_records.py`
> 筛选需要的数据。

Offer these choices:

1. Filter successful records now.
2. Generate the filtering command only.
3. Skip filtering.

If the user chooses filtering, read the successful-record selection section in
[batch-execution.md](references/batch-execution.md). Ask one question at a
time for any missing success-rate range, maximum records per retained scene,
output JSON path, optional split YAML path, and non-default target role. Reuse
the known asset root.

Do not imply that the selector copies, moves, or deletes MCAP files. It writes
a JSON selection manifest and a benchmark split YAML. After running it, report
the scenes kept and removed, selected MCAP count, JSON path, and split YAML
path.
