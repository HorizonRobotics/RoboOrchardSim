# Batch Planning and Execution

Use this reference after task YAML generation and per-scene data-volume
selection. One task YAML represents one scene. Map the user's requested data
samples per scene to the scheduler's `episodes_per_config`.

## Contents

- Create and validate the batch plan
- Run locally
- Prepare and validate cluster submission
- Verify synthesis outputs
- Select successful records

## Create the Batch Plan

Create `batch_plan.json` while the working directory is the portable bundle so
that `configs/yaml_list.txt` entries remain portable:

```bash
python3 <repo-root>/examples/manipulation-app/scripts/scm/\
schedule_config_groups.py \
  --task <task> \
  --config-list configs/yaml_list.txt \
  --configs-per-group <configs-per-group> \
  --episodes-per-config <data-samples-per-scene> \
  --seed <seed> \
  --output batch_plan.json
```

Require for every group:

```text
len(group.configs) * data_samples_per_scene <= 500
```

Validate that every config entry resolves inside the bundle's `configs/`
directory and exists. Reject duplicate paths, empty groups, and paths that
escape the bundle.

Run the bundled validator immediately after scheduling, even for local runs:

```bash
python3 .agents/skills/data-synthesis/scripts/\
validate_batch_bundle.py \
  --batch-plan <bundle>/batch_plan.json \
  --json
```

This replaces repeated manual checks for plan structure, relative YAML paths,
YAML readability, duplicate configs, and the 500-data-per-group limit. It does
not launch Isaac or mutate the bundle.

## Run Locally

Run each group from the repository root in a separate process:

```bash
python3 examples/manipulation-app/scripts/scm/run_multi_task_synthesis.py \
  --batch-plan logs/scm_batch_data_synthesis/<batch-id>/batch_plan.json \
  --group-id <group-id> \
  --asset-root <asset-root> \
  --output-root-dir <data-output-root>/<group-id>
```

Forward `--snapshot` or `--splits` when the generated configs require them.
Verify the group summary before launching the next group. A process boundary is
required between groups to release Isaac and GPU resources.

## Prepare Cluster Submission

Prepare a batch-specific template with the bundled helper instead of editing
JSON fields individually:

```bash
python3 .agents/skills/data-synthesis/scripts/\
prepare_submit_template.py \
  --base-template <selected-submit-template.json> \
  --output <bundle>/submit_template.json \
  --queue-name <queue-name> \
  --bundle-upload-path <bundle>
```

The helper preserves the base template, sets one worker and one GPU, maps the
queue, appends the complete bundle to `to_upload`, and prints the basename-
flattened workspace batch-plan path. Prefer a current, authorized repository
template as `--base-template` when it contains site-specific settings. The
skill also includes a sanitized fallback at
`assets/aidi-submit-template.json`; its `job_password` remains
`__REQUIRED__` unless supplied through the `AIDI_JOB_PASSWORD` environment
variable. Never put a real password in the bundled asset.

Preserve the user-facing cluster machine name and map it to the template's
concrete `queue_name`; keep both values in working state. If the user supplies
only a full `queue_name`, use it as the displayed cluster machine name. Check
the image, project, buckets, workspace, and wall time.

Derive the cluster registry root manually before rendering:

1. Read the selected template's `cmd` entry that exports `ORCHARD_ASSET`.
2. Strip any trailing slash from its concrete value and append `/OBJECTS`.
3. Pass the resulting concrete path to `submit_config_groups.py` as
   `--asset-root`.

For example, when the template contains:

```bash
export ORCHARD_ASSET=/horizon-bucket/robot_lab/assets/ROBO_ORCHARD_SIM
```

render with:

```bash
--asset-root /horizon-bucket/robot_lab/assets/ROBO_ORCHARD_SIM/OBJECTS
```

Do not pass the `ORCHARD_ASSET` parent directory itself. The registry scanner
expects the `OBJECTS` directory's domain/super-category/asset hierarchy; an
index created one level above may be valid but contain zero assets. Before
rendering, require `<derived-root>/asset_index.parquet` to exist and confirm
that every UUID pinned by the selected group is present in that index. Treat
`ORCHARD_ASSET_LIBRARY` as a separate generator/selector setting; do not use it
as a substitute for the submit template's `ORCHARD_ASSET` value.

Do not add or remove proxy commands in the rendered cluster job merely to
satisfy the submission rule below. That rule applies only to the host process
that invokes AIDI; proxy behavior inside the running cluster container is
independent and may remain unchanged unless the user explicitly requests it.

Append the complete repository-relative bundle path to `to_upload`, for
example:

```json
"logs/scm_batch_data_synthesis/<batch-id>"
```

Do not upload only `batch_plan.json`: every YAML referenced by the plan is a
runtime dependency.

Treat host paths and uploaded workspace paths as different namespaces.
`RoboOrchardJob-AIDISubmit` prepares its workspace with the equivalent of:

```bash
rsync -aL <to_upload-entry> <workspace-folder>
```

For a nested upload entry, `rsync` keeps only the source basename at the
workspace root; it does not recreate the source's parent directories. Thus:

```text
host bundle:
  logs/scm_batch_data_synthesis/<batch-id>/batch_plan.json

uploaded bundle:
  <batch-id>/batch_plan.json
```

Do not render the cluster command with
`logs/scm_batch_data_synthesis/<batch-id>/batch_plan.json` unless the upload
layout has independently been staged to preserve those parent directories.
Use `<batch-id>/batch_plan.json` when the bundle itself is the `to_upload`
entry.

Render one submit JSON per group with
`examples/manipulation-app/scripts/scm/submit_config_groups.py`. Pass the
repository-relative bundle path to `--batch-plan`, the shared data path to
`--cluster-output-root`, and the batch-specific template copy to
`--submit-template`. Pass the concrete `<ORCHARD_ASSET-value>/OBJECTS` path to
`--asset-root`.

The current CLI uses its `--batch-plan` value both to load the host file and
to render the cluster command. When upload flattening makes those paths
different, do not assume the CLI output is portable. Load the plan from its
real host path and render the job command with the computed workspace path,
using the script's public rendering helpers or a focused rewrite of only the
rendered submit JSONs. Never change the portable plan's relative `configs/`
entries.

Before real submission, run one workspace preparation with `execute: false`.
Inspect the resulting workspace rather than only the source repository, and
require all of the following:

1. The exact `--batch-plan` path from the rendered command exists beneath the
   prepared workspace.
2. Every relative config entry resolves from that uploaded plan's directory
   to an existing file beneath its `configs/` directory.
3. The generated `run_local.sh` contains the same batch-plan path.
4. The generated `run_local.sh` passes the concrete derived `/OBJECTS` path to
   `--asset-root`, and not the `ORCHARD_ASSET` parent directory.

Delete or ignore the preflight-only config after validation; it must keep
`execute: false` and must not create a cluster job.

Before rendering and again after any plan or template change, run:

```bash
python3 .agents/skills/data-synthesis/scripts/\
validate_batch_bundle.py \
  --batch-plan <bundle>/batch_plan.json \
  --submit-template <bundle>/submit_template.json \
  --workspace-batch-plan <batch-id>/batch_plan.json \
  --json
```

Treat a nonzero exit as a hard stop. This deterministic validator covers the
source bundle, plan workload, template placeholders, queue fields, one-GPU
settings, complete-bundle upload, and expected basename mapping. It does not
replace the `execute=false` inspection of the actual prepared AIDI workspace,
the concrete `/OBJECTS` asset-index check, or UUID validation.

Render without `--submit` first. Before submission, require all of the
following:

1. The batch plan and every referenced YAML exist in the source bundle and
   the prepared upload workspace.
2. All config paths are relative, resolve inside `configs/`, and remain valid
   from the uploaded plan location.
3. Every rendered JSON includes the whole bundle in `to_upload`.
4. Every job command points to the actual basename-flattened uploaded batch
   plan path, not an assumed repository path.
5. `num_workers` and `gpu_per_worker` are one for each group job.
6. Every group contains at most 500 planned data samples, corresponding to 500
   attempted episodes.
7. `queue_name` equals the queue mapped from the cluster machine selected by
   the user, and the stored cluster machine name matches that selection.
8. The data output root is visible from the cluster.
9. `--asset-root` equals the selected template's concrete `ORCHARD_ASSET`
   value plus `/OBJECTS`.
10. The derived asset index is non-empty and contains every UUID pinned by the
    rendered groups.

After these internal checks pass, show the user an AIDI submission confirmation
table containing only the AIDI job name, cluster machine name, Docker image,
GPU allocation, data output directory, and concrete `--asset-root` value. Show
one row per rendered job when group-specific values differ. Do not include the
queue, project, buckets, command, upload bundle, workspace paths, wall time, or
proxy status in this user-facing table unless the user asks for them. Add
`--submit` only after the user explicitly confirms these rendered jobs. If any
of the six displayed values changes, render and preflight again and request
confirmation again.

Run the submitting process itself without proxy variables. Prefix the final
submission command with `env -u` so the Python submitter and its
`RoboOrchardJob-AIDISubmit` subprocess inherit a proxy-free environment:

```bash
env \
  -u http_proxy -u https_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY \
  python3 examples/manipulation-app/scripts/scm/\
submit_config_groups.py \
  <the previously reviewed render arguments> \
  --submit
```

When submitting an already-rendered JSON directly, apply the same environment
rule:

```bash
env \
  -u http_proxy -u https_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY \
  RoboOrchardJob-AIDISubmit submit_from_config --config <submit.json>
```

Do not rely on changing the user's interactive shell globally. Refuse to
submit if any proxy-enabling variable remains in the host-side submission
process. Do not require a proxy-clearing command in the rendered cluster job.

## Verify Outputs

For each completed group, inspect:

- `batch_run_summary_<task>.json`;
- `all_episode_records_<task>.jsonl`;
- `all_mcap_paths_<task>.txt`;
- copied task configs and per-config recording directories.

Report planned data count, attempted episodes, successful records, success
rate, errors, and missing MCAP paths. Do not call a batch complete while
required groups or artifacts are absent.

## Select Successful Records

Offer this optional post-processing step only after all required group
summaries are present. The selector accepts either one
`batch_run_summary_<task>.json` or a run root containing summaries at
`group_*/batch_run_summary_<task>.json`.

Collect these settings one at a time when they are not already known:

- inclusive minimum and maximum scene success rates;
- `sample_num`, the maximum successful records to keep from each retained
  scene summary;
- output JSON path;
- optional split YAML path;
- target asset role, which defaults to `pick`;
- asset root, reusing the synthesis asset root or `ORCHARD_ASSET_LIBRARY`.

Reuse the concrete synthesis registry root ending in `/OBJECTS`. If deriving
it again from `ORCHARD_ASSET`, append `/OBJECTS`; do not pass the parent root.

Use the synthesis data output root as the input path:

```bash
python3 examples/manipulation-app/scripts/scm/\
select_successful_records.py \
  <data-output-root> \
  --output <data-output-root>/selected_records.json \
  --sample-num <records-per-retained-scene> \
  --min-success-rate <minimum-rate> \
  --max-success-rate <maximum-rate> \
  --asset-root <asset-root>
```

Add `--split-output <path>` when the user requests a non-default split path,
and add `--target-role <role>` when the target is not `pick`. Without
`--split-output`, the selector writes a YAML next to the JSON using the same
stem.

Explain these semantics before execution:

- success-rate bounds are inclusive;
- records are selected only from summaries inside the requested rate range;
- `sample_num` takes the first N successful record paths per retained scene,
  not a random sample and not N records across the whole batch;
- the output JSON contains selected scene/asset metadata and MCAP paths;
- the split YAML contains selected UUIDs under `seen`;
- the command writes manifests only and does not copy, move, or delete MCAP
  data.

If a cluster output path is not mounted locally, generate the command for an
environment where that path is accessible instead of claiming to run it.

After selection, validate both manifest files. Report retained and removed
scene counts, total selected MCAP paths, the JSON path, and the split YAML
path. If no summary is found, report the missing input instead of running with
the script's unrelated default input path.
