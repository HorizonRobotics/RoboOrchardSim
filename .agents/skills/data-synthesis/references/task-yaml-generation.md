# Task and YAML Generation

Use this reference during task discovery, embodiment selection, and task YAML
generation. Re-discover repository state instead of assuming the current
catalog remains unchanged.

## Contents

- Discover registered tasks and embodiments
- Interpret user-defined YAML requirements
- Build a portable batch bundle
- Validate generated YAMLs

## Discover Tasks

Inspect all of the following together:

- task registrations and namespaces under `robo_orchard_sim/benchmark/`;
- task-definition `config_path` values;
- YAMLs under `robo_orchard_sim/benchmark/**/configs/`;
- `build_atomic_action_plan` overrides for synthesis readiness.

The current synthesis-ready task namespaces are:

- `pick_category`
- `pick_attribute`
- `pick_disambiguation`
- `place_a2b_easy`
- `place_a2b_hard`
- `spatial_pick_easy`
- `spatial_place_a2b_easy`

`affordance` and `joint_direction` are registered, but currently inherit the
empty base atomic action plan. Show them as unavailable unless the current code
adds a real plan.

Do not infer a task merely from an arbitrary YAML filename. Confirm that a
task-definition class registers the namespace and points at the YAML.

## Discover Embodiments

Inspect `_bootstrap_embodiment_registry` in
`robo_orchard_sim/benchmark/base.py`. The current registered types are:

- `dualarm_piper`
- `dualarm_piperx`
- `franka_panda`
- `panda_droid`

Directories such as `cobot_magic`, `fr3`, `galaxea`, and `zr_h1pro` are not
selectable until they are registered. Check task action-plan dispatch before
claiming compatibility.

When changing the base YAML's embodiment, replace the full mapping. Start from
the selected embodiment's supported defaults and add only explicitly verified
parameters. Never carry `init_joint_pos` across robot families.

## Interpret the User's YAML Requirement

Start from the user's requested output behavior. Decide implementation only
after understanding the desired variations and overrides.

1. Use an existing generator unchanged when its behavior exactly matches.
2. Reuse its public functions or patterns when only part of the behavior
   matches.
3. Generate YAMLs directly from the selected base config when the requirement
   is custom.
4. Modify an SCM generator itself only when the user explicitly requests a
   reusable repository feature, not merely to finish one synthesis batch.

Useful existing implementations include:

- `generate_graspable_asset_configs.py`: discover tagged assets, optionally
  filter by snapshot, split, or grasp success, and pin a target role by UUID.
- `generate_layout_task_configs.py`: expand a base config over a count, range,
  or tail selection of layout JSONs.
- `generate_pick_attribute_configs.py`: generate category/UUID and attribute
  variants with optional distractor validation.

These scripts do not restrict valid user requests. Combine filters, edit other
task fields, or use a custom deterministic expansion when required.

## Build a Portable Bundle

Use the fixed bundle root:

```text
logs/scm_batch_data_synthesis/<batch-id>/
```

Run config generation with the bundle as the working directory and use
`configs` as the output directory, or normalize the resulting list afterward.
Require `configs/yaml_list.txt` entries to look like:

```text
configs/task_config_0000.yaml
configs/task_config_0001.yaml
```

This is important because `load_batch_plan` resolves relative config entries
against the directory containing `batch_plan.json`. Entries that repeat the
bundle prefix will resolve incorrectly after upload.

## Validate Generated YAMLs

Before scheduling:

1. Parse every YAML as a top-level mapping.
2. Confirm every filename is unique and every listed path exists.
3. Confirm the selected task accepts the YAML's asset roles and layout mode.
4. Confirm `embodiment.type` equals the user's selection.
5. Confirm robot-specific joint fields belong to that embodiment.
6. Confirm user-requested filters, splits, instructions, and randomization
   settings are present.
7. Confirm `yaml_list.txt` is sorted, contains no duplicates, and contains only
   paths relative to the bundle root.

Show at least one representative generated YAML or focused diff before bulk
generation, and summarize the final config count afterward.
