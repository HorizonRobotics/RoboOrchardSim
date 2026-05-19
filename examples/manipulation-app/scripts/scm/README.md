# SCM Batch Data Synthesis

This directory provides a small workflow for batch data synthesis:

1. Generate one task YAML per graspable asset.
2. Group those YAML files into a batch plan.
3. Render AIDI submit JSON files, and optionally submit them.
4. Run one group locally when debugging.

## Step 1: Generate Task YAMLs

Input:

- A base task config.
- An asset library root.

Command:

```bash
python3 examples/manipulation-app/scripts/scm/generate_graspable_asset_configs.py \
  --task pick_category \
  --base-config configs/data_synthesis/pick_category.yaml \
  --output-dir logs/multi_task_gen/configs \
  --asset-root /path/to/asset_library
```

Notes:

- `--asset-root` defaults to `ORCHARD_ASSET_LIBRARY` when that env var is set.
- `--target-role` defaults to `pick`.
- `--graspable-tag` defaults to `is_graspable`.

Output:

```text
logs/multi_task_gen/configs/*.yaml
logs/multi_task_gen/configs/yaml_list.txt
```

## Step 2: Create Batch Plan

Input:

- The `yaml_list.txt` from step 1.
- The number of configs to run per group.
- The number of episodes to run per config.

Command:

```bash
python3 examples/manipulation-app/scripts/scm/schedule_config_groups.py \
  --task pick_category \
  --config-list logs/multi_task_gen/configs/yaml_list.txt \
  --configs-per-group 10 \
  --episodes-per-config 100 \
  --seed 0 \
  --output logs/multi_task_gen/batch_plan.json
```

`--batch-id` is optional. If omitted, a timestamped batch id is used.

Output:

```text
logs/multi_task_gen/batch_plan.json
```

## Step 3: Render Submit JSONs

Input:

- The `batch_plan.json` from step 2.
- An AIDI submit template JSON.

Command:

```bash
python3 examples/manipulation-app/scripts/scm/submit_config_groups.py \
  --batch-plan logs/multi_task_gen/batch_plan.json \
  --submit-template examples/manipulation-app/scripts/scm/submit_data_gen_5090.json \
  --submit-json-dir logs/multi_task_gen/submit_jsons \
  --cluster-output-root logs/multi_task_gen \
  --asset-root /path/to/asset_library
```

Notes:

- `--submit-json-dir` defaults to `submit_jsons`.
- `--cluster-output-root` is the output root passed into cluster jobs.
- Generated jobs call `run_multi_task_synthesis.py` with `--batch-plan` and
  `--output-root-dir`.
- `submit_config_groups.py` only uses Python standard library modules.

Output:

```text
logs/multi_task_gen/submit_jsons/*.json
```

To submit after rendering, add `--submit`:

```bash
python3 examples/manipulation-app/scripts/scm/submit_config_groups.py \
  --batch-plan logs/multi_task_gen/batch_plan.json \
  --submit-template examples/manipulation-app/scripts/scm/submit_data_gen_5090.json \
  --submit-json-dir logs/multi_task_gen/submit_jsons \
  --cluster-output-root logs/multi_task_gen \
  --asset-root /path/to/asset_library \
  --submit
```

## Local Group Run

Use this when debugging one group without AIDI submission.

```bash
python3 examples/manipulation-app/scripts/scm/run_multi_task_synthesis.py \
  --batch-plan logs/multi_task_gen/batch_plan.json \
  --group-id group_0000 \
  --asset-root /path/to/asset_library \
  --output-root-dir logs/multi_task_gen/group_0000
```

Output is written under `--output-root-dir`, including group summaries, copied
configs, record paths, and mcap path lists.

