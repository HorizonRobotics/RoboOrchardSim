---
name: usd-asset-batch-labelling
description: Use when processing a batch of raw USD assets that need to be organized by taxonomy, labelled with URDF/renders via asset_labeller, annotated with grasp interaction data via generate_interaction.py, tagged with semantic capability tags via tag_labeller, and given caption candidates via caption_labeller.
---

# USD Asset Batch Labelling Pipeline

## Overview

Five-stage pipeline for raw USD assets: **organize → label (with AABB) → annotate → tag → caption**.
Produces a labelled directory per asset:
```
<domain>/<super_category>/<asset_name>/
├── <asset_name>.usd          # source mesh
├── textures/                 # source textures
├── mesh/                     # labeller output
├── renders/                  # labeller output (6 views)
├── <asset_name>.urdf         # labeller output (extended by tag + caption)
├── interaction.json          # generate_interaction.py output
└── caption_candidates.json   # caption_labeller output
```

The URDF's `<extra_info>` block accumulates fields across stages:
- Step 2 writes the core fields (`category`, `description`, `shape`, `material`, dimensions, ...)
- Step 2 also writes `<aabb>` (local-frame axis-aligned bounding box, computed from the USD via pxr)
- Step 4 adds `<tags>` (e.g. `is_container,is_graspable`)
- Step 5 adds a `<caption_candidates>` link to the JSON file

---

## Taxonomy Reference

When building `CATEGORY_MAP` for a new batch, infer each asset's `(domain, super_category)` by matching its name against this table. The asset name gives the semantic cue; use the English name column to match.

| domain | super_category | English asset names (examples) |
|--------|---------------|-------------------------------|
| `desktop_supplies` | `office_stationery` | stapler, ballpoint_pen, pen_holder, glue_stick, eraser, correction_tape, marker, sharpener, notebook |
| `desktop_supplies` | `office_tools` | scissors, scotch_tape, magnifier, cutter_knife, stamp, staples_box, sticky_notes |
| `desktop_supplies` | `electronic_accessories` | mouse, tv_remote, ac_remote, presenter_clicker, gamepad, earbud_case, charging_head, charging_cable, usb_drive |
| `desktop_supplies` | `decorations` | mini_sculpture, phone_stand, vase |
| `kitchen_supplies` | `cutting_tools` | fruit_knife, peeler, garlic_press, wire_cutters |
| `kitchen_supplies` | `tableware` | bowl, chopsticks, spoon, fork, coaster |
| `kitchen_supplies` | `ware` | wine_glass, mug, coffee_cup, thermos, baby_bottle |
| `kitchen_supplies` | `utensils` | spice_jar, bottle_opener, spatula |
| `bedroom_supplies` | `storage` | storage_box, jewelry_box, glasses_case, pill_organizer, remote_holder |
| `bedroom_supplies` | `small_electronics` | flashlight, alarm_clock |
| `bathroom_supplies` | `personal_care` | toothbrush, toothpaste, mouthwash_cup, comb, razor, nail_clipper, dental_floss, hair_clip |
| `bathroom_supplies` | `bath_products` | shampoo_bottle, soap_dish, lotion, hand_sanitizer, sponge, facial_cleanser |
| `livingroom_supplies` | `ornaments` | houseplant, watering_can, diffuser_bottle, photo_frame, candle_holder |
| `tools` | `repair_tools` | screwdriver, wrench, hex_key, measuring_tape, pliers, mini_hammer, glue_gun |
| `food` | `fruits` | apple, lemon, banana, pear, strawberry, peach |
| `food` | `ingredients` | potato, tomato, egg, bread, garlic, carrot, chili |
| `food` | `beverages` | juice, milk_carton, canned_coke, beverage_bottle |
| `food` | `snack` | candy, chewing_gum, biscuit, potato_chips, chocolate, cheese_sticks |

**Inference rule:** strip the numeric suffix (`_001`, `_002` …) and match the base name. When ambiguous, prefer the more specific super_category (e.g. `fruit_knife` → `cutting_tools`, not `tools`).

---

## Step 1 — Organize

Write a temporary `scripts/organize_<batch>.py`. Key patterns:

```python
# CATEGORY_MAP: asset_name (without numeric suffix stripped) -> (domain, super_category)
# Infer this mapping from the Taxonomy Reference table above before writing the script.
CATEGORY_MAP = {
    "apple_002":  ("food", "fruits"),
    "coke_003":   ("food", "beverages"),
    "carrot_002": ("food", "ingredients"),
    "biscuit_002":("food", "snack"),
    "mouse_001":  ("desktop_supplies", "electronic_accessories"),
    # ...
}

TYPO_FIXES = {
    "strwberry_002": "strawberry_002",   # fix typos here; source retains typo, dest is clean
}

for src in sorted(SRC_DIR.iterdir()):
    if not src.name.endswith("_usd"):
        continue
    asset_name = src.name[:-len("_usd")]
    dest_name = TYPO_FIXES.get(asset_name, asset_name)
    domain, super_cat = CATEGORY_MAP[asset_name]
    dst = DST_ROOT / domain / super_cat / dest_name
    shutil.copytree(src, dst)
    # rename .usd file if dest_name differs from asset_name
    if dest_name != asset_name:
        (dst / f"{asset_name}.usd").rename(dst / f"{dest_name}.usd")
```

**Typo rule:** correct the name at copy time (rename both folder and `.usd` file). Source retains the typo; destination is clean.

**Cleanup:** After the organize script runs successfully, delete it — it is a one-time throwaway:
```bash
rm scripts/organize_<batch>.py
```

---

## Step 2 — Label

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> \
    python3 tools/asset_pipeline/run_labeller.py \
    --input-dir <dst_root>/ \
    --output-root <dst_root>/ \
    --format usd \
    2>&1 | tee nohup_labeller_<batch>.out
```

**Key flags:**
| Flag | Value | Why |
|------|-------|-----|
| `--input-dir == --output-root` | same dir | writes mesh/renders/urdf into each asset folder |
| `--format usd` | `usd` | triggers `copy_source=True` for textures |
| `CUDA_VISIBLE_DEVICES` | idle GPU | avoid contention; check with `nvidia-smi` first |

**Check idle GPUs first:**
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
```

**GPT API config** (`tools/asset_pipeline/configs/gpt_config.yaml`) must have real credentials. Environment variables `ENDPOINT`, `API_KEY`, `MODEL_NAME` take precedence over the file.

**Bottleneck:** GPT Vision API latency (~2 min/asset), not GPU. Expect ~2 min × N assets.

**AABB side-effect:** for each processed asset the labeller writes a structured
`<aabb><min>x y z</min><max>x y z</max></aabb>` block into the URDF's
`<extra_info>`, computed from the just-written USD via
`pxr.UsdGeom.BBoxCache.ComputeUntransformedBound`. Values are in the asset's
local frame, meters, 6 decimal places. Consumed downstream by
`build_asset_index` (parquet columns `aabb_{x,y,z}_{min,max}`) and by
`PoseResetTerm.auto_z_clearance` to prevent spawn penetration.

## Step 3 — Annotate Interactions

```bash
python3 tools/asset_pipeline/generate_interaction.py <dst_root>/ \
    [--gripper_width 0.09] [--center_mode obb]
```

Writes `interaction.json` next to each `.usd` file. Status is `[OK]` if `passive_pick_body > 0`, `[FAIL]` if all approach directions exceed `gripper_width`.

**FAIL means the object is too large for the gripper, not a bug.** Re-run with larger `--gripper_width` for specific assets if needed:
```bash
python3 tools/asset_pipeline/generate_interaction.py \
    <dst_root>/food/fruits/apple_002/ \
    --gripper_width 0.12
```

---

## Step 4 — Tag (semantic capability tags)

```bash
python3 tools/asset_pipeline/run_tag_labeller.py \
    --asset-root <dst_root>/ \
    --max-workers 4 \
    2>&1 | tee nohup_tag_<batch>.out
```

Reads each URDF's `<extra_info>` fields (text-only, no rendered images), asks GPT to judge each tag in `tools/asset_pipeline/tag_labeller/tags.yaml` (default vocab: `is_container`, `is_graspable`), and writes a `<tags>` element into the URDF.

**Key flags:**
| Flag | Default | Why |
|------|---------|-----|
| `--asset-root` | (required) | Recursively scans for `*.urdf` |
| `--tags-vocab` | `tools/asset_pipeline/tag_labeller/tags.yaml` | Vocab + criteria for GPT |
| `--force` | off | Re-tag URDFs that already have a `<tags>` element |
| `--only-tags A,B` | off | Evaluate only listed tags; preserve existing other tags. Implies `--force` for those. |
| `--max-workers` | 4 | Parallel GPT calls |
| `--dry-run` | off | List URDFs without calling GPT |

**Idempotent by default**: presence of `<tags>` element (even empty `<tags></tags>`) marks an asset as processed; re-runs skip it. Use `--force` to overwrite.

**Adding a new tag**: append an entry to `tag_labeller/tags.yaml` with `description` + `criteria` (use common-sense framing, no robot-specific jargon — see file comment), then re-run with `--only-tags new_tag` to backfill without touching existing tag verdicts.

**Failure handling**: GPT errors / unparseable JSON / unknown vocab keys → URDF left unmodified (so the next run retries automatically); failure recorded as one JSONL row in `<dst_root>/tag_labeller_errors.jsonl` with a UTC `run_id` per run.

---

## Step 5 — Caption candidates

```bash
python3 tools/asset_pipeline/run_caption_labeller.py \
    --asset-root <dst_root>/ \
    --seen-count 15 \
    --unseen-count 5 \
    --max-workers 4 \
    2>&1 | tee nohup_caption_<batch>.out
```

Multimodal: reads each asset's `renders/` images and asks GPT for `seen_count + unseen_count` short noun phrases (color/texture/shape framings) suitable as VLM training labels. Phrases are deduplicated (Jaccard ≥ 0.85 on normalized tokens), then sorted and split deterministically by uuid-seeded RNG into `seen` (training) and `unseen` (held-out) lists. Writes `caption_candidates.json` next to the URDF and adds a `<caption_candidates>` link element to `<extra_info>`.

**Key flags:**
| Flag | Default | Why |
|------|---------|-----|
| `--asset-root` | (required) | Recursively scans for `*.urdf` |
| `--seen-count` | 15 | Phrases per asset written to the `seen` list |
| `--unseen-count` | 5 | Phrases per asset written to the `unseen` list |
| `--force` | off | Regenerate even if `caption_candidates.json` exists |
| `--max-workers` | 4 | Parallel GPT calls |
| `--dry-run` | off | List URDFs without calling GPT |

**Prerequisite**: the asset must have a `renders/` directory from Step 2. URDFs without renders are skipped (logged as error). Re-run Step 2 first if the renders are missing.

---

## Verification

After all five steps, spot-check output completeness (N = total asset count):

```bash
N=$(find <dst_root> -maxdepth 5 -name "*.usd" | wc -l)
echo "Assets:       $N"
echo "URDFs:        $(find <dst_root> -maxdepth 5 -name "*.urdf" | wc -l)  (expect $N)"
echo "interactions: $(find <dst_root> -maxdepth 5 -name "interaction.json" | wc -l)  (expect $N)"
echo "renders:      $(find <dst_root> -maxdepth 5 -type d -name "renders" | wc -l)  (expect $N)"
echo "tags:         $(grep -rl '<tags>' <dst_root> --include='*.urdf' | wc -l)  (expect $N)"
echo "captions:     $(find <dst_root> -maxdepth 5 -name caption_candidates.json | wc -l)  (expect $N)"
echo "aabbs:        $(grep -rl '<aabb>' <dst_root> --include='*.urdf' | wc -l)  (expect $N)"
```

All seven counts should equal N. A mismatch pinpoints which stage failed.

For assets with `passive_pick_body=0`, check `outputs/labeller_runs/label_summary.json` (relative to CWD) for the full per-asset status log.

For tag failures, inspect `<dst_root>/tag_labeller_errors.jsonl` (one JSONL row per failure, tagged with UTC `run_id`).

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Running labeller on GPU 0-3 when occupied | Check `nvidia-smi`, set `CUDA_VISIBLE_DEVICES` to idle GPU |
| GPT API 503 / ProxyError | Check proxy env; real endpoint required in `gpt_config.yaml` |
| Typo in source folder name carried to dest | Add to `TYPO_FIXES`; rename folder AND `.usd` file at copy time |
| `[FAIL]` in generate_interaction summary | Expected for large objects; increase `--gripper_width` if needed |
| Asset name matches multiple super_categories | Prefer more specific one (e.g. `fruit_knife` → `cutting_tools` not `tools`) |
| Re-run of Step 4 silently skips everything | URDFs already have `<tags>` element — add `--force` or `--only-tags X` to override |
| Step 5 fails with "missing renders" | Step 2's renders weren't generated for that asset — re-run Step 2 first |
| New tag added to vocab but old assets don't have it | Run Step 4 with `--only-tags new_tag` to backfill without touching existing tags |
