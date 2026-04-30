# asset_pipeline

CLI tools for preparing USD assets for the simulator.

For the full label-→-tag-→-caption pipeline see
`.agents/skills/usd-asset-batch-labelling/SKILL.md`. This README documents
**`generate_interaction.py`**, the step that emits per-asset
`interaction.json` annotations.

> **Scope:** `generate_interaction.py` only annotates **pick objects** —
> i.e. it writes `interaction.active.place.body` and
> `interaction.passive.pick.body`. Annotations for **place objects**, the
> `interaction.passive.place.body` field that describes where another
> object can be deposited on top, must be authored manually.

---

## generate_interaction.py

### What it does

Given a USD/USDC asset (or a directory of assets), writes an
`interaction.json` describing where and how the gripper should approach
the object:

- `interaction.active.place.body` — one entry: where to *place* the
  object, given as a local-frame `xyz`, `direction` (placement-down
  axis), and `ref_frame` (in-plane reference axis).
- `interaction.passive.pick.body` — zero or more entries: each is a
  feasible *pick* configuration with `xyz`, approach `direction`,
  `ref_frame`, and the source axis labels (`dir_axis`, `ref_axis`).

A pick triplet `(direction, ref_frame)` is emitted only when the
object's extent along the third orthogonal axis (the axis the gripper
fingers close along) is `<= --gripper_width`. So the output count
reflects how many ways the gripper can grasp the object at the chosen
center.

The output file is placed next to the source USD as `interaction.json`,
unless `--out` points to a specific `.json` path.

### Requirements

- `usd-core` (`pip install usd-core`) — the script imports `pxr`.

### Usage

Single asset, default settings (gripper width 9 cm, OBB center):

```bash
python3 tools/asset_pipeline/generate_interaction.py path/to/asset.usd
```

Whole directory tree (recurses through subdirs for `.usd` / `.usdc`):

```bash
python3 tools/asset_pipeline/generate_interaction.py <dst_root>/
```

Custom gripper width (use this for larger objects that fail the default
9 cm check):

```bash
python3 tools/asset_pipeline/generate_interaction.py \
    <dst_root>/food/fruits/apple_002/ \
    --gripper_width 0.12
```

Pick a different center heuristic:

```bash
python3 tools/asset_pipeline/generate_interaction.py asset.usd \
    --center_mode centroid    # or: obb (default), bbox
```

Write to a specific JSON file instead of the default `<usd_dir>/interaction.json`:

```bash
python3 tools/asset_pipeline/generate_interaction.py asset.usd \
    --out /tmp/my_interaction.json
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `input` (positional, `nargs='+'`) | — | One or more USD/USDC files or directories. |
| `--root_prim` | stage's `defaultPrim` | Target prim path inside the USD stage. |
| `--out` | `.` | Output dir or `.json` path. `.` means write next to each USD. |
| `--gripper_width` | `0.09` | Maximum object extent (m) along the closing axis for a pick to be emitted. |
| `--center_mode` | `obb` | Local-frame center used for both place and pick `xyz`. One of `obb`, `bbox`, `centroid`. |

### Output format

```json
{
    "interaction": {
        "active": {
            "place": {
                "body": [
                    {"xyz": [...], "direction": [...], "ref_frame": [...]}
                ]
            }
        },
        "passive": {
            "pick": {
                "body": [
                    {
                        "xyz": [...],
                        "direction": [...],
                        "ref_frame": [...],
                        "dir_axis": "-z",
                        "ref_axis": "+x"
                    }
                ]
            }
        }
    }
}
```

All vectors are in the root prim's local frame after applying any
ancestor `unitsResolve` transforms (so the output is in meters).

### Per-asset status line

For each USD the script prints one of:

```
[OK]   wrote <out>  (input=..., root=..., meshes=N, passive_pick_body=K)
[FAIL] wrote <out>  (input=..., root=..., meshes=N, passive_pick_body=0)
```

`[FAIL]` means **no pick configuration satisfied `--gripper_width`** —
the object is too wide for the gripper at every approach. This is
expected for large objects; rerun the specific path with a larger
`--gripper_width` if you want a result.

A `[SUMMARY]` line at the end aggregates per-input counts.

---

## Other tools in this directory

- `asset_labeller/`, `caption_labeller/`, `tag_labeller/` — GPT-driven
  labelling stages. Driven by the three `run_*.py` scripts.

See `.agents/skills/usd-asset-batch-labelling/SKILL.md` for how these
fit together end-to-end.

---

## Manual validation

The repository CI does not run tests for this tool because its full workflow
depends on tool-specific packages and runtime services such as GPT, CUDA,
`nvdiffrast`, CoACD, and USD. Install those dependencies first:

```bash
python3 -m pip install -r tools/asset_pipeline/requirements.txt
```

Then run the relevant pipeline step on a small asset sample.
