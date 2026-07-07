# mcap → LeRobot v2 converter

Converts `robo_orchard_sim` mcap recordings into
[LeRobot v2 datasets](https://github.com/huggingface/lerobot).

Currently **dualarm_piperx** is fully vali end-to-end.
**franka_panda** preset is wired up (config + features) but not yet
end-to-end tested (available mcap sample is corrupted).

---

## Layout

```
tools/mcap_to_lerobot/
├── config.py           # Pydantic config dataclasses
├── config_presets.py   # DUALARM_PIPERX_CONFIG, FRANKA_PANDA_CONFIG, EMBODIMENT_REGISTRY
├── data_types.py       # RawEpisode, CameraCalib
├── reader.py           # McapEpisodeReader
├── writer.py           # LeRobotEpisodePacker + build_features
├── convert.py          # CLI: discover mcaps → reader → packer
├── verify.py           # CLI: smoke + cross-check verification
└── README.md
```

---

## Feature schema

### dualarm_piperx

| LeRobot key | shape | dtype | notes |
|---|---|---|---|
| `observation.state` | `(14,)` | float32 | `[L_arm×6, L_grip, R_arm×6, R_grip]`; gripper from action topic ×2 |
| `action` | `(14,)` | float32 | same layout as state |
| `observation.velocity` | `(14,)` | float32 | arm from obs topic; gripper from action topic ×2 |
| `observation.effort` | `(14,)` | float32 | same layout |
| `observation.images.{cam}` | `(H, W, 3)` | uint8 | 4 cameras: `static_camera`, `left_hand_camera`, and_camera`, `vis_camera`; video (mp4) or PNG |
| `observation.images.{cam}_depth` | `(H, W, 1)` | uint16 | 16-bit depth PNG (never video) |

Camera calibration (`intrinsic_K`, `intrinsic_P`, `distortion_D`, `extrinsic`) is stored in `meta/info.json` under the key `"cameras"`.

### franka_panda *(preset only, not end-to-end validated)*

| LeRobot key | shape | dtype | notes |
|---|---|---|---|
| `observation.state` | `(8,)` | float32 | `[arm×7, gripper]`; gripper from action topic ×2 |
| `action` | `(8,)` | float32 | same layout |
| `observation.velocity` | `(8,)` | float32 | |
| `observation.effort` | `(8,)` | float32 | |
| `observation.images.{cam}` | `(H, W, 3)` | uint8 | 3 cameras: `ext1_camera`, `ext2_camera`, `wrist_camera` |
| `observation.images.{cam}_depth` | `(H, W, 1)` | uint16 | |

---

## Quick start

### 1 · Smoke test (1 mcap, PNG mode, fast)

```bash
rm -rf /tmp/piperx_smoke
python3 -m tools.mcap_to_lerobot.convert \
    --embodiment dualarm_piperx \
    --mcap-root logs/anymove_figure/pick_attribute_20260529_133352_715/data/episode_0000_seed_1 \
    --out /tmp/piperx_smoke \
    --repo-id robo_orchard/piperx_smoke \
    --no-videos \
    --limit 1
```

### 2 · Full batch (mp4 video mode)

```bash
python3 -m tools.mcap_to_lerobot.convert \
    --embodiment dualarm_piperx \
    --mcap-root logs/anymove_figure/pick_attribute_20260529_133352_715/data \
    --out /path/to/piperx_pick_attribute \
    --repo-id robo_orchard/piperx_pick_attribute
```

### 3 · With a manifest (per-episode instructions)

Manifest format (compatible with `robo_orchard_lab` arrow packer):

```json
[
  {"mcap_path": "logs/.../episode0/env0_data.mcap", "instruction": "Pick the red marker."},
  {"mcap_path": "logs/.../episode1/env0_data.mcap", "instruction": "Pick the blue cup."}
]
```

```bash
python3 -m tools.mcap_to_lerobot.convert \
    --embodiment dualarm_piperx \
    --mcap-root logs/.../data \
    --manifest selected_records.json \
    --out /path/to/dataset \
    --repo-id robo_orchard/piperx_pick
```

**Task priority (D6):** manifest instruction → `/meta_data.instruction` → `--default-task`.

### CLI flags — convert.py

| Flag | Default | Description |
|---|---|---|
| `--embodiment` | `dualarm_piperx` | Robot preset (`dualarm_piperx` or `franka_panda`) |
| `--mcap-root` | *(required)* | Directory scanned recursively for `env*_data.mcap` |
| `--out` | *(required)* | Output LeRobotDataset root (must not exist) |
| `--repo-id` | *(required)* | LeRobot repo_id, e.g. `robo_orchard/my_dataset` |
| `--manifest` | `None` | JSON manifest with per-episode instructions |
| `--default-task` | `manipulation` | Fallback task string |
| `--no-videos` | `False` | Store color images as PNG instead of mp4 |
| `--limit N` | `None` | Process at most N mcaps (smoke tests) |
| `--include-depth` | `False` | Pack per-frame depth as PNG (1 file/frame/cam). Off by default; depth dominates filesystem cost on object-store backends. |
| `--num-workers N` | `1` | Parallel worker processes. >1 fan-outs to shard dirs then merges; `0` or negative = `os.cpu_count()//2`. |
| `--tmp-dir PATH` | `<out>.tmp_shards` | Per-worker shard directory parent. **Must live on same fs as `--out`** so merge can `os.rename`. |
| `--keep-shards` | `False` | Retain shard dirs after merge (debugging). |
| `--best-effort` | `False` | If any mcap fails, merge the successful shards anyway. Exit code 2. |
| `--upload-to URI` | `None` | After merge, `juicefs sync` (or `rsync -a` fallback) the finished dataset. |
| `-v / --verbose` | `False` | DEBUG logging |

---

## Verification

### Smoke check

```bash
python3 -m tools.mcap_to_lerobot.verify \
    --dataset-root /path/to/dataset \
    --repo-id robo_orchard/my_dataset
```

Prints summary stats, samples one episode/frame, validates feature shapes,
and checks camera calibration in `meta/info.json`.

### Cross-check against source mcap

```bash
python3 -m tools.mcap_to_lerobot.verify \
    --dataset-root /path/to/dataset \
    --repo-id robo_orchard/my_dataset \
    --cross-check \
    --mcap-root logs/.../data \
    --embodiment dualarm_piperx \
    --dump-video-sample /tmp/ep0_check.mp4
```

Re-reads the first source mcap and compares frame count, first-frame joint
values (tolerance 1e-4), and task string against the stored dataset.
`--dump-video-sample` additionally renders episode 0 to an mp4 for visual
inspection.

**Exit codes:** `0` = all checks pass · `1` = argument/load error · `2` = at
least one check failed.

---

## Data processing pipeline

```
mcap file
  │
  ├─ _read_buckets()       read all wanted topics into (timestamp, payload) lists
  ├─ _pick_base_time()     master clock = config.master_clock_topic (action arm)
  ├─ _time_sync()          argmin nearest-neighbour align every stream to base_time
  ├─ _filter_frames()      static-frame filter (action Δpos < threshold) + head/tail trim
  ├─ _build_frames()       assemble per-frame dicts:
  │                          obs state/vel/effort, action, color images, depth images
  │                          gripper = action_gripper topic joint1 × gripper_scale (×2)
  └─ _extract_calib()      pull intrinsic K/P, distortion D, extrinsic tf from first frame
```

### Gripper encoding

Following the `robo_orchard_lab` arrow packer convention:
- Both dual-finger joints are **collapsed to 1 scalar** = `joint1.position × 2`.
- This applies to both `observation.state` (gripper uses action topic) and `action`.
- The scale factor `gripper_scale=2.0` is configurable per arm in `JointGroupSpec`.

---

## Adding a new embodiment

1. Add a new `EmbodimentConfig` entry in `config_presets.py` with the correct
   topic strings, `arm_dim`, camera names, and joint name lists.
2. Register it in `EMBODIMENT_REGISTRY`.
3. Run `--limit 1` smoke test to validate feature shapes.
4. Run `verify --cross-check` to confirm alignment.

No changes to `reader.py`, `writer.py`, or `convert.py` are required for
standard embodiments that follow the same topic layout.

---

## Known limitations

- **No parallelism / no resumption.** Single-process, fails-soft per mcap.
  For large batches, wrap `convert.py` with `ProcessPoolExecutor`, writing
  to separate shard directories and merging afterwards.
- **No image rescaling.** `PackConfig.image_scale` is reserved but not yet
  applied; images are stored at full resolution.
- **No `task_status.json` filtering.** Failed-action segments are not removed
  (no `task_status.json` available at time of writing).
- **Franka panda not end-to-end validated.** The available sample mcap is
  corrupted (`RecordLengthLimitExceeded`). The preset and feature schema are
  correct; re-validate once a healthy mcap is available.
- **Depth stored as `dtype=image` (PNG), never video.** LeRobot's image writer
  supports 16-bit PNG via `PIL.Image` passthrough; depth is always lossless.
- **No `push_to_hub`.** Run `lerobot.push_to_hub` separately if needed.
