# Holobrain Policy

This document describes how to use the `holobrain` policy in
`robo_orchard_sim`.

## Environment Setup

Use the following Docker image for holobrain evaluation:

- `hub.hobot.cc/auto/robot_lab/mengao.zhao:ubuntu22.04-gcc11.4-py3.11-cuda12.8-isaac_lab-v2.0.2-sem-ext-v0.2`

## Export Model

The `holobrain` model export flow is maintained in `RoboOrchardLab`.

Reference:

- `https://github.com/HorizonRobotics/RoboOrchardLab/tree/master/projects/holobrain`
- Section: **5. Export Model and Processors and Pipeline**

After export, make sure the exported model directory is accessible from the
runtime environment where evaluation will run.

In this repository, the model directory can be provided in either of the
following ways:

1. Set `model_dir` in `robo_orchard_sim/policy/configs/holobrain.yaml`
2. Export `ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR`

Example:

```bash
export ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR=/path/to/exported/holobrain_model
```

## Evaluation Config

The evaluation entrypoint reads an eval config YAML. Use the example below as
the starting point:

- `examples/manipulation-app/configs/eval_example.yaml`

The key fields are:

- `policy.model_type`: policy type, for example `holobrain`
- `policy.model_yaml`: optional policy config path
- `defaults`: shared task defaults
- `tasks`: evaluation tasks

Recommended policy config:

```yaml
policy:
  model_type: holobrain
  model_yaml: robo_orchard_sim/policy/configs/holobrain.yaml
```

If `policy.model_yaml` is omitted, the default config is resolved from
`robo_orchard_sim/policy/configs/{model_type}.yaml`. For `holobrain`, this is
`robo_orchard_sim/policy/configs/holobrain.yaml`.

Recommended workflow:

1. Copy `examples/manipulation-app/configs/eval_example.yaml` to a new file
2. Set `policy.model_type` to `holobrain`
3. Optionally set `policy.model_yaml` to
   `robo_orchard_sim/policy/configs/holobrain.yaml`
4. Update the task entries for your evaluation target

## Run Evaluation Locally

This mode runs `examples/manipulation-app/scripts/eval_policy.py` directly in
the current Docker environment.

Before running:

1. Start the Docker container
2. In the container, start a virtual display
3. Export the model by following the reference above
4. Copy `examples/manipulation-app/configs/eval_example.yaml` to your own eval
   config file
5. Set `policy.model_type: holobrain`
6. Export `PYTHONPATH` to include `robo_orchard_lab`
7. Export `NV_ASSET_ROOT_DIR`
8. Set `ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR`, or update `model_dir` in
   `robo_orchard_sim/policy/configs/holobrain.yaml`

Start the virtual display in the container:

```bash
Xvfb :<id> -screen 0 1920x1200x24 -ac +extension GLX +render -noreset &
x11vnc -display :<id> -forever -bg -ncache
```

Here `<id>` is the X display id used by both `Xvfb` and `DISPLAY`.

Example:

```bash
export PYTHONPATH=path/to/robo_orchard_lab:$PYTHONPATH
export NV_ASSET_ROOT_DIR=/horizon-bucket/robot_lab/assets/NVIDIA/Assets/Isaac/4.1
export ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR=/path/to/exported/holobrain_model

DISPLAY=:<id> python3 examples/manipulation-app/scripts/eval_policy.py \
  --eval-config path/to/holobrain_eval.yaml \
  --output-dir eval_result/holobrain_local_run \
  --gpus 0 \
  --enable-recording
```

## Run Evaluation on Cluster

This mode submits evaluation jobs through the SCM submit config:

- `examples/manipulation-app/scripts/scm/submit_eval_policy.json`

Related reference:

- `examples/manipulation-app/scripts/scm/README.md`

Submit with:

```bash
RoboOrchardJob-AIDISubmit submit_from_config \
  --config examples/manipulation-app/scripts/scm/submit_eval_policy.json
```

Before submitting:

1. Prepare an eval config derived from
   `examples/manipulation-app/configs/eval_example.yaml`
2. Ensure the eval config uses `policy.model_type: holobrain`
3. Ensure `gpu_per_worker`, `--gpus`, and the number of tasks are consistent
