# RoboOrchard Server

A standalone model-side policy server. Requires Python 3.11+, NumPy and
websockets; install your model dependencies separately.

## Install

Run from the repository root in the model environment:

```bash
python3 -m pip install ./robo_orchard_server
```

For wheel delivery:

```bash
python3 -m pip install build
python3 -m build --wheel ./robo_orchard_server
python3 -m pip install robo_orchard_server/dist/robo_orchard_server-0.2.3-py3-none-any.whl
```

For development, use `python3 -m pip install -e ./robo_orchard_server` and
restart the server after edits.

## Start a dummy server

Client connects directly to the listening server:

```bash
python3 -m robo_orchard_server.policy.dummy.main \
  --host 0.0.0.0 --port 8765 --action-steps 15
```

Or connect the model to a supervisor through the platform WSS URL:

```bash
python3 -m robo_orchard_server.policy.dummy.main \
  --supervisor-uri 'wss://GATEWAY/api/v1/sessions/SESSION/ws' \
  --action-steps 15
```

`--action-steps` defaults to 1 and must be positive. A 15-step response lets the
Client execute 15 actions before sending another observation. Dummy returns
constant joint targets for communication benchmarks. Use 1 for the baseline.
Legacy supervisors requesting authentication also require `--team-id` and
`--evaluation-id` (or `TEAM_ID` and `SESSION_ID`).

## Connect the simulation Client

For direct evaluation, set the server address in
[`robo_orchard_sim/policy/configs/server.yaml`](../robo_orchard_sim/policy/configs/server.yaml)
and use this policy section in your eval YAML:

```yaml
policy:
  model_type: server
  model_yaml: robo_orchard_sim/policy/configs/server.yaml
```

Use the model machine's address for remote connections. Run the existing
`eval_policy.py` with your eval YAML. Supervisor evaluation generates its own
Client connection configuration; use `--remote-policy-type full` for dummy.

## HoloBrain 0.5

Prepare the model dependencies and a checkpoint matching the task robot:

```bash
python3 -m robo_orchard_server.policy.holobrain.main \
  --model-dir /path/to/model --device cuda:0 --host 0.0.0.0 --port 8765
```

Use `--supervisor-uri` instead of listening for reverse connections. The
adapter accepts one environment per observation.

This entrypoint uses the HoloBrain 0.5 pipeline and observation types.
Install the HoloBrain 0.5 model dependencies and use a matching 0.5 export;
legacy HoloBrain exports are no longer supported by this entrypoint.
`--model-dir` must contain the model weights, model config, and the inference
config selected by `inference_prefix` in the policy YAML.

The default policy settings are in
[`config.yaml`](robo_orchard_server/policy/holobrain/config.yaml): `model_dir`
sets the model directory, `inference_prefix` selects the robot's inference
config, `valid_action_step` limits the number of action steps returned per
prediction, and `logging_tag` labels server logs. Use `--model-yaml /path/to/config.yaml`
to load a custom configuration; `--model-dir` overrides its model directory.

## Custom models

Subclass [`BasePolicy`](robo_orchard_server/server/policy.py):

| Method | Contract |
| --- | --- |
| `reset()` | Clear this session's episode state. |
| `act(obs)` | Return one named joint-target action. |
| `act_sequence(obs)` | Optionally return multiple actions in time order. |
| `close()` | Optionally release session resources. |

Observations contain `instruction`, `cameras`, `manipulators` and `action_layout`.
Each action supplies all layout `joint_names` exactly once and finite float32
`values` of shape `[B, J]`; use radians for revolute joints and meters for
prismatic joints. See [types](robo_orchard_server/server/types.py) and the
[dummy implementation](robo_orchard_server/policy/dummy/main.py).

Pass your factory to `PolicyWebsocketServer(policy_factory=create_policy)` and
call `run()` or `run_reverse(uri)`. Each Client session gets its own policy
state; model weights may be shared. Requests are processed serially.
