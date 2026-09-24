# RoboOrchard Server

A lightweight policy server for RoboOrchard evaluation. Run your model in its
own Python environment and connect it to the simulator over WebSocket, on the
same machine or across machines.

```text
Policy environment          WebSocket          Simulation environment
Model + policy server  <------------------>  Evaluation client + Isaac Sim
```

Run the commands below from the repository root in the indicated environment.

## Installation

In your **policy environment**, install the package with Python 3.11 or later:

```bash
python -m pip install ./robo_orchard_server
```

The base package requires only NumPy and `websockets>=12`; install model
dependencies separately. For development, use an editable installation:

```bash
python -m pip install --config-settings editable_mode=strict -e ./robo_orchard_server
```

Restart the server after editing source files; reinstall after adding files.
Prepare the simulation environment and assets using the
[repository installation guide](../README.md#1-installation).

## Evaluate the Holobrain Baseline

### 1. Start the policy server

Prepare `robo_orchard_lab` and the Holobrain dependencies matching your exported
checkpoint (Torch, Pydantic, SciPy, PyYAML, core, and PyTorch3D).
Install the server package as shown above, then run:

```bash
export ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR=/path/to/exported/holobrain

python -m robo_orchard_server.policy.holobrain.main \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8765
```

Prefer setting `ROBO_ORCHARD_HOLOBRAIN_MODEL_DIR` to your exported checkpoint
directory so the same launch command works across machines. For a one-off
override, pass `--model-dir /another/model/path`. Use
`--model-yaml /path/to/config.yaml` to override the [default model
configuration](robo_orchard_server/policy/holobrain/config.yaml).

Match the checkpoint to the task's robot: `franka_panda`, `dualarm_piper`, or
`dualarm_piperx`. The adapter accepts one environment per observation (B=1).

### 2. Configure the evaluation client

In your **simulation environment**, create `holobrain_server.yaml`:

```yaml
policy: server
host: 127.0.0.1
port: 8765
logging_tag: holobrain
remote_policy_type: full
```

Use `127.0.0.1` locally or the server's reachable address remotely.
Match the server port; `0.0.0.0` is only the server's listening address.

Copy the evaluation template:

```bash
cp examples/manipulation-app/configs/eval_example.yaml eval_holobrain.yaml
```

Replace the `policy` section in `eval_holobrain.yaml` with:

```yaml
policy:
  model_type: server
  model_yaml: /absolute/path/to/holobrain_server.yaml
```

Set `model_yaml` to the connection file's absolute path. Keep and adjust
`defaults` and `tasks` for your assets, tasks, and checkpoint's robot.

### 3. Run evaluation

With `ORCHARD_ASSET` and `NV_ASSET_ROOT_DIR` set as described in the installation
guide, run in your **simulation environment**:

```bash
python examples/manipulation-app/scripts/eval_policy.py \
  --eval-config eval_holobrain.yaml \
  --output-dir eval_result/holobrain \
  --gpus 0
```

`--gpus` selects simulation GPUs; the server's `--device` selects the model GPU.
Task results go to `eval_result/holobrain/<task>/`, with the aggregate report at
`eval_result/holobrain/summary.json`. Add `--enable-recording` for MCAP recordings.

## Add Your Own Policy

### 1. Implement the policy interface

Subclass `BasePolicy` and implement synchronous methods:

| Method | Requirement | Purpose |
|---|---|---|
| `reset()` | Required | Clear the session's episode history and action cache. |
| `act(obs)` | Required | Return one `JointAction` with physical joint targets. |
| `act_sequence(obs)` | Recommended for multi-step models | Return a nonempty list of actions in time order; defaults to `[self.act(obs)]`. |
| `close()` | Optional | Release resources when the session closes; defaults to a no-op. |

Save this runnable template as `my_policy.py`. **It returns fixed joint targets
to test connectivity; replace the output with model inference to solve tasks.**

```python
import numpy as np
from robo_orchard_server.server import BasePolicy, PolicyWebsocketServer
from robo_orchard_server.server.types import JointAction, Observation


class MyPolicy(BasePolicy):
    def reset(self) -> None:
        # Clear this session's history and pending actions here.
        pass

    def act(self, obs: Observation) -> JointAction:
        layout = obs["action_layout"]
        joint_names = []
        for slot in layout["manipulator_order"]:
            arm = layout["manipulators"][slot]
            joint_names.extend(arm["arm_joint_names"])
            joint_names.extend(arm["gripper_joint_names"])
        first_slot = layout["manipulator_order"][0]
        batch_size = obs["manipulators"][first_slot]["joint_position"].shape[0]

        # Preprocess obs, run your model, and decode physical joint targets.
        return {
            "joint_names": joint_names,
            "values": np.zeros(
                (batch_size, len(joint_names)), dtype=np.float32
            ),
        }


if __name__ == "__main__":
    PolicyWebsocketServer(
        policy_factory=MyPolicy, host="0.0.0.0", port=8765
    ).run()
```

Observations are dictionaries containing NumPy arrays:

| Field | Content |
|---|---|
| `instruction` | Task instruction, or `None`. |
| `cameras` | Camera slots with RGB and optional depth, intrinsics, and poses. |
| `manipulators` | Arm slots with `joint_position` of shape `[B,N]`. |
| `action_layout` | Robot type, arm order, joint names, and gripper metadata. |

Return `joint_names` covering every arm and gripper joint exactly once, and
finite `float32` `values` of shape `[B,J]`. B must match the observation batch;
column j must match `joint_names[j]`. Targets use radians for revolute joints
and meters for prismatic joints. Your adapter handles normalization, delta or
end-effector conversion, and gripper decoding. Multi-step models should
override `act_sequence()` to reduce requests; every action follows this contract.

See the [type definitions](robo_orchard_server/server/types.py) for all fields
and the [dummy policy](robo_orchard_server/policy/dummy/main.py) for a CLI example.

### 2. Start your server

Run `python my_policy.py` in your **policy environment**.

`policy_factory` creates a new policy per Client session without arguments. Load
weights outside the factory and pass them to new instances. Keep history and
action caches per session; `reset()` and `close()` should retain shared weights.

### 3. Evaluate your policy

Reuse the [evaluation setup](#2-configure-the-evaluation-client) above with your
server's address and port, and choose a new output directory. The evaluation
client still uses `model_type: server`.


## Connect to a Supervisor (reverse mode)

Both the Dummy and Holobrain entry points support `--supervisor-uri`. Without
this option, the existing `--host` / `--port` listening mode is unchanged.
With it, no listening port is opened; `--host` / `--port` are unused.

```bash
TEAM_ID=local SESSION_ID=native-server-001 \
python3 -m robo_orchard_server.policy.dummy.main \
  --supervisor-uri ws://WORKER_IP:8765
```

For Holobrain, use the same connection arguments in the model environment:

```bash
TEAM_ID=YOUR_TEAM SESSION_ID=YOUR_EVALUATION \
python3 -m robo_orchard_server.policy.holobrain.main \
  --model-dir /path/to/exported/model --device cuda:0 \
  --supervisor-uri ws://WORKER_IP:8765
```

Alternatively, pass `--team-id` and `--evaluation-id`; explicit arguments
replace the environment values. `SESSION_ID` is sent as `evaluation_id`.
Both IDs must match the worker's expected strings. Missing IDs are rejected
before model loading. Neither URI nor IDs are written to the new auth log.

The model process connects once and waits for the supervisor to initiate:

```text
supervisor -> model: {"type":"authenticate"}
model -> supervisor: {"type":"authenticate","team_id":"...","evaluation_id":"..."}
supervisor -> model: {"type":"authenticated"}
supervisor -> model: {"type":"policy_protocol","protocol":"sessions-v1"}
model -> supervisor: {"type":"policy_protocol_ready","protocol":"sessions-v1"}
supervisor -> model: routed binary reset / act / act_sequence / close_session
model -> supervisor: routed binary response
```

A policy session is created on the first reset for its `session_id`, after
identity and protocol negotiation. Requests from all Clients arrive over one
WebSocket and are processed strictly in receive order, using independent
factory-created policies. Close requests release individual sessions; socket
closure releases all remaining sessions. No automatic reconnect or replay is
attempted.
Authentication waiting has no separate READY timeout; the supervisor owns its
total walltime and closes the connection on expiry. Transport opening is
limited to 10 seconds and closing to 1 second.

Programmatic usage:

```python
server = PolicyWebsocketServer(policy_factory=create_policy)
server.run_reverse(
    "ws://WORKER_IP:8765", team_id="TEAM", evaluation_id="EVALUATION"
)
```

The Dummy policy returns one action step. Use `--remote-policy-type full` on
the supervisor for this policy. `--expected-steps N` is optional; by default
any valid nonempty action sequence is accepted.

### Single connection, multiple Client sessions

Use server 0.2.0 and supervisor 0.2.0 together. The reverse protocol now requires
`sessions-v1` negotiation and is incompatible with the 0.1.x reverse protocol.
The passive listening protocol is unchanged. The server opens exactly one
connection, regardless of the supervisor's GPU count.

Each routed binary frame retains the canonical length-prefixed JSON header and
tensor blocks. A reserved header field carries routing metadata:

```json
{"__supervisor__":{"protocol":"sessions-v1","session_id":"client-uuid","request_id":"request-uuid"}}
```

The original reset/act/act_sequence fields stay in the same header. Responses
carry the same IDs; tensor block indices and bytes are unchanged. A new
session must start with reset. `close_session` returns `{"ok":true}` after
releasing that session. At most 64 sessions may be active on one connection.
A supplied shared policy must remain stateless; use `policy_factory` for
stateful models. Model weights may be shared by factory-created sessions.

The receive loop processes one complete request before reading the next.
Reset, inference and session cleanup all use the existing model lock. Thus
observations from several Clients may queue on the connection, while actual
model inference remains serial. Supervisor inference timeout includes this
queueing time.
