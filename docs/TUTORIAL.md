# Tutorial: Define Your Own `env_task` and Run Evaluation

This tutorial shows the shortest supported path to do two things:

1. define a new user-facing `env_task`
2. run an evaluation on that task

If you are new to this repository, use `place_a2b` as the reference implementation. The core idea is simple:

```text
TaskDefinition -> OrchardEnv -> Evaluator -> EvaluationResult
```

In other words, you first make your task build a valid `OrchardEnv`, then you point the evaluator at that task by name.

## What You Will Build

From a user point of view, a runnable `env_task` needs four pieces:

- a `TaskDefinition` with a unique `namespace`
- a `build()` path that returns an `OrchardEnv`
- runtime registration so the task can be found by name
- a task validator so evaluation can report success and progress

The reference files are:

- task entry: [place_a2b_env.py](../robo_orchard_sim/task_suite/manipulation/place_a2b/place_a2b_env.py)
- task registration: [registration.py](../robo_orchard_sim/task_suite/registration.py)
- runtime bootstrap: [registry.py](../robo_orchard_sim/task_suite/registry.py)
- evaluator entry: [eval_policy.py](../examples/manipulation-app/scripts/eval_policy.py)

## Part 1: Define a New `env_task`

Start by adding a new task definition under `robo_orchard_sim/task_suite/...`. The repository already does this for `place_a2b`, which is registered with `@register_task` and builds an `OrchardEnv` explicitly in [place_a2b_env.py](../robo_orchard_sim/task_suite/manipulation/place_a2b/place_a2b_env.py).

### Step 1: Create a `TaskDefinition`

Your task definition is the user-facing entry point. It gives the task a name and returns the default `OrchardEnv`.

```python
from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
from robo_orchard_sim.task_suite.base import TaskDefinition
from robo_orchard_sim.task_suite.registration import register_task


@register_task
class MyTaskDefinition(TaskDefinition):
    namespace = "my_task"
    config_path = "my_task.yaml"

    @classmethod
    def build(cls) -> OrchardEnv:
        task = ...
        return OrchardEnv(
            scene=cls.resolve_scene(),
            embodiment=cls.resolve_embodiment(),
            task=task,
        )
```

What matters here:

- `namespace` is the task name users pass into evaluation
- `build()` is required; `TaskDefinition` does not provide a default implementation
- `resolve_scene()` and `resolve_embodiment()` are helper methods you can call inside `build()`
- `config_path` can point to a YAML file that provides `scene`, `embodiment`, and optional `instruction` config
- the task object you construct inside `build()` must be a `TaskBase` with task-specific assets and logic
- `@register_task` adds the class to the task registry in [registration.py](../robo_orchard_sim/task_suite/registration.py)
- `scene` supports either a registered string name or a `SceneBase` instance; `embodiment` supports either a registered string name or an `EmbodimentBase` instance

Note: `TaskDefinition` currently parses `instruction` config, but the current
`place_a2b` build path does not pass instruction into `OrchardEnv`, so treat it
as optional task-definition metadata unless your own task wires it through
explicitly.

The current `TaskDefinition` YAML shape is:

```yaml
scene:
  type: plane_table
  num_envs: 1
  env_spacing: 2.5
  physics_fps: 600
  render_fps: 30
  step_fps: 30
  params: {}

embodiment:
  type: dualarm_piper
  initial_pos: [0.0, 0.3, 0.0]
  params: {}

instruction:
  template: place_a2b_default
  template_mode: raw
```

### Step 2: Assemble the `OrchardEnv`

An `OrchardEnv` is built from:

- `scene`: world layout and simulator timing
- `embodiment`: robot definition
- `task`: task-specific assets, reset logic, and validator

The reference composition is in [place_a2b_env.py](../robo_orchard_sim/task_suite/manipulation/place_a2b/place_a2b_env.py). For your first custom task, the safest path is to reuse an existing scene and embodiment, and only change the task-specific assets and validator logic.

### Step 3: Implement Reset and Validation Logic

The task object is where evaluation becomes meaningful. In the `place_a2b` example, [place_a2b_task.py](../robo_orchard_sim/orchard_env/tasks/place_a2b_task.py) defines:

- `get_event_cfg()`: how objects are reset at episode start
- `build_validator()`: how success and progress are measured

A minimal pattern looks like this:

```python
class MyTask(TaskBase):
    def get_event_cfg(self) -> EventManagerCfg:
        return EventManagerCfg(
            terms={
                "random_object_pose": ...,
            }
        )

    def build_validator(self) -> Validator:
        return Validator(
            actors=["target_object"],
            criteria=[
                ...,
            ],
            criteria_name=[
                ...,
            ],
        )
```

If `build_validator()` is missing or too weak, the evaluator can still run, but the result will not reflect the task you actually care about.

### Step 4: Register the Task for Runtime Lookup

Defining the class is not enough. The evaluator resolves tasks by name at runtime, so your module must also be imported during bootstrap.

Update [_bootstrap_task_definitions()](../robo_orchard_sim/task_suite/registry.py) in [registry.py](../robo_orchard_sim/task_suite/registry.py):

```python
def _bootstrap_task_definitions() -> None:
    from robo_orchard_sim.task_suite.manipulation import (
        my_task as _my_task,
        place_a2b as _place_a2b,
    )

    del _my_task
    del _place_a2b
```

This step is required because:

- `@register_task` fills the in-memory registry
- `_bootstrap_task_definitions()` makes sure your module is imported so the decorator actually runs

If you forget this import, evaluation will fail with `Unknown task name`.

### Step 5: Do a Quick Env Sanity Check

Before running a full evaluation, make sure your task can at least build, reset, and step. The reference script is the `place_a2b` example [simple_orchard_env_example.py](../examples/manipulation-app/scripts/simple_orchard_env_example.py).

```bash
python3 examples/manipulation-app/scripts/simple_orchard_env_example.py
```

For your own task, the equivalent local sanity check is:

- build the task
- call `to_isaac_env_cfg()`
- open the env
- run `reset()`
- run a few `step()` calls

Do this before evaluation. It is much easier to debug env construction failures here than inside a rollout loop.

For the current `place_a2b` reference, the example script simply calls
`PlaceA2BTaskDefinition.build()`. It does not override scene/task parameters in
the script itself.

## Part 2: Plug In Your Own Model

Once the task side is ready, the next question is how to run evaluation with your own policy instead of the built-in dummy policy. The repository already shows the expected interface in [DummyPolicy.py](../robo_orchard_sim/policy/DummyPolicy.py).

### Step 1: Implement a Policy Class

Your runtime policy class should inherit from `PolicyMixin`. The required methods are:

- `reset(...)`: clear recurrent state or per-episode caches
- `act(obs)`: map one observation payload to one action payload

Minimal structure:

```python
from typing import Any

import gymnasium as gym
import torch
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType


class MyPolicy(PolicyMixin[dict[str, Any], dict[str, torch.Tensor]]):
    def __init__(
        self,
        cfg: "MyPolicyCfg",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.model = ...

    def reset(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def act(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "left_robot_joint_position": ...,
            "left_robot_gripper_control": ...,
            "right_robot_joint_position": ...,
            "right_robot_gripper_control": ...,
        }


class MyPolicyCfg(PolicyConfig[MyPolicy]):
    class_type: ClassType[MyPolicy] = MyPolicy
    checkpoint: str
```

Use `PolicyConfig` when you want the evaluator to construct your policy from a config object. This is the same pattern used by `DummyPolicyCfg`.

At runtime, the evaluator calls `policy.reset()` at the start of each episode,
so any recurrent state or cached rollout state should be cleared there.

### Step 2: Match the Observation Interface

Your `act(obs)` implementation receives the environment observation dictionary. In the current dual-arm example, the robot observation group includes keys such as:

- `obs["/robot"]["left_joint_position"]`
- `obs["/robot"]["right_joint_position"]`
- `obs["/tf"][...]`
- optionally `obs["/camera"][...]` when cameras are enabled

The exact observation groups come from the embodiment and task config. For the reference dual-arm robot, they are defined in [dualarm_piper/embodiment.py](../robo_orchard_sim/orchard_env/embodiments/dualarm_piper/embodiment.py).

### Step 3: Match the Action Interface

The action payload must match the environment action terms. For the current `DualArmPiperEmbodiment`, [get_action_cfg()](../robo_orchard_sim/orchard_env/embodiments/dualarm_piper/embodiment.py) defines four action keys:

- `left_robot_joint_position`: shape `(batch, 6)`
- `left_robot_gripper_control`: shape `(batch, 2)`
- `right_robot_joint_position`: shape `(batch, 6)`
- `right_robot_gripper_control`: shape `(batch, 2)`

This is why the reference dummy policy returns:

```python
{
    "left_robot_joint_position": torch.tensor([[...]]),
    "left_robot_gripper_control": torch.tensor([[...]]),
    "right_robot_joint_position": torch.tensor([[...]]),
    "right_robot_gripper_control": torch.tensor([[...]]),
}
```

In other words, the current action mode is joint-position control, expressed as a dictionary of named action tensors. If you change the embodiment or task action config, your policy output must change with it.

### Step 4: Replace the Dummy Policy in Evaluation

The example script currently builds:

```python
policy = DummyPolicyCfg()
```

To deploy your own model, replace that line with your config:

```python
policy = MyPolicyCfg(checkpoint="checkpoints/my_model.pt")
```

Then run:

```bash
python3 examples/manipulation-app/scripts/eval_policy.py \
  --task-name my_task \
  --episode-num 3 \
  --max-steps 100 \
  --output eval_result/isaac_eval/my_model_eval.json
```

If your model needs extra assets such as checkpoints, tokenizers, or normalization stats, load them inside your policy class so evaluation stays a single-entry workflow.

## Part 3: Run an Evaluation

Once your task can build and step, and your policy interface is ready, evaluation is straightforward: pass the task name into `EvaluatorCfg`, run several episodes, and save the aggregated result.

### Step 1: Point the Evaluator at Your Task

The example script [eval_policy.py](../examples/manipulation-app/scripts/eval_policy.py) builds an evaluator like this:

```python
evaluator_cfg = EvaluatorCfg(
    task_name="my_task",
    launch=LaunchConfig(
        headless=True,
        enable_cameras=True,
        virtual_display=False,
    ),
    seed=0,
    episode_num=3,
    max_steps=100,
)
```

The main field you must change is `task_name`. It must match the `namespace` you registered earlier.

### Step 2: Run the Evaluation Loop

The evaluator implementation is in [evaluator.py](../robo_orchard_sim/evaluator/evaluator.py). At runtime it will:

1. resolve your task by `task_name`
2. build the `OrchardEnv`
3. convert it to an Isaac env config
4. open the env
5. run the policy for `episode_num` episodes
6. call the task validator every step
7. aggregate the final `EvaluationResult`

The per-episode loop in [_run_episode()](../robo_orchard_sim/evaluator/evaluator.py) stops when one of these happens:

- validator reports success
- environment terminates
- environment truncates
- `max_steps` is reached

### Step 3: Run the Example Command

If you temporarily wire `eval_policy.py` to your task, you can run:

```bash
python3 examples/manipulation-app/scripts/eval_policy.py \
  --task-name my_task \
  --seed 0 \
  --episode-num 3 \
  --max-steps 100 \
  --output eval_result/isaac_eval/my_task_eval.json
```

By default, the example script uses `DummyPolicyCfg`, so this is best treated as a pipeline check:

- can the task be found by name
- can the env launch successfully
- can the validator produce metrics
- can results be serialized to JSON

### Step 4: Read the Output

The script writes a serialized `EvaluationResult` JSON. The evaluator computes these top-level fields in [evaluate()](../robo_orchard_sim/evaluator/evaluator.py):

- `episode_num`
- `seed_start`
- `success_rate`
- `average_progress`
- `episode_results`

If the output file exists and those fields look reasonable, your evaluation has completed the basic path successfully.

## Common Failure Points

When a new task does not evaluate correctly, the problem is usually one of these:

- `task_name` does not match `namespace`
- the task file was decorated with `@register_task` but never imported in `registry.py`
- `build()` does not return a valid `OrchardEnv`
- reset events are invalid, so the episode starts in a broken state
- validator logic does not match the task, so progress or success never updates

In practice, fix env construction and reset issues first, then fix evaluation logic.

## Summary

To add your own `env_task`, define a `TaskDefinition`, return an `OrchardEnv`, register it, and verify that the env can reset and step. To run an evaluation, pass that task name into `EvaluatorCfg` or `--task-name`, execute the rollout loop, and inspect the serialized `EvaluationResult`.

To deploy your own model, implement a `PolicyMixin`, optionally wrap it in a `PolicyConfig`, and make sure `act(obs)` returns action tensors that exactly match the action terms of the target embodiment. If you want the fastest path, copy the structure of `place_a2b` and `DummyPolicy`, change only the task-specific and model-specific pieces first, and keep your first evaluation focused on end-to-end correctness rather than policy quality.
