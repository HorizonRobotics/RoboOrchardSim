Release Notes
#############

This document provides an overview of the main changes in this release.


Version 1.0.0
=============

Highlights
----------

* Published the first stable release of ``robo_orchard_sim``.
* Reorganized the package into clearer benchmark, pipeline, policy, asset, and
  Isaac extension boundaries.
* Added a configuration-driven evaluation workflow with open-source HoloBrain
  policy integration.
* Expanded the benchmark and asset systems for layout-driven tasks and
  reproducible asset selection.


Naming and Package Changes
--------------------------

* Moved Isaac-facing wrappers from ``cfg_wrappers``, ``envs``, ``models``, and
  ``sim`` into the ``robo_orchard_sim.ext`` namespace.
* Renamed ``task_suite`` to ``benchmark`` and ``tasks`` to
  ``task_components``.
* Moved evaluation APIs from ``evaluator`` to ``pipeline.evaluator``.
* Added focused top-level modules for ``asset_manager``, ``benchmark``,
  ``contracts``, ``pipeline``, ``policy``, and ``task_components``.


Evaluation Workflow
-------------------

* Added a unified YAML evaluation configuration for policy settings, shared
  defaults, task lists, asset splits, and batch plans.
* Added single-task and multi-task evaluation with consistent seed handling,
  recording, result output, and summary generation.
* Added HoloBrain policy configuration, observation and action adaptation, and
  integration with the evaluation entry point. HoloBrain is the policy
  integration included in this open-source release.


Benchmark and Asset Support
---------------------------

* Added semantic pick, spatial pick, spatial place, and place-a2b benchmark
  configurations built through the benchmark registry.
* Added layout-driven task construction and per-episode asset pools.
* Added asset registry, resolver, immutable snapshots, and UUID-based splits
  for reproducible asset filtering and sampling.


Migration Notes
---------------

* Update imports to the new ``ext``, ``benchmark``, ``pipeline``, and
  ``task_components`` namespaces.
* Update task definitions to use the YAML and layout-driven configuration
  flow.
* Update asset split entries to use registry UUIDs.
* Run evaluations through
  ``examples/manipulation-app/scripts/eval_policy.py`` with a unified
  evaluation configuration.


Version 0.1.0
=============

Highlights
----------

* Added a data synthesis and collection pipeline based on atomic action
  executors, allowing pick-and-place trajectories to be generated directly from
  registered tasks and exported as recorded data.
* Added the ``tools/asset_pipeline`` asset labelling toolkit, covering tag
  generation, interaction pose generation, semantic tag completion, and
  caption generation.
* Refactored the validator and instruction pipelines so evaluation metadata,
  task instructions, and asset descriptions are all based on runtime asset
  snapshots and caption asset information.
* Introduced ``AssetRegistry``, ``AssetResolver``, ``AssetFilter``, and
  ``AssetSplits`` to support declarative asset filtering from the asset library
  by taxonomy, attributes, tags, and split rules.


New Features
------------

Data Synthesis
~~~~~~~~~~~~~~

* Added the ``robo_orchard_sim/task_components/trajs_gen`` atomic action framework, which
  provides ``AtomicActionManager``, ``BaseExecutor``, and executors such as
  ``PickExecutor``, ``PlaceExecutor``, ``MoveExecutor``, ``GripperExecutor``,
  and ``BackToDefaultExecutor`` for generating pick-and-place trajectories.
* Added a default atomic action plan in
  ``robo_orchard_sim/benchmark/manipulation/place_a2b/action_plan.py`` that
  binds executors to the left and right arms according to task state.
* Added
  ``examples/manipulation-app/scripts/collect_data_example.py``, which can
  assemble an environment, execute an action plan, and record data directly
  from the task registry, task YAML, and asset library resolver.
* ``examples/manipulation-app/scripts/eval_policy.py`` and ``Evaluator`` now
  support ``--asset-root``, ``--config``, ``--enable-recording``, and
  ``--record-dir`` arguments so the evaluation flow can reuse the same task
  assembly and recording path.


Asset Labelling Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

* Added the ``tools/asset_pipeline`` directory with standalone dependencies,
  a README, and multi-stage CLI entry points for asset processing.
* Added ``run_labeller.py``, ``run_tag_labeller.py``, ``run_caption_labeller.py``,
  and ``generate_interaction.py`` for asset label generation, capability tag
  generation, caption generation, and interaction annotation generation.
* The ``asset_labeller`` submodule now includes mesh processing, rendering,
  CoACD decomposition, and GPT integration to fill in structured labels for raw
  USD assets.
* ``generate_interaction.py`` generates ``interaction.active.place.body`` and
  ``interaction.passive.pick.body`` for pick objects. The
  ``interaction.passive.place.body`` field for place objects still needs manual
  completion.


Validator and Instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``Validator`` now accepts runtime ``ValidatorActor`` snapshots, and actor
  metadata preserves ``uuid``, ``category``, ``actor_type``, and initial and
  final poses.
* Validator criteria now support ``env_idx``, which is a better fit for
  multi-environment evaluation and per-environment metric aggregation.
* ``Evaluator`` now writes episode metadata in the recording flow, including
  task success, progress, actor metadata, initial and final poses, and
  instruction text.
* The instruction system has been refactored into a template registry plus
  caption-driven descriptions. ``InstructionWrapper`` now supports
  ``template_mode = fixed | variants`` and
  ``actor_description_mode = raw | seen | unseen``, with descriptions sourced
  from each asset's ``caption_candidates.json``.
* Added three semantic pick task types: ``pick_category``, ``pick_attribute``,
  and ``pick_disambiguation``. They now all use YAML task definitions, caption
  descriptions, and registry-backed instruction configuration.


Asset Library Filtering
~~~~~~~~~~~~~~~~~~~~~~~

* Added ``AssetRegistry``, which can automatically generate
  ``asset_index.parquet`` the first time it points to a new asset library root,
  and rebuild the index automatically when the schema version changes.
* The asset index now consistently includes taxonomy, color, shape, material,
  size and mass ranges, path information, capability tags, and generation
  provenance.
* ``AssetFilter`` now supports ``tags`` AND matching, plus filtering by
  ``super_category``, ``category``, ``color``, ``shape``, ``material``,
  ``size_bucket``, ``only_in``, and ``exclude``.
* ``AssetResolver`` now supports declarative ``target`` and ``distractor``
  configurations, along with relative sampling rules using ``anchor`` plus
  ``match`` or ``differ``.
* Added an ``AssetSplits`` YAML loader with support for ``seen``,
  ``unseen_category``, and ``unseen_instance`` benchmark splits.
* Task YAML and example scripts can now filter assets declaratively through
  ``asset_configs`` instead of hard-coding object definitions in code.


Improvements
------------

Task Assembly and Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``TaskDefinition`` has been refactored into a strongly typed YAML assembly
  flow, standardizing how ``scene``, ``embodiment``, ``instruction``,
  ``asset_configs``, and ``task.params`` are loaded.
* ``examples/manipulation-app/scripts/simple_orchard_env_example.py`` now acts
  as a generic registry-driven example. Use ``--task`` to select a task,
  ``--config`` to override the default YAML, and ``--asset-root`` to point to
  the asset library.
* The ``place_a2b`` task configuration has been split into
  ``place_a2b_easy.yaml`` and ``place_a2b_hard.yaml``, and both now follow the
  same configuration style as the semantic pick tasks.


Migration Notes
---------------

* Tasks, evaluation flows, and examples that depend on the asset library now
  require ``--asset-root`` explicitly, or the ``ORCHARD_ASSET_LIBRARY``
  environment variable.
* The first run of a registry-backed task will generate ``asset_index.parquet``
  in the asset library directory. Before releasing, confirm that the target
  asset library contains the required ``.urdf``, ``interaction.json``, and
  ``caption_candidates.json`` files.
* The legacy ``instruction.json`` for ``place_a2b`` has been replaced by a YAML
  ``instruction`` block. New fields include ``template``, ``template_mode``,
  and ``actor_description_mode``.
* The asset labelling pipeline depends on additional runtime components such as
  GPT, CUDA, ``nvdiffrast``, CoACD, and USD. The current repository CI does not
  cover the full pipeline, so manual validation on a small asset sample is
  recommended before release.
