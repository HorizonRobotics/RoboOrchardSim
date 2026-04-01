# RoboOrchardSim

`robo_orchard_sim` is the simulation repository for RoboOrchard. The Python
package published from this repository is `robo_orchard_sim`, which depends
on `robo_orchard_core` as an external pip package.

## Prerequisites

- Python 3.10
- Access to the package sources required by `isaacsim`, `isaaclab`, and
  `robo_orchard_core`

If your environment does not resolve these dependencies from the default pip
index, configure your internal package source first.

## Install

After cloning the repository, install the package from the repository root:

```bash
git clone <your-repo-url> robo_orchard_sim
cd robo_orchard_sim
python3 -m pip install -e .
```

You can use the repository `Makefile` instead:

```bash
make install-editable
```

## Development Environment

Install development tools with:

```bash
make dev-env
```

Common local commands:

```bash
make check-lint
make type-check
make test
```

Test entry points:

```bash
# Local full test run. Uses serial execution for stability.
make test

# Cluster-oriented full test run. Uses the original pytest-xdist parallelism.
make test-cluster
```
