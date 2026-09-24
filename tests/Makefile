# Put it first so that "make" without argument is like "make test".
PYTEST_SERIAL_OPT := -rf -x
PYTEST_CLUSTER_OPT := -rf -x -n 2 --dist loadgroup
PYTEST_ISAAC_ROOT := python/robo_orchard_sim_ut
PYTEST_AGENT_ROOT := agent

test:
	export ORCHARD_CI=1 && \
	export NV_ASSET_ROOT_DIR=/horizon-bucket/robot_lab2/robo_orchard_workspace/assets/NVIDIA/Assets/Isaac/4.1 && \
	env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
	pytest -x -c pytest.ini $(PYTEST_SERIAL_OPT) python $(PYTEST_AGENT_ROOT)

test-cluster:
	export ORCHARD_CI=1 && \
	export NV_ASSET_ROOT_DIR=/horizon-bucket/robot_lab2/robo_orchard_workspace/assets/NVIDIA/Assets/Isaac/4.1 && \
	env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
	pytest -x -c pytest.ini $(PYTEST_CLUSTER_OPT) \
		--ignore=$(PYTEST_ISAAC_ROOT) python $(PYTEST_AGENT_ROOT)
	export ORCHARD_CI=1 && \
	export NV_ASSET_ROOT_DIR=/horizon-bucket/robot_lab2/robo_orchard_workspace/assets/NVIDIA/Assets/Isaac/4.1 && \
	env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
	pytest -x -c pytest.ini $(PYTEST_SERIAL_OPT) $(PYTEST_ISAAC_ROOT)

.PHONY: test test-cluster
