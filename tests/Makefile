# Put it first so that "make" without argument is like "make test".
PYTEST_SERIAL_OPT := -rf -x
PYTEST_CLUSTER_OPT := -rf -x -n 2 --dist loadgroup

test:
	export ORCHARD_CI=1 && \
	export NV_ASSET_ROOT_DIR=/horizon-bucket/robot_lab2/robo_orchard_workspace/assets/NVIDIA/Assets/Isaac/4.1 && \
	env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
	pytest -x -c pytest.ini $(PYTEST_SERIAL_OPT) python

test-cluster:
	export ORCHARD_CI=1 && \
	export NV_ASSET_ROOT_DIR=/horizon-bucket/robot_lab2/robo_orchard_workspace/assets/NVIDIA/Assets/Isaac/4.1 && \
	env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
	pytest -x -c pytest.ini $(PYTEST_CLUSTER_OPT) python

.PHONY: test test-cluster
