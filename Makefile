# Put it first so that "make" without argument is like "make help".
PIP_ARGS =
BUILD_ARGS =
INSTALL_CONSTRAINTS = scm/install_constraints.txt

help:
	@echo "Available targets:"
	@echo "  make install"
	@echo "  make install-editable"
	@echo "  make install-runtime-constraints"
	@echo "  make dev-env"
	@echo "  make auto-format"
	@echo "  make check-lint"
	@echo "  make type-check"
	@echo "  make test"
	@echo "  make test-cluster"

version:
	@echo "version: $$(cat VERSION)"

install-runtime-constraints:
	python3 -m pip install -c $(INSTALL_CONSTRAINTS) \
		protobuf robo_orchard_schemas $(PIP_ARGS)

install: version
	python3 -m pip install . $(BUILD_ARGS) $(PIP_ARGS)
	$(MAKE) install-runtime-constraints PIP_ARGS='$(PIP_ARGS)'

install-editable: version
	python3 -m pip install --config-settings editable_mode=compat -e . $(BUILD_ARGS) $(PIP_ARGS)
	$(MAKE) install-runtime-constraints PIP_ARGS='$(PIP_ARGS)'

dev-env:
	python3 -m pip install -r scm/requirements.txt $(PIP_ARGS)
	pre-commit install

auto-format:
	ruff check . --fix
	ruff format .

check-lint:
	ruff check .

type-check:
	pyright

test:
	make -C tests

test-cluster:
	make -C tests test-cluster

show-args:
	@echo "PIP_ARGS: $(PIP_ARGS)"
	@echo "BUILD_ARGS: $(BUILD_ARGS)"

.PHONY: help version install install-editable install-runtime-constraints dev-env auto-format check-lint type-check test test-cluster show-args
