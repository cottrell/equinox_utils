.PHONY: help install test test-core test-full smoke clean

PYTHON ?= python
PYTEST ?= pytest
ENV := PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

help:
	@sed -n '1,120p' Makefile

install:
	uv pip install -e .

smoke:
	PYTHONPATH=src $(PYTHON) -c "import equinox_utils as eu; print(eu.__file__)"

test: test-core

test-core:
	$(ENV) $(PYTEST) -q src/equinox_utils/test_models.py -k 'tree_serialise_leaves'

test-full:
	$(ENV) $(PYTEST) -q src/equinox_utils/test_models.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
