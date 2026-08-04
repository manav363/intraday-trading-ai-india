.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install test lint fmt typecheck imports up down clean

help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv
	python3 -m venv .venv && $(PIP) install --upgrade pip

install: venv  ## Install every package and service in editable mode
	$(PIP) install -e "packages/contracts[dev]"
	@for s in services/*/; do \
		[ -f "$$s/pyproject.toml" ] && $(PIP) install -e "$$s[dev]" || true; \
	done

test:  ## Run all tests, excluding ones that need the network
	$(PY) -m pytest -m "not network"

lint:  ## Lint
	$(PY) -m ruff check .

fmt:  ## Format
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck:  ## Type-check
	$(PY) -m mypy packages services

imports:  ## Assert every service imports standalone.
	@# Three lines that catch the entire undeclared-dependency class — the one
	@# that passes every test in a shared venv and dies at container start.
	@for s in services/*/src/*/; do \
		m=$$(basename $$s); \
		$(PY) -c "import $$m" && echo "ok   $$m" || { echo "FAIL $$m"; exit 1; }; \
	done

up:  ## Start the stack
	docker compose -f infra/docker-compose.yml up --build

down:  ## Stop the stack
	docker compose -f infra/docker-compose.yml down

clean:  ## Remove caches and build artefacts
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
