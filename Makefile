# NMN — developer task runner
#
# All targets invoke tools via `python3 -m <tool>` so they work inside a
# virtualenv without relying on globally-installed console scripts. Install the
# dev tooling first with `make install` (or `pip install -e ".[dev]"`).
#
# Run `make help` (the default target) to list everything.

# Use the active interpreter; override with `make PYTHON=python3.11 <target>`.
PYTHON ?= python3
PIP    := $(PYTHON) -m pip

# Source roots used by the linters / formatters / type checker.
SRC  := src/nmn
TEST := tests

.DEFAULT_GOAL := help

.PHONY: help install test test-fast \
        test-torch test-nnx test-linen test-keras test-tf test-mlx \
        lint format format-check typecheck build docs clean

help: ## Show this help (default target)
	@echo "NMN developer commands — usage: make <target>"
	@echo ""
	@echo "Setup"
	@echo "  install        Editable install with dev extras ('.[dev]')"
	@echo ""
	@echo "Testing"
	@echo "  test           Run the full test suite"
	@echo "  test-fast      Run tests excluding slow ones (-m 'not slow')"
	@echo "  test-torch     Run the PyTorch tests        (tests/test_torch/)"
	@echo "  test-nnx       Run the Flax NNX tests        (tests/test_nnx/)"
	@echo "  test-linen     Run the Flax Linen tests      (tests/test_linen/)"
	@echo "  test-keras     Run the Keras tests           (tests/test_keras/)"
	@echo "  test-tf        Run the TensorFlow tests      (tests/test_tf/)"
	@echo "  test-mlx       Run the MLX tests             (tests/test_mlx/)"
	@echo ""
	@echo "Quality"
	@echo "  lint           flake8 errors (E9,F63,F7,F82) + advisory style pass"
	@echo "  format         Auto-format with black + isort"
	@echo "  format-check   Check formatting without writing (black/isort --check)"
	@echo "  typecheck      mypy on the torch and nnx backends"
	@echo ""
	@echo "Build & docs"
	@echo "  build          Build the sdist + wheel (python -m build)"
	@echo "  docs           Where to find the documentation"
	@echo "  clean          Remove build/test/cache artifacts"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Editable install with dev extras
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Testing
#
# Each backend is an independent optional install; only the matching extra has
# to be present for its test target (e.g. `pip install -e ".[dev,torch]"`).
# ---------------------------------------------------------------------------

test: ## Run the full test suite
	$(PYTHON) -m pytest $(TEST)

test-fast: ## Run tests excluding slow ones
	$(PYTHON) -m pytest $(TEST) -m "not slow"

test-torch: ## PyTorch backend tests
	$(PYTHON) -m pytest $(TEST)/test_torch/

test-nnx: ## Flax NNX backend tests
	$(PYTHON) -m pytest $(TEST)/test_nnx/

test-linen: ## Flax Linen backend tests
	$(PYTHON) -m pytest $(TEST)/test_linen/

test-keras: ## Keras backend tests
	$(PYTHON) -m pytest $(TEST)/test_keras/

test-tf: ## TensorFlow backend tests
	$(PYTHON) -m pytest $(TEST)/test_tf/

test-mlx: ## MLX backend tests
	$(PYTHON) -m pytest $(TEST)/test_mlx/

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

# Mirrors CI: errors (E9,F63,F7,F82) fail the build; the wider style pass is
# advisory (--exit-zero). flake8 reads its config from setup.cfg.
lint: ## flake8 errors + advisory style pass
	$(PYTHON) -m flake8 $(SRC) --count --select=E9,F63,F7,F82 --show-source --statistics
	$(PYTHON) -m flake8 $(SRC) --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Auto-format with black + isort
	$(PYTHON) -m black $(SRC) $(TEST)
	$(PYTHON) -m isort $(SRC) $(TEST)

format-check: ## Check formatting without modifying files
	$(PYTHON) -m black --check --diff $(SRC) $(TEST)
	$(PYTHON) -m isort --check-only --diff $(SRC) $(TEST)

typecheck: ## mypy on the torch and nnx backends
	$(PYTHON) -m mypy $(SRC)/torch --ignore-missing-imports
	$(PYTHON) -m mypy $(SRC)/nnx --ignore-missing-imports

# ---------------------------------------------------------------------------
# Build & docs
# ---------------------------------------------------------------------------

build: ## Build sdist + wheel
	$(PYTHON) -m build

docs: ## Pointer to the documentation
	@echo "Guides live under docs/ — start with docs/README.md."
	@echo "Per-framework guides: docs/guides/<framework>.md"
	@echo "The docs website is built with Docusaurus under website/docusaurus/ (npm run start)."

clean: ## Remove build/test/cache artifacts
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml
	rm -rf .pytest_cache/ .mypy_cache/
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
