# Makefile for Remote Development on Perlmutter

-include .remote_config

REMOTE_HOST ?= perlmutter.nersc.gov
REMOTE_DIR ?= zeus

.DEFAULT_GOAL := help

.PHONY: help build format lint test \
	optimize-rosenbrock optimize-rastrigin optimize-ackley \
	optimize-goldstein-price optimize-himmelblau run-examples \
	remote-sync remote-build remote-test remote-test-dual remote-test-non-null remote-clean remote-shell

help:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║                     Zeus Build Targets                         ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "LOCAL DEVELOPMENT:"
	@echo "  make build                - Build Zeus library and examples locally"
	@echo "  make format               - Reformat code (C++, CUDA) using clang-format"
	@echo "  make lint                 - Run all linting checks"
	@echo "  make test                 - Build and run the local test suite"
	@echo ""
	@echo "OPTIMIZATION EXAMPLES (< 1 second each):"
	@echo "  make optimize-rosenbrock       - Run Rosenbrock function optimizer"
	@echo "  make optimize-rastrigin        - Run Rastrigin function optimizer"
	@echo "  make optimize-ackley           - Run Ackley function optimizer"
	@echo "  make optimize-goldstein-price  - Run Goldstein-Price function optimizer"
	@echo "  make optimize-himmelblau       - Run Himmelblau function optimizer"
	@echo "  make run-examples              - Run all optimizer examples"
	@echo ""
	@echo "REMOTE DEVELOPMENT (Perlmutter):"
	@echo "  make remote-sync          - Mirror local changes to $(REMOTE_HOST):$(REMOTE_DIR)"
	@echo "  make remote-build         - Build Zeus on $(REMOTE_HOST) via SSH"
	@echo "  make remote-test          - Run ctest on $(REMOTE_HOST) via SSH"
	@echo "  make remote-test-dual     - Run only [dual] Catch2 tests on $(REMOTE_HOST)"
	@echo "  make remote-test-non-null - Build and run non_null tests on $(REMOTE_HOST)"
	@echo "  make remote-clean         - Remove build directory on $(REMOTE_HOST)"
	@echo "  make remote-shell         - Open interactive SSH shell on $(REMOTE_HOST)"
	@echo ""

build:
	@mkdir -p build && cd build && cmake -G "Unix Makefiles" -DZEUS_BUILD_EXAMPLES=ON -DZEUS_BUILD_TESTS=ON .. && cmake --build .

format:
	@clang-format -i $$(find . -name "*.cpp" -o -name "*.cu" -o -name "*.hpp" -o -name "*.cuh" -o -name "*.h" | grep -v build)

lint:
	@echo "Running linting checks..."

test: build
	@cd build && ctest --output-on-failure

optimize-rosenbrock: build
	@./build/optimize_rosenbrock -5.0 5.0 10 5 1 2 0.01 42 1

optimize-rastrigin: build
	@./build/optimize_rastrigin -5.0 5.0 10 5 1 2 0.01 42 1

optimize-ackley: build
	@./build/optimize_ackley -5.0 5.0 10 5 1 2 0.01 42 1

optimize-goldstein-price: build
	@./build/optimize_goldstein_price -2.0 2.0 10 5 1 2 0.01 42 1

optimize-himmelblau: build
	@./build/optimize_himmelblau -5.0 5.0 10 5 1 2 0.01 42 1

run-examples: optimize-rosenbrock optimize-rastrigin optimize-ackley optimize-goldstein-price optimize-himmelblau
	@echo "✅ All examples completed successfully"

remote-sync:
	@echo "🚀 Syncing local files to $(REMOTE_HOST):$(REMOTE_DIR)..."
	rsync -avz --delete \
	    --filter=':- .gitignore' \
	    --exclude '.git/' \
	    --exclude 'build/' \
	    --exclude '.remote_config' \
	    ./ $(REMOTE_HOST):$(REMOTE_DIR)/

remote-clean:
	@echo "🛠️ Cleaning up on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && rm -rf build"


remote-build:
	@echo "🛠️ Building on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "set -e; cd $(REMOTE_DIR); if [ -f remote_env.sh ]; then . ./remote_env.sh; fi; mkdir -p build; cd build; cmake -G 'Unix Makefiles' -DZEUS_BUILD_TESTS=ON ..; cmake --build . -j"

remote-test:
	@echo "🧪 Running tests on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && [ -f remote_env.sh ] && . ./remote_env.sh; cd build && ctest --output-on-failure"

remote-test-dual:
	@echo "🧪 Running only [dual] tests on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && [ -f remote_env.sh ] && . ./remote_env.sh; cd build && ./unit_test '[dual]'"

remote-test-non-null: remote-sync
	@echo "🧪 Running non_null tests on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && [ -f remote_env.sh ] && . ./remote_env.sh; mkdir -p build && cd build && cmake -G 'Unix Makefiles' -DZEUS_BUILD_TESTS=ON .. && cmake --build . -j && ./non_null_tests"

remote-shell:
	ssh -t $(REMOTE_HOST) "cd $(REMOTE_DIR); bash --login"
