# Makefile for Remote Development on Perlmutter

-include .remote_config

REMOTE_HOST ?= perlmutter.nersc.gov
REMOTE_DIR ?= zeus
REMOTE_CTEST_JOBS ?= auto
PERLMUTTER_CTEST_RESOURCE_SPEC ?= ctest-resources.generated.json

define REMOTE_CTEST_SETUP
'set -e' \
'cd $(REMOTE_DIR)' \
'[ -f remote_env.sh ] && . ./remote_env.sh' \
'cd build' \
'CTEST_JOBS="$(REMOTE_CTEST_JOBS)"' \
'if [ "$$CTEST_JOBS" = auto ] || [ -z "$$CTEST_JOBS" ]; then CTEST_JOBS=$$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1); fi' \
'CTEST_RESOURCE_SPEC="$(PERLMUTTER_CTEST_RESOURCE_SPEC)"' \
'VISIBLE_DEVICES="$$CUDA_VISIBLE_DEVICES"' \
'if [ -n "$$VISIBLE_DEVICES" ]; then OLD_IFS="$$IFS"; IFS=,; set -- $$VISIBLE_DEVICES; IFS="$$OLD_IFS"; GPU_COUNT=$$#; else GPU_COUNT=$$(nvidia-smi -L 2>/dev/null | grep -c '^"'"'GPU '"'"' || true); fi' \
'printf '"'"'{\n  "version": {\n    "major": 1,\n    "minor": 0\n  },\n  "local": [\n    {\n      "gpus": [\n'"'"' > "$$CTEST_RESOURCE_SPEC"' \
'gpu_index=0' \
'while [ $$gpu_index -lt $$GPU_COUNT ]; do if [ $$gpu_index -gt 0 ]; then printf '"'"',\n'"'"' >> "$$CTEST_RESOURCE_SPEC"; fi; printf '"'"'        {\n          "id": "%s",\n          "slots": 1\n        }'"'"' "$$gpu_index" >> "$$CTEST_RESOURCE_SPEC"; gpu_index=$$((gpu_index + 1)); done' \
'printf '"'"'\n      ]\n    }\n  ]\n}\n'"'"' >> "$$CTEST_RESOURCE_SPEC"' \
'echo "Generated $$CTEST_RESOURCE_SPEC with $$GPU_COUNT visible GPU(s)."'
endef

.DEFAULT_GOAL := help

.PHONY: help build format lint test \
	optimize-rosenbrock optimize-rastrigin optimize-ackley \
	optimize-goldstein-price optimize-himmelblau run-examples \
	remote-sync remote-build remote-test remote-test-dual remote-test-non-null \
	remote-optimize remote-clean remote-shell

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
	@echo "                             Auto-detects visible GPUs and CPU count unless overridden"
	@echo "  make remote-test-dual     - Run only dual_tests ctest entries on $(REMOTE_HOST)"
	@echo "  make remote-test-non-null - Run only non_null_tests ctest entries on $(REMOTE_HOST)"
	@echo "  make remote-optimize      - Run optimization on $(REMOTE_HOST) and copy results back (compressed)"
	@echo "                             Usage: make remote-optimize FUNC=rosenbrock ARGS=\"...\" FILE=out.tsv"
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


remote-build: remote-sync
	@echo "🛠️ Building on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "set -e; cd $(REMOTE_DIR); if [ -f remote_env.sh ]; then . ./remote_env.sh; fi; mkdir -p build; cd build; cmake -G Ninja -DZEUS_BUILD_TESTS=ON ..; cmake --build . -j8"

remote-test: remote-build
	@echo "🧪 Running tests on $(REMOTE_HOST)..."
	@printf '%s\n' $(REMOTE_CTEST_SETUP) 'ctest -j"$$CTEST_JOBS" --resource-spec-file "$$CTEST_RESOURCE_SPEC" --output-on-failure' | ssh $(REMOTE_HOST) /bin/sh

remote-test-dual: remote-build
	@echo "🧪 Running only [dual] tests on $(REMOTE_HOST)..."
	@printf '%s\n' $(REMOTE_CTEST_SETUP) 'ctest -j"$$CTEST_JOBS" -L '^"'"'^dual_tests$$'"'"' --resource-spec-file "$$CTEST_RESOURCE_SPEC" --output-on-failure' | ssh $(REMOTE_HOST) /bin/sh

remote-test-non-null: remote-build
	@echo "🧪 Running non_null tests on $(REMOTE_HOST)..."
	@printf '%s\n' $(REMOTE_CTEST_SETUP) 'ctest -j"$$CTEST_JOBS" -L '^"'"'^non_null_tests$$'"'"' --resource-spec-file "$$CTEST_RESOURCE_SPEC" --output-on-failure' | ssh $(REMOTE_HOST) /bin/sh

remote-optimize: remote-build
	@if [ -z "$(FUNC)" ] || [ -z "$(ARGS)" ] || [ -z "$(FILE)" ]; then \
		echo "Usage: make remote-optimize FUNC=<name> ARGS=\"<args>\" FILE=<filename>"; \
		exit 1; \
	fi
	@echo "🏃 Running optimize_$(FUNC) on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "set -e; cd $(REMOTE_DIR); if [ -f remote_env.sh ]; then . ./remote_env.sh; fi; cd build && ./optimize_$(FUNC) $(ARGS) --save-trajectories $(FILE) && bzip2 -f $(FILE)"
	@echo "📥 Copying $(FILE).bz2 back to local machine..."
	scp $(REMOTE_HOST):$(REMOTE_DIR)/build/$(FILE).bz2 ./$(FILE).bz2

remote-shell:
	ssh -t $(REMOTE_HOST) "cd $(REMOTE_DIR); bash --login"
