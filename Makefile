# Makefile for Remote Development on Perlmutter

-include .remote_config

REMOTE_HOST ?= perlmutter.nersc.gov
REMOTE_DIR ?= zeus

.PHONY: help remote-sync remote-build remote-test remote-test-non-null remote-clean remote-shell

help:
	@echo "Available targets:"
	@echo "  remote-sync           : Mirror local changes to Perlmutter using rsync"
	@echo "  remote-build         : Run cmake and make on Perlmutter via SSH"
	@echo "  remote-test          : Run ctest on Perlmutter via SSH"
	@echo "  remote-test-non-null : Build and run non_null tests on Perlmutter"
	@echo "  remote-clean         : Remove build directory on Perlmutter"
	@echo "  remote-shell         : Open an interactive SSH shell on Perlmutter"

remote-sync:
	@echo "🚀 Syncing local files to $(REMOTE_HOST):$(REMOTE_DIR)..."
	rsync -avz --exclude '.git/' --exclude 'build/' --exclude '.remote_config' ./ $(REMOTE_HOST):$(REMOTE_DIR)/

remote-clean:
	@echo "🛠️ Cleaning up on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && rm -rf build"


remote-build:
	@echo "🛠️ Building on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && [ -f remote_env.sh ] && . ./remote_env.sh; mkdir -p build && cd build && cmake -G 'Unix Makefiles' -DZEUS_BUILD_TESTS=ON .. && cmake --build . -j"

remote-test:
	@echo "🧪 Running tests on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && [ -f remote_env.sh ] && . ./remote_env.sh; cd build && ctest --output-on-failure"

remote-test-non-null: remote-sync
	@echo "🧪 Running non_null tests on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && [ -f remote_env.sh ] && . ./remote_env.sh; mkdir -p build && cd build && cmake -G 'Unix Makefiles' -DZEUS_BUILD_TESTS=ON .. && cmake --build . -j && ./non_null_tests"

remote-shell:
	ssh -t $(REMOTE_HOST) "cd $(REMOTE_DIR); bash --login"
