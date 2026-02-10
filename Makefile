# Makefile for Remote Development on Perlmutter

-include .remote_config

REMOTE_HOST ?= perlmutter.nersc.gov
REMOTE_DIR ?= global-optimizer-gpu

.PHONY: help remote-sync remote-build remote-test remote-shell

help:
	@echo "Available targets:"
	@echo "  remote-sync  : Mirror local changes to Perlmutter using rsync"
	@echo "  remote-build : Run cmake and make on Perlmutter via SSH"
	@echo "  remote-test  : Run ctest on Perlmutter via SSH"
	@echo "  remote-shell : Open an interactive SSH shell on Perlmutter"

remote-sync:
	@echo "🚀 Syncing local files to $(REMOTE_HOST):$(REMOTE_DIR)..."
	rsync -avz --exclude '.git/' --exclude 'build/' --exclude '.remote_config' ./ $(REMOTE_HOST):$(REMOTE_DIR)/

remote-build:
	@echo "🛠️ Building on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && [ -f remote_env.sh ] && . ./remote_env.sh; mkdir -p build && cd build && cmake -G 'Unix Makefiles' .. && cmake --build . -j"

remote-test:
	@echo "🧪 Running tests on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && [ -f remote_env.sh ] && . ./remote_env.sh; cd build && ctest --output-on-failure"

remote-test-non-null: remote-sync
	@echo "🧪 Running non_null tests on $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && [ -f remote_env.sh ] && . ./remote_env.sh; mkdir -p build && cd build && cmake -G 'Unix Makefiles' .. && cmake --build . -j && ./non_null_tests"

remote-shell:
	ssh -t $(REMOTE_HOST) "cd $(REMOTE_DIR); bash --login"
