.DEFAULT_GOAL := help

.PHONY: help test

help: ## Show the available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-6s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

test: ## Run the test suite (pytest)
	uv run pytest -q
