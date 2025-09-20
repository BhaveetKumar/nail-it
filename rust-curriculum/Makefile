SHELL := /bin/bash

.PHONY: all fmt lint check

all: fmt lint

fmt:
	cargo fmt --all

lint:
	cargo clippy --all-targets -- -D warnings || true

check:
	echo "Lessons are standalone snippets; see each lesson for commands"
