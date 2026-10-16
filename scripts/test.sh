#!/bin/sh
# Full verification: formatting, lints at deny-level, and the test suite.
# Usage: scripts/test.sh
set -eu
cd "$(dirname "$0")/.."

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

# Test labels: portable tests always run; isolated-worker tests (which spawn
# the built binary) are mandatory here and in CI — a missing prerequisite
# fails the gate; real-weight reference tests are machine-local and skip
# without the pinned llama.cpp library and artifacts.
cargo build -p tuckmill-cli
TUCKMILL_TEST_WORKER_BIN="$PWD/target/debug/tuckmill" \
TUCKMILL_REQUIRE_ISOLATED_WORKER=1 \
cargo test --workspace

echo "all checks passed"
