#!/bin/sh
# Full verification: formatting, lints at deny-level, and the test suite.
# Usage: scripts/test.sh
set -eu
cd "$(dirname "$0")/.."

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

# The CLI end-to-end suite runs the isolated worker against a temporary data
# directory; pointing the whole suite at the built binary also enables the
# supervisor's isolated-worker coverage (worker lifecycle, warm reuse).
TUCKMILL_TEST_WORKER_BIN="$PWD/target/debug/tuckmill" cargo test --workspace

echo "all checks passed"
