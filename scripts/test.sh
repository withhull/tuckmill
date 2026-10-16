#!/bin/sh
# Full verification: formatting, lints at deny-level, and the test suite.
# Usage: scripts/test.sh
set -eu
cd "$(dirname "$0")/.."

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# The CLI end-to-end suite runs the isolated worker against a temporary data
# directory; the supervisor's isolated-worker test additionally runs when it
# is pointed at this binary.
TUCKMILL_TEST_WORKER_BIN="$PWD/target/debug/tuckmill" cargo test -p tuckmill-supervisor isolated_worker_round_trip_when_binary_available

echo "all checks passed"
