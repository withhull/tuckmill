#!/bin/sh
# Build the workspace (debug or release) and the native executor stub check.
# Usage: scripts/build.sh [release]
set -eu
cd "$(dirname "$0")/.."

if [ "${1:-}" = "release" ]; then
    cargo build --workspace --release
else
    cargo build --workspace
fi

# Native ABI stub: compiled as a syntax/ABI check only; it is not linked into
# the Rust worker in this build.
if command -v clang++ >/dev/null 2>&1; then
    mkdir -p target/native-check
    clang++ -std=c++17 -Wall -Wextra -Werror \
        -I native/executor \
        -c native/executor/reference_worker_stub.cpp \
        -o target/native-check/reference_worker_stub.o
    echo "native ABI stub compiled cleanly (target/native-check/reference_worker_stub.o)"
else
    echo "clang++ not found; skipping native ABI stub check"
fi
