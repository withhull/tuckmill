#!/bin/sh
# End-to-end demonstration on a temporary data directory. Generates a tiny
# valid GGUF fixture (not a real model), then exercises doctor, inspect,
# import, plan, run, pack, bench, cache, and the serve probe.
#
# Usage: scripts/demo.sh
set -eu
cd "$(dirname "$0")/.."

BIN="$PWD/target/debug/tuckmill"
if [ ! -x "$BIN" ]; then
    cargo build --workspace
fi

WORK="$(mktemp -d)"
export TUCKMILL_DATA_DIR="$WORK/store"
FIXTURE="$WORK/tiny-qwen3.gguf"

echo "### generate a tiny fixture (fixture proves code paths, not model behavior)"
cargo run -q -p tuckmill-manifest --example gen_fixture -- "$FIXTURE"

echo
echo "### doctor"
"$BIN" doctor

echo
echo "### inspect"
"$BIN" inspect "$FIXTURE"

echo
echo "### import"
"$BIN" import "$FIXTURE" --alias local-model

echo
echo "### plan (2GiB budget, 2048 context)"
"$BIN" plan local-model --memory-budget 2GiB --context 2048

echo
echo "### run without the gate (must refuse; this build has no certified engine)"
"$BIN" run local-model --prompt "hello" --max-output 4 || true

echo
echo "### run with --allow-test-engine (synthetic output, isolated worker)"
"$BIN" run local-model --prompt "hello" --max-output 6 --allow-test-engine

echo
echo "### run with --engine tuckcore (experimental executor: real weights, scalar kernels)"
"$BIN" run local-model --prompt "hello core" --max-output 6 --engine tuckcore

echo
echo "### verify: resident vs forced-streaming equivalence (bit-exact logits, evictions proven)"
"$BIN" verify local-model

echo
echo "### pack, bench, cache"
"$BIN" pack local-model
"$BIN" bench local-model --input 64 --output 32 --repetitions 1 --allow-test-engine
"$BIN" bench local-model --input 32 --output 16 --repetitions 1 --engine tuckcore
"$BIN" cache list
"$BIN" cache prune --dry-run

echo
echo "### serve probe (binds loopback, prints token file, exits)"
"$BIN" serve --probe-only --allow-test-engine

echo
echo "demo complete; data dir: $WORK"
