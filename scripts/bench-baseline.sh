#!/bin/sh
# Phase 0 baseline: the same artifact through TuckCore and the pinned
# llama.cpp reference under one harness, plus the pinned llama-bench tool
# and an MLX availability probe. Writes one result directory per run under
# benchmarks/results/<tier>/ (records, environment, summary.md).
#
# Usage: scripts/bench-baseline.sh <model.gguf> [label]
#
# Environment (all optional):
#   ROUNDS=3            paired rounds; engine order alternates (ABBA)
#   INPUT=512 OUTPUT=256 REPS=3   standard-v1 shape per bench invocation
#   CONTEXT=2048        admitted context
#   MEMORY_BUDGET=      process budget passed to every run (e.g. 4GiB;
#                       default: the planner's hardware-adaptive default)
#   KV_PRECISION=       KV cache precision passed to BOTH engines (f32 or
#                       f16; default: each engine's own default). Set it
#                       whenever the engines' defaults differ (dense Qwen3:
#                       TuckCore f32, llama.cpp f16) — identity rules.
#   MIN_RECLAIMABLE_MIB=1536      refuse to start a model process below this
#   SWAP_DISCARD_MIB=64 a run that grows swap by more than this is discarded
#                       (and retried once after a pause; both attempts are listed)
#   TUCKMILL_LLAMA_LIB  pinned shim (default .deps/llama/libtm_llama.dylib)
#   SKIP_LLAMA_BENCH=1  skip the llama-bench vendor-tool row
#
# Rules (benchmarks/README.md): one model process at a time; ABBA ordering;
# the first repetition of each invocation includes worker spawn and load
# and is labeled cold-load, never mixed into warm medians; numbers are
# locally-calibrated smoke evidence, never claims.
set -eu
cd "$(dirname "$0")/.."
ROOT="$PWD"

GGUF="${1:?usage: scripts/bench-baseline.sh <model.gguf> [label]}"
LABEL="${2:-$(basename "$GGUF" .gguf)}"
ROUNDS="${ROUNDS:-3}"
INPUT="${INPUT:-512}"
OUTPUT="${OUTPUT:-256}"
REPS="${REPS:-3}"
CONTEXT="${CONTEXT:-2048}"
MEMORY_BUDGET="${MEMORY_BUDGET:-}"
KV_PRECISION="${KV_PRECISION:-}"
MIN_RECLAIMABLE_MIB="${MIN_RECLAIMABLE_MIB:-1536}"
SWAP_DISCARD_MIB="${SWAP_DISCARD_MIB:-64}"
export TUCKMILL_LLAMA_LIB="${TUCKMILL_LLAMA_LIB:-$ROOT/.deps/llama/libtm_llama.dylib}"
LLAMA_BENCH="$ROOT/.deps/llama/llama-bench"
MLX_PY="$ROOT/.deps/mlx-venv/bin/python"

[ -f "$GGUF" ] || { echo "no such artifact: $GGUF" >&2; exit 2; }
[ -f "$TUCKMILL_LLAMA_LIB" ] || { echo "missing $TUCKMILL_LLAMA_LIB; run scripts/fetch-llama.sh" >&2; exit 2; }

cargo build -q --release -p tuckmill-cli
BIN="$ROOT/target/release/tuckmill"

# ---- tier and identity ----
mem_bytes=$(sysctl -n hw.memsize)
mem_gib=$((mem_bytes / 1073741824))
if [ "$mem_gib" -le 8 ]; then tier="8gb"
elif [ "$mem_gib" -le 24 ]; then tier="16-24gb"
elif [ "$mem_gib" -le 48 ]; then tier="36-48gb"
else tier="64gb-plus"; fi
chip=$(sysctl -n machdep.cpu.brand_string | tr ' ' '-')
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ROOT/benchmarks/results/$tier/$stamp-$chip-$LABEL"
mkdir -p "$OUT/records"

sha=$(shasum -a 256 "$GGUF" | cut -d' ' -f1)
commit=$(git rev-parse HEAD)
dirty=$(git status --porcelain | grep -qv '^??' && echo true || echo false)
cat > "$OUT/environment.json" <<EOF
{
  "schema": "tuckmill-baseline-env-v1",
  "started_utc": "$stamp",
  "tier": "$tier",
  "chip": "$(sysctl -n machdep.cpu.brand_string)",
  "memory_bytes": $mem_bytes,
  "macos": "$(sw_vers -productVersion) ($(sw_vers -buildVersion))",
  "power_source": "$(pmset -g batt | head -1 | sed 's/.*from //; s/'"'"'//g')",
  "tuckmill_commit": "$commit",
  "tuckmill_tree_dirty": $dirty,
  "llama_cpp": "b11695 (pinned, scripts/fetch-llama.sh)",
  "artifact": "$(basename "$GGUF")",
  "artifact_bytes": $(stat -f %z "$GGUF"),
  "artifact_sha256": "$sha",
  "memory_budget": "${MEMORY_BUDGET:-planner default}",
  "kv_precision": "${KV_PRECISION:-engine defaults}",
  "shape": {"input": $INPUT, "output": $OUTPUT, "repetitions": $REPS, "context": $CONTEXT, "rounds": $ROUNDS}
}
EOF

# ---- guards ----
reclaimable_mib() {
    vm_stat | awk '/Pages (free|inactive|speculative|purgeable)/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*16384/1048576}'
}
swap_used_mib() {
    sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1
}
wait_for_memory() {
    tries=0
    while [ "$(reclaimable_mib)" -lt "$MIN_RECLAIMABLE_MIB" ]; do
        tries=$((tries + 1))
        if [ "$tries" -gt 12 ]; then
            echo "reclaimable memory stayed below ${MIN_RECLAIMABLE_MIB} MiB; aborting (close other apps or lower MIN_RECLAIMABLE_MIB)" >&2
            exit 3
        fi
        sleep 5
    done
}
no_model_processes() {
    if pgrep -f "tuckmill __worker|llama-bench|llama-server|mlx_lm" >/dev/null 2>&1; then
        echo "another model process is running; one model process at a time" >&2
        exit 3
    fi
}

DATA="$(mktemp -d)"
trap 'rm -rf "$DATA"; pkill -f "tuckmill __worker" 2>/dev/null || true' EXIT
export TUCKMILL_DATA_DIR="$DATA"
"$BIN" import "$GGUF" --alias bench-model >/dev/null

RUNS="$OUT/runs.tsv"
printf 'round\tslot\tengine\tstatus\treclaimable_mib_before\tswap_delta_mib\trecord\n' > "$RUNS"

run_engine() { # round slot engine; a swap-discarded attempt is retried once
    attempt=1
    while :; do
        run_engine_once "$1" "$2" "$3" "$attempt"
        if [ "$status" != discarded-swap ] || [ "$attempt" -ge 2 ]; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 10
    done
}

run_engine_once() { # round slot engine attempt
    round=$1; slot=$2; engine=$3; attempt=$4
    no_model_processes
    wait_for_memory
    before_mem=$(reclaimable_mib)
    swap0=$(swap_used_mib)
    rec="$OUT/records/r${round}-${slot}-${engine}.json"
    if [ "$attempt" -gt 1 ]; then
        rec="$OUT/records/r${round}-${slot}-${engine}-attempt${attempt}.json"
    fi
    case "$engine" in
        tuckcore-metal) set -- --engine tuckcore --kernels metal ;;
        llama-metal) set -- --engine llama --gpu-layers 99 ;;
    esac
    if [ -n "$MEMORY_BUDGET" ]; then
        set -- "$@" --memory-budget "$MEMORY_BUDGET"
    fi
    if [ -n "$KV_PRECISION" ]; then
        set -- "$@" --kv-precision "$KV_PRECISION"
    fi
    if "$BIN" bench bench-model "$@" --input "$INPUT" --output "$OUTPUT" \
        --repetitions "$REPS" --context "$CONTEXT" --json > "$rec" 2> "$rec.stderr"; then
        status=ok
    else
        status=failed
    fi
    swap1=$(swap_used_mib)
    delta=$((swap1 - swap0))
    if [ "$status" = ok ] && [ "$delta" -gt "$SWAP_DISCARD_MIB" ]; then
        status=discarded-swap
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$round" "$slot" "$engine" "$status" "$before_mem" "$delta" "$(basename "$rec")" >> "$RUNS"
    echo "round $round slot $slot $engine: $status (reclaimable ${before_mem} MiB, swap +${delta} MiB)"
}

r=1
while [ "$r" -le "$ROUNDS" ]; do
    if [ $((r % 2)) -eq 1 ]; then
        run_engine "$r" A tuckcore-metal
        run_engine "$r" B llama-metal
    else
        run_engine "$r" A llama-metal
        run_engine "$r" B tuckcore-metal
    fi
    r=$((r + 1))
done

# ---- vendor-tool row: pinned llama-bench (own prompt tokens; not identity-matched) ----
if [ "${SKIP_LLAMA_BENCH:-0}" != 1 ] && [ -x "$LLAMA_BENCH" ]; then
    no_model_processes
    wait_for_memory
    /usr/bin/time -l "$LLAMA_BENCH" -m "$GGUF" -p "$INPUT" -n "$OUTPUT" -r "$REPS" -ngl 99 -o json \
        > "$OUT/llama-bench.json" 2> "$OUT/llama-bench.stderr" || echo "llama-bench failed (see llama-bench.stderr)"
fi

# ---- MLX probe: only the identical artifact counts (four-way identity) ----
if [ -x "$MLX_PY" ]; then
    HF_HUB_OFFLINE=1 "$MLX_PY" -I "$ROOT/scripts/bench_mlx_probe.py" "$GGUF" > "$OUT/mlx.json" 2> "$OUT/mlx.stderr" || true
else
    echo '{"available": false, "reason": "mlx-lm not installed (.deps/mlx-venv absent)"}' > "$OUT/mlx.json"
fi

python3 -I "$ROOT/scripts/bench_summary.py" "$OUT"
echo "results: $OUT"
