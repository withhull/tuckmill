#!/bin/sh
# Fetch the pinned llama.cpp reference build (ADR-005/ADR-006), verify its
# checksum, compile the Tuckmill shim against the pinned headers, and print
# the environment needed to use the resulting library.
#
# The llama.cpp build is a *build dependency*: it is downloaded into .deps/
# (gitignored), never vendored into the repository.
#
# Usage: scripts/fetch-llama.sh
# Then:  export TUCKMILL_LLAMA_LIB="$PWD/.deps/llama/libtm_llama.dylib"
set -eu
cd "$(dirname "$0")/.."

TAG="b11695"
ARCHIVE="llama-${TAG}-bin-macos-arm64.tar.gz"
SHA256="087505ef3ac701704f8807e1c31f848b23a7d00eb8c570a1d4d2e411aa7b304e"
BASE_URL="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}"
RAW_BASE="https://raw.githubusercontent.com/ggml-org/llama.cpp/${TAG}"

DEPS=".deps"
LLAMA_DIR="${DEPS}/llama"

mkdir -p "${DEPS}"

if [ ! -f "${DEPS}/${ARCHIVE}" ]; then
    echo "downloading ${ARCHIVE} (pinned ${TAG})..."
    curl -fsSL -o "${DEPS}/${ARCHIVE}" "${BASE_URL}/${ARCHIVE}"
fi

echo "verifying sha256..."
echo "${SHA256}  ${DEPS}/${ARCHIVE}" | shasum -a 256 -c -

if [ ! -d "${LLAMA_DIR}" ]; then
    echo "extracting..."
    tar xzf "${DEPS}/${ARCHIVE}" -C "${DEPS}"
    mv "${DEPS}/llama-${TAG}" "${LLAMA_DIR}"
fi

echo "fetching pinned headers..."
mkdir -p "${LLAMA_DIR}/include"
curl -fsSL -o "${LLAMA_DIR}/include/llama.h"          "${RAW_BASE}/include/llama.h"
curl -fsSL -o "${LLAMA_DIR}/include/ggml.h"           "${RAW_BASE}/ggml/include/ggml.h"
curl -fsSL -o "${LLAMA_DIR}/include/ggml-cpu.h"       "${RAW_BASE}/ggml/include/ggml-cpu.h"
curl -fsSL -o "${LLAMA_DIR}/include/ggml-backend.h"   "${RAW_BASE}/ggml/include/ggml-backend.h"
curl -fsSL -o "${LLAMA_DIR}/include/ggml-opt.h"       "${RAW_BASE}/ggml/include/ggml-opt.h"
curl -fsSL -o "${LLAMA_DIR}/include/gguf.h"           "${RAW_BASE}/ggml/include/gguf.h"
curl -fsSL -o "${LLAMA_DIR}/include/ggml-alloc.h"     "${RAW_BASE}/ggml/include/ggml-alloc.h"
curl -fsSL -o "${LLAMA_DIR}/include/ggml-cpp.h"       "${RAW_BASE}/ggml/include/ggml-cpp.h"

echo "compiling the Tuckmill shim against the pinned headers..."
clang -O2 -fPIC -shared \
    -o "${LLAMA_DIR}/libtm_llama.dylib" \
    native/llamaref/tm_llama_shim.c \
    -I "${LLAMA_DIR}/include" \
    -L "${LLAMA_DIR}" \
    -lllama \
    -Wl,-rpath,@loader_path

echo
echo "llama.cpp reference build ready (revision ${TAG})."
echo "  export TUCKMILL_LLAMA_LIB=\"$(pwd)/${LLAMA_DIR}/libtm_llama.dylib\""
