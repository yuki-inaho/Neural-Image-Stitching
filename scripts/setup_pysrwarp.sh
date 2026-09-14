#!/usr/bin/env bash
# Build and install pysrwarp (https://github.com/sanghyun-son/pysrwarp) into the active pixi env.
#
# The upstream Makefile hardcodes `NVCC = /usr/local/cuda-11.2/bin/nvcc` and
# `-gencode=arch=compute_75`, so we override both from the host.
# CUDA >= 12 CUB headers declare template parameters named NUM_THREADS; the
# kernel sources define a macro with the same name, so it is renamed first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${PYSRWARP_DIR:-$REPO_ROOT/.cache/pysrwarp}"
COMMIT="${PYSRWARP_COMMIT:-803125536675a22add19eb31a7e0c296f60b0ff2}"
REPO_URL="https://github.com/sanghyun-son/pysrwarp.git"

if [ ! -d "$SRC_DIR/.git" ]; then
    mkdir -p "$(dirname "$SRC_DIR")"
    git init -q "$SRC_DIR"
    git -C "$SRC_DIR" remote add origin "$REPO_URL"
fi

echo "==> fetching pysrwarp $COMMIT"
git -C "$SRC_DIR" fetch -q --depth 1 origin "$COMMIT"
git -C "$SRC_DIR" checkout -q --force --detach FETCH_HEAD

echo "==> renaming NUM_THREADS macro (CUB collision on CUDA >= 12)"
sed -i 's/\bNUM_THREADS\b/SVF_NUM_THREADS/g' "$SRC_DIR"/cuda/*.cu "$SRC_DIR"/cuda/*.cuh

# Absolute library_dirs so the built C++ extension records a resolvable path
# to the three pre-compiled CUDA kernel libraries.
sed -i "s|library_dirs=\[path.join('.', target_dir)\],|library_dirs=[path.abspath(path.join('.', target_dir))],|" \
    "$SRC_DIR/setup.py"

if [ -z "${NVCC:-}" ]; then
    if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
        NVCC="$CUDA_HOME/bin/nvcc"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        NVCC=/usr/local/cuda/bin/nvcc
    else
        NVCC="$(command -v nvcc || true)"
    fi
fi
if [ -z "$NVCC" ] || [ ! -x "$NVCC" ]; then
    echo "error: nvcc not found. Set CUDA_HOME or NVCC to a CUDA toolkit >= 11." >&2
    exit 1
fi
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}"

CAP="${PYSRWARP_CUDA_ARCH:-}"
if [ -z "$CAP" ]; then
    CAP="$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}.{minor}")
except Exception:
    pass
PY
)"
fi
if [ -z "$CAP" ]; then
    CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"
fi
if [ -z "$CAP" ]; then
    CAP="7.5"
fi
ARCH="${CAP/./}"

echo "==> building CUDA kernels with $NVCC for sm_$ARCH (CUDA_HOME=$CUDA_HOME)"
make -C "$SRC_DIR/cuda" clean
make -C "$SRC_DIR/cuda" \
    NVCC="$NVCC" \
    INCFLAGS="-I$CUDA_HOME/include" \
    CUDAFLAGS="-shared -O2 -gencode=arch=compute_${ARCH},code=sm_${ARCH} -std=c++17 -Wno-deprecated-gpu-targets" \
    CUDAFLAGSADD='--compiler-options "-fPIC"'

echo "==> installing pysrwarp into $(python -c 'import sys; print(sys.executable)')"
python -m pip install --no-build-isolation --no-deps --force-reinstall "$SRC_DIR"

python -c "import torch; import srwarp_cuda; from srwarp import transform, warp; print('pysrwarp ok:', srwarp_cuda.__file__)"
