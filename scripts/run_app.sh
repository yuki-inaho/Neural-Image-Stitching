#!/usr/bin/env bash
# Launch the gradio demo (http://127.0.0.1:7860).
#
# The app lives in the optional gradio environment:
#   pixi run -e gradio start   # first time: build pysrwarp, download weights, launch
#   pixi run -e gradio app     # launch only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! python -c "import gradio" >/dev/null 2>&1; then
    echo "error: gradio is not installed in the active environment ($(python -c 'import sys; print(sys.executable)'))." >&2
    echo "run:  pixi run -e gradio start   # prepares the env and launches the app" >&2
    exit 1
fi

if ! python -c "import torch, srwarp_cuda" >/dev/null 2>&1; then
    echo "error: pysrwarp is not installed in the active environment." >&2
    echo "run:  pixi run -e gradio start   # builds pysrwarp and downloads the weights" >&2
    exit 1
fi

exec python app.py
