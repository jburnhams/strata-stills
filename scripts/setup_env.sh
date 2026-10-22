#!/bin/bash
# Build the stratastills environment from scratch (macOS Apple silicon or Linux, Python 3.13).
#   scripts/setup_env.sh [--no-torch] [--with-vision]
# Environment variables: VENV (default .venv), PYTHON (default python3)
# On macOS the default PyTorch wheel includes the Metal (MPS) backend; on Linux with an NVIDIA card install the CUDA build yourself:
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"; PYTHON="${PYTHON:-python3}"
TORCH=1; VISION=0
for a in "$@"; do case "$a" in --no-torch) TORCH=0;; --with-vision) VISION=1;; *) echo "unknown option $a"; exit 2;; esac; done
say() { printf '\n==> %s\n' "$*"; }
"$PYTHON" -c 'import sys; assert sys.version_info[:2] >= (3, 11), "Python 3.11+ required (tested on 3.13)"'
say "virtual environment: $VENV"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip >/dev/null
say "requirements"
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"
if [ "$TORCH" = 1 ]; then
  say "torch (the pixel work runs on the GPU when there is one: CUDA or Metal; the CPU build works everywhere)"
  "$VENV/bin/pip" install torch
fi
if [ "$VISION" = 1 ]; then
  say "detector and denoising models"
  "$VENV/bin/pip" install -r "$ROOT/requirements-vision.txt"
fi
say "checking"
STRATA_PYTHON="$VENV/bin/python" "$ROOT/stratastills" doctor

# ---- web app (React + Tailwind, built into src/stratastills/server/static) ---------------------------------------------------------
if command -v npm >/dev/null 2>&1; then
  ( cd "$ROOT/web" && npm install --silent && npm run build ) && echo "web app built"
else
  echo "npm not found: install Node.js to build the web app (cd web && npm install && npm run build)"
fi
