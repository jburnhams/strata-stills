#!/bin/bash
# Build the stratastills environment from scratch (macOS Apple silicon or Linux, Python 3.13).
#   scripts/setup_env.sh [--no-torch] [--with-vision] [--report]
# --report: at the end, build the visual report on the candidate photos (scripts/build_report.py into out/report/; skipped when there is no candidates/ folder)
# Environment variables: VENV (default .venv), PYTHON (default python3)
# On macOS the default PyTorch wheel includes the Metal (MPS) backend; on Linux with an NVIDIA card install the CUDA build yourself, BEFORE the learned denoise step needs it (re-run the script after):
#   .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128      (a Pascal card, GTX 10xx, needs cu126 and "torch<2.10": the newer wheels have no kernels for it)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"; PYTHON="${PYTHON:-python3}"
TORCH=1; VISION=0; REPORT=0
for a in "$@"; do case "$a" in --no-torch) TORCH=0;; --with-vision) VISION=1;; --report) REPORT=1;; *) echo "unknown option $a"; exit 2;; esac; done
say() { printf '\n==> %s\n' "$*"; }
"$PYTHON" -c 'import sys; assert sys.version_info[:2] >= (3, 11), "Python 3.11+ required (tested on 3.13)"'
say "virtual environment: $VENV"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip >/dev/null
say "requirements"
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"
if [ "$TORCH" = 1 ]; then
  say "torch (the pixel work runs on the GPU when there is one: CUDA or Metal; the CPU build works everywhere)"
  "$VENV/bin/pip" install torch torchvision
  say "learned denoise (Real-ESRGAN weights, 9 MB, into models/): spandrel without its dependencies, so that it cannot replace the torch just installed"
  "$VENV/bin/pip" install einops safetensors && "$VENV/bin/pip" install --no-deps spandrel==0.4.2
  PYTHONPATH="$ROOT/src" "$VENV/bin/python" -c "from stratastills import denoise_ai; denoise_ai.fetch()"
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

# ---- optional: the visual report on the candidate photos ---------------------------------------------------------------------------
if [ "$REPORT" = 1 ]; then
  PYTHONPATH="$ROOT/src" "$VENV/bin/python" -c "from stratastills import externaltools as E; E.require()" || exit 1      # darktable and RawTherapee are needed: the report is a comparison with them
  if [ -d "$ROOT/candidates" ]; then PYTHONPATH="$ROOT/src" "$VENV/bin/python" "$ROOT/scripts/build_report.py" && echo "report: $ROOT/out/report/index.html"
  else echo "no candidates/ folder: report skipped (scripts/copy_candidates.py)"; fi
fi
