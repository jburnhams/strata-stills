#!/bin/bash
# Build the stratastills environment from scratch (macOS Apple silicon or Linux, Python 3.13).
#   scripts/setup_env.sh [--no-torch] [--with-vision] [--report]
# --report: at the end, build the visual report on the candidate photos (scripts/build_report.py into out/report/neutral/; skipped when there is no candidates/ folder)
# Needs uv (https://docs.astral.sh/uv/). Environment variables: VENV (default .venv), PYTHON (default 3.13)
# torch: scripts/setup_torch.py picks the build for the machine (CUDA = cu128|cu126|rocm64|cpu in the environment overrides it).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"; PYTHON="${PYTHON:-3.13}"
TORCH=1; VISION=0; REPORT=0
for a in "$@"; do case "$a" in --no-torch) TORCH=0;; --with-vision) VISION=1;; --report) REPORT=1;; *) echo "unknown option $a"; exit 2;; esac; done
say() { printf '\n==> %s\n' "$*"; }
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
say "virtual environment: $VENV (with pip: scripts/setup_torch.py installs torch with it)"
uv venv --seed --python "$PYTHON" "$VENV"
say "dependencies from uv.lock (--inexact: keeps pip and whatever setup_torch.py installs; torch is not locked, the right build depends on the card)"
export UV_PROJECT_ENVIRONMENT="$VENV"
(cd "$ROOT" && uv sync --locked --no-dev --inexact)
if [ "$TORCH" = 1 ]; then
  say "torch (the newest build the hardware can run: scripts/setup_torch.py looks at the NVIDIA card or the AMD GPU, installs, repairs a clashing cuDNN and probes it; CUDA, ROCm, Metal, or the CPU build)"
  "$VENV/bin/python" "$ROOT/scripts/setup_torch.py" --python "$VENV/bin/python" ${CUDA:+--cuda "$CUDA"}
  say "learned denoise (Real-ESRGAN weights, 9 MB, into models/): spandrel without its dependencies, so that it cannot replace the torch just installed"
  "$VENV/bin/python" -m pip install einops safetensors && "$VENV/bin/python" -m pip install --no-deps spandrel==0.4.2
  PYTHONPATH="$ROOT/src" "$VENV/bin/python" -c "from stratastills import denoise_ai; denoise_ai.fetch()"
fi
if [ "$VISION" = 1 ]; then
  say "detector and denoising models"
  (cd "$ROOT" && uv sync --locked --no-dev --inexact --group vision)
  "$VENV/bin/python" "$ROOT/scripts/setup_torch.py" --python "$VENV/bin/python" ${CUDA:+--cuda "$CUDA"}      # the detector packages pull their own torch: the right one again
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
  if [ -d "$ROOT/candidates" ]; then PYTHONPATH="$ROOT/src" "$VENV/bin/python" "$ROOT/scripts/build_report.py" && echo "report: $ROOT/out/report/neutral/index.html"
  else echo "no candidates/ folder: report skipped (scripts/copy_candidates.py)"; fi
fi
