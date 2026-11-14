"""Moved to src/stratastills/look/gpu.py (the pipeline's look stage uses it); this name is kept for the scripts that import it."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills.look import gpu as _gpu

globals().update({k: v for k, v in vars(_gpu).items() if not k.startswith("__")})
