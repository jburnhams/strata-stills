"""Shared pytest configuration. Every test is hermetic through strata-python's pytest plugin, configured in pyproject.toml (`strata_env_prefix`, `strata_tuning_vars`,
`strata_env`): its own home, app home and projects folder, the app's switches off, no tuning variables, no network. Tests under tests/integration get the `integration`
marker from the plugin too."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
