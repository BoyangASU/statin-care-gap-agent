"""MIMIC-IV port of the statin care-gap agent.

The prototype package `statin_agent` (under src/) is reused unchanged: its
guideline rules, evaluation metrics and pipeline dataclasses. If the repository
is not pip-installed, add src/ to the import path here so `python -m mimic.x`
works from the repository root.
"""
import sys
from pathlib import Path

try:
    import statin_agent  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
