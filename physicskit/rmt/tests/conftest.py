"""Ensures tests/ is importable as a plain module path (for
``from cache_utils import cached_sample``) regardless of how pytest is
invoked (rootdir vs tests/ as cwd)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
