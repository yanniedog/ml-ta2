"""Utility subpackage to resolve name conflict between `src.utils` package
(directory) and original `src/utils.py` module.

This `__init__.py` dynamically loads the original `utils.py` *module* that lives
next to this directory (i.e. `src/utils.py`) and re-exports all of its public
symbols. This allows both package-style imports (`import src.utils as u`) and
module-style imports (`from src.utils import ensure_directory`) to work
transparently for the rest of the codebase and tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# ---------------------------------------------------------------------------
# Dynamically load original `utils.py` that resides next to the `utils/` dir.
# ---------------------------------------------------------------------------
_current_dir = Path(__file__).resolve().parent
_src_root = _current_dir.parent  # Path to `src/`
_original_utils_path = _src_root / "utils.py"

_spec = importlib.util.spec_from_file_location(
    "_mlta_utils_module", _original_utils_path
)
_original_utils: ModuleType = importlib.util.module_from_spec(_spec)  # type: ignore
assert _spec.loader is not None
_spec.loader.exec_module(_original_utils)  # type: ignore

# Register the module under an alternate key to avoid recursive import issues
sys.modules["src._mlta_utils_module"] = _original_utils

# ---------------------------------------------------------------------------
# Re-export public symbols
# ---------------------------------------------------------------------------
__all__ = getattr(_original_utils, "__all__", []) or [
    name for name in dir(_original_utils) if not name.startswith("_")
]

globals().update({name: getattr(_original_utils, name) for name in __all__})
