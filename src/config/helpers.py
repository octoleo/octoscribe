"""
src/config/helpers.py — Small pure parsing/validation helpers for config loading.

These functions are intentionally free of any knowledge of OctoScribe's config
shape: they convert raw strings (from the environment or the INI file) into
typed Python values, and report fatal misconfiguration consistently.  They are
unit-tested directly (see ``tests/test_config.py``) and re-exported from the
package root, so they remain importable as ``from config import _parse_bool``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


def _require_int(value: Any, name: str) -> int:
    """Parse *value* as an integer, or call :func:`_die` with a clear message."""
    if value == "" or value is None:
        _die([
            f"{name} is required but not set. "
            "Export it as an environment variable or add it to your .env file."
        ])
    try:
        return int(value)
    except (ValueError, TypeError):
        _die([
            f"{name} must be a valid integer, got {value!r}. "
            "Check your environment variable or .env file."
        ])


def _optional_int(value: Any, name: str) -> Optional[int]:
    """
    Parse *value* as an integer when present.

    Returns ``None`` when *value* is empty/unset.  A present-but-invalid value
    is logged and treated as unset rather than fatal, because the caller has
    indicated the field is not required in the current mode.
    """
    if value == "" or value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        log.warning("%s is set but is not a valid integer (%r); ignoring.", name, value)
        return None


def _parse_bool(value: Any) -> bool:
    """Parse a boolean from a string, int, or bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean from {value!r}")


def _resolve_path(raw: str | Path, base: Path) -> Path:
    """
    Return an absolute path.

    If *raw* is already absolute (or starts with ``~``), expand and return it
    as-is.  Otherwise resolve it relative to *base*.
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _die(errors: list[str]) -> None:
    """Print all error messages to stderr and exit with status 1."""
    print("OctoScribe configuration error(s):", file=sys.stderr)
    for msg in errors:
        print(f"  • {msg}", file=sys.stderr)
    sys.exit(1)
