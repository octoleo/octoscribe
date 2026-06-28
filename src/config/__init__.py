"""
src/config — OctoScribe configuration package.

This package replaces the former single ``src/config.py`` module.  It is split
by responsibility for readability and testability:

* :mod:`src.config.models`  — typed, logic-free configuration value objects.
* :mod:`src.config.helpers` — small pure parsing/validation helpers.
* :mod:`src.config.loader`  — the multi-source loader (precedence + validation).
* :mod:`src.config.root`    — the aggregate :class:`Config` and its factory.

Loading priority (highest wins):
  1. CLI overrides (kwargs passed to ``Config.load()``)
  2. Environment variables (including those sourced from a .env file)
  3. INI file values
  4. Built-in defaults

Secrets are sourced exclusively from environment variables (never from the INI
file).  The .env file is loaded first so its values appear in ``os.environ``.

Everything that previously lived in ``src/config.py`` is re-exported here, so
existing imports such as ``from src.config import Config`` and
``from config import _parse_bool, _resolve_path`` continue to work unchanged.
"""

from __future__ import annotations

from src.config.helpers import (
    _die,
    _optional_int,
    _parse_bool,
    _require_int,
    _resolve_path,
)
from src.config.loader import _ConfigLoader, _DEFAULT_INI
from src.config.models import (
    DataRepoConfig,
    DownloadConfig,
    SourceConfig,
    TelegramConfig,
    TranscribeConfig,
)
from src.config.root import Config

__all__ = [
    "Config",
    "SourceConfig",
    "TelegramConfig",
    "DownloadConfig",
    "TranscribeConfig",
    "DataRepoConfig",
    # Internal helpers kept importable for tests and backwards compatibility.
    "_ConfigLoader",
    "_DEFAULT_INI",
    "_parse_bool",
    "_resolve_path",
    "_require_int",
    "_optional_int",
    "_die",
]
