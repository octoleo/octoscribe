"""
src/config/root.py — The aggregate :class:`Config` object.

:class:`Config` is the single typed object the rest of OctoScribe depends on.
It is a thin aggregate of the value objects in :mod:`src.config.models` plus
two conveniences: the :meth:`Config.load` factory (which delegates all the
multi-source loading work to :class:`src.config.loader._ConfigLoader`) and
:meth:`Config.redacted_repr` for safe logging.

Construction logic lives in the loader, not here, so this module stays focused
on *representing* a fully-resolved configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from src.config.models import (
    DataRepoConfig,
    DownloadConfig,
    SourceConfig,
    TelegramConfig,
    TranscribeConfig,
)

# Field coordinates whose values must never appear in logs or reprs.
_SECRET_FIELDS = {
    ("telegram", "api_hash"),
    ("telegram", "phone"),
    ("telegram", "api_id"),
    ("transcribe", "api_key"),
    ("data_repo", "url"),
}


@dataclass
class Config:
    """
    Root configuration object for OctoScribe.

    Instantiate via :meth:`Config.load` rather than directly; ``load`` applies
    the documented precedence (CLI overrides > environment/.env > INI >
    built-in defaults) and validates the result.
    """

    telegram: TelegramConfig
    download: DownloadConfig
    transcribe: TranscribeConfig
    data_repo: DataRepoConfig
    ini_path: Path
    # Defaulted so existing call sites that predate the source feature keep
    # working; Config.load() always sets this explicitly.
    source: SourceConfig = field(
        default_factory=lambda: SourceConfig(mode="telegram", folder=None, recursive=True)
    )

    # ------------------------------------------------------------------
    # Public factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        ini_path: Optional[str | Path] = None,
        env_file: Optional[str | Path] = None,
        **overrides: Any,
    ) -> "Config":
        """
        Load configuration from all sources and return a validated ``Config``.

        Parameters
        ----------
        ini_path:
            Path to the INI file.  Overridden by the ``OCTOSCRIBE_CONFIG``
            environment variable (which may itself be set in the .env file).
        env_file:
            Path to the .env file.  Overridden by the ``OCTOSCRIBE_ENV``
            environment variable.  Defaults to ``.env`` in the current
            working directory.
        **overrides:
            Flat key=value pairs that override everything else.  Keys use
            dot notation, e.g. ``telegram__group="my-group"`` (double
            underscore as separator) or simply ``group="my-group"`` for
            top-level keys.  Section-prefixed keys are preferred to avoid
            ambiguity, e.g. ``download__workers=8``.
        """
        # Imported lazily so this module does not depend on the loader at import
        # time (the loader, in turn, constructs Config), keeping the two
        # decoupled and free of import cycles.
        from src.config.loader import _ConfigLoader

        loader = _ConfigLoader(ini_path=ini_path, env_file=env_file, overrides=overrides)
        return loader.build()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def redacted_repr(self) -> str:
        """Return a human-readable string with all secret values replaced by ***."""
        lines: list[str] = ["Config("]

        def _fmt_sub(section_name: str, sub_obj: Any) -> None:
            lines.append(f"  {section_name}=(")
            for f in fields(sub_obj):  # type: ignore[arg-type]
                val = getattr(sub_obj, f.name)
                if (section_name, f.name) in _SECRET_FIELDS and val is not None:
                    display = "***"
                else:
                    display = repr(val)
                lines.append(f"    {f.name}={display},")
            lines.append("  ),")

        _fmt_sub("source", self.source)
        _fmt_sub("telegram", self.telegram)
        _fmt_sub("download", self.download)
        _fmt_sub("transcribe", self.transcribe)
        _fmt_sub("data_repo", self.data_repo)
        lines.append(f"  ini_path={self.ini_path!r},")
        lines.append(")")
        return "\n".join(lines)
