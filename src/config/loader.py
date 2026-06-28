"""
src/config/loader.py — Multi-source configuration loader.

:class:`_ConfigLoader` is the single place that knows *how* an OctoScribe
configuration is assembled.  It applies the documented precedence

    CLI overrides  >  environment / .env  >  INI file  >  built-in defaults

builds each typed section (see :mod:`src.config.models`), validates the result,
and returns a :class:`~src.config.root.Config`.  Secrets are read exclusively
from the environment, never from the INI file.

This loader is deliberately internal (underscore-prefixed): callers go through
:meth:`Config.load`, which keeps the public surface small and the assembly
strategy swappable.
"""

from __future__ import annotations

import configparser
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Optional

from src.config.helpers import (
    _die,
    _optional_int,
    _parse_bool,
    _require_int,
    _resolve_path,
)
from src.config.models import (
    DataRepoConfig,
    DownloadConfig,
    SourceConfig,
    TelegramConfig,
    TranscribeConfig,
)
from src.config.root import Config

try:
    from dotenv import load_dotenv  # python-dotenv
    _HAVE_DOTENV = True
except ImportError:  # pragma: no cover
    _HAVE_DOTENV = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default INI text – also documents every tunable option.
# ---------------------------------------------------------------------------

_DEFAULT_INI: str = """\
[source]
mode = telegram
folder =
recursive = true

[telegram]
group =

[download]
workers = 4
resume = true
deduplicate = true

[transcribe]
backend = openai
model = gpt-4o-transcribe
language = en
workers = 4
retry_attempts = 3
retry_base_delay = 2.5
retry_max_delay = 30.0

[local_transcribe]
model = large-v3
device = cuda
compute_type = int8_float16
beam_size = 5
best_of = 5
repetition_penalty = 1.1
vad_filter = true
vad_min_silence_ms = 500
vad_speech_pad_ms = 400

[data_repo]
path = ~/.octoscribe/data
branch = main
auto_push = true

[paths]
audio_dir = audio
transcriptions_dir = transcriptions
manifest_file = manifest.json
session_dir = .session
"""


class _ConfigLoader:
    """Encapsulates the multi-source loading logic."""

    def __init__(
        self,
        ini_path: Optional[str | Path],
        env_file: Optional[str | Path],
        overrides: dict[str, Any],
    ) -> None:
        self._overrides = overrides
        self._ini_path = ini_path
        self._env_file = env_file

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(self) -> Config:
        # Step 1 – resolve & load .env so its values land in os.environ.
        env_file = self._resolve_env_file()
        if env_file is not None:
            self._load_dotenv(env_file)

        # Step 2 – resolve the INI file path (env var may now be set).
        resolved_ini_path = self._resolve_ini_path()

        # Step 3 – parse the INI file (missing file → pure defaults).
        ini = self._parse_ini(resolved_ini_path)

        # Step 4 – build each sub-config.
        # data_repo must be built first because other builders need data_repo.path
        # as the base for resolving relative paths (including session_dir).
        # source must be built before telegram so we know whether Telegram
        # credentials are required (they are not in folder mode).
        data_repo = self._build_data_repo(ini)
        source = self._build_source(ini)
        telegram = self._build_telegram(
            ini, data_repo.path, require_secrets=(source.mode == "telegram")
        )
        download = self._build_download(ini, data_repo.path)
        transcribe = self._build_transcribe(ini, data_repo.path)

        # Step 5 – validate.
        self._validate(source, telegram, transcribe, data_repo)

        return Config(
            telegram=telegram,
            download=download,
            transcribe=transcribe,
            data_repo=data_repo,
            ini_path=resolved_ini_path,
            source=source,
        )

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------

    def _resolve_env_file(self) -> Optional[Path]:
        # Explicit kwarg beats env var beats default.
        if self._env_file is not None:
            return Path(self._env_file).expanduser().resolve()
        env_override = os.environ.get("OCTOSCRIBE_ENV", "").strip()
        if env_override:
            return Path(env_override).expanduser().resolve()
        candidate = Path.cwd() / ".env"
        return candidate if candidate.exists() else None

    def _resolve_ini_path(self) -> Path:
        if self._ini_path is not None:
            return Path(self._ini_path).expanduser().resolve()
        env_override = os.environ.get("OCTOSCRIBE_CONFIG", "").strip()
        if env_override:
            return Path(env_override).expanduser().resolve()
        conf_candidate = Path.cwd() / "conf" / "octoscribe.ini"
        if conf_candidate.exists():
            return conf_candidate
        return Path.cwd() / "octoscribe.ini"

    # ------------------------------------------------------------------
    # .env loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dotenv(path: Path) -> None:
        if not _HAVE_DOTENV:
            log.warning(
                "python-dotenv is not installed; .env file %s will not be loaded. "
                "Install it with: pip install python-dotenv",
                path,
            )
            return
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)
            log.debug("Loaded .env from %s", path)
        else:
            log.debug(".env file not found at %s; skipping", path)

    # ------------------------------------------------------------------
    # INI parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ini(path: Path) -> configparser.ConfigParser:
        parser = configparser.ConfigParser(
            interpolation=None,
            inline_comment_prefixes=("#", ";"),
        )
        # Load the built-in defaults first.
        parser.read_string(_DEFAULT_INI)
        if path.exists():
            parser.read(path, encoding="utf-8")
            log.debug("Loaded INI from %s", path)
        else:
            log.debug("INI file %s not found; using defaults", path)
        return parser

    # ------------------------------------------------------------------
    # Override helper
    # ------------------------------------------------------------------

    def _override(self, section: str, key: str, value: Any) -> Any:
        """
        Return the override value for ``section.key`` if present, else ``value``.

        Overrides may be supplied as ``section__key=val`` or (for unambiguous
        keys) as plain ``key=val``.
        """
        compound = f"{section}__{key}"
        if compound in self._overrides:
            return self._overrides[compound]
        if key in self._overrides:
            return self._overrides[key]
        return value

    # ------------------------------------------------------------------
    # Sub-config builders
    # ------------------------------------------------------------------

    def _build_source(self, ini: configparser.ConfigParser) -> SourceConfig:
        mode = str(
            self._override("source", "mode", ini.get("source", "mode", fallback="telegram"))
        ).strip().lower()
        recursive = _parse_bool(
            self._override(
                "source", "recursive", ini.get("source", "recursive", fallback="true")
            )
        )
        folder_raw = self._override(
            "source", "folder", ini.get("source", "folder", fallback="")
        )
        folder: Optional[Path] = None
        if folder_raw and str(folder_raw).strip():
            folder = Path(str(folder_raw).strip()).expanduser().resolve()

        return SourceConfig(mode=mode, folder=folder, recursive=recursive)

    def _build_telegram(
        self, ini: configparser.ConfigParser, data_root: Path, require_secrets: bool = True
    ) -> TelegramConfig:
        # --- secrets from env only ---
        api_id_raw = self._override("telegram", "api_id", os.environ.get("TELEGRAM_API_ID", ""))
        api_hash = self._override("telegram", "api_hash", os.environ.get("TELEGRAM_API_HASH", ""))
        phone = self._override("telegram", "phone", os.environ.get("TELEGRAM_PHONE", ""))

        # Validate api_id early so we get a clear error — but only when Telegram
        # is the active source.  In folder mode the credentials are optional.
        if require_secrets:
            api_id = _require_int(api_id_raw, "TELEGRAM_API_ID")
        else:
            api_id = _optional_int(api_id_raw, "TELEGRAM_API_ID")

        # --- non-secret settings from INI ---
        group = self._override("telegram", "group", ini.get("telegram", "group", fallback=""))
        session_dir_raw = self._override(
            "paths", "session_dir", ini.get("paths", "session_dir", fallback=".session")
        )

        return TelegramConfig(
            api_id=api_id,
            api_hash=str(api_hash),
            phone=str(phone),
            group=str(group),
            session_dir=_resolve_path(session_dir_raw, data_root),
        )

    def _build_data_repo(self, ini: configparser.ConfigParser) -> DataRepoConfig:
        url = self._override("data_repo", "url", os.environ.get("DATA_REPO_URL") or None)
        path_raw = self._override(
            "data_repo", "path", ini.get("data_repo", "path", fallback="~/.octoscribe/data")
        )
        branch = self._override(
            "data_repo", "branch", ini.get("data_repo", "branch", fallback="main")
        )
        auto_push = _parse_bool(
            self._override(
                "data_repo", "auto_push", ini.get("data_repo", "auto_push", fallback="true")
            )
        )

        return DataRepoConfig(
            url=str(url) if url else None,
            path=Path(path_raw).expanduser().resolve(),
            branch=str(branch),
            auto_push=auto_push,
        )

    def _build_download(self, ini: configparser.ConfigParser, data_root: Path) -> DownloadConfig:
        workers = int(
            self._override("download", "workers", ini.getint("download", "workers", fallback=4))
        )
        resume = _parse_bool(
            self._override("download", "resume", ini.get("download", "resume", fallback="true"))
        )
        deduplicate = _parse_bool(
            self._override(
                "download", "deduplicate", ini.get("download", "deduplicate", fallback="true")
            )
        )
        audio_dir_raw = self._override(
            "paths", "audio_dir", ini.get("paths", "audio_dir", fallback="audio")
        )
        manifest_raw = self._override(
            "paths", "manifest_file", ini.get("paths", "manifest_file", fallback="manifest.json")
        )

        return DownloadConfig(
            workers=workers,
            resume=resume,
            deduplicate=deduplicate,
            audio_dir=_resolve_path(audio_dir_raw, data_root),
            manifest_file=_resolve_path(manifest_raw, data_root),
        )

    def _build_transcribe(
        self, ini: configparser.ConfigParser, data_root: Path
    ) -> TranscribeConfig:
        backend = self._override(
            "transcribe", "backend", ini.get("transcribe", "backend", fallback="openai")
        )
        model = self._override(
            "transcribe", "model", ini.get("transcribe", "model", fallback="gpt-4o-transcribe")
        )
        language = self._override(
            "transcribe", "language", ini.get("transcribe", "language", fallback="en")
        )
        workers = int(
            self._override(
                "transcribe", "workers", ini.getint("transcribe", "workers", fallback=4)
            )
        )
        retry_attempts = int(
            self._override(
                "transcribe",
                "retry_attempts",
                ini.getint("transcribe", "retry_attempts", fallback=3),
            )
        )
        retry_base_delay = float(
            self._override(
                "transcribe",
                "retry_base_delay",
                ini.getfloat("transcribe", "retry_base_delay", fallback=2.5),
            )
        )
        retry_max_delay = float(
            self._override(
                "transcribe",
                "retry_max_delay",
                ini.getfloat("transcribe", "retry_max_delay", fallback=30.0),
            )
        )

        # Secret from env only
        api_key = self._override(
            "transcribe", "api_key", os.environ.get("OPENAI_API_KEY") or None
        )

        # Local Whisper settings
        local_model = self._override(
            "local_transcribe", "model", ini.get("local_transcribe", "model", fallback="large-v3")
        )
        device = self._override(
            "local_transcribe", "device", ini.get("local_transcribe", "device", fallback="cuda")
        )
        compute_type = self._override(
            "local_transcribe",
            "compute_type",
            ini.get("local_transcribe", "compute_type", fallback="int8_float16"),
        )
        beam_size = int(
            self._override(
                "local_transcribe",
                "beam_size",
                ini.getint("local_transcribe", "beam_size", fallback=5),
            )
        )
        best_of = int(
            self._override(
                "local_transcribe",
                "best_of",
                ini.getint("local_transcribe", "best_of", fallback=5),
            )
        )
        repetition_penalty = float(
            self._override(
                "local_transcribe",
                "repetition_penalty",
                ini.getfloat("local_transcribe", "repetition_penalty", fallback=1.1),
            )
        )
        vad_filter = _parse_bool(
            self._override(
                "local_transcribe",
                "vad_filter",
                ini.get("local_transcribe", "vad_filter", fallback="true"),
            )
        )
        vad_min_silence_ms = int(
            self._override(
                "local_transcribe",
                "vad_min_silence_ms",
                ini.getint("local_transcribe", "vad_min_silence_ms", fallback=500),
            )
        )
        vad_speech_pad_ms = int(
            self._override(
                "local_transcribe",
                "vad_speech_pad_ms",
                ini.getint("local_transcribe", "vad_speech_pad_ms", fallback=400),
            )
        )

        transcriptions_dir_raw = self._override(
            "paths",
            "transcriptions_dir",
            ini.get("paths", "transcriptions_dir", fallback="transcriptions"),
        )
        manifest_raw = self._override(
            "paths", "manifest_file", ini.get("paths", "manifest_file", fallback="manifest.json")
        )

        return TranscribeConfig(
            backend=str(backend),
            model=str(model),
            language=str(language),
            workers=workers,
            retry_attempts=retry_attempts,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            api_key=str(api_key) if api_key else None,
            local_model=str(local_model),
            device=str(device),
            compute_type=str(compute_type),
            beam_size=beam_size,
            best_of=best_of,
            repetition_penalty=repetition_penalty,
            vad_filter=vad_filter,
            vad_min_silence_ms=vad_min_silence_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            transcriptions_dir=_resolve_path(transcriptions_dir_raw, data_root),
            manifest_file=_resolve_path(manifest_raw, data_root),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        source: SourceConfig,
        telegram: TelegramConfig,
        transcribe: TranscribeConfig,
        data_repo: DataRepoConfig,
    ) -> None:
        errors: list[str] = []

        # Source mode validation
        valid_modes = {"telegram", "folder"}
        if source.mode not in valid_modes:
            errors.append(
                f"source.mode must be one of {sorted(valid_modes)!r}, "
                f"got {source.mode!r}."
            )

        if source.mode == "folder":
            # Folder mode only needs a folder path — no Telegram credentials.
            if source.folder is None:
                errors.append(
                    "source.folder is required when source.mode='folder'. "
                    "Set it in the [source] section of your INI file or pass "
                    "--folder PATH on the command line."
                )
        elif source.mode == "telegram":
            # Required Telegram secrets (only when Telegram is the active source).
            if not telegram.api_id:
                errors.append(
                    "TELEGRAM_API_ID is required but not set. "
                    "Export it as an environment variable or add it to your .env file."
                )
            if not telegram.api_hash:
                errors.append(
                    "TELEGRAM_API_HASH is required but not set. "
                    "Export it as an environment variable or add it to your .env file."
                )
            if not telegram.phone:
                errors.append(
                    "TELEGRAM_PHONE is required but not set. "
                    "Export it as an environment variable or add it to your .env file."
                )

        # Backend validation
        valid_backends = {"openai", "local"}
        if transcribe.backend not in valid_backends:
            errors.append(
                f"transcribe.backend must be one of {sorted(valid_backends)!r}, "
                f"got {transcribe.backend!r}."
            )

        # OpenAI key required when backend=openai
        if transcribe.backend == "openai" and not transcribe.api_key:
            errors.append(
                "OPENAI_API_KEY is required when transcribe.backend='openai' but not set. "
                "Export it as an environment variable or add it to your .env file."
            )

        if errors:
            _die(errors)

        # Non-fatal warning: data_repo inside the project directory.
        # __file__ is src/config/loader.py, so the project root is three
        # levels up (src/config -> src -> project root).
        project_dir = Path(__file__).resolve().parents[2]
        try:
            data_repo.path.relative_to(project_dir)
            warnings.warn(
                f"data_repo.path ({data_repo.path}) is inside the project directory "
                f"({project_dir}).  Consider using a path outside the project tree, "
                "e.g. ~/.octoscribe/data.",
                stacklevel=4,
            )
        except ValueError:
            pass  # Good – it is outside the project directory.
