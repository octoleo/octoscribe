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
import urllib.parse
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
model = gpt-transcribe
language = en
workers = 4
retry_attempts = 1
retry_base_delay = 2.5
retry_max_delay = 30.0
providers =
primary_provider =
provider_timeout_seconds = 900

[chunking]
target_seconds = 480
max_seconds = 600
overlap_seconds = 12
silence_search_seconds = 45
silence_threshold_db = -35
silence_min_ms = 500
max_chunk_megabytes = 24

[quality]
disagreement_retry_limit = 1
arbitration_limit = 1

[xai]
base_url = https://api.x.ai/v1/stt

[meta_asr]
url =
model = omniASR_LLM_Unlimited_7B_v2
language = eng_Latn

[local_transcribe]
model = large-v3
device = cuda
compute_type = int8_float16
beam_size = 5
best_of = 5
repetition_penalty = 1.0
vad_filter = false
vad_min_silence_ms = 500
vad_speech_pad_ms = 400

[data_repo]
path = ~/.octoscribe/data
branch = main
auto_push = true

[audio_repo]
path = ~/.octoscribe/audio-data
branch = main
auto_push = true

[transcript_repo]
path = ~/.octoscribe/transcript-data
branch = main
auto_push = true

[paths]
audio_dir = audio
transcriptions_dir = transcriptions
manifest_file = manifest.json
session_dir = ~/.octoscribe/session
artifacts_dir = candidates
reports_dir = reports
"""


class _ConfigLoader:
    """Encapsulates the multi-source loading logic."""

    def __init__(
        self,
        ini_path: Optional[str | Path],
        env_file: Optional[str | Path],
        overrides: dict[str, Any],
        validation_profile: str = "run",
    ) -> None:
        self._overrides = overrides
        self._ini_path = ini_path
        self._env_file = env_file
        self._validation_profile = validation_profile
        self._user_ini = configparser.ConfigParser(interpolation=None)

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
        # Repositories must be built first because path-bearing sections use
        # their worktrees as resolution roots.  Audio is immutable evidence;
        # transcripts, reports, and the manifest live in the text repository.
        # source must be built before telegram so we know whether Telegram
        # credentials are required (they are not in folder mode).
        data_repo, transcript_repo = self._build_repositories(ini)
        source = self._build_source(ini)
        telegram = self._build_telegram(
            ini,
            data_repo.path,
            require_secrets=(
                self._validation_profile in {"run", "download", "debug"}
                and (
                    self._validation_profile == "debug"
                    or source.mode == "telegram"
                )
            ),
        )
        download = self._build_download(
            ini, audio_root=data_repo.path, text_root=transcript_repo.path
        )
        transcribe = self._build_transcribe(ini, transcript_repo.path)

        # Step 5 – validate.
        self._validate(
            source,
            telegram,
            download,
            transcribe,
            data_repo,
            transcript_repo,
            validation_profile=self._validation_profile,
        )

        return Config(
            telegram=telegram,
            download=download,
            transcribe=transcribe,
            data_repo=data_repo,
            ini_path=resolved_ini_path,
            source=source,
            transcript_repo=transcript_repo,
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

    def _parse_ini(self, path: Path) -> configparser.ConfigParser:
        parser = configparser.ConfigParser(
            interpolation=None,
            inline_comment_prefixes=("#", ";"),
        )
        # Load the built-in defaults first.
        parser.read_string(_DEFAULT_INI)
        if path.exists():
            self._user_ini.read(path, encoding="utf-8")
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

    def _secret_override(self, section: str, key: str, value: Any) -> Any:
        """Apply only a section-qualified override to a credential.

        Several providers use the field name ``api_key``.  Accepting a plain
        ``api_key=...`` override would copy one provider's credential into
        every configured endpoint, so credentials deliberately require the
        unambiguous ``section__api_key`` form.
        """
        compound = f"{section}__{key}"
        return self._overrides.get(compound, value)

    def _is_explicit(self, section: str, key: str, *env_names: str) -> bool:
        """Whether a value came from user config, CLI, or environment."""
        return (
            self._user_ini.has_option(section, key)
            or f"{section}__{key}" in self._overrides
            or key in self._overrides
            or any(bool(os.environ.get(name)) for name in env_names)
        )

    @staticmethod
    def _provider_list(raw: Any) -> tuple[str, ...]:
        """Parse and canonicalise a comma-separated provider list."""
        aliases = {
            "grok": "xai",
            "local": "whisper",
            "local_whisper": "whisper",
            "omnilingual": "meta",
            "meta_asr": "meta",
        }
        if isinstance(raw, (tuple, list)):
            values = [str(v) for v in raw]
        else:
            values = str(raw or "").split(",")
        result: list[str] = []
        for value in values:
            name = aliases.get(value.strip().lower(), value.strip().lower())
            if name and name not in result:
                result.append(name)
        return tuple(result)

    # ------------------------------------------------------------------
    # Sub-config builders
    # ------------------------------------------------------------------

    def _build_source(self, ini: configparser.ConfigParser) -> SourceConfig:
        mode = str(
            self._override(
                "source",
                "mode",
                os.environ.get("OCTOSCRIBE_SOURCE_MODE")
                or os.environ.get("SOURCE_MODE")
                or ini.get("source", "mode", fallback="telegram"),
            )
        ).strip().lower()
        recursive = _parse_bool(
            self._override(
                "source",
                "recursive",
                os.environ.get("OCTOSCRIBE_SOURCE_RECURSIVE")
                or ini.get("source", "recursive", fallback="true"),
            )
        )
        folder_raw = self._override(
            "source",
            "folder",
            os.environ.get("OCTOSCRIBE_SOURCE_FOLDER")
            or os.environ.get("SOURCE_FOLDER")
            or ini.get("source", "folder", fallback=""),
        )
        folder: Optional[Path] = None
        if folder_raw and str(folder_raw).strip():
            folder = Path(str(folder_raw).strip()).expanduser().resolve()

        return SourceConfig(mode=mode, folder=folder, recursive=recursive)

    def _build_telegram(
        self, ini: configparser.ConfigParser, data_root: Path, require_secrets: bool = True
    ) -> TelegramConfig:
        # --- secrets from env only ---
        api_id_raw = self._secret_override(
            "telegram", "api_id", os.environ.get("TELEGRAM_API_ID", "")
        )
        api_hash = self._secret_override(
            "telegram", "api_hash", os.environ.get("TELEGRAM_API_HASH", "")
        )
        phone = self._secret_override(
            "telegram", "phone", os.environ.get("TELEGRAM_PHONE", "")
        )

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

    def _build_repositories(
        self, ini: configparser.ConfigParser
    ) -> tuple[DataRepoConfig, DataRepoConfig]:
        """Build audio and transcript repositories with legacy migration.

        A user-supplied ``[data_repo]`` section (or DATA_REPO_* environment
        variables) keeps the historical combined layout unless either new
        repository is explicitly configured.  Fresh configurations default to
        two worktrees so large audio history never bloats transcript history.
        """

        legacy_explicit = (
            self._user_ini.has_section("data_repo")
            or any(k.startswith("data_repo__") for k in self._overrides)
            or bool(os.environ.get("DATA_REPO_URL") or os.environ.get("DATA_REPO_PATH"))
        )
        split_explicit = (
            self._user_ini.has_section("audio_repo")
            or self._user_ini.has_section("transcript_repo")
            or any(
                k.startswith("audio_repo__") or k.startswith("transcript_repo__")
                for k in self._overrides
            )
            or bool(
                os.environ.get("AUDIO_REPO_URL")
                or os.environ.get("AUDIO_REPO_PATH")
                or os.environ.get("TRANSCRIPT_REPO_URL")
                or os.environ.get("TRANSCRIPT_REPO_PATH")
            )
        )

        if legacy_explicit and not split_explicit:
            legacy = self._build_repository(
                ini,
                section="data_repo",
                env_prefix="DATA_REPO",
                default_path="~/.octoscribe/data",
            )
            return legacy, legacy

        audio = self._build_repository(
            ini,
            section="audio_repo",
            env_prefix="AUDIO_REPO",
            default_path="~/.octoscribe/audio-data",
        )
        transcript = self._build_repository(
            ini,
            section="transcript_repo",
            env_prefix="TRANSCRIPT_REPO",
            default_path="~/.octoscribe/transcript-data",
        )
        return audio, transcript

    def _build_repository(
        self,
        ini: configparser.ConfigParser,
        *,
        section: str,
        env_prefix: str,
        default_path: str,
    ) -> DataRepoConfig:
        url = self._override(
            section,
            "url",
            os.environ.get(f"{env_prefix}_URL")
            or ini.get(section, "url", fallback="")
            or None,
        )
        path_raw = self._override(
            section,
            "path",
            os.environ.get(f"{env_prefix}_PATH")
            or ini.get(section, "path", fallback=default_path),
        )
        branch = self._override(
            section,
            "branch",
            os.environ.get(f"{env_prefix}_BRANCH")
            or ini.get(section, "branch", fallback="main"),
        )
        auto_push = _parse_bool(
            self._override(
                section,
                "auto_push",
                os.environ.get(f"{env_prefix}_AUTO_PUSH")
                or ini.get(section, "auto_push", fallback="true"),
            )
        )
        return DataRepoConfig(
            url=str(url) if url else None,
            path=Path(str(path_raw)).expanduser().resolve(),
            branch=str(branch),
            auto_push=auto_push,
        )

    def _build_download(
        self,
        ini: configparser.ConfigParser,
        audio_root: Path,
        text_root: Path,
    ) -> DownloadConfig:
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
            audio_dir=_resolve_path(audio_dir_raw, audio_root),
            manifest_file=_resolve_path(manifest_raw, text_root),
        )

    def _build_transcribe(
        self, ini: configparser.ConfigParser, data_root: Path
    ) -> TranscribeConfig:
        backend = self._override(
            "transcribe",
            "backend",
            os.environ.get("TRANSCRIBE_BACKEND")
            or ini.get("transcribe", "backend", fallback="openai"),
        )
        model = self._override(
            "transcribe",
            "model",
            os.environ.get("OPENAI_TRANSCRIBE_MODEL")
            or os.environ.get("TRANSCRIBE_MODEL")
            or ini.get("transcribe", "model", fallback="gpt-transcribe"),
        )
        language = self._override(
            "transcribe",
            "language",
            os.environ.get("TRANSCRIBE_LANGUAGE")
            or ini.get("transcribe", "language", fallback="en"),
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
                ini.getint("transcribe", "retry_attempts", fallback=1),
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
        api_key = self._secret_override(
            "transcribe", "api_key", os.environ.get("OPENAI_API_KEY") or None
        )
        xai_api_key = self._secret_override(
            "xai", "api_key", os.environ.get("XAI_API_KEY") or None
        )
        xai_base_url = self._override(
            "xai",
            "base_url",
            os.environ.get("XAI_STT_URL")
            or ini.get("xai", "base_url", fallback="https://api.x.ai/v1/stt"),
        )
        meta_asr_url = self._override(
            "meta_asr",
            "url",
            os.environ.get("META_ASR_URL")
            or ini.get("meta_asr", "url", fallback="")
            or None,
        )
        meta_asr_api_key = self._secret_override(
            "meta_asr", "api_key", os.environ.get("META_ASR_API_KEY") or None
        )
        meta_asr_model = self._override(
            "meta_asr",
            "model",
            os.environ.get("META_ASR_MODEL")
            or ini.get(
                "meta_asr", "model", fallback="omniASR_LLM_Unlimited_7B_v2"
            ),
        )
        meta_asr_language = self._override(
            "meta_asr",
            "language",
            os.environ.get("META_ASR_LANGUAGE")
            or ini.get("meta_asr", "language", fallback="eng_Latn"),
        )

        providers_raw = self._override(
            "transcribe",
            "providers",
            os.environ.get("OCTOSCRIBE_ASR_PROVIDERS")
            or ini.get("transcribe", "providers", fallback=""),
        )
        backend_name = str(backend).strip().lower()
        backend_explicit = self._is_explicit(
            "transcribe", "backend", "TRANSCRIBE_BACKEND"
        )
        providers_explicit = self._is_explicit(
            "transcribe", "providers", "OCTOSCRIBE_ASR_PROVIDERS"
        )
        if providers_explicit:
            providers = self._provider_list(providers_raw)
        elif backend_explicit:
            # Preserve the historical single-backend contract for existing
            # configurations that deliberately selected one backend.
            providers = self._provider_list((backend_name,))
        else:
            discovered: list[str] = []
            if api_key:
                discovered.append("openai")
            if xai_api_key:
                discovered.append("xai")
            if meta_asr_url:
                discovered.append("meta")
            if backend_name in {"local", "whisper"}:
                discovered.append("whisper")
            providers = tuple(discovered)

        primary_raw = self._override(
            "transcribe",
            "primary_provider",
            os.environ.get("OCTOSCRIBE_PRIMARY_ASR")
            or ini.get("transcribe", "primary_provider", fallback=""),
        )
        primary_provider = self._provider_list((primary_raw,))
        primary = primary_provider[0] if primary_provider else (providers[0] if providers else "")
        provider_timeout_seconds = float(
            self._override(
                "transcribe",
                "provider_timeout_seconds",
                os.environ.get("ASR_TIMEOUT_SECONDS")
                or ini.getfloat(
                    "transcribe", "provider_timeout_seconds", fallback=900.0
                ),
            )
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
                ini.getfloat("local_transcribe", "repetition_penalty", fallback=1.0),
            )
        )
        vad_filter = _parse_bool(
            self._override(
                "local_transcribe",
                "vad_filter",
                ini.get("local_transcribe", "vad_filter", fallback="false"),
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

        chunk_target_seconds = int(
            self._override(
                "chunking",
                "target_seconds",
                ini.getint("chunking", "target_seconds", fallback=480),
            )
        )
        chunk_max_seconds = int(
            self._override(
                "chunking",
                "max_seconds",
                ini.getint("chunking", "max_seconds", fallback=600),
            )
        )
        chunk_overlap_seconds = int(
            self._override(
                "chunking",
                "overlap_seconds",
                ini.getint("chunking", "overlap_seconds", fallback=12),
            )
        )
        silence_search_seconds = int(
            self._override(
                "chunking",
                "silence_search_seconds",
                ini.getint("chunking", "silence_search_seconds", fallback=45),
            )
        )
        silence_threshold_db = float(
            self._override(
                "chunking",
                "silence_threshold_db",
                ini.getfloat("chunking", "silence_threshold_db", fallback=-35.0),
            )
        )
        silence_min_ms = int(
            self._override(
                "chunking",
                "silence_min_ms",
                ini.getint("chunking", "silence_min_ms", fallback=500),
            )
        )
        max_chunk_megabytes = int(
            self._override(
                "chunking",
                "max_chunk_megabytes",
                ini.getint("chunking", "max_chunk_megabytes", fallback=24),
            )
        )
        disagreement_retry_limit = int(
            self._override(
                "quality",
                "disagreement_retry_limit",
                ini.getint("quality", "disagreement_retry_limit", fallback=1),
            )
        )
        arbitration_limit = int(
            self._override(
                "quality",
                "arbitration_limit",
                ini.getint("quality", "arbitration_limit", fallback=1),
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
        artifacts_raw = self._override(
            "paths", "artifacts_dir", ini.get("paths", "artifacts_dir", fallback="candidates")
        )
        reports_raw = self._override(
            "paths", "reports_dir", ini.get("paths", "reports_dir", fallback="reports")
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
            providers=providers,
            primary_provider=primary,
            xai_api_key=str(xai_api_key) if xai_api_key else None,
            xai_base_url=str(xai_base_url),
            meta_asr_url=str(meta_asr_url) if meta_asr_url else None,
            meta_asr_api_key=(
                str(meta_asr_api_key) if meta_asr_api_key else None
            ),
            meta_asr_model=str(meta_asr_model),
            meta_asr_language=str(meta_asr_language),
            provider_timeout_seconds=provider_timeout_seconds,
            chunk_target_seconds=chunk_target_seconds,
            chunk_max_seconds=chunk_max_seconds,
            chunk_overlap_seconds=chunk_overlap_seconds,
            silence_search_seconds=silence_search_seconds,
            silence_threshold_db=silence_threshold_db,
            silence_min_ms=silence_min_ms,
            max_chunk_megabytes=max_chunk_megabytes,
            disagreement_retry_limit=disagreement_retry_limit,
            arbitration_limit=arbitration_limit,
            artifacts_dir=_resolve_path(artifacts_raw, data_root),
            reports_dir=_resolve_path(reports_raw, data_root),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        source: SourceConfig,
        telegram: TelegramConfig,
        download: DownloadConfig,
        transcribe: TranscribeConfig,
        data_repo: DataRepoConfig,
        transcript_repo: DataRepoConfig,
        *,
        validation_profile: str = "run",
    ) -> None:
        errors: list[str] = []

        known_profiles = {
            "run",
            "download",
            "transcribe",
            "sync",
            "status",
            "debug",
            "ci-export",
            "session",
        }
        if validation_profile not in known_profiles:
            errors.append(f"unknown validation profile: {validation_profile!r}")
        requires_source = validation_profile in {"run", "download"}
        requires_telegram = validation_profile == "debug" or (
            requires_source and source.mode == "telegram"
        )
        requires_asr = validation_profile in {"run", "transcribe"}

        def _is_within(path: Path, root: Path) -> bool:
            try:
                path.resolve().relative_to(root.resolve())
                return True
            except ValueError:
                return False

        # Source mode validation
        valid_modes = {"telegram", "folder"}
        if requires_source and source.mode not in valid_modes:
            errors.append(
                f"source.mode must be one of {sorted(valid_modes)!r}, "
                f"got {source.mode!r}."
            )

        if requires_source and source.mode == "folder":
            # Folder mode only needs a folder path — no Telegram credentials.
            if source.folder is None:
                errors.append(
                    "source.folder is required when source.mode='folder'. "
                    "Set it in the [source] section of your INI file or pass "
                    "--folder PATH on the command line."
                )
        elif requires_telegram:
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

        allowed_providers = {"openai", "xai", "meta", "whisper"}
        unknown_providers = sorted(set(transcribe.providers) - allowed_providers)
        if requires_asr and unknown_providers:
            errors.append(
                "transcribe.providers contains unsupported provider(s): "
                + ", ".join(unknown_providers)
            )
        if requires_asr and not transcribe.providers:
            errors.append(
                "No audio transcription provider is configured. Set at least one "
                "of OPENAI_API_KEY, XAI_API_KEY, META_ASR_URL, or explicitly "
                "select whisper with OCTOSCRIBE_ASR_PROVIDERS."
            )
        if requires_asr and len(transcribe.providers) > 3:
            errors.append(
                "transcribe.providers supports at most three providers so the "
                "resolution architecture remains bounded."
            )
        if requires_asr and transcribe.primary_provider not in transcribe.providers:
            errors.append(
                f"primary provider {transcribe.primary_provider!r} is not enabled in "
                f"{list(transcribe.providers)!r}."
            )

        # Provider credentials are required only for providers that will run.
        if requires_asr and "openai" in transcribe.providers and not transcribe.api_key:
            errors.append(
                "OPENAI_API_KEY is required when the openai provider is enabled. "
                "Export it as an environment variable or add it to your .env file."
            )
        if requires_asr and "xai" in transcribe.providers and not transcribe.xai_api_key:
            errors.append(
                "XAI_API_KEY is required when the xai provider is enabled."
            )
        if requires_asr and "meta" in transcribe.providers and not transcribe.meta_asr_url:
            errors.append(
                "META_ASR_URL is required when the meta provider is enabled."
            )
        if requires_asr and not transcribe.language.strip():
            errors.append("transcribe.language must not be empty.")
        if requires_asr and "openai" in transcribe.providers and not transcribe.model.strip():
            errors.append("transcribe.model must not be empty for OpenAI.")
        if (
            requires_asr
            and "openai" in transcribe.providers
            and transcribe.model.casefold().startswith(
                "gpt-4o-transcribe-diarize"
            )
        ):
            errors.append(
                "OpenAI diarization models are not supported by the verbatim "
                "chunk pipeline; use gpt-transcribe."
            )
        if requires_asr and "whisper" in transcribe.providers and not transcribe.local_model.strip():
            errors.append("local_transcribe.model must not be empty for Whisper.")
        if requires_asr and "meta" in transcribe.providers:
            if not transcribe.meta_asr_model.strip():
                errors.append("meta_asr.model must not be empty.")
            if not transcribe.meta_asr_language.strip():
                errors.append("meta_asr.language must not be empty.")
        if requires_asr and transcribe.meta_asr_url:
            meta_host = urllib.parse.urlsplit(transcribe.meta_asr_url).hostname
            if "openai" in transcribe.providers and meta_host == "api.openai.com":
                errors.append(
                    "Meta ASR cannot point at api.openai.com while OpenAI is also "
                    "enabled; cross-check providers must be independent."
                )
            if "xai" in transcribe.providers and meta_host == "api.x.ai":
                errors.append(
                    "Meta ASR cannot point at api.x.ai while xAI is also enabled; "
                    "cross-check providers must be independent."
                )

        if not (1 <= transcribe.chunk_target_seconds <= transcribe.chunk_max_seconds):
            errors.append("chunking target_seconds must be between 1 and max_seconds.")
        if (
            transcribe.chunk_target_seconds + transcribe.chunk_overlap_seconds
            > transcribe.chunk_max_seconds
        ):
            errors.append(
                "chunking target_seconds plus overlap_seconds must not exceed "
                "max_seconds."
            )
        if transcribe.chunk_max_seconds > 600 and "openai" in transcribe.providers:
            errors.append(
                "chunking max_seconds must be <= 600 when OpenAI is enabled so "
                "16 kHz mono WAV chunks stay below the 25 MB API limit."
            )
        if not (0 <= transcribe.chunk_overlap_seconds < transcribe.chunk_target_seconds / 2):
            errors.append("chunking overlap_seconds must be non-negative and below half the target.")
        if not (1 <= transcribe.max_chunk_megabytes <= 24):
            errors.append("chunking max_chunk_megabytes must be between 1 and 24.")
        if not (0 <= transcribe.silence_search_seconds < transcribe.chunk_target_seconds):
            errors.append(
                "chunking silence_search_seconds must be non-negative and below "
                "target_seconds."
            )
        if transcribe.silence_min_ms <= 0:
            errors.append("chunking silence_min_ms must be positive.")
        if not (1 <= transcribe.workers <= 32):
            errors.append("transcribe workers must be between 1 and 32.")
        if not (1 <= transcribe.retry_attempts <= 5):
            errors.append("transcribe retry_attempts must be between 1 and 5.")
        if not (0 <= transcribe.retry_base_delay <= transcribe.retry_max_delay):
            errors.append(
                "transcribe retry_base_delay must be non-negative and no greater "
                "than retry_max_delay."
            )
        if transcribe.retry_max_delay > 300:
            errors.append("transcribe retry_max_delay must not exceed 300 seconds.")
        if not (1 <= transcribe.provider_timeout_seconds <= 3600):
            errors.append(
                "transcribe provider_timeout_seconds must be between 1 and 3600."
            )
        if not (0 <= transcribe.disagreement_retry_limit <= 1):
            errors.append("quality disagreement_retry_limit must be 0 or 1.")
        if not (0 <= transcribe.arbitration_limit <= 1):
            errors.append("quality arbitration_limit must be 0 or 1.")

        for label, repo in (
            ("audio_repo", data_repo),
            ("transcript_repo", transcript_repo),
        ):
            if repo.url:
                parsed_repo_url = urllib.parse.urlsplit(repo.url)
                if parsed_repo_url.username or parsed_repo_url.password:
                    errors.append(
                        f"{label}.url must not embed credentials; use an SSH agent "
                        "or credential helper."
                    )
            branch = repo.branch
            if (
                not branch
                or branch.startswith("-")
                or branch.endswith(("/", ".", ".lock"))
                or any(token in branch for token in ("..", "//", "@{"))
                or any(
                    character.isspace()
                    or ord(character) < 32
                    or character in "~^:?*[\\"
                    for character in branch
                )
            ):
                errors.append(f"{label}.branch is not a safe Git branch name.")

        audio_root = data_repo.path.resolve()
        text_root = transcript_repo.path.resolve()
        if audio_root != text_root and (
            _is_within(audio_root, text_root)
            or _is_within(text_root, audio_root)
        ):
            errors.append(
                "audio_repo.path and transcript_repo.path must not be nested; "
                "use separate worktrees (or the exact same path for legacy mode)."
            )

        path_ownership = (
            ("paths.audio_dir", download.audio_dir, audio_root),
            ("paths.manifest_file", download.manifest_file, text_root),
            (
                "paths.transcriptions_dir",
                transcribe.transcriptions_dir,
                text_root,
            ),
            ("paths.artifacts_dir", transcribe.artifacts_dir, text_root),
            ("paths.reports_dir", transcribe.reports_dir, text_root),
        )
        for label, path, root in path_ownership:
            if path is not None and not _is_within(path, root):
                errors.append(
                    f"{label} must remain inside its configured evidence "
                    f"repository ({root})."
                )

        if errors:
            _die(errors)

        # Non-fatal warning: evidence repositories inside the project directory.
        # __file__ is src/config/loader.py, so the project root is three
        # levels up (src/config -> src -> project root).
        project_dir = Path(__file__).resolve().parents[2]
        for label, repo in (("data_repo", data_repo), ("transcript_repo", transcript_repo)):
            try:
                repo.path.relative_to(project_dir)
                warnings.warn(
                    f"{label}.path ({repo.path}) is inside the project directory "
                    f"({project_dir}). Consider using a path outside the project tree.",
                    stacklevel=4,
                )
            except ValueError:
                pass  # Good – it is outside the project directory.
