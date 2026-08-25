"""
Comprehensive pytest tests for src/config.py.

Run with:
    pytest tests/test_config.py -v
"""

from __future__ import annotations

import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import pytest

# Ensure the src package is importable when tests are run from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config, _parse_bool, _resolve_path  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_ENV: dict[str, str] = {
    "TELEGRAM_API_ID": "12345",
    "TELEGRAM_API_HASH": "abc123hash",
    "TELEGRAM_PHONE": "+15550001234",
}

_OPENAI_ENV: dict[str, str] = {
    **_REQUIRED_ENV,
    "OPENAI_API_KEY": "sk-test-openaikey",
}


def _write_ini(tmp_path: Path, content: str) -> Path:
    """Write an INI file to a temp directory and return its path."""
    ini_file = tmp_path / "octoscribe.ini"
    ini_file.write_text(textwrap.dedent(content), encoding="utf-8")
    return ini_file


def _minimal_env(monkeypatch: pytest.MonkeyPatch, extra: dict[str, str] | None = None) -> None:
    """
    Set up a clean environment that satisfies all *required* secrets plus any
    *extra* variables the caller needs.  All other env vars are left intact
    (we only set, never delete, because deleting arbitrary env vars can break
    the Python runtime itself).
    """
    for key, val in _OPENAI_ENV.items():
        monkeypatch.setenv(key, val)
    if extra:
        for key, val in extra.items():
            monkeypatch.setenv(key, val)


# ---------------------------------------------------------------------------
# 1. Loading from INI only (env provides required secrets)
# ---------------------------------------------------------------------------

class TestIniLoading:
    def test_ini_values_override_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Values present in the INI file replace built-in defaults."""
        _minimal_env(monkeypatch)
        ini = _write_ini(
            tmp_path,
            """\
            [telegram]
            group = my-church-group

            [download]
            workers = 8
            resume = false

            [transcribe]
            language = af
            workers = 2
            """,
        )
        cfg = Config.load(ini_path=ini)

        assert cfg.telegram.group == "my-church-group"
        assert cfg.download.workers == 8
        assert cfg.download.resume is False
        assert cfg.transcribe.language == "af"
        assert cfg.transcribe.workers == 2

    def test_ini_path_stored_on_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config.ini_path reflects the resolved path of the INI file."""
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "")
        cfg = Config.load(ini_path=ini)
        assert cfg.ini_path == ini.resolve()

    def test_missing_ini_uses_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing INI file is not an error; all defaults apply."""
        _minimal_env(monkeypatch)
        missing = tmp_path / "no_such.ini"
        cfg = Config.load(ini_path=missing)
        # Default backend
        assert cfg.transcribe.backend == "openai"
        # Default workers
        assert cfg.download.workers == 4

    def test_partial_ini_leaves_other_defaults_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An INI that only overrides one section leaves the others at defaults."""
        _minimal_env(monkeypatch)
        ini = _write_ini(
            tmp_path,
            """\
            [download]
            workers = 16
            """,
        )
        cfg = Config.load(ini_path=ini)

        assert cfg.download.workers == 16
        # transcribe section untouched → default
        assert cfg.transcribe.backend == "openai"
        assert cfg.transcribe.retry_attempts == 1
        # local_transcribe section untouched → default
        assert cfg.transcribe.local_model == "large-v3"
        assert cfg.transcribe.vad_filter is False


# ---------------------------------------------------------------------------
# 2. Env-var only (no INI file)
# ---------------------------------------------------------------------------

class TestEnvVarLoading:
    def test_secrets_loaded_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Telegram credentials come from environment variables."""
        monkeypatch.setenv("TELEGRAM_API_ID", "99999")
        monkeypatch.setenv("TELEGRAM_API_HASH", "hashfromenv")
        monkeypatch.setenv("TELEGRAM_PHONE", "+27820001234")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        missing_ini = tmp_path / "none.ini"

        cfg = Config.load(ini_path=missing_ini)

        assert cfg.telegram.api_id == 99999
        assert cfg.telegram.api_hash == "hashfromenv"
        assert cfg.telegram.phone == "+27820001234"
        assert cfg.transcribe.api_key == "sk-env-key"

    def test_repository_git_environment_is_not_part_of_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source-control settings belong to the calling workflow."""
        _minimal_env(
            monkeypatch,
            extra={
                "DATA_REPO_URL": "https://example.invalid/data",
                "DATA_REPO_BRANCH": "external-branch",
                "DATA_REPO_AUTO_PUSH": "true",
            },
        )
        cfg = Config.load(ini_path=tmp_path / "none.ini")
        assert not hasattr(cfg.data_repo, "url")
        assert not hasattr(cfg.data_repo, "branch")
        assert not hasattr(cfg.data_repo, "auto_push")

    def test_provider_auto_discovery_uses_only_configured_audio_services(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key, value in _REQUIRED_ENV.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        monkeypatch.delenv("META_ASR_URL", raising=False)

        cfg = Config.load(
            ini_path=tmp_path / "none.ini", env_file=tmp_path / "none.env"
        )

        assert cfg.transcribe.providers == ("xai",)
        assert cfg.transcribe.primary_provider == "xai"

    def test_provider_auto_discovery_orders_openai_xai_then_meta(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _minimal_env(
            monkeypatch,
            extra={
                "XAI_API_KEY": "xai-test",
                "META_ASR_URL": "http://localhost:8080",
            },
        )
        cfg = Config.load(ini_path=tmp_path / "none.ini")
        assert cfg.transcribe.providers == ("openai", "xai", "meta")
        assert cfg.transcribe.meta_asr_language == "eng_Latn"
        assert cfg.transcribe.model == "gpt-transcribe"

    def test_plain_api_key_override_cannot_cross_provider_boundaries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _minimal_env(
            monkeypatch,
            extra={
                "XAI_API_KEY": "xai-only",
                "META_ASR_URL": "http://localhost:8080",
                "META_ASR_API_KEY": "meta-only",
            },
        )

        cfg = Config.load(
            ini_path=tmp_path / "none.ini",
            api_key="ambiguous-must-be-ignored",
        )

        assert cfg.transcribe.api_key == "sk-test-openaikey"
        assert cfg.transcribe.xai_api_key == "xai-only"
        assert cfg.transcribe.meta_asr_api_key == "meta-only"

    def test_section_qualified_provider_secret_overrides_are_isolated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _minimal_env(monkeypatch)
        cfg = Config.load(
            ini_path=tmp_path / "none.ini",
            transcribe__api_key="openai-override",
            xai__api_key="xai-override",
        )
        assert cfg.transcribe.api_key == "openai-override"
        assert cfg.transcribe.xai_api_key == "xai-override"

    def test_workspace_environment_splits_audio_and_transcripts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _minimal_env(
            monkeypatch,
            extra={
                "AUDIO_REPO_PATH": str(tmp_path / "audio-repo"),
                "TRANSCRIPT_REPO_PATH": str(tmp_path / "text-repo"),
            },
        )
        cfg = Config.load(ini_path=tmp_path / "none.ini")
        assert cfg.audio_repo.path == (tmp_path / "audio-repo").resolve()
        assert cfg.text_repo.path == (tmp_path / "text-repo").resolve()
        assert cfg.download.audio_dir.parent == cfg.audio_repo.path
        assert cfg.transcribe.transcriptions_dir.parent == cfg.text_repo.path
        assert cfg.download.manifest_file.parent == cfg.text_repo.path

    def test_default_session_directory_is_outside_evidence_repositories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _minimal_env(monkeypatch)
        cfg = Config.load(ini_path=tmp_path / "none.ini")
        assert not cfg.telegram.session_dir.is_relative_to(cfg.audio_repo.path)
        assert not cfg.telegram.session_dir.is_relative_to(cfg.text_repo.path)


class TestCommandValidationProfiles:
    @staticmethod
    def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "TELEGRAM_API_ID",
            "TELEGRAM_API_HASH",
            "TELEGRAM_PHONE",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
            "META_ASR_URL",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_folder_download_requires_neither_telegram_nor_asr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_credentials(monkeypatch)
        source = tmp_path / "incoming"
        source.mkdir()
        ini = _write_ini(
            tmp_path,
            f"[source]\nmode = folder\nfolder = {source}\n",
        )
        cfg = Config.load(ini_path=ini, validation_profile="download")
        assert cfg.source.mode == "folder"
        assert cfg.transcribe.providers == ()

    def test_telegram_download_does_not_require_asr_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_credentials(monkeypatch)
        for key, value in _REQUIRED_ENV.items():
            monkeypatch.setenv(key, value)
        cfg = Config.load(
            ini_path=tmp_path / "none.ini",
            validation_profile="download",
        )
        assert cfg.source.mode == "telegram"
        assert cfg.transcribe.providers == ()

    def test_transcribe_does_not_require_telegram_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_credentials(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-only")
        cfg = Config.load(
            ini_path=tmp_path / "none.ini",
            validation_profile="transcribe",
        )
        assert cfg.transcribe.providers == ("openai",)
        assert cfg.telegram.api_id is None

    @pytest.mark.parametrize("profile", ["status", "session", "ci-export"])
    def test_non_processing_commands_require_no_service_credentials(
        self,
        profile: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._clear_credentials(monkeypatch)
        cfg = Config.load(
            ini_path=tmp_path / "none.ini",
            validation_profile=profile,
        )
        assert cfg.transcribe.providers == ()


# ---------------------------------------------------------------------------
# 3. .env file loading
# ---------------------------------------------------------------------------

class TestDotEnvLoading:
    def test_dotenv_file_populates_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Secrets written to a .env file are picked up via python-dotenv."""
        pytest.importorskip("dotenv", reason="python-dotenv not installed")

        # Ensure no conflicting env vars from the test runner.
        for key in _OPENAI_ENV:
            monkeypatch.delenv(key, raising=False)

        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join([
                "TELEGRAM_API_ID=55555",
                "TELEGRAM_API_HASH=dotenvhash",
                "TELEGRAM_PHONE=+15559999",
                "OPENAI_API_KEY=sk-dotenv-key",
            ]),
            encoding="utf-8",
        )

        cfg = Config.load(ini_path=tmp_path / "none.ini", env_file=env_file)

        assert cfg.telegram.api_id == 55555
        assert cfg.telegram.api_hash == "dotenvhash"
        assert cfg.transcribe.api_key == "sk-dotenv-key"

    def test_env_var_wins_over_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An env var already in os.environ takes precedence over .env values."""
        pytest.importorskip("dotenv", reason="python-dotenv not installed")

        # Pre-set the env var *before* loading (simulating an already-exported var).
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-from-shell")
        for key in ["TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_PHONE"]:
            monkeypatch.setenv(key, _REQUIRED_ENV[key])

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-dotenv-should-lose\n", encoding="utf-8")

        cfg = Config.load(ini_path=tmp_path / "none.ini", env_file=env_file)

        # python-dotenv uses override=False so the shell var wins.
        assert cfg.transcribe.api_key == "sk-real-key-from-shell"


# ---------------------------------------------------------------------------
# 4. CLI overrides
# ---------------------------------------------------------------------------

class TestCliOverrides:
    def test_override_beats_ini(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kwarg override takes precedence over the INI value."""
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "[download]\nworkers = 4\n")
        cfg = Config.load(ini_path=ini, download__workers=32)
        assert cfg.download.workers == 32

    def test_override_beats_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kwarg override takes precedence over an environment variable for non-secrets."""
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "")
        cfg = Config.load(ini_path=ini, transcribe__language="zu")
        assert cfg.transcribe.language == "zu"

    def test_multiple_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple simultaneous overrides all take effect."""
        _minimal_env(monkeypatch)
        cfg = Config.load(
            ini_path=tmp_path / "none.ini",
            download__workers=12,
            transcribe__backend="local",
            transcribe__language="fr",
        )
        assert cfg.download.workers == 12
        assert cfg.transcribe.backend == "local"
        assert cfg.transcribe.language == "fr"

    def test_override_group_plain_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plain (non-prefixed) keys in overrides are also respected."""
        _minimal_env(monkeypatch)
        cfg = Config.load(ini_path=tmp_path / "none.ini", group="plain-key-group")
        assert cfg.telegram.group == "plain-key-group"


# ---------------------------------------------------------------------------
# 5. Missing required secrets → SystemExit
# ---------------------------------------------------------------------------

class TestMissingSecrets:
    def test_missing_api_id_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent TELEGRAM_API_ID must cause a SystemExit."""
        monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
        monkeypatch.setenv("TELEGRAM_API_HASH", "hash")
        monkeypatch.setenv("TELEGRAM_PHONE", "+1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

        with pytest.raises(SystemExit):
            Config.load(ini_path=tmp_path / "none.ini", env_file=tmp_path / "none.env")

    def test_missing_api_hash_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent TELEGRAM_API_HASH must cause a SystemExit."""
        monkeypatch.setenv("TELEGRAM_API_ID", "1")
        monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
        monkeypatch.setenv("TELEGRAM_PHONE", "+1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

        with pytest.raises(SystemExit):
            Config.load(ini_path=tmp_path / "none.ini", env_file=tmp_path / "none.env")

    def test_missing_phone_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent TELEGRAM_PHONE must cause a SystemExit."""
        monkeypatch.setenv("TELEGRAM_API_ID", "1")
        monkeypatch.setenv("TELEGRAM_API_HASH", "hash")
        monkeypatch.delenv("TELEGRAM_PHONE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

        with pytest.raises(SystemExit):
            Config.load(ini_path=tmp_path / "none.ini", env_file=tmp_path / "none.env")

    def test_exit_message_mentions_variable_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The SystemExit error output should name the missing variable."""
        monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
        monkeypatch.setenv("TELEGRAM_API_ID", "1")
        monkeypatch.setenv("TELEGRAM_PHONE", "+1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

        with pytest.raises(SystemExit):
            Config.load(ini_path=tmp_path / "none.ini", env_file=tmp_path / "none.env")

        captured = capsys.readouterr()
        assert "TELEGRAM_API_HASH" in captured.err


# ---------------------------------------------------------------------------
# 6. Invalid api_id (non-integer) → SystemExit
# ---------------------------------------------------------------------------

class TestInvalidApiId:
    def test_non_integer_api_id_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-integer TELEGRAM_API_ID must cause a SystemExit."""
        monkeypatch.setenv("TELEGRAM_API_ID", "not-a-number")
        monkeypatch.setenv("TELEGRAM_API_HASH", "hash")
        monkeypatch.setenv("TELEGRAM_PHONE", "+1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

        with pytest.raises(SystemExit):
            Config.load(ini_path=tmp_path / "none.ini")

    def test_float_api_id_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A float string for TELEGRAM_API_ID must cause a SystemExit."""
        monkeypatch.setenv("TELEGRAM_API_ID", "3.14")
        monkeypatch.setenv("TELEGRAM_API_HASH", "hash")
        monkeypatch.setenv("TELEGRAM_PHONE", "+1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

        with pytest.raises(SystemExit):
            Config.load(ini_path=tmp_path / "none.ini")

    def test_error_message_names_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("TELEGRAM_API_ID", "abc")
        monkeypatch.setenv("TELEGRAM_API_HASH", "hash")
        monkeypatch.setenv("TELEGRAM_PHONE", "+1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

        with pytest.raises(SystemExit):
            Config.load(ini_path=tmp_path / "none.ini")

        captured = capsys.readouterr()
        assert "TELEGRAM_API_ID" in captured.err


# ---------------------------------------------------------------------------
# 7. Invalid backend → SystemExit
# ---------------------------------------------------------------------------

class TestInvalidBackend:
    def test_invalid_backend_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognised backend value must cause a SystemExit."""
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "[transcribe]\nbackend = whisperx\n")

        with pytest.raises(SystemExit):
            Config.load(ini_path=ini)

    def test_invalid_backend_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "[transcribe]\nbackend = azure\n")

        with pytest.raises(SystemExit):
            Config.load(ini_path=ini)

        captured = capsys.readouterr()
        assert "backend" in captured.err.lower()

    def test_openai_diarization_model_is_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _minimal_env(monkeypatch)
        ini = _write_ini(
            tmp_path,
            "[transcribe]\nmodel = gpt-4o-transcribe-diarize\n",
        )

        with pytest.raises(SystemExit):
            Config.load(ini_path=ini)

        assert "diarization models are not supported" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 8. OpenAI backend without API key → SystemExit
# ---------------------------------------------------------------------------

class TestOpenAiBackendWithoutKey:
    def test_openai_backend_requires_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """backend=openai without OPENAI_API_KEY must cause a SystemExit."""
        for key in ["TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_PHONE"]:
            monkeypatch.setenv(key, _REQUIRED_ENV[key])
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ini = _write_ini(tmp_path, "[transcribe]\nbackend = openai\n")

        with pytest.raises(SystemExit):
            Config.load(ini_path=ini, env_file=tmp_path / "none.env")

    def test_local_backend_does_not_require_openai_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """backend=local must succeed even without OPENAI_API_KEY."""
        for key in ["TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_PHONE"]:
            monkeypatch.setenv(key, _REQUIRED_ENV[key])
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ini = _write_ini(tmp_path, "[transcribe]\nbackend = local\n")

        cfg = Config.load(ini_path=ini, env_file=tmp_path / "none.env")
        assert cfg.transcribe.backend == "local"
        assert cfg.transcribe.api_key is None

    def test_error_message_mentions_openai_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        for key in ["TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_PHONE"]:
            monkeypatch.setenv(key, _REQUIRED_ENV[key])
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(SystemExit):
            Config.load(ini_path=tmp_path / "none.ini", env_file=tmp_path / "none.env")

        captured = capsys.readouterr()
        assert "OPENAI_API_KEY" in captured.err


# ---------------------------------------------------------------------------
# 9. All paths are absolute after loading
# ---------------------------------------------------------------------------

class TestAbsolutePaths:
    def test_all_path_fields_are_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every Path attribute on Config must be absolute."""
        _minimal_env(monkeypatch)
        cfg = Config.load(ini_path=tmp_path / "none.ini")

        path_attrs = [
            cfg.telegram.session_dir,
            cfg.download.audio_dir,
            cfg.download.manifest_file,
            cfg.transcribe.transcriptions_dir,
            cfg.transcribe.manifest_file,
            cfg.data_repo.path,
            cfg.ini_path,
        ]
        for p in path_attrs:
            assert p.is_absolute(), f"{p} is not absolute"

    def test_relative_paths_in_ini_become_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Relative [paths] entries are resolved against data_repo.path."""
        _minimal_env(monkeypatch)
        ini = _write_ini(
            tmp_path,
            """\
            [data_repo]
            path = /tmp/octoscribe_data_test

            [paths]
            audio_dir = my_audio
            transcriptions_dir = my_transcripts
            """,
        )
        cfg = Config.load(ini_path=ini)

        assert cfg.download.audio_dir == Path("/tmp/octoscribe_data_test/my_audio")
        assert cfg.transcribe.transcriptions_dir == Path(
            "/tmp/octoscribe_data_test/my_transcripts"
        )


# ---------------------------------------------------------------------------
# 10. Default data_repo path is outside the project directory
# ---------------------------------------------------------------------------

class TestDataRepoPath:
    def test_default_data_repo_outside_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default ~/.octoscribe/data path must lie outside the project tree."""
        _minimal_env(monkeypatch)
        cfg = Config.load(ini_path=tmp_path / "none.ini")

        project_dir = Path(__file__).parent.parent.resolve()
        try:
            cfg.data_repo.path.relative_to(project_dir)
            pytest.fail(
                f"data_repo.path ({cfg.data_repo.path}) is inside the project directory"
            )
        except ValueError:
            pass  # Good – it is outside.


class TestEvidencePathContainment:
    def test_rejects_nested_split_workspaces(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _minimal_env(monkeypatch)
        with pytest.raises(SystemExit):
            Config.load(
                ini_path=tmp_path / "none.ini",
                audio_repo__path=tmp_path / "evidence",
                transcript_repo__path=tmp_path / "evidence" / "text",
            )

    @pytest.mark.parametrize(
        "override",
        [
            "paths__audio_dir",
            "paths__manifest_file",
            "paths__transcriptions_dir",
            "paths__artifacts_dir",
            "paths__reports_dir",
        ],
    )
    def test_rejects_evidence_paths_outside_owning_workspace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        override: str,
    ) -> None:
        _minimal_env(monkeypatch)
        with pytest.raises(SystemExit):
            Config.load(
                ini_path=tmp_path / "none.ini",
                audio_repo__path=tmp_path / "audio-repo",
                transcript_repo__path=tmp_path / "text-repo",
                **{override: tmp_path / "outside"},
            )


class TestSecuritySensitiveValidation:
    def test_meta_provider_requires_language_identifier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _minimal_env(
            monkeypatch,
            extra={"META_ASR_URL": "http://127.0.0.1:9000"},
        )
        with pytest.raises(SystemExit):
            Config.load(
                ini_path=tmp_path / "none.ini",
                transcribe__providers="openai,meta",
                meta_asr__language="",
            )

    def test_custom_data_repo_path_respected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom data_repo.path set in the INI file is used as-is."""
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, f"[data_repo]\npath = {tmp_path / 'mydata'}\n")
        cfg = Config.load(ini_path=ini)
        assert cfg.data_repo.path == (tmp_path / "mydata").resolve()

    def test_data_repo_inside_project_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A data_repo.path inside the project tree should emit a UserWarning."""
        _minimal_env(monkeypatch)
        project_dir = Path(__file__).parent.parent.resolve()
        inside_path = project_dir / "data"

        ini = _write_ini(tmp_path, f"[data_repo]\npath = {inside_path}\n")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Config.load(ini_path=ini)

        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("project" in m.lower() or "data_repo" in m.lower() for m in messages), (
            f"Expected a project-directory warning, got: {messages}"
        )


# ---------------------------------------------------------------------------
# 11. redacted_repr() does not expose secrets
# ---------------------------------------------------------------------------

class TestRedactedRepr:
    def test_secrets_are_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """redacted_repr() must not contain the actual secret values."""
        _minimal_env(monkeypatch)
        cfg = Config.load(ini_path=tmp_path / "none.ini")
        redacted = cfg.redacted_repr()

        assert "abc123hash" not in redacted, "api_hash leaked"
        assert "+15550001234" not in redacted, "phone leaked"
        assert "sk-test-openaikey" not in redacted, "openai key leaked"

    def test_redacted_repr_uses_placeholder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The placeholder *** must appear in redacted_repr() output."""
        _minimal_env(monkeypatch)
        cfg = Config.load(ini_path=tmp_path / "none.ini")
        assert "***" in cfg.redacted_repr()

    def test_non_secret_values_visible(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-secret values (e.g. group, backend) must remain visible."""
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "[telegram]\ngroup = visible-group\n")
        cfg = Config.load(ini_path=ini)
        redacted = cfg.redacted_repr()
        assert "visible-group" in redacted
        assert "openai" in redacted


# ---------------------------------------------------------------------------
# 12. Boolean and numeric INI parsing correctness
# ---------------------------------------------------------------------------

class TestValueParsing:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_parse_bool_variants(self, raw: str, expected: bool) -> None:
        assert _parse_bool(raw) is expected

    def test_parse_bool_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_bool("maybe")

    def test_numeric_fields_parsed_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Numeric INI values are converted to the correct Python types."""
        _minimal_env(monkeypatch)
        ini = _write_ini(
            tmp_path,
            """\
            [transcribe]
            retry_base_delay = 5.0
            retry_max_delay = 60.0
            retry_attempts = 5

            [local_transcribe]
            repetition_penalty = 1.25
            beam_size = 10
            vad_min_silence_ms = 750
            """,
        )
        cfg = Config.load(ini_path=ini)

        assert cfg.transcribe.retry_base_delay == pytest.approx(5.0)
        assert cfg.transcribe.retry_max_delay == pytest.approx(60.0)
        assert cfg.transcribe.retry_attempts == 5
        assert cfg.transcribe.repetition_penalty == pytest.approx(1.25)
        assert cfg.transcribe.beam_size == 10
        assert cfg.transcribe.vad_min_silence_ms == 750


# ---------------------------------------------------------------------------
# 13. _resolve_path helper
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_absolute_path_returned_unchanged(self) -> None:
        base = Path("/some/base")
        absolute = Path("/absolute/path")
        assert _resolve_path(absolute, base) == absolute

    def test_relative_path_joined_to_base(self) -> None:
        base = Path("/some/base")
        result = _resolve_path("subdir/file.json", base)
        assert result == Path("/some/base/subdir/file.json")

    def test_tilde_expanded(self) -> None:
        base = Path("/irrelevant")
        result = _resolve_path("~/mydir", base)
        assert result.is_absolute()
        assert "~" not in str(result)


# ---------------------------------------------------------------------------
# 14. OCTOSCRIBE_CONFIG env var overrides the INI path
# ---------------------------------------------------------------------------

class TestOctoscribeConfigEnvVar:
    def test_env_var_points_to_ini(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OCTOSCRIBE_CONFIG env var selects which INI file to use."""
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "[download]\nworkers = 7\n")
        monkeypatch.setenv("OCTOSCRIBE_CONFIG", str(ini))

        # Do NOT pass ini_path= so the env var is the only source.
        cfg = Config.load()
        assert cfg.download.workers == 7
        assert cfg.ini_path == ini.resolve()


# ---------------------------------------------------------------------------
# 15. Combined loading: env + INI + overrides in correct priority order
# ---------------------------------------------------------------------------

class TestCombinedPriority:
    def test_priority_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Override > env > INI > default.

        We test with the ``transcribe.workers`` setting:
        - Default: 4
        - INI: 6
        - (No env var for this setting – it's not a secret.)
        - Override: 10
        Final value must be 10.
        """
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "[transcribe]\nworkers = 6\n")

        cfg = Config.load(ini_path=ini, transcribe__workers=10)
        assert cfg.transcribe.workers == 10

    def test_ini_beats_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """INI value beats the built-in default."""
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "[transcribe]\nworkers = 6\n")

        cfg = Config.load(ini_path=ini)
        assert cfg.transcribe.workers == 6

    def test_default_when_nothing_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Built-in default used when nothing else overrides."""
        _minimal_env(monkeypatch)
        cfg = Config.load(ini_path=tmp_path / "none.ini")
        assert cfg.transcribe.workers == 4  # built-in default


# ---------------------------------------------------------------------------
# 16. Audio source selection (telegram | folder)
# ---------------------------------------------------------------------------

def _no_telegram_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all Telegram/OpenAI secrets so folder mode can be tested clean."""
    for key in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_PHONE", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


class TestSourceConfig:
    def test_default_source_is_telegram(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When nothing is configured the source defaults to Telegram."""
        _minimal_env(monkeypatch)
        cfg = Config.load(ini_path=tmp_path / "none.ini")
        assert cfg.source.mode == "telegram"
        assert cfg.source.folder is None
        assert cfg.source.recursive is True

    def test_folder_mode_does_not_require_telegram_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Folder mode loads successfully with no Telegram credentials at all."""
        _no_telegram_env(monkeypatch)
        folder = tmp_path / "sermons"
        folder.mkdir()
        ini = _write_ini(
            tmp_path,
            f"""\
            [source]
            mode = folder
            folder = {folder}

            [transcribe]
            backend = local
            """,
        )
        cfg = Config.load(ini_path=ini, env_file=tmp_path / "none.env")

        assert cfg.source.mode == "folder"
        assert cfg.source.folder == folder.resolve()
        assert cfg.telegram.api_id is None  # not required, not set

    def test_folder_mode_requires_folder_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Folder mode without a folder path is a fatal configuration error."""
        _no_telegram_env(monkeypatch)
        ini = _write_ini(
            tmp_path,
            """\
            [source]
            mode = folder

            [transcribe]
            backend = local
            """,
        )
        with pytest.raises(SystemExit):
            Config.load(ini_path=ini, env_file=tmp_path / "none.env")

    def test_folder_mode_missing_folder_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _no_telegram_env(monkeypatch)
        ini = _write_ini(
            tmp_path, "[source]\nmode = folder\n\n[transcribe]\nbackend = local\n"
        )
        with pytest.raises(SystemExit):
            Config.load(ini_path=ini, env_file=tmp_path / "none.env")
        captured = capsys.readouterr()
        assert "source.folder" in captured.err

    def test_invalid_source_mode_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognised source.mode must cause a SystemExit."""
        _minimal_env(monkeypatch)
        ini = _write_ini(tmp_path, "[source]\nmode = ftp\n")
        with pytest.raises(SystemExit):
            Config.load(ini_path=ini)

    def test_folder_path_resolved_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative source.folder is expanded and made absolute."""
        _no_telegram_env(monkeypatch)
        ini = _write_ini(
            tmp_path,
            "[source]\nmode = folder\nfolder = ~/sermons\n\n[transcribe]\nbackend = local\n",
        )
        cfg = Config.load(ini_path=ini, env_file=tmp_path / "none.env")
        assert cfg.source.folder is not None
        assert cfg.source.folder.is_absolute()
        assert "~" not in str(cfg.source.folder)

    def test_folder_mode_via_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Folder mode can be selected purely via CLI-style overrides."""
        _no_telegram_env(monkeypatch)
        folder = tmp_path / "sermons"
        folder.mkdir()
        cfg = Config.load(
            ini_path=tmp_path / "none.ini",
            env_file=tmp_path / "none.env",
            source__mode="folder",
            source__folder=str(folder),
            transcribe__backend="local",
        )
        assert cfg.source.mode == "folder"
        assert cfg.source.folder == folder.resolve()

    def test_folder_mode_still_requires_openai_key_for_openai_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Folder mode does not exempt the OpenAI key when backend=openai."""
        _no_telegram_env(monkeypatch)
        folder = tmp_path / "sermons"
        folder.mkdir()
        ini = _write_ini(
            tmp_path,
            f"[source]\nmode = folder\nfolder = {folder}\n\n[transcribe]\nbackend = openai\n",
        )
        with pytest.raises(SystemExit):
            Config.load(ini_path=ini, env_file=tmp_path / "none.env")
