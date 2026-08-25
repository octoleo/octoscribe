"""
tests/test_transcribe_hardening.py — Stability guarantees of the transcriber.

Two robustness behaviours were added when the transcription package was
refactored, both aimed at never silently losing a transcript:

* colliding output names no longer overwrite each other; and
* an empty transcript is recorded as a failure (retried next run), not as a
  misleading "completed" empty file.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.transcribe import TranscriptionBackend, Transcriber


def _make_config(tmp_path: Path):
    transcribe = SimpleNamespace(
        backend="openai",
        workers=1,
        transcriptions_dir=tmp_path / "transcriptions",
        manifest_file=tmp_path / "manifest.json",
    )
    download = SimpleNamespace(
        audio_dir=tmp_path / "audio",
        manifest_file=tmp_path / "manifest.json",
    )
    return SimpleNamespace(transcribe=transcribe, download=download)


def _make_audio(tmp_path: Path, name: str) -> Path:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    p = audio_dir / name
    p.write_bytes(b"AUDIO")
    return p


def _make_manifest(pending: list[dict]) -> MagicMock:
    m = MagicMock()
    m.pending_transcription.return_value = pending
    return m


# ---------------------------------------------------------------------------
# Collision-safe output
# ---------------------------------------------------------------------------

def test_same_title_does_not_clobber(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _make_audio(tmp_path, "a.ogg")
    _make_audio(tmp_path, "b.ogg")
    pending = [
        {"telegram_msg_id": "1", "filename": "a.ogg", "title": "Same Title"},
        {"telegram_msg_id": "2", "filename": "b.ogg", "title": "Same Title"},
    ]
    manifest = _make_manifest(pending)

    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "openai"
    backend.transcribe.side_effect = ["First sermon.", "Second sermon."]

    stats = Transcriber(config, manifest, backend=backend).run()

    out_dir = config.transcribe.transcriptions_dir
    written = sorted(p.name for p in out_dir.iterdir())
    # Two distinct files, the second disambiguated by message id.
    assert written == ["Same Title.txt", "Same Title_2.txt"]
    assert stats.succeeded == 2
    contents = {(out_dir / n).read_text() for n in written}
    assert contents == {"First sermon.", "Second sermon."}


# ---------------------------------------------------------------------------
# Empty-result guard
# ---------------------------------------------------------------------------

def test_empty_transcript_is_failure(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _make_audio(tmp_path, "silent.ogg")
    pending = [{"telegram_msg_id": "9", "filename": "silent.ogg", "title": None}]
    manifest = _make_manifest(pending)

    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "openai"
    backend.transcribe.return_value = "   \n  \n"  # whitespace only

    stats = Transcriber(config, manifest, backend=backend).run()

    assert stats.failed == 1
    assert stats.succeeded == 0
    manifest.mark_transcribed.assert_not_called()
    manifest.mark_failed.assert_called_once()
    # No empty file should have been written.
    out_dir = config.transcribe.transcriptions_dir
    assert not any(out_dir.iterdir()) if out_dir.exists() else True


@pytest.mark.parametrize(
    "filename",
    ["../secret.mp3", "/etc/passwd.mp3", "nested/sermon.mp3", "secret.env"],
)
def test_tampered_manifest_path_is_never_uploaded(
    tmp_path: Path, filename: str
) -> None:
    config = _make_config(tmp_path)
    manifest = _make_manifest(
        [{"telegram_msg_id": "10", "filename": filename, "title": None}]
    )
    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "openai"

    stats = Transcriber(config, manifest, backend=backend).run()

    assert stats.failed == 1
    backend.transcribe.assert_not_called()
    manifest.mark_failed.assert_called_once()


def test_symlinked_manifest_audio_is_never_uploaded(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    audio_dir = config.download.audio_dir
    audio_dir.mkdir(parents=True)
    outside = tmp_path / "private.mp3"
    outside.write_bytes(b"PRIVATE")
    (audio_dir / "sermon.mp3").symlink_to(outside)
    manifest = _make_manifest(
        [{"telegram_msg_id": "11", "filename": "sermon.mp3", "title": None}]
    )
    backend = MagicMock(spec=TranscriptionBackend)
    backend.name = "openai"

    stats = Transcriber(config, manifest, backend=backend).run()

    assert stats.failed == 1
    backend.transcribe.assert_not_called()
    manifest.mark_failed.assert_called_once()
