"""
tests/test_telegram.py — Pytest suite for src/telegram.py.

Uses pytest-asyncio for async tests and unittest.mock / pytest-mock to stub
all Telethon calls.  No real network or filesystem interaction is required
beyond tmp_path.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import pytest_asyncio

from telethon.tl.types import (
    DocumentAttributeAudio,
    DocumentAttributeFilename,
    MessageMediaDocument,
)

from src.telegram import (
    AudioMetadata,
    DownloadStats,
    TelegramDownloader,
    build_filename,
    format_duration,
    get_audio_metadata,
    is_audio,
    sanitize_filename,
)


# ---------------------------------------------------------------------------
# Helpers for building fake Telethon message objects
# ---------------------------------------------------------------------------


def _make_document(
    mime_type: str = "audio/ogg",
    attributes: list | None = None,
) -> MagicMock:
    doc = MagicMock()
    doc.mime_type = mime_type
    doc.attributes = attributes or []
    return doc


def _make_media(document: MagicMock | None = None) -> MessageMediaDocument:
    media = MagicMock(spec=MessageMediaDocument)
    media.document = document
    return media


def _make_message(
    msg_id: int = 1,
    media=None,
    date_str: str = "2024-03-15",
) -> MagicMock:
    from datetime import datetime, timezone

    msg = MagicMock()
    msg.id = msg_id
    msg.media = media
    msg.date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return msg


def _make_audio_message(
    msg_id: int = 42,
    title: str | None = "My Track",
    performer: str | None = "DJ Test",
    duration: int = 180,
    mime: str = "audio/mpeg",
    original_filename: str | None = None,
) -> MagicMock:
    attrs: list = []
    audio_attr = MagicMock(spec=DocumentAttributeAudio)
    audio_attr.title = title
    audio_attr.performer = performer
    audio_attr.duration = duration
    attrs.append(audio_attr)

    if original_filename:
        fn_attr = MagicMock(spec=DocumentAttributeFilename)
        fn_attr.file_name = original_filename
        attrs.append(fn_attr)

    doc = _make_document(mime_type=mime, attributes=attrs)
    media = _make_media(doc)
    return _make_message(msg_id=msg_id, media=media)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_config(tmp_path: Path):
    """Return a minimal Config-like object without calling Config.load()."""
    from src.config import TelegramConfig, DownloadConfig, Config, TranscribeConfig, DataRepoConfig

    tg = TelegramConfig(
        api_id=12345,
        api_hash="testhash",
        phone="+10000000000",
        group="@testgroup",
        session_dir=tmp_path / "session",
    )
    dl = DownloadConfig(
        workers=2,
        resume=True,
        deduplicate=True,
        audio_dir=tmp_path / "audio",
        manifest_file=tmp_path / "manifest.json",
    )
    tr = TranscribeConfig(
        backend="openai",
        model="gpt-4o-transcribe",
        language="en",
        workers=2,
        retry_attempts=3,
        retry_base_delay=2.5,
        retry_max_delay=30.0,
        api_key=None,
        local_model="large-v3",
        device="cpu",
        compute_type="int8",
        beam_size=5,
        best_of=5,
        repetition_penalty=1.1,
        vad_filter=True,
        vad_min_silence_ms=500,
        vad_speech_pad_ms=400,
        transcriptions_dir=tmp_path / "transcriptions",
        manifest_file=tmp_path / "manifest.json",
    )
    dr = DataRepoConfig(path=tmp_path / "data")
    return Config(telegram=tg, download=dl, transcribe=tr, data_repo=dr, ini_path=tmp_path / "octoscribe.ini")


@pytest.fixture()
def mock_manifest():
    """Return a MagicMock that satisfies the Manifest interface."""
    m = MagicMock()
    m.is_downloaded.return_value = False
    m.get_entry.return_value = None
    m.all_entries.return_value = {}
    return m


# ---------------------------------------------------------------------------
# AudioMetadata value object
# ---------------------------------------------------------------------------


class TestAudioMetadata:
    def test_construction_full(self):
        meta = AudioMetadata(
            msg_id=1,
            title="Song",
            performer="Artist",
            duration=240,
            extension=".mp3",
            original_filename="song.mp3",
            date="2024-01-01",
        )
        assert meta.msg_id == 1
        assert meta.title == "Song"
        assert meta.performer == "Artist"
        assert meta.duration == 240
        assert meta.extension == ".mp3"
        assert meta.original_filename == "song.mp3"
        assert meta.date == "2024-01-01"

    def test_defaults(self):
        meta = AudioMetadata(msg_id=99)
        assert meta.title is None
        assert meta.performer is None
        assert meta.duration is None
        assert meta.extension == ".ogg"
        assert meta.original_filename is None
        assert meta.date is None

    def test_frozen(self):
        meta = AudioMetadata(msg_id=1)
        with pytest.raises(Exception):
            meta.title = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_removes_invalid_chars(self):
        result = sanitize_filename('bad<>:"/\\|?*name')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "\\" not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_removes_control_chars(self):
        result = sanitize_filename("hello\x00\x1f\x7fworld")
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "\x7f" not in result
        assert "helloworld" == result

    def test_strips_leading_trailing_whitespace_and_dots(self):
        assert sanitize_filename("  ..hello..  ") == "hello"

    def test_limits_to_200_chars(self):
        long_name = "a" * 300
        assert len(sanitize_filename(long_name)) == 200

    def test_normal_name_unchanged(self):
        assert sanitize_filename("My Great Track") == "My Great Track"


# ---------------------------------------------------------------------------
# build_filename
# ---------------------------------------------------------------------------


class TestBuildFilename:
    def test_uses_title_when_present(self):
        meta = AudioMetadata(msg_id=1, title="Best Song", extension=".mp3")
        assert build_filename(meta) == "Best Song.mp3"

    def test_falls_back_to_original_filename(self):
        meta = AudioMetadata(
            msg_id=2,
            title=None,
            original_filename="my_track.ogg",
            extension=".ogg",
        )
        result = build_filename(meta)
        assert result == "my_track.ogg"

    def test_skips_record_ogg_fallback(self):
        meta = AudioMetadata(
            msg_id=3,
            title=None,
            original_filename="record.ogg",
            date="2024-06-01",
            extension=".ogg",
        )
        result = build_filename(meta)
        assert result == "audio_2024-06-01_3.ogg"

    def test_falls_back_to_audio_date_msgid(self):
        meta = AudioMetadata(
            msg_id=7,
            title=None,
            original_filename=None,
            date="2024-12-25",
            extension=".mp3",
        )
        assert build_filename(meta) == "audio_2024-12-25_7.mp3"

    def test_sanitizes_title(self):
        meta = AudioMetadata(msg_id=5, title='Bad/Name<Here>', extension=".flac")
        result = build_filename(meta)
        assert "/" not in result
        assert "<" not in result
        assert result.endswith(".flac")


# ---------------------------------------------------------------------------
# get_audio_metadata
# ---------------------------------------------------------------------------


class TestGetAudioMetadata:
    def test_extracts_from_document_attribute_audio(self):
        msg = _make_audio_message(
            msg_id=10,
            title="Deep House",
            performer="DJ X",
            duration=300,
            mime="audio/mpeg",
        )
        meta = get_audio_metadata(msg)
        assert meta.title == "Deep House"
        assert meta.performer == "DJ X"
        assert meta.duration == 300
        assert meta.msg_id == 10

    def test_extension_from_mime_type(self):
        cases = [
            ("audio/mpeg", ".mp3"),
            ("audio/flac", ".flac"),
            ("audio/x-m4a", ".m4a"),
            ("audio/wav", ".wav"),
            ("audio/aac", ".aac"),
            ("audio/opus", ".opus"),
            ("application/ogg", ".ogg"),
        ]
        for mime, expected_ext in cases:
            msg = _make_audio_message(mime=mime)
            meta = get_audio_metadata(msg)
            assert meta.extension == expected_ext, f"MIME {mime!r} → expected {expected_ext!r}, got {meta.extension!r}"

    def test_date_extracted(self):
        msg = _make_audio_message(msg_id=5)
        meta = get_audio_metadata(msg)
        assert meta.date == "2024-03-15"

    def test_non_media_message_returns_defaults(self):
        msg = _make_message(media=None)
        meta = get_audio_metadata(msg)
        assert meta.title is None
        assert meta.extension == ".ogg"

    def test_original_filename_extracted(self):
        msg = _make_audio_message(
            msg_id=20,
            original_filename="concert_2024.mp3",
        )
        meta = get_audio_metadata(msg)
        assert meta.original_filename == "concert_2024.mp3"


# ---------------------------------------------------------------------------
# is_audio
# ---------------------------------------------------------------------------


class TestIsAudio:
    def test_true_for_document_attribute_audio(self):
        msg = _make_audio_message()
        assert is_audio(msg) is True

    def test_false_for_no_media(self):
        msg = _make_message(media=None)
        assert is_audio(msg) is False

    def test_false_for_non_document_media(self):
        msg = MagicMock()
        msg.media = MagicMock()  # Not a MessageMediaDocument instance
        # Ensure isinstance check fails
        with patch("src.telegram.MessageMediaDocument", MessageMediaDocument):
            assert is_audio(msg) is False

    def test_true_for_audio_mime(self):
        doc = _make_document(mime_type="audio/flac", attributes=[])
        media = _make_media(doc)
        msg = _make_message(media=media)
        assert is_audio(msg) is True

    def test_true_for_audio_extension_in_filename(self):
        fn_attr = MagicMock(spec=DocumentAttributeFilename)
        fn_attr.file_name = "track.mp3"
        doc = _make_document(mime_type="application/octet-stream", attributes=[fn_attr])
        media = _make_media(doc)
        msg = _make_message(media=media)
        assert is_audio(msg) is True


# ---------------------------------------------------------------------------
# DownloadStats.summary
# ---------------------------------------------------------------------------


class TestDownloadStats:
    def test_summary_is_non_empty_string(self):
        stats = DownloadStats(downloaded=3, skipped=1, duplicate=0, failed=1)
        summary = stats.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_includes_counts(self):
        stats = DownloadStats(downloaded=5, skipped=2, duplicate=1, failed=0)
        s = stats.summary()
        assert "5" in s
        assert "2" in s
        assert "1" in s


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_seconds_only(self):
        assert format_duration(75) == "1:15"

    def test_zero_seconds(self):
        assert format_duration(0) is None

    def test_none(self):
        assert format_duration(None) is None

    def test_with_hours(self):
        assert format_duration(3661) == "1:01:01"

    def test_exactly_one_hour(self):
        assert format_duration(3600) == "1:00:00"

    def test_short_duration(self):
        assert format_duration(59) == "0:59"


# ---------------------------------------------------------------------------
# TelegramDownloader.__init__ session path
# ---------------------------------------------------------------------------


class TestTelegramDownloaderInit:
    def test_session_path_inside_session_dir(self, minimal_config, mock_manifest, tmp_path):
        with patch("src.telegram.TelegramClient") as MockClient:
            dl = TelegramDownloader(minimal_config, mock_manifest)
            call_args = MockClient.call_args
            session_arg: str = call_args[0][0]
            assert session_arg.startswith(str(minimal_config.telegram.session_dir))
            assert "octoscribe" in session_arg


# ---------------------------------------------------------------------------
# TelegramDownloader.run() — async integration tests
# ---------------------------------------------------------------------------


def _make_mock_client(messages: list, entity=None):
    """Build a mock TelegramClient that returns *messages* from get_messages."""
    client = AsyncMock()

    async def _get_messages(ent, limit=100, offset_id=0):
        # Return all messages on first call, empty list on second to stop loop
        if offset_id == 0:
            batch = messages[:]
            # Simulate the offset_id progression
            if batch:
                return batch
        return []

    client.get_messages = _get_messages
    client.get_entity = AsyncMock(return_value=entity or MagicMock())
    client.download_media = AsyncMock()
    client.start = AsyncMock()
    client.disconnect = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_run_skips_already_downloaded(minimal_config, mock_manifest, tmp_path):
    """Resume skips only when the recorded hash matches the existing file."""
    minimal_config.download.audio_dir.mkdir(parents=True, exist_ok=True)
    existing_file = minimal_config.download.audio_dir / "existing.mp3"
    content = b"data"
    existing_file.write_bytes(content)

    msg = _make_audio_message(msg_id=100, title="Existing")
    mock_manifest.is_downloaded.return_value = True
    mock_manifest.get_entry.return_value = {
        "filename": "existing.mp3",
        "downloaded": True,
        "hash": hashlib.sha256(content).hexdigest(),
    }

    with patch("src.telegram.TelegramClient") as MockClient:
        mock_client = _make_mock_client([msg])
        MockClient.return_value = mock_client

        dl = TelegramDownloader(minimal_config, mock_manifest)
        dl._client = mock_client
        stats = await dl.run()

    assert stats.skipped == 1
    assert stats.downloaded == 0
    mock_client.download_media.assert_not_called()


@pytest.mark.asyncio
async def test_run_calls_mark_downloaded_for_new_files(minimal_config, mock_manifest, tmp_path):
    """New audio messages should be downloaded and recorded in the manifest."""
    minimal_config.download.audio_dir.mkdir(parents=True, exist_ok=True)
    downloaded_file = minimal_config.download.audio_dir / "My Track.mp3"

    msg = _make_audio_message(msg_id=200, title="My Track", mime="audio/mpeg")
    mock_manifest.is_downloaded.return_value = False
    mock_manifest.all_entries.return_value = {}

    async def fake_download_media(message, file=None):
        # Simulate Telethon writing the file and returning the path
        p = Path(file)
        p.write_bytes(b"audio content here")
        return str(p)

    with patch("src.telegram.TelegramClient") as MockClient:
        mock_client = _make_mock_client([msg])
        mock_client.download_media = fake_download_media
        MockClient.return_value = mock_client

        dl = TelegramDownloader(minimal_config, mock_manifest)
        dl._client = mock_client
        stats = await dl.run()

    assert stats.downloaded == 1
    mock_manifest.mark_downloaded.assert_called_once()
    call_kwargs = mock_manifest.mark_downloaded.call_args
    assert call_kwargs[0][0] == 200  # msg_id
    metadata_dict = call_kwargs[0][1]
    assert "filename" in metadata_dict
    assert "hash" in metadata_dict


@pytest.mark.asyncio
async def test_run_preserves_historical_ogg_manifest_contract(
    minimal_config, mock_manifest
):
    """Telegram OGG downloads retain the established manifest fields exactly."""
    from datetime import datetime, timezone

    minimal_config.download.audio_dir.mkdir(parents=True, exist_ok=True)
    content = b"deterministic telegram ogg fixture bytes"
    expected_hash = hashlib.sha256(content).hexdigest()
    msg = _make_audio_message(
        msg_id=10,
        title="Matthew 24:39-42",
        performer="Family Devotions",
        duration=1824,
        mime="audio/ogg",
        original_filename="record.ogg",
    )
    msg.date = datetime(2023, 2, 9, tzinfo=timezone.utc)
    mock_manifest.is_downloaded.return_value = False
    mock_manifest.all_entries.return_value = {}

    async def fake_download_media(message, file=None):
        path = Path(file)
        path.write_bytes(content)
        return str(path)

    with patch("src.telegram.TelegramClient") as client_type:
        client = _make_mock_client([msg])
        client.download_media = fake_download_media
        client_type.return_value = client

        downloader = TelegramDownloader(minimal_config, mock_manifest)
        downloader._client = client
        stats = await downloader.run()

    assert stats.downloaded == 1
    assert (minimal_config.download.audio_dir / "Matthew 2439-42.ogg").read_bytes() == content
    mock_manifest.mark_downloaded.assert_called_once_with(
        10,
        {
            "filename": "Matthew 2439-42.ogg",
            "title": "Matthew 24:39-42",
            "performer": "Family Devotions",
            "date": "2023-02-09",
            "duration": 1824,
            "duration_formatted": "30:24",
            "extension": ".ogg",
            "hash": expected_hash,
            "original_filename": "record.ogg",
        },
    )


@pytest.mark.asyncio
async def test_run_calls_mark_failed_on_download_error(minimal_config, mock_manifest, tmp_path):
    """A download exception should call mark_failed and count as failed."""
    minimal_config.download.audio_dir.mkdir(parents=True, exist_ok=True)
    msg = _make_audio_message(msg_id=300, title="Broken")
    mock_manifest.is_downloaded.return_value = False

    with patch("src.telegram.TelegramClient") as MockClient:
        mock_client = _make_mock_client([msg])
        mock_client.download_media = AsyncMock(side_effect=RuntimeError("network error"))
        MockClient.return_value = mock_client

        dl = TelegramDownloader(minimal_config, mock_manifest)
        dl._client = mock_client
        stats = await dl.run()

    assert stats.failed == 1
    mock_manifest.mark_failed.assert_called_once_with(300, "download", "network error")


@pytest.mark.asyncio
async def test_run_deduplication_removes_duplicate_file(minimal_config, mock_manifest, tmp_path):
    """A verified duplicate is removed and its Telegram message is recorded."""
    minimal_config.download.audio_dir.mkdir(parents=True, exist_ok=True)

    content = b"duplicate audio bytes"
    existing_hash = hashlib.sha256(content).hexdigest()
    original = minimal_config.download.audio_dir / "original.mp3"
    original.write_bytes(content)

    msg = _make_audio_message(msg_id=400, title="Dup Track")
    mock_manifest.is_downloaded.return_value = False
    mock_manifest.all_entries.return_value = {
        "1": {"hash": existing_hash, "downloaded": True, "filename": "original.mp3"}
    }

    async def fake_download_media(message, file=None):
        p = Path(file)
        p.write_bytes(content)
        return str(p)

    with patch("src.telegram.TelegramClient") as MockClient:
        mock_client = _make_mock_client([msg])
        mock_client.download_media = fake_download_media
        MockClient.return_value = mock_client

        dl = TelegramDownloader(minimal_config, mock_manifest)
        dl._client = mock_client
        stats = await dl.run()

    assert stats.duplicate == 1
    assert stats.downloaded == 0
    # Only the verified canonical file remains; the temporary duplicate is gone.
    remaining = list(minimal_config.download.audio_dir.iterdir())
    assert remaining == [original]
    duplicate_entry = mock_manifest.mark_downloaded.call_args.args[1]
    assert duplicate_entry["filename"] == "original.mp3"
    assert duplicate_entry["hash"] == existing_hash
    assert duplicate_entry["duplicate"] is True
    assert duplicate_entry["duplicate_of"] == 1


@pytest.mark.asyncio
async def test_resume_reacquires_hash_mismatch_without_losing_historical_metadata(
    minimal_config,
):
    """A corrupt local copy is preserved and replaced only by expected bytes."""
    from src.manifest import Manifest

    audio_dir = minimal_config.download.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    trusted_content = b"the original Telegram object"
    corrupt_content = b"locally corrupted bytes"
    trusted_hash = hashlib.sha256(trusted_content).hexdigest()
    target = audio_dir / "Historical Sermon.ogg"
    target.write_bytes(corrupt_content)

    manifest = Manifest(minimal_config.download.manifest_file)
    historical = {
        "date": "2024-11-25",
        "duration": 1412,
        "duration_formatted": "23:32",
        "extension": ".ogg",
        "filename": target.name,
        "hash": trusted_hash,
        "original_filename": "record.ogg",
        "performer": "Family Devotions",
        "title": "1 Timothy 1:5-6",
        "custom_historical_field": "must survive recovery",
    }
    manifest.mark_downloaded(856, historical)

    msg = _make_audio_message(
        msg_id=856,
        title="Changed Telegram title must not rewrite history",
        performer="Changed performer",
        duration=1,
        mime="audio/ogg",
        original_filename="different.ogg",
    )

    async def fake_download_media(message, file=None):
        path = Path(file)
        path.write_bytes(trusted_content)
        return str(path)

    with patch("src.telegram.TelegramClient") as client_type:
        client = _make_mock_client([msg])
        client.download_media = fake_download_media
        client_type.return_value = client
        downloader = TelegramDownloader(minimal_config, manifest)
        downloader._client = client
        stats = await downloader.run()

    assert stats.downloaded == 1
    assert stats.failed == 0
    assert target.read_bytes() == trusted_content
    quarantined = list(audio_dir.glob(f"{target.name}.integrity-mismatch-*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupt_content

    recovered = manifest.get_entry(856)
    assert recovered is not None
    for field, value in historical.items():
        assert recovered[field] == value
    assert recovered["telegram_msg_id"] == 856


@pytest.mark.asyncio
async def test_resume_rejects_remote_hash_mismatch_without_rewriting_evidence(
    minimal_config,
):
    """Changed Telegram bytes never replace a recorded source or its hash."""
    from src.manifest import Manifest

    audio_dir = minimal_config.download.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    trusted_hash = hashlib.sha256(b"historically trusted content").hexdigest()
    local_content = b"suspect local content"
    remote_content = b"Telegram now returned different content"
    remote_hash = hashlib.sha256(remote_content).hexdigest()
    target = audio_dir / "Sermon.ogg"
    target.write_bytes(local_content)

    manifest = Manifest(minimal_config.download.manifest_file)
    historical = {
        "filename": target.name,
        "hash": trusted_hash,
        "title": "Historical title",
        "performer": "Historical performer",
        "date": "2025-02-06",
        "duration": 1850,
        "duration_formatted": "30:50",
        "extension": ".ogg",
        "original_filename": "record.ogg",
    }
    manifest.mark_downloaded(990, historical)
    msg = _make_audio_message(msg_id=990, title="Sermon", mime="audio/ogg")

    async def fake_download_media(message, file=None):
        path = Path(file)
        path.write_bytes(remote_content)
        return str(path)

    with patch("src.telegram.TelegramClient") as client_type:
        client = _make_mock_client([msg])
        client.download_media = fake_download_media
        client_type.return_value = client
        downloader = TelegramDownloader(minimal_config, manifest)
        downloader._client = client
        stats = await downloader.run()

    assert stats.failed == 1
    assert stats.downloaded == 0
    assert target.read_bytes() == local_content
    assert not list(audio_dir.glob("*.corrupt"))
    assert not list(audio_dir.glob(".octoscribe-*"))

    rejected = manifest.get_entry(990)
    assert rejected is not None
    for field, value in historical.items():
        assert rejected[field] == value
    assert rejected["failed_stage"] == "download"
    assert trusted_hash in rejected["failed_error"]
    assert remote_hash in rejected["failed_error"]


@pytest.mark.asyncio
async def test_duplicate_message_is_persisted_and_not_downloaded_on_next_run(
    minimal_config,
):
    """Content deduplication is terminal acquisition state for each message ID."""
    from src.manifest import Manifest

    audio_dir = minimal_config.download.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    content = b"one canonical Telegram audio object"
    content_hash = hashlib.sha256(content).hexdigest()
    canonical = audio_dir / "Canonical.ogg"
    canonical.write_bytes(content)

    manifest = Manifest(minimal_config.download.manifest_file)
    manifest.mark_downloaded(
        100,
        {
            "filename": canonical.name,
            "hash": content_hash,
            "title": "Canonical",
            "performer": "Family Devotions",
            "date": "2025-01-01",
            "duration": 10,
            "duration_formatted": "0:10",
            "extension": ".ogg",
            "original_filename": "record.ogg",
        },
    )
    duplicate_msg = _make_audio_message(
        msg_id=101,
        title="Reposted sermon",
        performer="Family Devotions",
        duration=10,
        mime="audio/ogg",
        original_filename="record.ogg",
    )

    async def fake_download_media(message, file=None):
        path = Path(file)
        path.write_bytes(content)
        return str(path)

    with patch("src.telegram.TelegramClient") as client_type:
        first_client = _make_mock_client([duplicate_msg])
        first_client.download_media = AsyncMock(side_effect=fake_download_media)
        client_type.return_value = first_client
        first_downloader = TelegramDownloader(minimal_config, manifest)
        first_downloader._client = first_client
        first_stats = await first_downloader.run()

    assert first_stats.duplicate == 1
    assert first_client.download_media.await_count == 1
    duplicate_entry = manifest.get_entry(101)
    assert duplicate_entry is not None
    assert duplicate_entry["downloaded"] is True
    assert duplicate_entry["telegram_msg_id"] == 101
    assert duplicate_entry["filename"] == canonical.name
    assert duplicate_entry["hash"] == content_hash
    assert duplicate_entry["duplicate"] is True
    assert duplicate_entry["duplicate_of"] == 100
    assert duplicate_entry["title"] == "Reposted sermon"
    assert list(audio_dir.iterdir()) == [canonical]

    # The persisted duplicate entry now passes the same hash-verified resume
    # gate as its canonical message and requires no second network transfer.
    with patch("src.telegram.TelegramClient") as client_type:
        second_client = _make_mock_client([duplicate_msg])
        client_type.return_value = second_client
        second_downloader = TelegramDownloader(minimal_config, manifest)
        second_downloader._client = second_client
        second_stats = await second_downloader.run()

    assert second_stats.skipped == 1
    assert second_stats.duplicate == 0
    second_client.download_media.assert_not_called()


@pytest.mark.asyncio
async def test_run_saves_manifest_periodically(minimal_config, mock_manifest, tmp_path):
    """Manifest should be saved at least once during a run with downloads."""
    minimal_config.download.audio_dir.mkdir(parents=True, exist_ok=True)

    # Create 10 distinct messages to trigger a periodic save
    messages = [_make_audio_message(msg_id=i, title=f"Track {i}") for i in range(1, 11)]
    mock_manifest.is_downloaded.return_value = False
    mock_manifest.all_entries.return_value = {}

    call_count = 0

    async def fake_download_media(message, file=None):
        nonlocal call_count
        p = Path(file)
        # Use unique content per message to avoid dedup
        p.write_bytes(f"audio content {call_count}".encode())
        call_count += 1
        return str(p)

    # Patch get_messages to return all 10 at once
    entity = MagicMock()

    first_call = True

    async def _get_messages(ent, limit=100, offset_id=0):
        nonlocal first_call
        if first_call:
            first_call = False
            return messages[:]
        return []

    with patch("src.telegram.TelegramClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get_entity = AsyncMock(return_value=entity)
        mock_client.get_messages = _get_messages
        mock_client.download_media = fake_download_media
        mock_client.start = AsyncMock()
        mock_client.disconnect = AsyncMock()
        MockClient.return_value = mock_client

        dl = TelegramDownloader(minimal_config, mock_manifest)
        dl._client = mock_client
        stats = await dl.run()

    # At least the final save plus one periodic save (after 10 downloads)
    assert mock_manifest.save.call_count >= 2
    assert stats.downloaded == 10


# ---------------------------------------------------------------------------
# Session restore tests
# ---------------------------------------------------------------------------

class TestRestoreSessionFromEnv:
    def test_returns_false_when_env_var_not_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TELEGRAM_SESSION_B64", raising=False)
        result = TelegramDownloader._restore_session_from_env(tmp_path)
        assert result is False

    def test_returns_false_for_empty_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_SESSION_B64", "")
        result = TelegramDownloader._restore_session_from_env(tmp_path)
        assert result is False

    def test_writes_session_file_when_env_var_set(self, tmp_path, monkeypatch):
        import base64
        session_bytes = b"fake session data 12345"
        b64 = base64.b64encode(session_bytes).decode()
        monkeypatch.setenv("TELEGRAM_SESSION_B64", b64)
        result = TelegramDownloader._restore_session_from_env(tmp_path)
        assert result is True
        session_file = tmp_path / "octoscribe.session"
        assert session_file.exists()
        assert session_file.read_bytes() == session_bytes

    def test_creates_session_dir_if_missing(self, tmp_path, monkeypatch):
        import base64
        session_dir = tmp_path / "nested" / ".session"
        b64 = base64.b64encode(b"data").decode()
        monkeypatch.setenv("TELEGRAM_SESSION_B64", b64)
        result = TelegramDownloader._restore_session_from_env(session_dir)
        assert result is True
        assert session_dir.exists()

    def test_returns_false_for_invalid_base64(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_SESSION_B64", "!!!not-valid-base64!!!")
        result = TelegramDownloader._restore_session_from_env(tmp_path)
        assert result is False
        # No file should be created
        assert not (tmp_path / "octoscribe.session").exists()

    def test_init_calls_restore_session(self, tmp_path, monkeypatch):
        """TelegramDownloader.__init__ calls _restore_session_from_env."""
        import base64
        from unittest.mock import patch, MagicMock
        from src.config import (
            Config, TelegramConfig, DownloadConfig, TranscribeConfig, DataRepoConfig
        )

        session_bytes = b"session content"
        b64 = base64.b64encode(session_bytes).decode()
        monkeypatch.setenv("TELEGRAM_SESSION_B64", b64)

        session_dir = tmp_path / ".session"
        tg_cfg = TelegramConfig(
            api_id=12345,
            api_hash="hash",
            phone="+1234567890",
            group="@test",
            session_dir=session_dir,
        )
        dl_cfg = DownloadConfig(
            workers=2,
            resume=True,
            deduplicate=True,
            audio_dir=tmp_path / "audio",
            manifest_file=tmp_path / "manifest.json",
        )
        tr_cfg = TranscribeConfig(
            backend="openai",
            model="gpt-4o-transcribe",
            language="en",
            workers=2,
            retry_attempts=3,
            retry_base_delay=1.0,
            retry_max_delay=10.0,
            api_key="sk-test",
            local_model="large-v3",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            best_of=5,
            repetition_penalty=1.1,
            vad_filter=True,
            vad_min_silence_ms=500,
            vad_speech_pad_ms=400,
            transcriptions_dir=tmp_path / "transcriptions",
            manifest_file=tmp_path / "manifest.json",
        )
        dr_cfg = DataRepoConfig(path=tmp_path)
        config = Config(
            telegram=tg_cfg,
            download=dl_cfg,
            transcribe=tr_cfg,
            data_repo=dr_cfg,
            ini_path=tmp_path / "octoscribe.ini",
        )
        mock_manifest = MagicMock()

        with patch("src.telegram.TelegramClient"):
            TelegramDownloader(config, mock_manifest)

        session_file = session_dir / "octoscribe.session"
        assert session_file.exists()
        assert session_file.read_bytes() == session_bytes
