"""
tests/test_debug.py — Pytest suite for src/debug.py (DebugInspector).

All Telethon calls are mocked; no real network or filesystem access is needed
beyond the temp directories provided by conftest fixtures.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from telethon.tl.types import (
    DocumentAttributeAudio,
    DocumentAttributeFilename,
    MessageMediaDocument,
)

from src.debug import DebugInspector


# ---------------------------------------------------------------------------
# Message factory helpers
# ---------------------------------------------------------------------------


def _make_audio_attr(title: str = "Test Sermon", duration: int = 120) -> DocumentAttributeAudio:
    """Create a real DocumentAttributeAudio instance with minimal fields."""
    attr = MagicMock(spec=DocumentAttributeAudio)
    attr.title = title
    attr.performer = "Test Performer"
    attr.duration = duration
    attr.voice = False
    return attr


def _make_filename_attr(name: str = "sermon.mp3") -> DocumentAttributeFilename:
    attr = MagicMock(spec=DocumentAttributeFilename)
    attr.file_name = name
    return attr


def _make_document(
    mime_type: str = "audio/mpeg",
    attributes: list | None = None,
    size: int = 1024 * 1024,
) -> MagicMock:
    doc = MagicMock()
    doc.id = 987654321
    doc.access_hash = 111222333
    doc.mime_type = mime_type
    doc.size = size
    doc.attributes = attributes or [_make_audio_attr(), _make_filename_attr()]
    doc.date = None
    return doc


def _make_media(document: MagicMock | None = None) -> MessageMediaDocument:
    media = MagicMock(spec=MessageMediaDocument)
    media.document = document or _make_document()
    return media


def _make_message(
    msg_id: int = 42,
    with_audio: bool = True,
    forwarded: bool = False,
) -> MagicMock:
    """Create a fake Telethon message."""
    from datetime import datetime, timezone

    msg = MagicMock()
    msg.id = msg_id
    msg.date = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    msg.message = "Test message text"
    msg.sender_id = 999
    msg.post_author = None

    if with_audio:
        doc = _make_document()
        msg.media = _make_media(doc)
        msg.document = doc
        msg.audio = doc
        msg.voice = None
    else:
        msg.media = None
        msg.document = None
        msg.audio = None
        msg.voice = None

    msg.file = MagicMock()
    msg.file.name = "sermon.mp3"
    msg.file.ext = ".mp3"
    msg.file.mime_type = "audio/mpeg"
    msg.file.size = 1024 * 1024

    if forwarded:
        msg.forward = SimpleNamespace(
            date=msg.date,
            sender_id=888,
            sender_name="Original Sender",
            channel_id=777,
        )
    else:
        msg.forward = None

    return msg


# ---------------------------------------------------------------------------
# Fixture: pre-patched DebugInspector
# ---------------------------------------------------------------------------


@pytest.fixture
def inspector(sample_config):
    """A DebugInspector with its TelegramClient replaced by a MagicMock."""
    with patch("src.debug.TelegramClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        insp = DebugInspector(sample_config, scan_limit=3)
        # Expose the mock for assertions.
        insp._mock_client = mock_client
        yield insp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDebugInspectorInit:
    def test_instantiation_stores_config_and_scan_limit(self, sample_config):
        """DebugInspector can be instantiated with config and scan_limit."""
        with patch("src.debug.TelegramClient"):
            insp = DebugInspector(sample_config, scan_limit=5)

        assert insp._config is sample_config
        assert insp._scan_limit == 5

    def test_default_scan_limit_is_three(self, sample_config):
        with patch("src.debug.TelegramClient"):
            insp = DebugInspector(sample_config)

        assert insp._scan_limit == 3


class TestDebugInspectorRun:
    @pytest.mark.asyncio
    async def test_run_connects_and_disconnects(self, sample_config):
        """DebugInspector.run() connects to Telegram and disconnects after."""
        with patch("src.debug.TelegramClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            entity = MagicMock()
            entity.title = "Test Church"
            entity.id = 123456
            mock_client.get_entity = AsyncMock(return_value=entity)
            # Return empty batch so we don't need real messages.
            mock_client.get_messages = AsyncMock(return_value=[])

            insp = DebugInspector(sample_config, scan_limit=3)
            await insp.run()

        mock_client.start.assert_called_once()
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_prints_connection_info(self, sample_config, capsys):
        """DebugInspector.run() prints connection info (group title, group ID) to stdout."""
        with patch("src.debug.TelegramClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            entity = MagicMock()
            entity.title = "My Test Church"
            entity.id = 987654
            mock_client.get_entity = AsyncMock(return_value=entity)
            mock_client.get_messages = AsyncMock(return_value=[])

            insp = DebugInspector(sample_config, scan_limit=3)
            await insp.run()

        captured = capsys.readouterr()
        assert "My Test Church" in captured.out
        assert "987654" in captured.out

    @pytest.mark.asyncio
    async def test_run_handles_zero_audio_messages_gracefully(self, sample_config, capsys):
        """DebugInspector.run() handles a group with zero audio messages gracefully."""
        with patch("src.debug.TelegramClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            entity = MagicMock()
            entity.title = "Silent Group"
            entity.id = 111
            mock_client.get_entity = AsyncMock(return_value=entity)
            # Non-audio message only.
            non_audio_msg = _make_message(with_audio=False)
            non_audio_msg.id = 1
            batch = MagicMock()
            batch.__iter__ = MagicMock(return_value=iter([non_audio_msg]))
            batch.__len__ = MagicMock(return_value=1)
            batch.__getitem__ = MagicMock(side_effect=lambda i: non_audio_msg)
            mock_client.get_messages = AsyncMock(side_effect=[
                [non_audio_msg],
                [],  # second call returns empty → loop ends
            ])

            insp = DebugInspector(sample_config, scan_limit=3)
            await insp.run()

        captured = capsys.readouterr()
        assert "no audio messages found" in captured.out.lower() or "found 0" in captured.out

    @pytest.mark.asyncio
    async def test_run_processes_up_to_scan_limit_only(self, sample_config, capsys):
        """DebugInspector.run() processes up to scan_limit messages, not more."""
        with patch("src.debug.TelegramClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            entity = MagicMock()
            entity.title = "Big Church"
            entity.id = 555
            mock_client.get_entity = AsyncMock(return_value=entity)

            # Build 5 audio messages but scan_limit is 2.
            messages = [_make_message(msg_id=i, with_audio=True) for i in range(1, 6)]
            for msg in messages:
                msg.id = messages.index(msg) + 1

            mock_client.get_messages = AsyncMock(return_value=messages)

            insp = DebugInspector(sample_config, scan_limit=2)
            await insp.run()

        captured = capsys.readouterr()
        # Only 2 audio headers should appear.
        assert "AUDIO MESSAGE #1" in captured.out
        assert "AUDIO MESSAGE #2" in captured.out
        assert "AUDIO MESSAGE #3" not in captured.out

    @pytest.mark.asyncio
    async def test_run_disconnects_even_on_error(self, sample_config):
        """DebugInspector.run() disconnects even if scanning raises an exception."""
        with patch("src.debug.TelegramClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.get_entity = AsyncMock(side_effect=RuntimeError("boom"))

            insp = DebugInspector(sample_config, scan_limit=3)
            with pytest.raises(RuntimeError, match="boom"):
                await insp.run()

        mock_client.disconnect.assert_called_once()


class TestPrintMessageDebug:
    @pytest.mark.asyncio
    async def test_print_message_debug_outputs_message_id_and_date(
        self, sample_config, capsys
    ):
        """DebugInspector._print_message_debug() outputs message ID and date."""
        with patch("src.debug.TelegramClient"):
            insp = DebugInspector(sample_config, scan_limit=3)

        msg = _make_message(msg_id=9001)
        await insp._print_message_debug(msg, index=1)

        captured = capsys.readouterr()
        assert "9001" in captured.out
        assert "2024-01-15" in captured.out

    @pytest.mark.asyncio
    async def test_print_message_debug_outputs_audio_metadata_section(
        self, sample_config, capsys
    ):
        """_print_message_debug() prints the AudioMetadata section with title and extension."""
        with patch("src.debug.TelegramClient"):
            insp = DebugInspector(sample_config, scan_limit=3)

        msg = _make_message(msg_id=5000)
        await insp._print_message_debug(msg, index=1)

        captured = capsys.readouterr()
        assert "AUDIO METADATA" in captured.out
        assert "msg_id" in captured.out

    @pytest.mark.asyncio
    async def test_print_message_debug_shows_forward_info_when_forwarded(
        self, sample_config, capsys
    ):
        """_print_message_debug() shows forward info when the message is forwarded."""
        with patch("src.debug.TelegramClient"):
            insp = DebugInspector(sample_config, scan_limit=3)

        msg = _make_message(msg_id=7777, forwarded=True)
        await insp._print_message_debug(msg, index=1)

        captured = capsys.readouterr()
        assert "FORWARD INFO" in captured.out
        assert "Original Sender" in captured.out

    @pytest.mark.asyncio
    async def test_print_message_debug_shows_not_forwarded_for_plain_messages(
        self, sample_config, capsys
    ):
        """_print_message_debug() prints '(not forwarded)' for plain messages."""
        with patch("src.debug.TelegramClient"):
            insp = DebugInspector(sample_config, scan_limit=3)

        msg = _make_message(msg_id=8888, forwarded=False)
        await insp._print_message_debug(msg, index=1)

        captured = capsys.readouterr()
        assert "not forwarded" in captured.out
