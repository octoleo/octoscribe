"""
tests/test_telegram_client.py — Tests for src/telegram_client.py.

These shared helpers carry the session-bootstrap and entity-resolution logic
that the downloader and the debug inspector used to duplicate.  They depend on
no Telethon import, so they are tested directly with a mock client.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.telegram_client import (
    SESSION_NAME,
    resolve_group_entity,
    restore_session_from_env,
    session_base_path,
)


# ---------------------------------------------------------------------------
# session_base_path
# ---------------------------------------------------------------------------

def test_session_base_path(tmp_path: Path) -> None:
    result = session_base_path(tmp_path)
    assert result == str(tmp_path / SESSION_NAME)
    # No extension — Telethon appends ".session" itself.
    assert not result.endswith(".session")


# ---------------------------------------------------------------------------
# restore_session_from_env
# ---------------------------------------------------------------------------

class TestRestoreSessionFromEnv:
    def test_returns_false_when_unset(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("TELEGRAM_SESSION_B64", raising=False)
        assert restore_session_from_env(tmp_path) is False

    def test_returns_false_when_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("TELEGRAM_SESSION_B64", "  ")
        assert restore_session_from_env(tmp_path) is False

    def test_writes_session_file(self, tmp_path: Path, monkeypatch) -> None:
        payload = b"session bytes"
        monkeypatch.setenv("TELEGRAM_SESSION_B64", base64.b64encode(payload).decode())
        assert restore_session_from_env(tmp_path) is True
        assert (tmp_path / f"{SESSION_NAME}.session").read_bytes() == payload

    def test_creates_missing_dir(self, tmp_path: Path, monkeypatch) -> None:
        target = tmp_path / "nested" / ".session"
        monkeypatch.setenv("TELEGRAM_SESSION_B64", base64.b64encode(b"x").decode())
        assert restore_session_from_env(target) is True
        assert target.exists()

    def test_invalid_base64_returns_false_and_writes_nothing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("TELEGRAM_SESSION_B64", "!!!not base64!!!")
        assert restore_session_from_env(tmp_path) is False
        assert not (tmp_path / f"{SESSION_NAME}.session").exists()


# ---------------------------------------------------------------------------
# resolve_group_entity
# ---------------------------------------------------------------------------

class TestResolveGroupEntity:
    @pytest.mark.asyncio
    async def test_numeric_id_passed_as_int(self) -> None:
        client = AsyncMock()
        client.get_entity.return_value = "entity"
        result = await resolve_group_entity(client, "-1001234567890")
        assert result == "entity"
        client.get_entity.assert_awaited_once_with(-1001234567890)

    @pytest.mark.asyncio
    async def test_username_passed_as_str(self) -> None:
        client = AsyncMock()
        client.get_entity.return_value = "entity"
        await resolve_group_entity(client, "@my_group")
        client.get_entity.assert_awaited_once_with("@my_group")

    @pytest.mark.asyncio
    async def test_strips_whitespace(self) -> None:
        client = AsyncMock()
        await resolve_group_entity(client, "  @padded  ")
        client.get_entity.assert_awaited_once_with("@padded")
