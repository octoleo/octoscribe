"""
src/debug.py — Debug inspector for Telegram connection and audio metadata.

Used to diagnose connection issues and understand message structure before
running the full pipeline.  All output goes to stdout so it can be piped
or redirected by the caller.
"""

from __future__ import annotations

import logging
from typing import Any

from telethon import TelegramClient
from telethon.tl.types import MessageMediaDocument

from src.config import Config
from src.telegram import (
    get_audio_metadata,
    is_audio,
)
from src.telegram_client import (
    resolve_group_entity,
    restore_session_from_env,
    session_base_path,
)

log = logging.getLogger(__name__)


class DebugInspector:
    """
    Inspects Telegram connection and dumps audio message metadata.

    Used to diagnose connection issues and understand message structure
    before running the full pipeline.

    Usage::

        inspector = DebugInspector(config, scan_limit=3)
        await inspector.run()
    """

    def __init__(self, config: Config, scan_limit: int = 3) -> None:
        self._config = config
        self._scan_limit = scan_limit
        config.telegram.session_dir.mkdir(parents=True, exist_ok=True)
        # Reuse the shared session bootstrap rather than reaching into the
        # downloader's internals — the inspector and downloader are peers.
        restore_session_from_env(config.telegram.session_dir)
        self._client = TelegramClient(
            session_base_path(config.telegram.session_dir),
            config.telegram.api_id,
            config.telegram.api_hash,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Connect to Telegram, scan first scan_limit audio messages, print
        full metadata.  Always disconnects cleanly on completion or error.
        """
        try:
            await self._client.start(phone=self._config.telegram.phone)
            await self._scan()
        finally:
            await self._client.disconnect()
            log.debug("Telegram client disconnected")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _scan(self) -> None:
        """Resolve the group, iterate messages, dump up to scan_limit audio files."""
        cfg = self._config

        # Resolve the group entity (username, link, or numeric chat ID).
        entity = await resolve_group_entity(self._client, cfg.telegram.group)

        group_title = getattr(entity, "title", str(entity))
        group_id = getattr(entity, "id", None)

        print(f"\n{'=' * 60}")
        print("OctoScribe Debug Inspector")
        print(f"{'=' * 60}")
        print(f"  Group title:  {group_title}")
        print(f"  Group ID:     {group_id}")
        print(f"  Scan limit:   {self._scan_limit} audio messages")
        print(f"{'=' * 60}")

        audio_count = 0
        offset_id = 0

        while audio_count < self._scan_limit:
            batch = await self._client.get_messages(entity, limit=100, offset_id=offset_id)
            if not batch:
                break

            for msg in batch:
                if is_audio(msg):
                    audio_count += 1
                    await self._print_message_debug(msg, audio_count)
                    if audio_count >= self._scan_limit:
                        break

            offset_id = batch[-1].id

        if audio_count == 0:
            print("\n  (no audio messages found in this group)")

        print(f"\n{'=' * 60}")
        print(f"SCAN COMPLETE — found {audio_count} audio message(s)")
        print(f"{'=' * 60}\n")

    async def _print_message_debug(self, msg: Any, index: int) -> None:
        """Print comprehensive metadata for one audio message."""
        print(f"\n{'=' * 60}")
        print(f"AUDIO MESSAGE #{index}")
        print(f"{'=' * 60}")

        # --- Basic message info -----------------------------------------
        print("\n--- MESSAGE INFO ---")
        print(f"  msg.id:          {msg.id}")
        print(f"  msg.date:        {msg.date}")

        sender_id = getattr(msg, "sender_id", None)
        post_author = getattr(msg, "post_author", None)
        print(f"  msg.sender_id:   {sender_id}")
        print(f"  msg.post_author: {post_author}")

        raw_text = getattr(msg, "message", None)
        text_preview = repr(raw_text[:100]) if raw_text else "None"
        print(f"  msg.message:     {text_preview}")

        # --- Telethon convenience attributes ----------------------------
        print("\n--- TELETHON CONVENIENCE ATTRIBUTES ---")
        file_attr = getattr(msg, "file", None)
        print(f"  msg.file:        {file_attr}")
        if file_attr is not None:
            print(f"    .name:         {getattr(file_attr, 'name', None)}")
            print(f"    .ext:          {getattr(file_attr, 'ext', None)}")
            print(f"    .mime_type:    {getattr(file_attr, 'mime_type', None)}")
            print(f"    .size:         {getattr(file_attr, 'size', None)}")

        audio_attr = getattr(msg, "audio", None)
        print(f"  msg.audio:       {audio_attr}")

        voice_attr = getattr(msg, "voice", None)
        print(f"  msg.voice:       {voice_attr}")

        document_attr = getattr(msg, "document", None)
        doc_type_name = type(document_attr).__name__ if document_attr is not None else "None"
        print(f"  msg.document:    {doc_type_name}")

        # --- Raw media structure ----------------------------------------
        media = getattr(msg, "media", None)
        print("\n--- RAW MEDIA STRUCTURE ---")
        print(f"  msg.media type:  {type(media).__name__}")

        if isinstance(media, MessageMediaDocument):
            doc = getattr(media, "document", None)
            if doc is not None:
                print("\n  --- DOCUMENT ---")
                print(f"    id:            {getattr(doc, 'id', None)}")
                print(f"    access_hash:   {getattr(doc, 'access_hash', None)}")
                print(f"    date:          {getattr(doc, 'date', None)}")
                print(f"    mime_type:     {getattr(doc, 'mime_type', None)}")
                print(f"    size:          {getattr(doc, 'size', None)}")

                attributes = getattr(doc, "attributes", [])
                print(f"\n    --- ATTRIBUTES ({len(attributes)}) ---")
                for i, attr in enumerate(attributes):
                    print(f"\n      [{i}] {type(attr).__name__}:")
                    for field_name in dir(attr):
                        if field_name.startswith("_") or callable(getattr(attr, field_name)):
                            continue
                        value = getattr(attr, field_name)
                        if isinstance(value, bytes) and len(value) > 50:
                            value = f"<{len(value)} bytes>"
                        print(f"          .{field_name}: {value!r}")

        # --- Forward info -----------------------------------------------
        print("\n--- FORWARD INFO ---")
        forward = getattr(msg, "forward", None)
        if forward is not None:
            print(f"  msg.forward.date:        {getattr(forward, 'date', None)}")
            print(f"  msg.forward.sender_id:   {getattr(forward, 'sender_id', None)}")
            print(f"  msg.forward.sender_name: {getattr(forward, 'sender_name', None)}")
            print(f"  msg.forward.channel_id:  {getattr(forward, 'channel_id', None)}")
        else:
            print("  (not forwarded)")

        # --- AudioMetadata extracted by the pipeline helper -------------
        print("\n--- AUDIO METADATA (via get_audio_metadata) ---")
        metadata = get_audio_metadata(msg)
        print(f"  msg_id:            {metadata.msg_id}")
        print(f"  title:             {metadata.title!r}")
        print(f"  performer:         {metadata.performer!r}")
        print(f"  duration:          {metadata.duration}")
        print(f"  extension:         {metadata.extension!r}")
        print(f"  original_filename: {metadata.original_filename!r}")
        print(f"  date:              {metadata.date!r}")
