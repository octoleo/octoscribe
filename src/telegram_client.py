"""
src/telegram_client.py — Shared Telegram session and entity helpers.

Both :class:`src.telegram.TelegramDownloader` and
:class:`src.debug.DebugInspector` need to do three identical things before they
can talk to Telegram:

1. Restore a saved session from the ``TELEGRAM_SESSION_B64`` environment
   variable (so CI can authenticate without an interactive phone-code prompt).
2. Derive the on-disk session base path (``<session_dir>/octoscribe``).
3. Resolve a group reference — a ``@username``, an invite link, or a numeric
   chat ID — into a Telethon entity.

Previously this logic was duplicated, and the debug inspector reached into the
*private* ``TelegramDownloader._restore_session_from_env`` method.  Extracting
these framework-agnostic helpers here removes the duplication and the
cross-class private access, leaving each consumer to own only its own concerns.

Note: this module intentionally does **not** construct the ``TelegramClient``
itself.  Client construction stays in the consumer modules so that their tests
can patch ``TelegramClient`` at the point of use, and so that this module has no
hard dependency on Telethon being importable.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Basename (without extension) used for the Telethon SQLite session file.
SESSION_NAME = "octoscribe"

#: Environment variable carrying a base64-encoded session file for CI/CD.
SESSION_ENV_VAR = "TELEGRAM_SESSION_B64"


def session_base_path(session_dir: Path) -> str:
    """
    Return the Telethon session base path as a string.

    Telethon appends the ``.session`` suffix itself, so callers pass the path
    *without* an extension, e.g. ``<session_dir>/octoscribe``.
    """
    return str(Path(session_dir) / SESSION_NAME)


def restore_session_from_env(session_dir: Path) -> bool:
    """
    Restore a Telegram session file from ``TELEGRAM_SESSION_B64`` if present.

    When the environment variable is set, its base64 payload is decoded and
    written to ``<session_dir>/octoscribe.session``.  This enables
    non-interactive authentication in CI/CD environments (e.g. GitHub Actions)
    where the interactive phone-code login is impossible.

    Returns ``True`` if a session was restored, ``False`` if the variable is
    unset, empty, or could not be decoded.  Never raises: a malformed value is
    logged and treated as "no session provided" so the caller can fall back to
    interactive login.
    """
    b64 = os.environ.get(SESSION_ENV_VAR, "").strip()
    if not b64:
        return False

    try:
        session_bytes = base64.b64decode(b64)
    except Exception as exc:  # noqa: BLE001 — any decode failure is non-fatal
        log.warning("%s is set but could not be decoded: %s", SESSION_ENV_VAR, exc)
        return False

    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    session_file = session_dir / f"{SESSION_NAME}.session"
    session_file.write_bytes(session_bytes)
    log.info(
        "Telegram session restored from %s (%d bytes)",
        SESSION_ENV_VAR,
        len(session_bytes),
    )
    return True


async def resolve_group_entity(client: Any, group: str) -> Any:
    """
    Resolve *group* into a Telethon entity using *client*.

    Accepts any reference Telethon understands: a ``@username``, a ``t.me``
    invite link, or a numeric chat ID (optionally negative, e.g. ``-100…``).
    Purely numeric references are converted to ``int`` first so Telethon treats
    them as chat IDs rather than usernames.
    """
    group_raw = group.strip()
    if group_raw.lstrip("-").isdigit():
        return await client.get_entity(int(group_raw))
    return await client.get_entity(group_raw)
