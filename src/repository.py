"""Filesystem workspace preparation for OctoScribe evidence.

The module name is retained for import compatibility, but OctoScribe does not
clone, pull, commit, or push repositories. A calling workflow owns all source
control operations and supplies already-available filesystem paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import Config


class WorkspaceError(Exception):
    """Raised when configured evidence paths are unsafe or unusable."""


@dataclass(frozen=True)
class WorkspaceStatus:
    """Read-only status for one caller-supplied filesystem workspace."""

    path: Path
    exists: bool
    writable: bool


class EvidenceWorkspaces:
    """Prepare and describe the audio and transcript filesystem workspaces."""

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def is_split(self) -> bool:
        """Return whether audio and text use different filesystem roots."""
        return self._config.audio_repo.path != self._config.text_repo.path

    def ensure_ready(self) -> None:
        """Create only the directories OctoScribe is authorised to write.

        The caller must create or check out the workspace roots. OctoScribe
        will not infer remote locations or perform source-control operations.
        """
        roots = {self._config.audio_repo.path, self._config.text_repo.path}
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise WorkspaceError(f"workspace path is not a directory: {root}")

        paths = (
            self._config.download.audio_dir,
            self._config.download.manifest_file.parent,
            self._config.transcribe.transcriptions_dir,
            self._config.transcribe.artifacts_dir,
            self._config.transcribe.reports_dir,
        )
        for path in paths:
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, WorkspaceStatus]:
        """Return path/existence/writeability without mutating either root."""
        audio = self._status(self._config.audio_repo.path)
        text = audio if not self.is_split else self._status(self._config.text_repo.path)
        return {"audio": audio, "transcripts": text}

    @staticmethod
    def _status(path: Path) -> WorkspaceStatus:
        exists = path.is_dir()
        # Avoid permission probes that create files. Parent writeability is
        # deliberately not guessed when the root does not exist.
        writable = exists and bool(path.stat().st_mode & 0o222)
        return WorkspaceStatus(path=path, exists=exists, writable=writable)


__all__ = ["EvidenceWorkspaces", "WorkspaceError", "WorkspaceStatus"]
