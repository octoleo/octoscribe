"""
OctoScribe data repository management.

Provides DataRepository — a class that manages the separate git repository
used to store audio files, transcriptions, and the manifest.  All git
operations are performed via subprocess; no third-party git libraries are used.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Any

from src.config import DataRepoConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GitResult
# ---------------------------------------------------------------------------

@dataclass
class GitResult:
    """Result of a git command execution."""

    returncode: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def success(self) -> bool:
        """Return True when the command exited with code 0."""
        return self.returncode == 0

    def __str__(self) -> str:
        cmd = " ".join(self.command)
        status = "ok" if self.success else f"exit {self.returncode}"
        parts = [f"[{status}] {cmd}"]
        if self.stdout:
            parts.append(f"  stdout: {self.stdout}")
        if self.stderr:
            parts.append(f"  stderr: {self.stderr}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# DataRepoError
# ---------------------------------------------------------------------------

class DataRepoError(Exception):
    """Raised when the data repository is in an unexpected state."""


# ---------------------------------------------------------------------------
# DataRepository
# ---------------------------------------------------------------------------

class DataRepository:
    """
    Manages the OctoScribe data repository.

    The data repository is a separate git repo (outside the project) that stores:
      - audio/            Downloaded audio files
      - transcriptions/   Transcribed text files
      - manifest.json     Progress tracker (version-controlled)
      - .session/         Telegram session files

    All git operations use subprocess.  Designed to be idempotent — calling
    ensure_ready() repeatedly is safe.
    """

    def __init__(self, config: DataRepoConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_ready(self) -> None:
        """
        Prepare the data repository for use.

        - If path doesn't exist and url is set: clone the repo.
        - If path doesn't exist and url is None: init a new local repo.
        - If path exists and is a git repo: pull latest changes.
        - If path exists but is NOT a git repo: raise DataRepoError.

        Creates required subdirectories (audio, transcriptions) after setup.
        """
        path = self._config.path

        if (path / ".git").exists():
            log.debug("Data repo already initialised at %s — pulling", path)
            self.pull()
        elif path.exists():
            # Directory exists but has no .git — bail out rather than corrupt it.
            raise DataRepoError(
                f"Path {path} exists but is not a git repository. "
                "Remove or relocate it, then re-run OctoScribe."
            )
        elif self._config.url:
            log.info("Cloning data repo from %s → %s", self._config.url, path)
            result = self._git_no_cwd(
                [
                    "clone",
                    "--branch", self._config.branch,
                    self._config.url,
                    str(path),
                ]
            )
            if not result.success:
                raise DataRepoError(
                    f"git clone failed:\n{result.stderr}"
                )
        else:
            log.info("Initialising new local data repo at %s", path)
            path.mkdir(parents=True, exist_ok=True)
            result = self._git_no_cwd(
                ["init", "-b", self._config.branch, str(path)]
            )
            if not result.success:
                raise DataRepoError(
                    f"git init failed:\n{result.stderr}"
                )
            self._ensure_git_identity()

        self._create_subdirectories()

    def pull(self) -> GitResult:
        """Pull latest changes from remote.  No-op (successful) if no remote configured."""
        if not self.has_remote():
            log.debug("No remote configured — skipping pull")
            return GitResult(
                returncode=0,
                stdout="No remote configured",
                stderr="",
                command=["git", "pull"],
            )

        log.debug("Pulling origin/%s", self._config.branch)
        result = self._git(["pull", "origin", self._config.branch])
        log.debug("pull stdout: %s", result.stdout)
        if result.stderr:
            log.debug("pull stderr: %s", result.stderr)
        return result

    def commit_and_push(self, message: str) -> GitResult:
        """
        Stage all changes, commit with message, optionally push.

        Returns the commit result (or push result if auto_push=True).
        If nothing to commit, returns a successful no-op result without pushing.
        """
        self._ensure_git_identity()

        # Stage everything.
        self._git(["add", "-A"], check=True)

        # Attempt the commit.
        commit_result = self._git(["commit", "-m", message])

        if not commit_result.success or "nothing to commit" in commit_result.stdout:
            log.debug("Nothing to commit — skipping push")
            # Return a success result regardless of the exact returncode from git.
            return GitResult(
                returncode=0,
                stdout=commit_result.stdout or "nothing to commit",
                stderr=commit_result.stderr,
                command=commit_result.command,
            )

        log.info("Committed: %s", message)

        if self._config.auto_push and self.has_remote():
            log.info("Pushing origin/%s", self._config.branch)
            push_result = self._git(["push", "origin", self._config.branch])
            log.debug("push stdout: %s", push_result.stdout)
            if push_result.stderr:
                log.debug("push stderr: %s", push_result.stderr)
            return push_result

        return commit_result

    def status(self) -> dict[str, Any]:
        """
        Return a status dictionary with the following keys:

          - is_git_repo (bool)
          - has_remote (bool)
          - branch (str)
          - uncommitted_changes (bool)
          - ahead_count (int)   — commits ahead of remote
          - path (str)
        """
        git_repo = self.is_git_repo()

        if not git_repo:
            return {
                "is_git_repo": False,
                "has_remote": False,
                "branch": "",
                "uncommitted_changes": False,
                "ahead_count": 0,
                "path": str(self._config.path),
            }

        remote = self.has_remote()

        porcelain = self._git(["status", "--porcelain"])
        uncommitted = bool(porcelain.stdout.strip())

        branch_result = self._git(["branch", "--show-current"])
        branch = branch_result.stdout.strip()

        ahead_count = 0
        if remote and branch:
            ahead_result = self._git(
                ["rev-list", "--count", f"HEAD..origin/{self._config.branch}"]
            )
            if ahead_result.success and ahead_result.stdout.strip().isdigit():
                ahead_count = int(ahead_result.stdout.strip())

        return {
            "is_git_repo": git_repo,
            "has_remote": remote,
            "branch": branch,
            "uncommitted_changes": uncommitted,
            "ahead_count": ahead_count,
            "path": str(self._config.path),
        }

    def is_git_repo(self) -> bool:
        """Return True if the configured path contains a git repository."""
        return (self._config.path / ".git").exists()

    def has_remote(self) -> bool:
        """Return True if the repository has an 'origin' remote."""
        result = self._git(["remote"])
        return "origin" in result.stdout.splitlines()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _git(self, args: list[str], check: bool = False) -> GitResult:
        """Run a git command with cwd set to the repository path."""
        result = subprocess.run(
            ["git"] + args,
            cwd=self._config.path,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise DataRepoError(f"git {args[0]} failed: {result.stderr.strip()}")
        return GitResult(
            returncode=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            command=["git"] + args,
        )

    def _git_no_cwd(self, args: list[str]) -> GitResult:
        """Run a git command without a cwd (used for clone/init before path exists)."""
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
        )
        return GitResult(
            returncode=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            command=["git"] + args,
        )

    def _ensure_git_identity(self) -> None:
        """Set git user.name and user.email if not already configured.

        This is necessary for commits to succeed in CI environments that lack
        a global git identity.
        """
        email_result = self._git(["config", "user.email"])
        if not email_result.success or not email_result.stdout.strip():
            self._git(["config", "user.email", "octoscribe@local"])
            log.debug("Set git user.email to octoscribe@local")

        name_result = self._git(["config", "user.name"])
        if not name_result.success or not name_result.stdout.strip():
            self._git(["config", "user.name", "OctoScribe"])
            log.debug("Set git user.name to OctoScribe")

    def _create_subdirectories(self) -> None:
        """Create audio/ and transcriptions/ subdirectories with .gitkeep files."""
        for subdir_name in ("audio", "transcriptions"):
            subdir = self._config.path / subdir_name
            subdir.mkdir(parents=True, exist_ok=True)
            gitkeep = subdir / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
                log.debug("Created %s", gitkeep)
