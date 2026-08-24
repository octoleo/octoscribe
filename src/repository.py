"""
OctoScribe data repository management.

Provides DataRepository — a class that manages the separate git repository
used to store audio files, transcriptions, and the manifest.  All git
operations are performed via subprocess; no third-party git libraries are used.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Config, DataRepoConfig

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

    def __init__(
        self,
        config: DataRepoConfig,
        *,
        subdirectories: tuple[str, ...] = ("audio", "transcriptions"),
    ) -> None:
        self._config = config
        self._subdirectories = subdirectories

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
            result = self.pull()
            if not result.success:
                raise DataRepoError(f"git pull failed:\n{result.stderr}")
        elif path.exists():
            # Directory exists but has no .git — bail out rather than corrupt it.
            raise DataRepoError(
                f"Path {path} exists but is not a git repository. "
                "Remove or relocate it, then re-run OctoScribe."
            )
        elif self._config.url:
            log.info("Cloning configured evidence repository → %s", path)
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

        self._assert_expected_branch()
        self._protect_sensitive_files()
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

    def commit_and_push(
        self,
        message: str,
        *,
        push: bool | None = None,
    ) -> GitResult:
        """
        Stage all changes, commit with message, optionally push.

        Returns the commit result (or push result if auto_push=True).
        If nothing to commit, returns a successful no-op result without pushing.
        """
        self._protect_sensitive_files()
        self._ensure_git_identity()

        # Stage everything.
        self._git(["add", "-A"], check=True)

        # Attempt the commit.
        commit_result = self._git(["commit", "-m", message])

        combined_output = f"{commit_result.stdout}\n{commit_result.stderr}".lower()
        nothing_to_commit = (
            "nothing to commit" in combined_output
            or "no changes added to commit" in combined_output
        )
        if nothing_to_commit:
            log.debug("Nothing new to commit; checking whether an earlier commit needs push")
            no_op = GitResult(
                returncode=0,
                stdout=commit_result.stdout or "nothing to commit",
                stderr=commit_result.stderr,
                command=commit_result.command,
            )
            # A previous push may have failed after its commit succeeded. A
            # later no-op must still publish that ahead commit before dependent
            # transcript evidence can be allowed through the split-repo gate.
            should_push = self._config.auto_push and push is not False
            if should_push and self.has_remote():
                return self._git(["push", "origin", self._config.branch])
            return no_op
        if not commit_result.success:
            log.error("git commit failed: %s", commit_result.stderr)
            return commit_result

        log.info("Committed: %s", message)

        should_push = self._config.auto_push and push is not False
        if should_push and self.has_remote():
            log.info("Pushing origin/%s", self._config.branch)
            push_result = self._git(["push", "origin", self._config.branch])
            log.debug("push stdout: %s", push_result.stdout)
            if push_result.stderr:
                log.debug("push stderr: %s", push_result.stderr)
            return push_result

        return commit_result

    def head_revision(self) -> str | None:
        """Return the current Git commit identity, or ``None`` before first commit."""
        if not self.is_git_repo():
            return None
        result = self._git(["rev-parse", "HEAD"])
        revision = result.stdout.strip().lower()
        if result.success and len(revision) in {40, 64} and all(
            character in "0123456789abcdef" for character in revision
        ):
            return revision
        return None

    def assert_tracked_clean(self, files: list[Path]) -> None:
        """Require each evidence file to be present, tracked, and unmodified."""
        if not self.is_git_repo():
            raise DataRepoError(f"evidence repository is not initialized: {self._config.path}")
        root = self._config.path.resolve()
        for file_path in files:
            resolved = Path(file_path).resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError as exc:
                raise DataRepoError(
                    f"audio evidence lies outside its repository: {resolved}"
                ) from exc
            if not resolved.is_file():
                raise DataRepoError(f"audio evidence is missing: {relative}")
            tracked = self._git(["ls-files", "--error-unmatch", "--", str(relative)])
            if not tracked.success:
                raise DataRepoError(
                    f"audio evidence is ignored or untracked: {relative}"
                )
            status = self._git(["status", "--porcelain", "--", str(relative)])
            if status.stdout.strip():
                raise DataRepoError(
                    f"audio evidence differs from committed bytes: {relative}"
                )

    def verify_append_only_directory(self, directory: Path) -> None:
        """Reject edits/deletions to committed evidence while allowing new files."""
        if not self.is_git_repo() or self.head_revision() is None:
            return
        root = self._config.path.resolve()
        resolved = Path(directory).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise DataRepoError(
                f"evidence directory lies outside repository: {resolved}"
            ) from exc
        for args in (
            ["diff", "--name-status", "--diff-filter=MDRTCUXB", "HEAD", "--", str(relative)],
            ["diff", "--cached", "--name-status", "--diff-filter=MDRTCUXB", "HEAD", "--", str(relative)],
        ):
            changed = self._git(args)
            if not changed.success:
                raise DataRepoError(
                    f"could not verify append-only evidence: {changed.stderr}"
                )
            if changed.stdout.strip():
                raise DataRepoError(
                    "committed evidence is immutable; modification/deletion detected: "
                    f"{changed.stdout.strip()}"
                )

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
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self._config.path,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(
                ["git"] + args,
                124,
                stdout=exc.stdout or "",
                stderr="git command timed out after 300 seconds",
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
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(
                ["git"] + args,
                124,
                stdout=exc.stdout or "",
                stderr="git command timed out after 300 seconds",
            )
        recorded_args = list(args)
        if recorded_args and recorded_args[0] == "clone" and len(recorded_args) >= 3:
            recorded_args[-2] = "<redacted-repository-url>"
        return GitResult(
            returncode=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            command=["git"] + recorded_args,
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

    def _assert_expected_branch(self) -> None:
        """Refuse to write evidence on an unintended branch."""
        current = self._git(["branch", "--show-current"])
        branch = current.stdout.strip()
        if not current.success or branch != self._config.branch:
            raise DataRepoError(
                f"evidence repository must be on branch {self._config.branch!r}; "
                f"current branch is {branch or '(detached)'}"
            )

    def _create_subdirectories(self) -> None:
        """Create configured evidence subdirectories with .gitkeep files."""
        for subdir_name in self._subdirectories:
            subdir = self._config.path / subdir_name
            subdir.mkdir(parents=True, exist_ok=True)
            gitkeep = subdir / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
                log.debug("Created %s", gitkeep)

    def _protect_sensitive_files(self) -> None:
        """Exclude credentials/session databases and reject tracked copies."""
        tracked = self._git(["ls-files"])
        sensitive: list[str] = []
        for value in tracked.stdout.splitlines():
            name = Path(value).name
            if (
                name == ".env"
                or (name.startswith(".env.") and name != ".env.example")
                or name.endswith(".session")
                or ".session-" in name
                or "/.session/" in f"/{value}"
            ):
                sensitive.append(value)
        if sensitive:
            raise DataRepoError(
                "sensitive credential/session file is already tracked in evidence "
                f"repository {self._config.path}: {', '.join(sensitive)}"
            )

        exclude_result = self._git(["rev-parse", "--git-path", "info/exclude"])
        if exclude_result.success and exclude_result.stdout:
            exclude_path = Path(exclude_result.stdout)
            if not exclude_path.is_absolute():
                exclude_path = self._config.path / exclude_path
        else:
            # Compatibility fallback for mocked/minimal repositories.
            exclude_path = self._config.path / ".git" / "info" / "exclude"
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        existing = (
            exclude_path.read_text(encoding="utf-8")
            if exclude_path.exists()
            else ""
        )
        patterns = (
            ".env",
            "**/.env",
            "**/.env.*",
            "!**/.env.example",
            ".session/",
            "**/.session/",
            "*.session",
            "*.session-journal",
            "*.session-wal",
            "*.session-shm",
        )
        missing = [pattern for pattern in patterns if pattern not in existing.splitlines()]
        if missing:
            prefix = "" if not existing or existing.endswith("\n") else "\n"
            block = "# OctoScribe sensitive files\n" + "\n".join(missing) + "\n"
            exclude_path.write_text(existing + prefix + block, encoding="utf-8")


class EvidenceRepositories:
    """Coordinate separate audio and transcript repositories safely.

    Existing single-repository configurations are detected by resolved path
    and operated on exactly once.  With split repositories, audio is always
    committed before transcript metadata so a transcript can never be pushed
    ahead of the source evidence it references by SHA-256.
    """

    def __init__(self, config: Config) -> None:
        self._shared = config.audio_repo.path == config.text_repo.path
        download = getattr(config, "download", None)
        self._audio_dir = getattr(
            download,
            "audio_dir",
            config.audio_repo.path / "audio",
        )
        transcribe = getattr(config, "transcribe", None)
        text_root = config.text_repo.path
        self._transcript_evidence_dirs = tuple(
            path
            for path in (
                getattr(transcribe, "transcriptions_dir", text_root / "transcriptions"),
                getattr(transcribe, "artifacts_dir", None) or text_root / "candidates",
                getattr(transcribe, "reports_dir", None) or text_root / "reports",
            )
            if path is not None
        )
        if self._shared:
            shared = DataRepository(
                config.audio_repo,
                subdirectories=("audio", "transcriptions", "candidates", "reports"),
            )
            self.audio = shared
            self.transcripts = shared
        else:
            self.audio = DataRepository(
                config.audio_repo,
                subdirectories=("audio",),
            )
            self.transcripts = DataRepository(
                config.text_repo,
                subdirectories=("transcriptions", "candidates", "reports"),
            )

    @property
    def is_split(self) -> bool:
        """Return whether two distinct git worktrees are configured."""
        return not self._shared

    def ensure_ready(self) -> None:
        """Clone/init/pull every configured evidence repository."""
        self.audio.ensure_ready()
        if not self._shared:
            self.transcripts.ensure_ready()

    def pull(self) -> dict[str, GitResult]:
        """Pull audio and transcript repositories, de-duplicating shared mode."""
        results = {"audio": self.audio.pull()}
        if self._shared:
            results["transcripts"] = results["audio"]
        else:
            results["transcripts"] = self.transcripts.pull()
        return results

    def commit_and_push(self, message: str) -> dict[str, GitResult]:
        """Commit audio first, then transcript artifacts, with a hard order."""
        results = {"audio": self.commit_audio(message)}
        results["transcripts"] = (
            results["audio"] if self._shared else self.commit_transcripts(message)
        )
        return results

    def commit_audio(
        self,
        message: str,
        *,
        push: bool | None = None,
    ) -> GitResult:
        """Commit/push source evidence before any dependent text publication."""
        self.audio.verify_append_only_directory(self._audio_dir)
        result = self.audio.commit_and_push(f"{message} (audio)", push=push)
        if not result.success:
            raise DataRepoError(
                "audio repository update failed; transcript evidence was not "
                "committed or pushed"
            )
        return result

    def commit_transcripts(
        self,
        message: str,
        *,
        push: bool | None = None,
    ) -> GitResult:
        """Commit/push transcript artifacts after the audio gate succeeds."""
        for directory in self._transcript_evidence_dirs:
            self.transcripts.verify_append_only_directory(directory)
        result = self.transcripts.commit_and_push(
            f"{message} (transcripts)",
            push=push,
        )
        if not result.success:
            raise DataRepoError(
                "transcript evidence repository update failed; local evidence "
                "is retained for a later retry"
            )
        return result

    def assert_audio_tracked(self, files: list[Path]) -> None:
        """Prove pending source files are represented by the current audio commit."""
        self.audio.assert_tracked_clean(files)

    def audio_revision(self) -> str | None:
        """Return the commit that contains the source-audio evidence."""
        return self.audio.head_revision()

    def status(self) -> dict[str, dict[str, Any]]:
        """Return status for both logical repositories."""
        audio_status = self.audio.status()
        transcript_status = (
            audio_status if self._shared else self.transcripts.status()
        )
        return {"audio": audio_status, "transcripts": transcript_status}
