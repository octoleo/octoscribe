"""
Tests for src/repository.py

Uses:
  - tmp_path  for real filesystem operations
  - unittest.mock.patch  for mocking subprocess.run where real git is undesirable
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.config import DataRepoConfig
from src.repository import DataRepoError, DataRepository, GitResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    path: Path,
    *,
    url: str | None = None,
    branch: str = "main",
    auto_push: bool = False,
) -> DataRepoConfig:
    return DataRepoConfig(url=url, path=path, branch=branch, auto_push=auto_push)


def _fake_run_success(stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock subprocess.CompletedProcess with returncode 0."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = stderr
    return m


def _fake_run_failure(stdout: str = "", stderr: str = "error") -> MagicMock:
    """Return a mock subprocess.CompletedProcess with returncode 1."""
    m = MagicMock()
    m.returncode = 1
    m.stdout = stdout
    m.stderr = stderr
    return m


def _init_real_repo(path: Path) -> None:
    """Initialise a real git repository at *path* (used for integration tests)."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"],
                   check=True, capture_output=True)
    # Disable GPG signing for this ephemeral test repo so commits work in CI
    # and headless environments where /dev/tty is unavailable.
    subprocess.run(["git", "-C", str(path), "config", "commit.gpgsign", "false"],
                   check=True, capture_output=True)


# ---------------------------------------------------------------------------
# GitResult
# ---------------------------------------------------------------------------

class TestGitResult:
    def test_success_true_when_returncode_zero(self) -> None:
        result = GitResult(returncode=0, stdout="ok", stderr="", command=["git", "status"])
        assert result.success is True

    def test_success_false_when_nonzero_returncode(self) -> None:
        result = GitResult(returncode=1, stdout="", stderr="fatal", command=["git", "push"])
        assert result.success is False

    def test_str_contains_command_and_status(self) -> None:
        result = GitResult(returncode=0, stdout="on branch main", stderr="", command=["git", "status"])
        text = str(result)
        assert "git status" in text
        assert "ok" in text

    def test_str_shows_failure_exit_code(self) -> None:
        result = GitResult(returncode=128, stdout="", stderr="not a repo", command=["git", "pull"])
        text = str(result)
        assert "exit 128" in text


# ---------------------------------------------------------------------------
# is_git_repo / has_remote  (real filesystem)
# ---------------------------------------------------------------------------

class TestIsGitRepo:
    def test_returns_true_for_real_git_repo(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        repo = DataRepository(make_config(tmp_path))
        assert repo.is_git_repo() is True

    def test_returns_false_for_empty_directory(self, tmp_path: Path) -> None:
        repo = DataRepository(make_config(tmp_path))
        assert repo.is_git_repo() is False


# ---------------------------------------------------------------------------
# ensure_ready  (mocked subprocess)
# ---------------------------------------------------------------------------

class TestEnsureReady:
    def test_clones_when_url_set_and_path_missing(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "data"  # does not exist yet
        config = make_config(repo_path, url="https://example.com/data.git")
        repo = DataRepository(config)

        def fake_run(cmd, **kwargs):
            # After "clone" the directory will appear; we fake that here.
            if cmd[1] == "clone":
                # Simulate git creating the .git dir so subsequent calls work.
                (repo_path / ".git").mkdir(parents=True, exist_ok=True)
            return _fake_run_success()

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            repo.ensure_ready()

        called_cmds = [c.args[0] for c in mock_run.call_args_list]
        assert any("clone" in cmd for cmd in called_cmds)

    def test_inits_when_url_none_and_path_missing(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "data"
        config = make_config(repo_path, url=None)
        repo = DataRepository(config)

        def fake_run(cmd, **kwargs):
            if cmd[1] == "init":
                (repo_path / ".git").mkdir(parents=True, exist_ok=True)
            return _fake_run_success()

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            repo.ensure_ready()

        called_cmds = [c.args[0] for c in mock_run.call_args_list]
        assert any("init" in cmd for cmd in called_cmds)

    def test_pulls_when_git_dir_exists(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        config = make_config(tmp_path)
        repo = DataRepository(config)

        with patch.object(repo, "pull", return_value=GitResult(0, "", "", ["git", "pull"])) as mock_pull:
            # Still need subdirs to be created; patch _create_subdirectories too.
            with patch.object(repo, "_create_subdirectories"):
                repo.ensure_ready()

        mock_pull.assert_called_once()

    def test_raises_when_path_exists_but_not_git_repo(self, tmp_path: Path) -> None:
        # tmp_path exists but has no .git
        config = make_config(tmp_path)
        repo = DataRepository(config)

        with pytest.raises(DataRepoError, match="not a git repository"):
            repo.ensure_ready()

    def test_creates_audio_and_transcriptions_subdirs(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        config = make_config(tmp_path)
        repo = DataRepository(config)

        # Stub pull so we don't need a real remote.
        with patch.object(repo, "pull", return_value=GitResult(0, "", "", ["git", "pull"])):
            repo.ensure_ready()

        assert (tmp_path / "audio").is_dir()
        assert (tmp_path / "transcriptions").is_dir()

    def test_raises_when_clone_fails(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "data"
        config = make_config(repo_path, url="https://example.com/data.git")
        repo = DataRepository(config)

        with patch("subprocess.run", return_value=_fake_run_failure(stderr="repository not found")):
            with pytest.raises(DataRepoError, match="clone failed"):
                repo.ensure_ready()

    def test_raises_when_init_fails(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "data"
        config = make_config(repo_path, url=None)
        repo = DataRepository(config)

        with patch("subprocess.run", return_value=_fake_run_failure(stderr="cannot init")):
            with pytest.raises(DataRepoError, match="init failed"):
                repo.ensure_ready()


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

class TestPull:
    def test_returns_success_result_when_no_remote(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        config = make_config(tmp_path, url=None)
        repo = DataRepository(config)

        result = repo.pull()

        assert result.success is True
        assert "No remote configured" in result.stdout

    def test_runs_git_pull_when_remote_configured(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        config = make_config(tmp_path, url="https://example.com/data.git")
        repo = DataRepository(config)

        # Add a real origin remote so that has_remote() works without mocking.
        subprocess.run(
            ["git", "-C", str(tmp_path), "remote", "add", "origin", "https://example.com/data.git"],
            check=True, capture_output=True,
        )

        # Only mock the actual pull call; let git remote run for real.
        def fake_run(cmd, **kwargs):
            if "pull" in cmd:
                return _fake_run_success("Already up to date.")
            # Let everything else (e.g. git remote) run for real.
            return subprocess.run.__wrapped__(cmd, **kwargs) if hasattr(subprocess.run, "__wrapped__") else _real_subprocess_run(cmd, **kwargs)

        _real_subprocess_run = subprocess.run

        with patch("src.repository.subprocess.run", side_effect=fake_run) as mock_run:
            result = repo.pull()

        assert result.success is True
        pull_calls = [c for c in mock_run.call_args_list if "pull" in c.args[0]]
        assert pull_calls, "Expected at least one git pull call"


# ---------------------------------------------------------------------------
# commit_and_push
# ---------------------------------------------------------------------------

class TestCommitAndPush:
    def test_returns_noop_result_when_nothing_to_commit(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        config = make_config(tmp_path)
        repo = DataRepository(config)

        def fake_run(cmd, **kwargs):
            if "commit" in cmd:
                return _fake_run_success("nothing to commit, working tree clean")
            return _fake_run_success()

        with patch("subprocess.run", side_effect=fake_run):
            result = repo.commit_and_push("test commit")

        assert result.success is True
        assert "nothing to commit" in result.stdout

    def test_pushes_when_auto_push_true_and_has_remote(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        # Add a real origin remote so has_remote() works without mocking.
        subprocess.run(
            ["git", "-C", str(tmp_path), "remote", "add", "origin", "https://example.com/data.git"],
            check=True, capture_output=True,
        )
        config = make_config(tmp_path, url="https://example.com/data.git", auto_push=True)
        repo = DataRepository(config)

        _real_subprocess_run = subprocess.run

        def fake_run(cmd, **kwargs):
            if "commit" in cmd:
                return _fake_run_success("1 file changed")
            if "push" in cmd:
                return _fake_run_success("")
            # Let git add, git remote, git config, etc. run for real.
            return _real_subprocess_run(cmd, **kwargs)

        with patch("src.repository.subprocess.run", side_effect=fake_run) as mock_run:
            repo.commit_and_push("add files")

        push_calls = [c for c in mock_run.call_args_list if "push" in c.args[0]]
        assert push_calls, "Expected a git push call when auto_push=True"

    def test_does_not_push_when_auto_push_false(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        subprocess.run(
            ["git", "-C", str(tmp_path), "remote", "add", "origin", "https://example.com/data.git"],
            check=True, capture_output=True,
        )
        config = make_config(tmp_path, url="https://example.com/data.git", auto_push=False)
        repo = DataRepository(config)

        _real_subprocess_run = subprocess.run

        def fake_run(cmd, **kwargs):
            if "commit" in cmd:
                return _fake_run_success("1 file changed")
            if "push" in cmd:
                return _fake_run_success("")
            return _real_subprocess_run(cmd, **kwargs)

        with patch("src.repository.subprocess.run", side_effect=fake_run) as mock_run:
            repo.commit_and_push("add files")

        push_calls = [c for c in mock_run.call_args_list if "push" in c.args[0]]
        assert not push_calls, "Expected no git push call when auto_push=False"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_returns_dict_with_expected_keys(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        config = make_config(tmp_path)
        repo = DataRepository(config)

        result = repo.status()

        expected_keys = {"is_git_repo", "has_remote", "branch", "uncommitted_changes",
                         "ahead_count", "path"}
        assert expected_keys == set(result.keys())

    def test_is_git_repo_false_for_empty_dir(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        repo = DataRepository(config)

        result = repo.status()

        assert result["is_git_repo"] is False
        assert result["has_remote"] is False
        assert result["branch"] == ""

    def test_path_matches_config(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        config = make_config(tmp_path)
        repo = DataRepository(config)

        result = repo.status()

        assert result["path"] == str(tmp_path)


# ---------------------------------------------------------------------------
# _ensure_git_identity
# ---------------------------------------------------------------------------

class TestEnsureGitIdentity:
    def test_sets_identity_when_not_configured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _init_real_repo(tmp_path)

        # Point GIT_CONFIG_GLOBAL at a blank temp file so the user's real global
        # config (which may already have user.email/user.name) is invisible to git
        # inside the test repo.  This lets us verify that _ensure_git_identity()
        # writes the fallback values when nothing is configured locally.
        blank_global = tmp_path / "blank_gitconfig"
        blank_global.write_text("")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(blank_global))

        # Also unset any local identity so the repo truly has no identity.
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "--local", "--unset", "user.email"],
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "--local", "--unset", "user.name"],
            capture_output=True,
        )

        config = make_config(tmp_path)
        repo = DataRepository(config)
        repo._ensure_git_identity()

        email = subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email"],
            capture_output=True, text=True,
            env={**__import__("os").environ, "GIT_CONFIG_GLOBAL": str(blank_global)},
        ).stdout.strip()
        name = subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name"],
            capture_output=True, text=True,
            env={**__import__("os").environ, "GIT_CONFIG_GLOBAL": str(blank_global)},
        ).stdout.strip()

        assert email == "octoscribe@local"
        assert name == "OctoScribe"


# ---------------------------------------------------------------------------
# _create_subdirectories
# ---------------------------------------------------------------------------

class TestCreateSubdirectories:
    def test_creates_dirs_with_gitkeep(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        config = make_config(tmp_path)
        repo = DataRepository(config)

        repo._create_subdirectories()

        assert (tmp_path / "audio").is_dir()
        assert (tmp_path / "audio" / ".gitkeep").exists()
        assert (tmp_path / "transcriptions").is_dir()
        assert (tmp_path / "transcriptions" / ".gitkeep").exists()

    def test_idempotent_when_dirs_already_exist(self, tmp_path: Path) -> None:
        _init_real_repo(tmp_path)
        config = make_config(tmp_path)
        repo = DataRepository(config)

        repo._create_subdirectories()
        # Second call must not raise.
        repo._create_subdirectories()

        assert (tmp_path / "audio").is_dir()
        assert (tmp_path / "transcriptions").is_dir()


# ---------------------------------------------------------------------------
# Integration: init → commit → status
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_init_commit_shows_clean_status(self, tmp_path: Path) -> None:
        """Initialise a local repo, write a file, commit it, verify clean status."""
        repo_path = tmp_path / "data_repo"
        config = make_config(repo_path, url=None, auto_push=False)
        repo = DataRepository(config)

        # Bootstrap the repo (init + create subdirs with .gitkeep files).
        repo.ensure_ready()

        # Disable GPG signing on this ephemeral test repo so commits work in
        # headless environments where /dev/tty is unavailable.
        subprocess.run(
            ["git", "-C", str(repo_path), "config", "commit.gpgsign", "false"],
            check=True, capture_output=True,
        )

        # Commit the initial scaffold (.gitkeep files) so the tree is clean.
        initial = repo.commit_and_push("scaffold: add subdirectories")
        assert initial.success is True

        # Write a new file and commit it.
        test_file = repo_path / "hello.txt"
        test_file.write_text("hello octoscribe\n")

        result = repo.commit_and_push("initial content")
        assert result.success is True

        # Status should now show no uncommitted changes.
        s = repo.status()
        assert s["is_git_repo"] is True
        assert s["uncommitted_changes"] is False
        assert s["branch"] == "main"
