#!/usr/bin/env python3
"""
octoscribe.py — CLI entry point for the OctoScribe pipeline.

Usage:
    octoscribe.py run              Full pipeline: sync→acquire→transcribe→sync
    octoscribe.py download         Acquire new audio from the configured source
                                   (Telegram group or local folder)
    octoscribe.py transcribe       Transcribe unprocessed audio only
    octoscribe.py sync             Sync data repository (pull or push)
    octoscribe.py status           Show pipeline status
    octoscribe.py debug            Inspect Telegram connection and audio metadata

Audio source:
    By default OctoScribe downloads audio from a Telegram group. To work with
    sermons that already live in a local folder instead, set source.mode=folder
    and source.folder in the INI file, or pass --folder PATH (which implies
    --source folder). In folder mode no Telegram credentials are required.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """Configure root logging: DEBUG when verbose, INFO otherwise."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------

def async_run(coro):
    """Run an async coroutine from synchronous context."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Config override builder
# ---------------------------------------------------------------------------

def build_overrides(args: argparse.Namespace) -> dict:
    """
    Translate CLI arguments into Config.load() override kwargs.

    Uses the double-underscore section__key convention expected by
    Config._override().
    """
    overrides: dict = {}

    # Data repo path override
    if getattr(args, "data_repo", None):
        overrides["data_repo__path"] = args.data_repo

    # Telegram group override (download / run)
    if getattr(args, "group", None):
        overrides["telegram__group"] = args.group

    # Transcription backend override (transcribe / run)
    if getattr(args, "backend", None):
        overrides["transcribe__backend"] = args.backend

    # Audio source overrides (download / run).
    source = getattr(args, "source", None)
    folder = getattr(args, "folder", None)
    if source:
        overrides["source__mode"] = source
    if folder:
        overrides["source__folder"] = folder
        # Passing a folder is a convenient shorthand for selecting folder mode.
        if not source:
            overrides["source__mode"] = "folder"

    return overrides


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level ArgumentParser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="octoscribe",
        description=(
            "OctoScribe — Download Telegram audio, transcribe verbatim, "
            "push to git."
        ),
    )

    # ------------------------------------------------------------------
    # Global options
    # ------------------------------------------------------------------
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=os.environ.get("OCTOSCRIBE_CONFIG"),
        help=(
            "Path to octoscribe.ini. "
            "Defaults to OCTOSCRIBE_CONFIG env var or ./octoscribe.ini."
        ),
    )
    parser.add_argument(
        "--env",
        metavar="PATH",
        default=os.environ.get("OCTOSCRIBE_ENV"),
        help=(
            "Path to .env file. "
            "Defaults to OCTOSCRIBE_ENV env var or ./.env."
        ),
    )
    parser.add_argument(
        "--data-repo",
        metavar="PATH",
        dest="data_repo",
        help="Override data_repo.path from config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )

    # ------------------------------------------------------------------
    # Subcommands
    # ------------------------------------------------------------------
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- run -----------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: sync → download → transcribe → sync.",
    )
    run_parser.add_argument(
        "--group",
        metavar="GROUP",
        help="Override Telegram group (username, link, or numeric ID).",
    )
    run_parser.add_argument(
        "--source",
        choices=["telegram", "folder"],
        help="Audio source: 'telegram' (default) or 'folder'.",
    )
    run_parser.add_argument(
        "--folder",
        metavar="PATH",
        help="Local folder to import audio from (implies --source folder).",
    )
    run_parser.add_argument(
        "--backend",
        metavar="BACKEND",
        choices=["openai", "local"],
        help="Override transcription backend.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Preview files to transcribe without running the pipeline.",
    )
    run_parser.add_argument(
        "--no-push",
        action="store_true",
        dest="no_push",
        help="Skip the final git push.",
    )

    # ---- download ------------------------------------------------------
    dl_parser = subparsers.add_parser(
        "download",
        help="Acquire new audio from the configured source (Telegram or folder).",
    )
    dl_parser.add_argument(
        "--group",
        metavar="GROUP",
        help="Override Telegram group.",
    )
    dl_parser.add_argument(
        "--source",
        choices=["telegram", "folder"],
        help="Audio source: 'telegram' (default) or 'folder'.",
    )
    dl_parser.add_argument(
        "--folder",
        metavar="PATH",
        help="Local folder to import audio from (implies --source folder).",
    )

    # ---- transcribe ----------------------------------------------------
    tr_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe unprocessed audio only.",
    )
    tr_parser.add_argument(
        "--backend",
        metavar="BACKEND",
        choices=["openai", "local"],
        help="Override transcription backend.",
    )
    tr_parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Preview files to transcribe without running.",
    )

    # ---- sync ----------------------------------------------------------
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync data repository (pull and/or push).",
    )
    sync_mutex = sync_parser.add_mutually_exclusive_group()
    sync_mutex.add_argument(
        "--pull-only",
        action="store_true",
        dest="pull_only",
        help="Pull only, do not push.",
    )
    sync_mutex.add_argument(
        "--push-only",
        action="store_true",
        dest="push_only",
        help="Push only, do not pull.",
    )

    # ---- status --------------------------------------------------------
    subparsers.add_parser(
        "status",
        help="Show pipeline status.",
    )

    # ---- debug ---------------------------------------------------------
    debug_parser = subparsers.add_parser(
        "debug",
        help="Inspect Telegram connection and audio message metadata.",
    )
    debug_parser.add_argument(
        "--scan-limit",
        metavar="N",
        type=int,
        default=3,
        dest="scan_limit",
        help="Number of audio messages to inspect (default: 3).",
    )

    # ---- ci-export -----------------------------------------------------
    subparsers.add_parser(
        "ci-export",
        help=(
            "Print all secrets and variables needed for CI/CD. "
            "LOCAL USE ONLY — blocked in CI environments."
        ),
    )

    # ---- session -------------------------------------------------------
    session_parser = subparsers.add_parser(
        "session",
        help="Manage Telegram session files (export for CI, check status).",
    )
    session_sub = session_parser.add_subparsers(dest="session_action", metavar="ACTION")
    session_sub.add_parser(
        "export",
        help=(
            "Print base64-encoded session file to stdout. "
            "Store the output as the TELEGRAM_SESSION_B64 GitHub Secret."
        ),
    )
    session_sub.add_parser(
        "check",
        help="Show whether a session file exists and its basic info.",
    )

    return parser


# ---------------------------------------------------------------------------
# Source acquisition
# ---------------------------------------------------------------------------

def acquire_audio(config, manifest):
    """
    Acquire new audio from the configured source.

    Dispatches on ``config.source.mode``:
      - ``"folder"``  — import audio from a local folder (no Telegram needed).
      - ``"telegram"`` — download audio from a Telegram group.

    Prints a one-line status describing the source and returns a stats object
    exposing a ``.summary()`` method.
    """
    if config.source.mode == "folder":
        from src.folder import FolderImporter

        print(f"Importing audio from folder: {config.source.folder}")
        return FolderImporter(config, manifest).run()

    from src.telegram import TelegramDownloader

    print("Downloading audio from Telegram...")

    async def _download():
        async with TelegramDownloader(config, manifest) as dl:
            return await dl.run()

    return async_run(_download())


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace, config) -> None:
    """Full pipeline: sync → download → transcribe → commit/push."""
    from src.manifest import Manifest
    from src.repository import DataRepository, DataRepoError
    from src.transcribe import Transcriber

    # 1. Prepare the data repository (clone or pull).
    try:
        repo = DataRepository(config.data_repo)
        repo.ensure_ready()
    except DataRepoError as exc:
        print(f"ERROR: Data repository error: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Set up manifest.
    manifest = Manifest(config.download.manifest_file)

    # 3. Acquire audio from the configured source.
    if getattr(args, "dry_run", False):
        pending = manifest.pending_transcription()
        print(f"Dry run — {len(pending)} file(s) pending transcription:")
        for entry in pending:
            print(f"  {entry.get('filename', '(unknown)')}")
        return

    try:
        src_stats = acquire_audio(config, manifest)
        print(src_stats.summary())
    except Exception as exc:
        print(f"ERROR: Audio acquisition failed: {exc}", file=sys.stderr)
        log.debug("Acquisition error detail", exc_info=True)
        sys.exit(1)

    # 4. Transcribe.
    print("Transcribing audio...")
    try:
        tr_stats = Transcriber(config, manifest).run()
        print(tr_stats.summary())
    except Exception as exc:
        print(f"ERROR: Transcription failed: {exc}", file=sys.stderr)
        log.debug("Transcription error detail", exc_info=True)
        sys.exit(1)

    # 5. Commit and push.
    if not getattr(args, "no_push", False):
        commit_msg = f"OctoScribe update {datetime.now().date()}"
        print(f"Committing and pushing: {commit_msg}")
        try:
            result = repo.commit_and_push(commit_msg)
            if result.success:
                print("Repository updated successfully.")
            else:
                print(f"WARNING: git operation returned non-zero: {result}", file=sys.stderr)
        except Exception as exc:
            print(f"ERROR: Repository push failed: {exc}", file=sys.stderr)
            log.debug("Push error detail", exc_info=True)
            sys.exit(1)
    else:
        print("Skipping push (--no-push).")


def cmd_download(args: argparse.Namespace, config) -> None:
    """Acquire new audio from the configured source (Telegram or folder)."""
    from src.manifest import Manifest

    manifest = Manifest(config.download.manifest_file)

    try:
        stats = acquire_audio(config, manifest)
        print(stats.summary())
    except Exception as exc:
        print(f"ERROR: Audio acquisition failed: {exc}", file=sys.stderr)
        log.debug("Acquisition error detail", exc_info=True)
        sys.exit(1)


def cmd_transcribe(args: argparse.Namespace, config) -> None:
    """Transcribe unprocessed audio only."""
    from src.manifest import Manifest
    from src.transcribe import Transcriber

    manifest = Manifest(config.download.manifest_file)

    if getattr(args, "dry_run", False):
        pending = manifest.pending_transcription()
        print(f"Dry run — {len(pending)} file(s) pending transcription:")
        for entry in pending:
            print(f"  {entry.get('filename', '(unknown)')}")
        return

    print(f"Transcribing audio with backend '{config.transcribe.backend}'...")
    try:
        stats = Transcriber(config, manifest).run()
        print(stats.summary())
    except Exception as exc:
        print(f"ERROR: Transcription failed: {exc}", file=sys.stderr)
        log.debug("Transcription error detail", exc_info=True)
        sys.exit(1)


def cmd_sync(args: argparse.Namespace, config) -> None:
    """Sync data repository: pull and/or push."""
    from src.repository import DataRepository, DataRepoError

    pull_only = getattr(args, "pull_only", False)
    push_only = getattr(args, "push_only", False)

    try:
        repo = DataRepository(config.data_repo)

        if not push_only:
            print("Pulling from remote...")
            result = repo.pull()
            if result.success:
                print(f"  {result.stdout or 'Already up to date.'}")
            else:
                print(f"WARNING: pull returned non-zero:\n  {result.stderr}", file=sys.stderr)

        if not pull_only:
            commit_msg = f"OctoScribe sync {datetime.now().date()}"
            print("Committing and pushing...")
            result = repo.commit_and_push(commit_msg)
            if result.success:
                print(f"  {result.stdout or 'Nothing to commit.'}")
            else:
                print(f"WARNING: push returned non-zero:\n  {result.stderr}", file=sys.stderr)

    except DataRepoError as exc:
        print(f"ERROR: Repository error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_status(args: argparse.Namespace, config) -> None:
    """Show pipeline status."""
    from src.manifest import Manifest
    from src.repository import DataRepository, DataRepoError

    # Repository status.
    try:
        repo = DataRepository(config.data_repo)
        repo_status = repo.status()
    except DataRepoError as exc:
        repo_status = {
            "path": str(config.data_repo.path),
            "branch": config.data_repo.branch,
            "has_remote": bool(config.data_repo.url),
            "uncommitted_changes": False,
        }
        log.debug("Could not read repo status: %s", exc)

    remote_display = config.data_repo.url or "(none)"
    branch_display = repo_status.get("branch") or config.data_repo.branch
    uncommitted_display = "Yes" if repo_status.get("uncommitted_changes") else "No"

    # Manifest stats.
    manifest_path = config.download.manifest_file
    if manifest_path.exists():
        manifest = Manifest(manifest_path)
        stats = manifest.stats()
        pending_count = len(manifest.pending_transcription())
    else:
        stats = {"total": 0, "downloaded": 0, "transcribed": 0, "failed": 0}
        pending_count = 0

    # Config display.
    ini_path_display = str(config.ini_path)
    backend_display = config.transcribe.backend
    model_display = config.transcribe.model

    if config.source.mode == "folder":
        source_display = f"folder ({config.source.folder})"
    else:
        source_display = f"telegram ({config.telegram.group or '(no group set)'})"

    print("OctoScribe Status")
    print("=================")
    print(f"Source:        {source_display}")
    print(f"Data repo:     {repo_status.get('path', config.data_repo.path)} (branch: {branch_display})")
    print(f"Remote:        {remote_display}")
    print(f"Uncommitted:   {uncommitted_display}")
    print()
    print("Manifest:")
    print(f"  Total entries:    {stats['total']}")
    print(f"  Downloaded:       {stats['downloaded']}")
    print(f"  Transcribed:      {stats['transcribed']}")
    print(f"  Failed:           {stats['failed']}")
    print(f"  Pending transcription: {pending_count}")
    print()
    print(f"Config: {ini_path_display}")
    print(f"Backend: {backend_display} ({model_display})")


_CI_ENV_MARKERS = ("CI", "GITHUB_ACTIONS", "GITEA_ACTIONS", "GITLAB_CI", "CIRCLECI", "TRAVIS")


def cmd_ci_export(args: argparse.Namespace, config) -> None:
    """Print all secrets and variables needed for CI/CD. Blocked in CI environments."""
    import base64

    # Safety guard: refuse to run inside any CI/CD environment.
    active = [m for m in _CI_ENV_MARKERS if os.environ.get(m)]
    if active:
        print(
            f"ERROR: ci-export is blocked in CI environments "
            f"(detected: {', '.join(active)}).",
            file=sys.stderr,
        )
        print(
            "This command is for local use only to help you configure secrets.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Collect session B64 -------------------------------------------
    session_file = config.telegram.session_dir / "octoscribe.session"
    if session_file.exists():
        session_b64 = base64.b64encode(session_file.read_bytes()).decode()
    else:
        session_b64 = "(not found — run `octoscribe.py download` first)"

    # --- Secrets (store as repository Secrets) -------------------------
    secrets = {
        "TELEGRAM_API_ID":    str(config.telegram.api_id),
        "TELEGRAM_API_HASH":  config.telegram.api_hash,
        "TELEGRAM_PHONE":     config.telegram.phone,
        "OPENAI_API_KEY":     config.transcribe.api_key or "(not set)",
        "DATA_REPO_URL":      config.data_repo.url or "(not set)",
        "TELEGRAM_SESSION_B64": session_b64,
    }

    # --- Variables (store as repository Variables, not Secrets) --------
    variables = {
        "SOURCE_MODE":         config.source.mode,
        "SOURCE_FOLDER":       str(config.source.folder) if config.source.folder else "(not set)",
        "TELEGRAM_GROUP":      config.telegram.group or "(not set)",
        "TRANSCRIBE_BACKEND":  config.transcribe.backend,
        "TRANSCRIBE_MODEL":    config.transcribe.model,
        "TRANSCRIBE_LANGUAGE": config.transcribe.language,
        "DATA_REPO_BRANCH":    config.data_repo.branch,
    }

    w = 24  # column width for alignment

    print("=" * 64)
    print("OctoScribe CI/CD Export")
    print("LOCAL USE ONLY — never share this output publicly.")
    print("=" * 64)

    print("\n--- SECRETS (add as repository Secrets) ---\n")
    for key, val in secrets.items():
        print(f"  {key:<{w}} = {val}")

    print("\n--- VARIABLES (add as repository Variables) ---\n")
    for key, val in variables.items():
        print(f"  {key:<{w}} = {val}")

    print()
    print("=" * 64)
    print("Tip: use `octoscribe.py session export` to get just the")
    print("     TELEGRAM_SESSION_B64 value for copying to clipboard.")
    print("=" * 64)


def cmd_session(args: argparse.Namespace, config) -> None:
    """Export or check the Telegram session file."""
    import base64

    action = getattr(args, "session_action", None) or "check"
    session_file = config.telegram.session_dir / "octoscribe.session"

    if action == "export":
        if not session_file.exists():
            print(f"ERROR: No session file found at {session_file}", file=sys.stderr)
            print(
                "Authenticate first by running:  python octoscribe.py download",
                file=sys.stderr,
            )
            sys.exit(1)
        data = session_file.read_bytes()
        print(base64.b64encode(data).decode())

    else:  # check (default)
        if session_file.exists():
            stat = session_file.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Session file : {session_file}")
            print(f"  Size       : {stat.st_size} bytes")
            print(f"  Modified   : {mtime}")
            print()
            print("To export for CI:  python octoscribe.py session export")
        else:
            print(f"No session file at {session_file}")
            print("Run `python octoscribe.py download` to authenticate interactively.")


def cmd_debug(args: argparse.Namespace, config) -> None:
    """Inspect Telegram connection and audio metadata."""
    from src.debug import DebugInspector

    scan_limit = getattr(args, "scan_limit", 3)
    inspector = DebugInspector(config, scan_limit=scan_limit)
    try:
        async_run(inspector.run())
    except Exception as exc:
        print(f"ERROR: Debug inspection failed: {exc}", file=sys.stderr)
        log.debug("Debug error detail", exc_info=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    setup_logging(args.verbose)

    # Build config overrides from CLI args.
    overrides = build_overrides(args)

    # Load configuration — Config.load() calls sys.exit(1) on missing secrets.
    # We catch SystemExit so we can print a friendlier top-level hint.
    try:
        from src.config import Config

        config = Config.load(
            ini_path=args.config if args.config else None,
            env_file=args.env if args.env else None,
            **overrides,
        )
    except SystemExit:
        # src/config.py already printed detailed error messages to stderr.
        print(
            "\nTip: Set secrets in a .env file or export them as environment variables.\n"
            "     Run 'octoscribe --help' for usage information.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Dispatch to the appropriate command handler.
    commands = {
        "run": cmd_run,
        "download": cmd_download,
        "transcribe": cmd_transcribe,
        "sync": cmd_sync,
        "status": cmd_status,
        "debug": cmd_debug,
        "ci-export": cmd_ci_export,
        "session": cmd_session,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
