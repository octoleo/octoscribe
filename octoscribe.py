#!/usr/bin/env python3
"""
octoscribe.py — CLI entry point for the OctoScribe pipeline.

Usage:
    octoscribe.py run              Full pipeline: acquire → transcribe
    octoscribe.py download         Acquire new audio from the configured source
                                   (Telegram group or local folder)
    octoscribe.py transcribe       Transcribe unprocessed audio only
    octoscribe.py verify           Compare transcripts to reference text
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
        # GitHub Actions captures stdout as the primary step log. Keeping all
        # operational logging here makes every decision visible to callers.
        stream=sys.stdout,
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

def direct_path_overrides(
    *,
    audio_path: str | Path | None,
    transcript_path: str | Path | None,
    manifest_path: str | Path | None,
    reference_path: str | Path | None = None,
    comparison_report_path: str | Path | None = None,
    cwd: Path | None = None,
) -> dict[str, Path]:
    """Map the simple path interface onto the existing typed configuration."""
    base = (cwd or Path.cwd()).resolve()

    def resolve(value: str | Path | None, default: str | Path) -> Path:
        candidate = Path(value if value is not None else default).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate.resolve()

    audio = resolve(audio_path, "audio")
    transcripts = resolve(transcript_path, "transcriptions")
    manifest = resolve(manifest_path, "manifest.json")
    text_parent = transcripts.parent
    references = resolve(
        reference_path,
        text_parent / "reference-transcripts",
    )
    comparisons = resolve(
        comparison_report_path,
        text_parent / "comparison-reports",
    )
    candidates = text_parent / "candidates"
    reports = text_parent / "reports"

    audio_root = audio.parent
    text_root = Path(
        os.path.commonpath(
            (
                transcripts.parent,
                manifest.parent,
                references.parent,
                comparisons.parent,
                candidates.parent,
                reports.parent,
            )
        )
    )
    try:
        audio_root.relative_to(text_root)
        roots_are_nested = audio_root != text_root
    except ValueError:
        try:
            text_root.relative_to(audio_root)
            roots_are_nested = audio_root != text_root
        except ValueError:
            roots_are_nested = False
    if roots_are_nested:
        shared_root = Path(os.path.commonpath((audio_root, text_root)))
        audio_root = shared_root
        text_root = shared_root

    return {
        "audio_repo__path": audio_root,
        "transcript_repo__path": text_root,
        "paths__audio_dir": audio,
        "paths__transcriptions_dir": transcripts,
        "paths__manifest_file": manifest,
        "paths__artifacts_dir": candidates,
        "paths__reports_dir": reports,
        "paths__reference_dir": references,
        "paths__comparison_reports_dir": comparisons,
    }


def build_overrides(args: argparse.Namespace) -> dict:
    """
    Translate CLI arguments into Config.load() override kwargs.

    Uses the double-underscore section__key convention expected by
    Config._override().
    """
    overrides: dict = {}

    # Repository path overrides. --data-repo is the legacy shared-layout alias.
    if getattr(args, "data_repo", None):
        overrides["data_repo__path"] = args.data_repo
    if getattr(args, "audio_repo", None):
        overrides["audio_repo__path"] = args.audio_repo
    if getattr(args, "transcript_repo", None):
        overrides["transcript_repo__path"] = args.transcript_repo

    # The primary path contract is deliberately direct: callers name the audio
    # directory, transcript directory, and manifest file they want.  Repository
    # roots remain compatibility aliases and are derived internally when any
    # direct path is supplied.
    direct_values = (
        getattr(args, "audio_path", None),
        getattr(args, "transcript_path", None),
        getattr(args, "manifest_path", None),
        getattr(args, "reference_path", None),
        getattr(args, "comparison_report_path", None),
    )
    if any(value is not None for value in direct_values):
        overrides.update(
            direct_path_overrides(
                audio_path=getattr(args, "audio_path", None),
                transcript_path=getattr(args, "transcript_path", None),
                manifest_path=getattr(args, "manifest_path", None),
                reference_path=getattr(args, "reference_path", None),
                comparison_report_path=getattr(
                    args,
                    "comparison_report_path",
                    None,
                ),
            )
        )

    # Telegram group override (download / run)
    if getattr(args, "group", None):
        overrides["telegram__group"] = args.group

    # Transcription backend override (transcribe / run)
    if getattr(args, "backend", None):
        overrides["transcribe__backend"] = args.backend
    if getattr(args, "providers", None):
        overrides["transcribe__providers"] = args.providers
    if getattr(args, "primary_provider", None):
        overrides["transcribe__primary_provider"] = args.primary_provider

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


def _max_word_error_rate_arg(value: str) -> float:
    """Argparse adapter for the verifier's inclusive 0..1 threshold."""
    from src.transcript_compare import (
        ComparisonInputError,
        validate_max_word_error_rate,
    )

    try:
        return validate_max_word_error_rate(value)
    except ComparisonInputError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level ArgumentParser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="octoscribe",
        description=(
            "OctoScribe — Download Telegram audio, transcribe verbatim, "
            "and write durable filesystem evidence."
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
        "--audio-repo",
        metavar="PATH",
        dest="audio_repo",
        help="Override audio_repo.path (immutable source audio).",
    )
    parser.add_argument(
        "--transcript-repo",
        metavar="PATH",
        dest="transcript_repo",
        help="Override transcript_repo.path (text, candidates, and reports).",
    )
    parser.add_argument(
        "--audio-path",
        default=os.environ.get("AUDIO_PATH"),
        help="Audio directory; relative paths resolve from the current workspace.",
    )
    parser.add_argument(
        "--transcript-path",
        default=os.environ.get("TRANSCRIPT_PATH"),
        help="Generated transcript directory; relative paths resolve from cwd.",
    )
    parser.add_argument(
        "--manifest-path",
        default=os.environ.get("MANIFEST_PATH"),
        help="Durable manifest file; relative paths resolve from cwd.",
    )
    parser.add_argument(
        "--reference-path",
        default=os.environ.get("REFERENCE_PATH"),
        help="Reference transcript directory used by verify.",
    )
    parser.add_argument(
        "--comparison-report-path",
        default=os.environ.get("COMPARISON_REPORT_PATH"),
        help="JSON comparison-report directory used by verify.",
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
        help="Full pipeline: acquire audio → transcribe.",
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
        "--providers",
        metavar="LIST",
        help="Ordered ASR providers, e.g. openai,xai,meta (maximum three).",
    )
    run_parser.add_argument(
        "--primary-provider",
        choices=["openai", "xai", "meta", "whisper"],
        help="Canonical transcript provider; must also be enabled.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Preview files to transcribe without running the pipeline.",
    )
    run_parser.add_argument(
        "--audio-revision",
        default=os.environ.get("OCTOSCRIBE_AUDIO_REVISION"),
        help="Optional caller-supplied source revision recorded as provenance.",
    )
    run_parser.add_argument(
        "--audio-repository-branch",
        default=os.environ.get("OCTOSCRIBE_AUDIO_REPOSITORY_BRANCH"),
        help="Optional caller-supplied source branch recorded as provenance.",
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
        "--providers",
        metavar="LIST",
        help="Ordered ASR providers, e.g. openai,xai,meta (maximum three).",
    )
    tr_parser.add_argument(
        "--primary-provider",
        choices=["openai", "xai", "meta", "whisper"],
        help="Canonical transcript provider; must also be enabled.",
    )
    tr_parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Preview files to transcribe without running.",
    )
    tr_parser.add_argument(
        "--audio-revision",
        default=os.environ.get("OCTOSCRIBE_AUDIO_REVISION"),
        help="Optional caller-supplied source revision recorded as provenance.",
    )
    tr_parser.add_argument(
        "--audio-repository-branch",
        default=os.environ.get("OCTOSCRIBE_AUDIO_REPOSITORY_BRANCH"),
        help="Optional caller-supplied source branch recorded as provenance.",
    )

    # ---- verify --------------------------------------------------------
    verify_parser = subparsers.add_parser(
        "verify",
        help="Compare generated transcripts to reference text word-for-word.",
    )
    verify_parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Override the configured directory of committed reference .txt files.",
    )
    verify_parser.add_argument(
        "--comparison-reports-dir",
        type=Path,
        help="Override the configured JSON comparison-report directory.",
    )
    verify_parser.add_argument(
        "--allow-missing-references",
        action="store_false",
        dest="reference_required",
        default=True,
        help=(
            "Bootstrap provenance only: record references as optional in the "
            "summary. Missing references still do not verify successfully."
        ),
    )
    verify_parser.add_argument(
        "--capture-reference",
        action="store_true",
        help=(
            "Copy generated transcripts into an empty reference directory; "
            "intended only for deliberate manual baseline capture."
        ),
    )
    verify_parser.add_argument(
        "--max-word-error-rate",
        type=_max_word_error_rate_arg,
        default=os.environ.get("MAX_WORD_ERROR_RATE", "0"),
        help=(
            "Maximum accepted word error rate from 0 to 1 (default: "
            "MAX_WORD_ERROR_RATE or strict 0). Numeric and negation changes "
            "always fail regardless of this tolerance."
        ),
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

def _maybe_print_dry_run(args: argparse.Namespace, manifest) -> bool:
    """
    Handle the ``--dry-run`` flag shared by ``run`` and ``transcribe``.

    When the flag is set, print the files currently pending transcription and
    return ``True`` so the caller can return early.  Returns ``False`` (and
    prints nothing) otherwise.  Centralised here so the two commands cannot
    drift apart.
    """
    if not getattr(args, "dry_run", False):
        return False
    pending = manifest.pending_transcription()
    print(f"Dry run — {len(pending)} file(s) pending transcription:")
    for entry in pending:
        print(f"  {entry.get('filename', '(unknown)')}")
    return True


def cmd_run(args: argparse.Namespace, config) -> None:
    """Acquire and transcribe using caller-supplied filesystem workspaces."""
    from src.manifest import Manifest
    from src.repository import EvidenceWorkspaces, WorkspaceError
    from src.transcribe import Transcriber

    try:
        EvidenceWorkspaces(config).ensure_ready()
    except (OSError, WorkspaceError) as exc:
        print(f"ERROR: Workspace preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    manifest = Manifest(config.download.manifest_file)
    if _maybe_print_dry_run(args, manifest):
        return

    log.info(
        "Pipeline workspaces: audio=%s text=%s manifest=%s",
        config.audio_repo.path,
        config.text_repo.path,
        config.download.manifest_file,
    )
    acquisition_error: Exception | None = None
    acquisition_failed = False
    try:
        src_stats = acquire_audio(config, manifest)
        print(src_stats.summary())
        acquisition_failed = bool(getattr(src_stats, "failed", 0))
    except Exception as exc:
        acquisition_error = exc
        print(f"ERROR: Audio acquisition failed: {exc}", file=sys.stderr)
        log.debug("Acquisition error detail", exc_info=True)
    finally:
        # Both source adapters already save periodically; this final flush is
        # authoritative even when acquisition partially fails.
        manifest.save()

    print("Transcribing audio...")
    try:
        tr_stats = Transcriber(
            config,
            manifest,
            audio_revision=getattr(args, "audio_revision", None),
            audio_repository_branch=getattr(
                args, "audio_repository_branch", None
            ),
        ).run()
        print(tr_stats.summary())
    except Exception as exc:
        print(f"ERROR: Transcription failed: {exc}", file=sys.stderr)
        log.debug("Transcription error detail", exc_info=True)
        sys.exit(1)

    if acquisition_error or acquisition_failed or tr_stats.failed or tr_stats.skipped:
        print(
            "ERROR: Pipeline completed with one or more failed or skipped "
            "recordings; durable manifest evidence was retained for retry.",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_download(args: argparse.Namespace, config) -> None:
    """Acquire new audio from the configured source (Telegram or folder)."""
    from src.manifest import Manifest
    from src.repository import EvidenceWorkspaces, WorkspaceError

    try:
        EvidenceWorkspaces(config).ensure_ready()
    except (OSError, WorkspaceError) as exc:
        print(f"ERROR: Workspace preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)
    manifest = Manifest(config.download.manifest_file)

    try:
        stats = acquire_audio(config, manifest)
        print(stats.summary())
        if getattr(stats, "failed", 0):
            print(
                "ERROR: Audio acquisition completed with failed items; "
                "manifest evidence was retained for retry.",
                file=sys.stderr,
            )
            sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Audio acquisition failed: {exc}", file=sys.stderr)
        log.debug("Acquisition error detail", exc_info=True)
        sys.exit(1)
    finally:
        manifest.save()


def cmd_transcribe(args: argparse.Namespace, config) -> None:
    """Transcribe unprocessed audio only."""
    from src.manifest import Manifest
    from src.repository import EvidenceWorkspaces, WorkspaceError
    from src.transcribe import Transcriber

    try:
        EvidenceWorkspaces(config).ensure_ready()
    except (OSError, WorkspaceError) as exc:
        print(f"ERROR: Workspace preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)
    manifest = Manifest(config.download.manifest_file)

    if _maybe_print_dry_run(args, manifest):
        return

    providers = getattr(config.transcribe, "providers", ())
    display = ",".join(providers) if providers else config.transcribe.backend
    print(f"Transcribing audio with provider(s) '{display}'...")
    try:
        stats = Transcriber(
            config,
            manifest,
            audio_revision=getattr(args, "audio_revision", None),
            audio_repository_branch=getattr(
                args, "audio_repository_branch", None
            ),
        ).run()
        print(stats.summary())
        if stats.failed or stats.skipped:
            print(
                "ERROR: One or more recordings were not transcribed.",
                file=sys.stderr,
            )
            sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Transcription failed: {exc}", file=sys.stderr)
        log.debug("Transcription error detail", exc_info=True)
        sys.exit(1)


def cmd_verify(args: argparse.Namespace, config) -> None:
    """Compare generated transcripts to references without changing either."""
    from src.transcript_compare import (
        ComparisonInputError,
        compare_transcript_directories,
        comparison_output_lines,
    )

    try:
        summary = compare_transcript_directories(
            config.transcribe.transcriptions_dir,
            args.reference_dir or config.transcribe.reference_dir,
            args.comparison_reports_dir or config.transcribe.comparison_reports_dir,
            reference_required=args.reference_required,
            capture_reference=args.capture_reference,
            max_word_error_rate=getattr(args, "max_word_error_rate", 0.0),
        )
    except (ComparisonInputError, OSError, ValueError) as exc:
        print(f"ERROR: Transcript verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
    for line in comparison_output_lines(summary):
        print(line)
    if not summary["success"]:
        sys.exit(1)


def cmd_status(args: argparse.Namespace, config) -> None:
    """Show pipeline status."""
    from src.manifest import Manifest
    from src.repository import EvidenceWorkspaces

    workspace_statuses = EvidenceWorkspaces(config).status()
    audio_status = workspace_statuses["audio"]
    text_status = workspace_statuses["transcripts"]

    # Manifest stats.
    manifest_path = config.download.manifest_file
    if manifest_path.exists():
        manifest = Manifest(manifest_path)
        stats = manifest.stats()
        quality_stats = manifest.quality_stats()
        pending_count = len(manifest.pending_transcription())
    else:
        stats = {"total": 0, "downloaded": 0, "transcribed": 0, "failed": 0}
        quality_stats = {
            "machine_transcribed": 0,
            "cross_checked": 0,
            "needs_review": 0,
            "human_verified": 0,
            "legacy_completed": 0,
        }
        pending_count = 0

    # Config display.
    ini_path_display = str(config.ini_path)
    providers = getattr(config.transcribe, "providers", ())
    backend_display = ",".join(providers) if providers else config.transcribe.backend
    model_display = config.transcribe.model

    if config.source.mode == "folder":
        source_display = f"folder ({config.source.folder})"
    else:
        source_display = f"telegram ({config.telegram.group or '(no group set)'})"

    print("OctoScribe Status")
    print("=================")
    print(f"Source:        {source_display}")
    print(f"Audio workspace: {audio_status.path}")
    print(f"Text workspace:  {text_status.path}")
    print(
        "Workspace ready: "
        f"audio={'Yes' if audio_status.exists and audio_status.writable else 'No'}, "
        f"text={'Yes' if text_status.exists and text_status.writable else 'No'}"
    )
    print()
    print("Manifest:")
    print(f"  Total entries:    {stats['total']}")
    print(f"  Downloaded:       {stats['downloaded']}")
    print(f"  Transcribed:      {stats['transcribed']}")
    print(f"  Failed:           {stats['failed']}")
    print(f"  Pending transcription: {pending_count}")
    print(f"  Machine only:     {quality_stats['machine_transcribed']}")
    print(f"  Cross-checked:    {quality_stats['cross_checked']}")
    print(f"  Needs review:     {quality_stats['needs_review']}")
    print(f"  Human verified:   {quality_stats['human_verified']}")
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
        "XAI_API_KEY":        config.transcribe.xai_api_key or "(not set)",
        "META_ASR_API_KEY":   config.transcribe.meta_asr_api_key or "(not set)",
        "TELEGRAM_SESSION_B64": session_b64,
    }

    # --- Variables (store as repository Variables, not Secrets) --------
    variables = {
        "OCTOSCRIBE_SOURCE_MODE": config.source.mode,
        "OCTOSCRIBE_SOURCE_FOLDER": str(config.source.folder) if config.source.folder else "(not set)",
        "TELEGRAM_GROUP":      config.telegram.group or "(not set)",
        "TRANSCRIBE_BACKEND":  config.transcribe.backend,
        "TRANSCRIBE_MODEL":    config.transcribe.model,
        "TRANSCRIBE_LANGUAGE": config.transcribe.language,
        "OCTOSCRIBE_ASR_PROVIDERS": ",".join(config.transcribe.providers),
        "OCTOSCRIBE_PRIMARY_ASR": config.transcribe.primary_provider,
        "XAI_STT_URL":         config.transcribe.xai_base_url,
        "META_ASR_URL":        config.transcribe.meta_asr_url or "(not set)",
        "META_ASR_MODEL":      config.transcribe.meta_asr_model,
        "META_ASR_LANGUAGE":   config.transcribe.meta_asr_language,
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
            validation_profile=args.command,
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
        "verify": cmd_verify,
        "status": cmd_status,
        "debug": cmd_debug,
        "ci-export": cmd_ci_export,
        "session": cmd_session,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
