# OctoScribe Testing Guide

This guide explains how the OctoScribe test suite is structured, how to run it,
and how to add to it. It is aimed at contributors.

For how the code itself works, see [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Table of contents

1. [Philosophy](#philosophy)
2. [Setup](#setup)
3. [Running the tests](#running-the-tests)
4. [What each test file covers](#what-each-test-file-covers)
5. [Shared fixtures (`conftest.py`)](#shared-fixtures-conftestpy)
6. [How external services are mocked](#how-external-services-are-mocked)
7. [Writing a new test](#writing-a-new-test)
8. [How tests run in CI](#how-tests-run-in-ci)

---

## Philosophy

The suite follows one rule above all: **a test must never touch the real
world.** No network calls, no Telegram login, no OpenAI billing, no GPU, no
files outside a temporary directory. Every external dependency is mocked, and
every file operation happens under pytest's `tmp_path`.

This is what lets the whole suite run in well under a second on any machine —
including CI runners with no credentials and no hardware — and makes every test
deterministic. The three external systems are stubbed as follows:

* **Telethon** (Telegram) — patched with `AsyncMock`/`MagicMock`.
* **OpenAI** — the client is patched; transcription returns canned text or
  raises canned errors to exercise the retry logic.
* **Faster-Whisper** — the model is patched; the import-time CUDA setup is never
  actually run.

Because the source/backend seams are small interfaces, tests frequently inject a
`MagicMock(spec=TranscriptionBackend)` straight into the `Transcriber` and assert
the orchestration without any real backend at all.

---

## Setup

The test dependencies are part of `requirements.txt`, so a normal install
already gives you everything you need:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The relevant test packages are `pytest`, `pytest-asyncio` (for the async
Telegram code), `pytest-mock`, and `pytest-cov`.

---

## Running the tests

From the project root:

```bash
# Run the whole suite
pytest

# Verbose — show every test name
pytest -v

# Run one file
pytest tests/test_transcribe.py

# Run one test by name
pytest tests/test_transcribe.py::test_openai_backend_retries_on_rate_limit

# Run every test whose name matches a keyword
pytest -k retry
```

### With coverage

```bash
# Terminal summary of which lines in src/ are not covered
pytest --cov=src --cov-report=term-missing

# Full HTML report
pytest --cov=src --cov-report=html
# then open htmlcov/index.html
```

This mirrors what CI runs (see [below](#how-tests-run-in-ci)).

---

## What each test file covers

| Test file | Subject under test | Focus |
| --------- | ------------------ | ----- |
| `test_config.py` | `src/config/` | Multi-source loading, precedence, source selection, path resolution, and that every missing-required setting produces a clear error. |
| `test_persistence.py` | `src/persistence.py` | Atomic-write tmp-then-rename behaviour, cleanup on failure, and `PeriodicSaver` cadence. |
| `test_telegram_client.py` | `src/telegram_client.py` | Session restore from `TELEGRAM_SESSION_B64`, session path derivation, and entity resolution (username vs numeric id). |
| `test_telegram.py` | `src/telegram.py` | Audio detection, metadata extraction, filename building, the download loop, dedup, and resume — all with Telethon mocked. |
| `test_folder.py` | `src/folder.py` | Folder scan (recursive/flat), content-hash dedup, resume, copy behaviour, and error handling. Uses the real filesystem. |
| `test_manifest.py` | `src/manifest.py` | State transitions, the `pending_*` queries, stats, JSON integrity, and thread-safety. |
| `test_transcribe.py` | `src/transcribe/` | Backend factory, OpenAI and local backends, retry on transient errors, no-retry on permanent errors, and the orchestrator's success/failure/skip paths. |
| `test_transcribe_hardening.py` | `src/transcribe/transcriber.py` | The two stability guarantees: collision-safe output names and the empty-result-is-a-failure guard. |
| `test_retry.py` | `src/transcribe/backends/retry.py` | The `ErrorClassifier` patterns and the `RetryPolicy` loop (backoff, exhaustion, immediate raises) — with `time.sleep` patched so it runs instantly. |
| `test_repository.py` | `src/repository.py` | `ensure_ready` clone/init/pull branches, commit/push, nothing-to-commit, and status — with `subprocess.run` mocked. |
| `test_source_cmd.py` | `octoscribe.py` | `build_overrides` flag→config translation and the `acquire_audio` dispatch between folder and Telegram. |
| `test_session_cmd.py` | `octoscribe.py` | The `session export`/`check` subcommand. |
| `test_ci_export_cmd.py` | `octoscribe.py` | The `ci-export` output and its guard that refuses to run inside a CI environment. |
| `test_debug.py` | `src/debug.py` | The `DebugInspector` metadata dump, with Telethon mocked. |

---

## Shared fixtures (`conftest.py`)

`tests/conftest.py` provides the fixtures most tests build on:

* **`tmp_data_dir`** — a temporary directory pre-populated with `audio/` and
  `transcriptions/` subdirs, standing in for the data repository.
* **`sample_config`** — a fully populated `Config` with fake-but-valid
  credentials and all paths pointing inside `tmp_data_dir`. Constructing it has
  no side effects, so tests can use it freely.
* **`sample_audio_folder`** — a temp folder containing two audio files (one
  nested) plus a non-audio file that the importer must ignore.
* **`folder_config`** — `sample_config` flipped into `source.mode == "folder"`,
  pointing at `sample_audio_folder`.
* **`tmp_manifest`** — a fresh, empty `Manifest` backed by a temp file.
* **`populated_manifest`** — a `Manifest` with three downloaded entries, two of
  them already transcribed and one pending, for exercising the `pending_*`
  queries.

Because `sample_config` already wires every path into a temp directory, most
tests are just "take the fixture, run the unit, assert on the manifest or the
files it produced."

---

## How external services are mocked

A quick reference for the patterns you'll see and should reuse:

* **Telegram (async).** Telethon's `TelegramClient` is replaced with a
  `MagicMock` whose async methods (`start`, `get_messages`, `download_media`,
  `get_entity`) are `AsyncMock`s. Messages are built as `SimpleNamespace`/typed
  Telethon objects so `is_audio`/`get_audio_metadata` can inspect them.
* **OpenAI.** The backend imports `openai` lazily in its constructor, so tests
  patch `openai.OpenAI` (or inject a pre-built mock client) and set
  `client.audio.transcriptions.create` to return canned text or raise a
  classified error.
* **Faster-Whisper.** Patched at `faster_whisper.WhisperModel`; the local
  backend's `transcribe` is driven with fake segment objects.
* **git.** `subprocess.run` is patched so `DataRepository` can be tested without
  a real git binary or network remote.
* **Time.** `time.sleep` is patched in the retry tests so backoff is exercised
  without actually waiting.

---

## Writing a new test

1. Start from a fixture. If you need a config, take `sample_config`; if you need
   state, take `tmp_manifest` or `populated_manifest`.
2. Keep everything inside `tmp_path` — never write to a real path.
3. Mock at the seam, not deep inside. For transcription, prefer injecting a
   `MagicMock(spec=TranscriptionBackend)` into the `Transcriber` over patching
   `openai`. For a source, patch the Telethon client method you depend on.
4. Assert on observable outcomes: the manifest entry that was written, the file
   that appeared in `transcriptions/`, or the stats summary returned — not on
   internal call order unless that *is* the contract (as in
   `test_source_cmd.py`).
5. Mark async tests with `@pytest.mark.asyncio`.

When you add a new backend or source (see *Extending OctoScribe* in the
[architecture doc](ARCHITECTURE.md#extending-octoscribe)), add both an
orchestration test (mock injected) and a unit test for the new component itself.

---

## How tests run in CI

`.github/workflows/ci.yml` runs the suite on every push and pull request to the
`main` and `v1` branches. It:

1. Checks out the repo.
2. Sets up Python — on a **matrix of 3.11 and 3.12** — with pip caching.
3. Installs `requirements.txt`.
4. Runs `pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
   --tb=short`.
5. Uploads the coverage report to Codecov (only on the 3.12 leg;
   non-blocking).

`fail-fast: false` means a failure on one Python version still lets the other
finish, so you see the full picture. This workflow is entirely separate from the
scheduled *pipeline* workflow that actually downloads and transcribes audio —
see [Running under GitHub Actions](ARCHITECTURE.md#running-under-github-actions).
