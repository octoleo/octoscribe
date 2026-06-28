# OctoScribe Architecture

This document explains how OctoScribe is organised and the design principles
behind that organisation. It is aimed at contributors who want to understand
*why* the code is shaped the way it is before changing it.

## Pipeline at a glance

```
            ┌─────────────┐
 source ───▶│  acquire    │  TelegramDownloader  |  FolderImporter
            └──────┬──────┘        (audio source Strategy)
                   │ writes audio + manifest entries
                   ▼
            ┌─────────────┐
            │  manifest   │  Manifest (single source of truth for state)
            └──────┬──────┘
                   │ pending_transcription()
                   ▼
            ┌─────────────┐
            │ transcribe  │  Transcriber + TranscriptionBackend Strategy
            └──────┬──────┘        (OpenAI | local Whisper)
                   │ writes .txt + marks transcribed
                   ▼
            ┌─────────────┐
            │ repository  │  DataRepository (git add/commit/push)
            └─────────────┘
```

`octoscribe.py` is the thin CLI shell: it parses arguments, loads a `Config`,
and wires these components together. It contains orchestration, not logic.

## Module map

| Area | Module(s) | Responsibility |
| --- | --- | --- |
| Configuration | `src/config/` | Load, validate and represent settings. |
| Audio helpers | `src/audio.py` | Pure, framework-agnostic file helpers. |
| Persistence | `src/persistence.py` | Atomic writes + periodic-save cadence. |
| Telegram shared | `src/telegram_client.py` | Session restore, entity resolution. |
| Audio sources | `src/telegram.py`, `src/folder.py` | Acquire audio into the manifest. |
| Transcription | `src/transcribe/` | Turn audio into verbatim text. |
| State | `src/manifest.py` | Track per-item download/transcription state. |
| Data repo | `src/repository.py` | Manage the git data repository. |
| Diagnostics | `src/debug.py` | Inspect Telegram messages/metadata. |

## How the SOLID principles show up

### Single Responsibility
The two largest modules were split into packages so each file has one reason to
change:

* **`src/config/`** — `models.py` (data shapes), `helpers.py` (parsing),
  `loader.py` (assembly + validation), `root.py` (the aggregate object). You can
  change the precedence rules without touching the data shapes, and vice versa.
* **`src/transcribe/`** — `prompt.py`, `normalize.py`, `results.py`,
  `backends/` and `transcriber.py`. The retry/backoff behaviour lives in its own
  `backends/retry.py` and is unit-tested in isolation.

### Open/Closed
Adding a new transcription backend means implementing
`TranscriptionBackend` and registering it in `Transcriber.create_backend`; the
orchestrator and CLI do not change. The same Strategy shape lets the audio
source be Telegram or a local folder without the rest of the pipeline knowing.

### Liskov Substitution
`OpenAIBackend` and `LocalWhisperBackend` are interchangeable behind
`TranscriptionBackend`. The `Transcriber` treats either identically; tests
substitute a `MagicMock(spec=TranscriptionBackend)` and the orchestrator behaves
the same.

### Interface Segregation
`TranscriptionBackend` is deliberately tiny — `transcribe(path) -> str` plus a
`name`. The orchestrator depends on nothing more. `PeriodicSaver` depends only
on a structural `save()` method (`_Saveable` protocol), not on the concrete
`Manifest`.

### Dependency Inversion
`Transcriber` depends on the `TranscriptionBackend` abstraction, not on OpenAI
or Whisper. The debug inspector and downloader both depend on the shared
`telegram_client` helpers rather than reaching into each other's internals (the
inspector previously called a *private* downloader method — that coupling is
gone).

## Shared building blocks (DRY)

Three pieces of logic that were previously copy-pasted now live in one place:

* **Atomic writes** — `persistence.atomic_write_text/bytes` (used by the
  manifest and the transcriber) guarantee a reader never sees a half-written
  file: write to `<name>.tmp`, then `os.replace`.
* **Periodic saving** — `persistence.PeriodicSaver` replaces three identical
  "save every N items" counter loops in the downloader, importer and
  transcriber.
* **Telegram session/entity** — `telegram_client.restore_session_from_env`,
  `session_base_path` and `resolve_group_entity` are shared by the downloader
  and the debug inspector.

## Transcription stability guarantees

Transcription is the part users care about most, so the orchestrator makes two
promises designed to **never silently lose a transcript**:

1. **Collision-safe output.** If two recordings derive the same output filename
   (e.g. the same Telegram title), the second is disambiguated by message id via
   `audio.unique_filepath` instead of overwriting the first.
2. **Empty-result guard.** A blank transcript is treated as a *failure* (so the
   next run retries it) rather than written out as a misleading "completed"
   empty file.

On the backend side, `RetryPolicy` retries only *transient* errors (rate limits,
5xx, timeouts) with exponential backoff and jitter; *permanent* errors (auth,
invalid file) and *unknown* errors are raised immediately so we never hammer an
endpoint over a failure we do not understand. Per-item failures are isolated:
one bad file never aborts the batch.

## Backwards compatibility

The `config` and `transcribe` packages re-export every name their former
single-module versions exposed (including private helpers such as
`_normalize_text` and `_parse_bool`). Existing imports — `from src.transcribe
import Transcriber`, `from config import _parse_bool` — continue to work
unchanged, which is why the refactor preserved the full test suite.

## Testing

`pytest` with `pytest-asyncio` and `pytest-mock`. Tests mock all external
services (Telethon, OpenAI, Faster-Whisper), so the suite needs no network,
credentials or GPU. New unit suites cover the extracted building blocks:
`test_persistence.py`, `test_retry.py`, `test_telegram_client.py`, and
`test_transcribe_hardening.py`.
