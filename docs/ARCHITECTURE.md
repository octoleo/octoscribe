# OctoScribe Architecture

This document explains **how OctoScribe actually works under the hood** — what
happens, in order, from the moment you type a command to the moment your
transcripts are committed and pushed. It traces the real execution path through
the code: which class is constructed, which method it calls next, what decision
that method makes, and *why* the code is shaped that way.

If you only want to install and run OctoScribe, read the
[README](../README.md). This document is for people who want to **understand or
change the code**.

> For how the automated test suite is structured and run, see
> [`docs/TESTING.md`](TESTING.md).

---

## Table of contents

1. [The 30-second mental model](#the-30-second-mental-model)
2. [Module map — what lives where](#module-map--what-lives-where)
3. [Stage 0: Configuration loading](#stage-0-configuration-loading)
4. [Stage 1: The CLI entry point and dispatch](#stage-1-the-cli-entry-point-and-dispatch)
5. [The `run` pipeline, end to end](#the-run-pipeline-end-to-end)
6. [Acquiring audio: the Telegram path](#acquiring-audio-the-telegram-path)
7. [Acquiring audio: the folder path](#acquiring-audio-the-folder-path)
8. [The manifest: the single source of truth](#the-manifest-the-single-source-of-truth)
9. [The transcription pipeline](#the-transcription-pipeline)
10. [The data repository: git lifecycle](#the-data-repository-git-lifecycle)
11. [The other commands and what they trigger](#the-other-commands-and-what-they-trigger)
12. [Command-line options and their outcomes](#command-line-options-and-their-outcomes)
13. [Durability: atomic writes and periodic saves](#durability-atomic-writes-and-periodic-saves)
14. [Concurrency model](#concurrency-model)
15. [Running under GitHub Actions](#running-under-github-actions)
16. [Why the code is shaped this way](#why-the-code-is-shaped-this-way)
17. [Extending OctoScribe](#extending-octoscribe)

---

## The 30-second mental model

OctoScribe is a **pipeline with four stages**. Each stage has exactly one job
and hands its result to the next through a shared state file (`manifest.json`):

```
   ┌──────────┐     ┌──────────┐     ┌─────────────┐     ┌────────────┐
   │  SYNC    │ ──▶ │ ACQUIRE  │ ──▶ │ TRANSCRIBE  │ ──▶ │   SYNC     │
   │ (pull)   │     │ audio    │     │ to text     │     │ (push)     │
   └──────────┘     └──────────┘     └─────────────┘     └────────────┘
   DataRepository   Telegram OR      Transcriber +        DataRepository
   .ensure_ready()  Folder source    a backend            .commit_and_push()
        │                │                  │                   │
        └────────────────┴──────────────────┴───────────────────┘
                              manifest.json
                    (records what has been done so far)
```

* **The audio source is pluggable.** Stage 2 is either a `TelegramDownloader`
  or a `FolderImporter`, chosen by one config setting (`source.mode`). The rest
  of the pipeline does not know or care which one ran.
* **The transcription backend is pluggable.** Stage 3 runs through either the
  `OpenAIBackend` (cloud) or the `LocalWhisperBackend` (local GPU), chosen by
  `transcribe.backend`. Everything else is identical.
* **The manifest is the memory.** Nothing else remembers state. If a run is
  interrupted, the next run reads the manifest and resumes exactly where it
  stopped — that is what makes every command safe to re-run.
* **The data lives in a separate git repository.** Audio, transcripts and the
  manifest are committed to their own repo, not the code repo.

`octoscribe.py` itself contains *no* business logic. It is a thin shell that
parses arguments, loads configuration, and wires these components together.

---

## Module map — what lives where

```
octoscribe/
│
├── octoscribe.py              CLI shell: argument parsing + command dispatch only.
│
├── src/
│   ├── config/                Turns INI + .env + CLI flags into one typed Config.
│   │   ├── models.py            Typed, logic-free dataclasses (the shapes).
│   │   ├── helpers.py           Pure parsers: _parse_bool, _require_int, _resolve_path…
│   │   ├── loader.py            The assembly logic: precedence + validation.
│   │   └── root.py              The aggregate Config object + Config.load() factory.
│   │
│   ├── audio.py               Framework-agnostic helpers: hashing, filenames, durations.
│   ├── persistence.py         Atomic file writes + the periodic-save helper.
│   │
│   ├── telegram_client.py     Shared Telegram session/entity helpers (no client built here).
│   ├── telegram.py            TelegramDownloader: scans a group, downloads audio.
│   ├── folder.py              FolderImporter: copies audio from a local folder.
│   │
│   ├── manifest.py            Manifest: the thread-safe state file (manifest.json).
│   │
│   ├── transcribe/            Audio → verbatim text.
│   │   ├── prompt.py            The single verbatim instruction string.
│   │   ├── normalize.py         Whitespace-only, word-preserving cleanup.
│   │   ├── results.py           TranscriptionResult / BatchStats value objects.
│   │   ├── transcriber.py       Transcriber: the batch orchestrator + backend factory.
│   │   └── backends/
│   │       ├── base.py            TranscriptionBackend interface (one method).
│   │       ├── retry.py           RetryPolicy + ErrorClassifier.
│   │       ├── openai_backend.py  Cloud backend (gpt-4o-transcribe).
│   │       └── local_whisper.py   Local Faster-Whisper backend.
│   │
│   ├── repository.py          DataRepository: clone/init/pull/commit/push via git CLI.
│   └── debug.py               DebugInspector: dumps Telegram message metadata.
│
├── action.yml                 Composite GitHub Action wrapper.
└── .github/workflows/         CI (tests) and the scheduled pipeline workflow.
```

The dependency direction is strict and one-way: **leaf helpers** (`audio.py`,
`persistence.py`, `config/`) depend on nothing else in the project; the
**sources, transcriber and repository** depend on those leaves and on the
manifest; and `octoscribe.py` sits on top, depending on everything but depended
on by nothing. There are no import cycles.

---

## Stage 0: Configuration loading

Before any command runs, `main()` builds one `Config` object. Every later stage
reads its settings from this object and never re-parses files. Understanding the
config flow first makes the rest of the codebase obvious.

### What gets called

`octoscribe.py:main()` → `Config.load()` (`src/config/root.py`) →
`_ConfigLoader.build()` (`src/config/loader.py`). `build()` runs five steps in a
fixed order:

```
_resolve_env_file()      Find the .env file (CLI flag > $OCTOSCRIBE_ENV > ./.env)
        │                and load it into os.environ via python-dotenv.
        ▼
_resolve_ini_path()      Find the INI file (CLI flag > $OCTOSCRIBE_CONFIG >
        │                ./conf/octoscribe.ini > ./octoscribe.ini).
        ▼
_parse_ini()             Read the built-in _DEFAULT_INI string first, THEN
        │                overlay the user's INI on top. A missing INI = pure defaults.
        ▼
_build_*()               Build each typed section in a deliberate order (see below).
        ▼
_validate()              Collect ALL errors, then _die() once with the full list.
```

### Precedence

Every individual setting is resolved by `_ConfigLoader._override()`, which
applies this precedence (highest wins):

```
1. CLI overrides         (kwargs passed to Config.load, e.g. telegram__group=…)
2. Environment / .env    (os.environ, populated from the shell and the .env file)
3. INI file values       (conf/octoscribe.ini)
4. Built-in defaults     (the _DEFAULT_INI string in loader.py)
```

**Secrets are special:** API IDs, hashes, phone numbers, the OpenAI key and the
data-repo URL are read **only** from the environment, never from the INI file.
This is enforced in the `_build_*` methods (e.g. `os.environ.get("OPENAI_API_KEY")`)
and is why the INI file can be committed in spirit while the `.env` file cannot.

### Order matters in `_build_*`

```python
data_repo = self._build_data_repo(ini)     # built FIRST…
source    = self._build_source(ini)         # …then source…
telegram  = self._build_telegram(ini, data_repo.path,
                                 require_secrets=(source.mode == "telegram"))
download  = self._build_download(ini, data_repo.path)
transcribe = self._build_transcribe(ini, data_repo.path)
```

Two ordering decisions are load-bearing:

1. **`data_repo` is built first** because every other section resolves its
   relative paths (`audio/`, `transcriptions/`, `manifest.json`, `.session/`)
   *against the data-repo path* using `helpers._resolve_path`. So `audio_dir =
   audio` becomes `<data_repo.path>/audio`.
2. **`source` is built before `telegram`** because the source mode decides
   whether Telegram credentials are *required*. In `folder` mode,
   `require_secrets=False` and `_require_int` becomes `_optional_int` — you can
   run OctoScribe with no Telegram credentials at all.

### Validation

`_validate()` accumulates a list of human-readable errors (unknown source mode,
missing folder in folder mode, missing Telegram secrets in telegram mode,
unknown backend, missing OpenAI key when `backend=openai`) and, if any exist,
prints them all at once and exits. It also emits a non-fatal warning if the data
repo path is *inside* the project tree (you almost never want large audio binaries
in your source checkout).

The result is a single immutable-ish `Config` aggregating `SourceConfig`,
`TelegramConfig`, `DownloadConfig`, `TranscribeConfig` and `DataRepoConfig`
(all in `src/config/models.py`). `Config.redacted_repr()` exists so the whole
object can be logged with secrets masked.

---

## Stage 1: The CLI entry point and dispatch

`octoscribe.py:main()` is short and worth reading in full. Its sequence:

```python
parser = build_parser()             # argparse with one subparser per command
args   = parser.parse_args()
if args.command is None: print_help(); exit(0)
setup_logging(args.verbose)         # DEBUG if --verbose else INFO, to stderr
overrides = build_overrides(args)   # CLI flags → config override dict
config = Config.load(..., **overrides)
commands[args.command](args, config)  # dispatch table → cmd_run / cmd_download / …
```

`build_overrides()` is the bridge between argparse and the config system. It
translates flags into the `section__key` convention the loader expects:

| CLI flag        | Override key produced      |
| --------------- | -------------------------- |
| `--data-repo`   | `data_repo__path`          |
| `--group`       | `telegram__group`          |
| `--backend`     | `transcribe__backend`      |
| `--source`      | `source__mode`             |
| `--folder`      | `source__folder` (and sets `source__mode=folder` if `--source` was omitted) |

`commands` is a plain dict mapping command name → handler function
(`cmd_run`, `cmd_download`, `cmd_transcribe`, `cmd_sync`, `cmd_status`,
`cmd_debug`, `cmd_ci_export`, `cmd_session`). Each handler imports the heavy
modules it needs **lazily, inside the function** — so `octoscribe.py status`
never imports Telethon or OpenAI, and a missing optional dependency only bites
when you actually use the feature that needs it.

---

## The `run` pipeline, end to end

`cmd_run()` is the canonical full pipeline. Here is the exact call chain, with
the decision each step makes:

```
cmd_run(args, config)
│
├─ 1. repo = DataRepository(config.data_repo)
│     repo.ensure_ready()
│        ├─ if <path>/.git exists      → pull latest
│        ├─ elif path exists, no .git  → raise DataRepoError (refuse to clobber)
│        ├─ elif url is set            → git clone --branch <branch> <url> <path>
│        └─ else                       → git init -b <branch>; set git identity
│        └─ then create audio/ and transcriptions/ (+ .gitkeep)
│
├─ 2. manifest = Manifest(config.download.manifest_file)
│        └─ reads manifest.json into memory if it already exists
│
├─ 3. if args.dry_run:                  → print pending list and RETURN early
│        _maybe_print_dry_run(args, manifest)
│
├─ 4. src_stats = acquire_audio(config, manifest)
│        ├─ source.mode == "folder"   → FolderImporter(config, manifest).run()
│        └─ source.mode == "telegram" → async TelegramDownloader(...).run()
│        print(src_stats.summary())
│
├─ 5. tr_stats = Transcriber(config, manifest).run()
│        print(tr_stats.summary())
│
└─ 6. if not args.no_push:
         repo.commit_and_push(f"OctoScribe update {today}")
      else:
         print("Skipping push (--no-push).")
```

Each numbered step is wrapped so that a failure prints a clear `ERROR:` line to
stderr and exits non-zero; with `--verbose` the full traceback is logged too.
Importantly, **per-item failures inside steps 4 and 5 do not abort the run** —
they are recorded in the manifest as failures and the batch continues, so one
corrupt file never costs you the whole batch.

The next sections drill into steps 4, 5 and 1 (in that order, because that is the
order they run).

---

## Acquiring audio: the Telegram path

`acquire_audio()` dispatches on `config.source.mode`. For `telegram` it runs an
async coroutine that opens a `TelegramDownloader` as an async context manager:

```python
async with TelegramDownloader(config, manifest) as dl:
    return await dl.run()
```

### Construction (`TelegramDownloader.__init__`)

1. **`restore_session_from_env(session_dir)`** (from `src/telegram_client.py`)
   runs *first*. If the `TELEGRAM_SESSION_B64` environment variable is set, its
   base64 payload is decoded and written to `<session_dir>/octoscribe.session`.
   This is what lets CI authenticate without an interactive phone-code prompt.
   If the variable is unset or malformed, it is a silent no-op and login falls
   back to interactive.
2. A Telethon `TelegramClient` is constructed at
   `session_base_path(session_dir)` (i.e. `<session_dir>/octoscribe`, no
   extension — Telethon appends `.session`).
3. An `asyncio.Semaphore(config.download.workers)` caps concurrent downloads.

> **Design note:** the `TelegramClient` is *not* built inside
> `telegram_client.py`. That module holds only framework-agnostic helpers
> (session restore, path derivation, entity resolution) shared by both the
> downloader and the debug inspector. Keeping client construction in the
> consumer means tests can patch `TelegramClient` at the point of use.

### Connecting (`__aenter__` → `connect`)

`connect()` ensures the session directory exists and calls
`client.start(phone=…)`. On first ever run this triggers the interactive
Telegram login (a code is sent to your phone); thereafter the saved session is
reused.

### The scan-then-download run (`run`)

```
1. entity = resolve_group_entity(client, group)
      └─ purely-numeric group → get_entity(int);  else → get_entity(str)
2. Page through the whole group in batches of 100 (offset_id walks backwards),
   collecting every message for which is_audio(msg) is True.
3. If no audio found → return empty stats.
4. Ensure audio_dir exists; create a PeriodicSaver(manifest).
5. Build one _download_one(msg) coroutine per audio message and run them all
   with asyncio.gather (bounded by the semaphore).
6. manifest.save()  (final, authoritative flush)
```

`is_audio(msg)` (in `src/telegram.py`) recognises audio three ways: a
`DocumentAttributeAudio` attribute, a known audio file extension on a
`DocumentAttributeFilename`, or an audio MIME type (including `application/ogg`).

### Per-message download (`_download_one`) — runs under the semaphore

```
async with self._semaphore:
  1. RESUME check: if config.download.resume and manifest.is_downloaded(msg_id)
     and the recorded file still exists on disk → return "skipped".
  2. metadata = get_audio_metadata(msg)        # title, performer, duration, ext, date
  3. filename = build_filename(metadata)        # title → original stem → audio_<date>_<id>
  4. target  = unique_filepath(audio_dir, filename, msg_id)  # never overwrite
  5. client.download_media(msg, file=target)    # on exception → mark_failed, return "failed"
  6. DEDUP check: if config.download.deduplicate, sha256 the file and compare to
     every existing downloaded entry's hash. Match → delete file, return "duplicate".
  7. manifest.mark_downloaded(msg_id, {filename, title, performer, date,
     duration, duration_formatted, extension, hash, original_filename})
     return "downloaded".
```

The four possible return strings (`downloaded`, `skipped`, `duplicate`,
`failed`) are tallied into `DownloadStats`, whose `.summary()` is what
`cmd_run`/`cmd_download` print. The manifest key is the **Telegram message id**.

### `build_filename` priority

1. The audio **title** if present (sanitised);
2. else the **original filename's stem** (unless it is the useless default
   `record.ogg`);
3. else `audio_<date>_<msg_id>`.

`unique_filepath` then guarantees no two files collide: if the name is taken, it
appends the message id to the stem.

---

## Acquiring audio: the folder path

When `source.mode == "folder"`, `acquire_audio()` runs a `FolderImporter`
instead. It needs no Telegram credentials and imports no Telegram libraries —
that is the whole point of the folder source.

### The run (`FolderImporter.run`)

```
1. Validate config.source.folder: exists, is a directory (else raise).
2. files = _gather_files(folder, recursive)
      └─ rglob("*") if recursive else iterdir(); keep only AUDIO_EXTENSIONS; sorted.
3. Ensure audio_dir exists; create a PeriodicSaver(manifest).
4. For each file (SEQUENTIALLY): _import_one(path); tally the result.
5. manifest.save()  (final flush)
```

### Per-file import (`_import_one`)

```
1. sha256_hex = sha256_file(path)         # the content hash IS the manifest key
2. IN-RUN DEDUP: if deduplicate and hash already seen this run → "duplicate"
3. RESUME: if resume and manifest.is_downloaded(hash) and copied file exists → "skipped"
4. target = unique_filepath(audio_dir, "<sanitised stem><ext>", hash[:8])
5. shutil.copy2(path, target)             # COPY — the source file is never moved/modified
6. duration = _probe_duration(target)     # best-effort via mutagen; None on failure
7. manifest.mark_downloaded(hash, {filename, date (from mtime), duration, hash,
   original_filename, source: "folder", source_path: <original path>, …})
   return "imported".
```

### The key design difference from Telegram

The folder importer **keys the manifest by SHA-256 content hash**, not by a
message id. This single decision gives three properties for free:

* **Deduplication** — identical content maps to one entry, even across different
  filenames or folders.
* **Resume** — re-running over the same folder skips files already imported.
* **Idempotence** — the original files are *copied* (never moved), so your
  source folder is untouched and safe to re-scan forever.

Both sources converge on the same `Manifest.mark_downloaded(...)` call, so from
the transcriber's point of view a folder-imported file and a Telegram-downloaded
file are indistinguishable.

---

## The manifest: the single source of truth

`src/manifest.py` is small but central. `manifest.json` lives in the data repo
and is the *only* thing that remembers what has been done. Every other component
is stateless between runs.

### Shape

It is a JSON object keyed by string id (Telegram message id, or content hash for
folder imports). Each entry accumulates fields over its lifetime:

```jsonc
{
  "1234": {
    "telegram_msg_id": 1234,
    "downloaded": true,
    "filename": "Sunday Sermon.ogg",
    "hash": "9f86d0…",
    "title": "Sunday Sermon",
    "date": "2026-06-29",
    "duration": 2730,
    "transcription": {
      "status": "completed",         // or "failed"
      "output_file": "Sunday Sermon.txt",
      "model": "openai",
      "completed_at": "2026-06-29T06:03:11Z"
    }
  }
}
```

### State transitions

```
                mark_downloaded()              mark_transcribed()
   (new) ───────────────────────▶ downloaded ───────────────────────▶ transcribed
                                       │
                                       │ mark_failed("transcription", err)
                                       ▼
                                  failed (retried on next run)
```

The two query helpers drive the pipeline:

* **`pending_transcription()`** returns every entry that is `downloaded` but not
  yet `transcription.status == "completed"`. This is what the transcriber reads
  to decide its work list — and why a failed transcript is automatically retried
  next run.
* **`is_downloaded(id)`** powers the resume checks in both sources.

### Thread-safety and durability

Every method takes a `threading.Lock`, because the OpenAI transcription path runs
in a `ThreadPoolExecutor` and several threads call `mark_transcribed`/
`mark_failed` concurrently. `save()` serialises with `sort_keys=True` (so git
diffs are stable and reviewable) and writes through
`persistence.atomic_write_text` (so a crash mid-write never corrupts the file).
Using the class as a context manager saves automatically on exit.

---

## The transcription pipeline

Stage 3 is the part users care about most, so it has the most safety machinery.
The orchestrator is `Transcriber` (`src/transcribe/transcriber.py`); the actual
speech-to-text work is delegated to a *backend*.

### The backend interface

`src/transcribe/backends/base.py` defines a deliberately tiny contract:

```python
class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> str: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
```

One method plus an identifier. The orchestrator depends on nothing more, which
is exactly why a new backend can be added without touching the orchestrator.

### Backend selection (`Transcriber.create_backend`)

```python
if config.backend == "openai": return OpenAIBackend(config)
if config.backend == "local":  return LocalWhisperBackend(config)
raise ValueError(...)
```

A backend can also be **injected** into the `Transcriber` constructor — tests
pass a `MagicMock(spec=TranscriptionBackend)` and the orchestrator behaves
identically.

### The run (`Transcriber.run`)

```
1. pending = manifest.pending_transcription()   (empty → return immediately)
2. Create the backend (unless one was injected).
3. use_parallel = (backend.name == "openai" and transcribe.workers > 1)
4. _run_parallel(...)   OR   _run_sequential(...)
5. manifest.save()
```

The local backend is always run sequentially — a single GPU cannot meaningfully
parallelise, so threading it would only add contention. The OpenAI backend runs
in a `ThreadPoolExecutor(max_workers=transcribe.workers)` because each call is an
independent blocking HTTP request.

### Per-entry processing (`_process_entry`) — the heart of the stage

```
1. Resolve audio_path = audio_dir / entry.filename.
      └─ Missing file? → return None  (counted as "skipped", NOT a failure)
2. desired = _output_filename(audio_path, title)     # sanitised title, else stem
   output_path = unique_filepath(out_dir, desired, msg_id)   # COLLISION-SAFE
3. raw  = backend.transcribe(audio_path)             # the actual work
   text = normalize_text(raw)                         # whitespace-only cleanup
4. EMPTY-RESULT GUARD: if not text.strip() → raise RuntimeError
      └─ a blank transcript is treated as a FAILURE so it retries next run,
         rather than being written out as a misleading "completed" empty file.
5. atomic_write_text(output_path, text)
   manifest.mark_transcribed(msg_id, {output_file, model})
   return TranscriptionResult(success=True, …)

   On ANY exception:
   manifest.mark_failed(msg_id, "transcription", error)
   return TranscriptionResult(success=False, …)        # batch continues
```

Two guarantees are worth calling out because they exist specifically to **never
silently lose a transcript**:

* **Collision-safe output.** If two recordings derive the same output name (e.g.
  the same Telegram title), the second is disambiguated by message id via
  `unique_filepath` instead of overwriting the first.
* **Empty-result guard.** An empty transcript is almost always a backend hiccup,
  not a genuinely silent recording — so it is failed and retried, not recorded
  as done.

### The OpenAI backend (`openai_backend.py`)

The OpenAI client is created lazily in `__init__` (the `openai` package is
imported there), so the dependency only matters when this backend is selected.
`transcribe()` builds a single-attempt closure and hands it to a `RetryPolicy`:

```python
def _attempt() -> str:
    with open(audio_path, "rb") as f:          # opened fresh each attempt
        result = client.audio.transcriptions.create(
            file=f, model=cfg.model, language=cfg.language,
            prompt=VERBATIM_PROMPT, response_format="text")
    return str(result)
return self._retry.run(_attempt, label=audio_path.name)
```

The file is reopened on every attempt because a retried HTTP request needs a
fresh, rewound stream.

### Retry and error classification (`backends/retry.py`)

Resilience is its own unit, not buried inside the API call:

* **`ErrorClassifier`** inspects the lower-cased exception text. **Permanent**
  patterns (401/403, "unauthorized", "invalid api key", "invalid audio",
  "corrupt"…) are checked *first*; **retryable** patterns (rate limit, 429,
  timeouts, 5xx, connection/network/DNS errors…) second.
* **`RetryPolicy.run`** loops up to `attempts + 1` times:
  * a **permanent** error re-raises immediately (no point retrying);
  * a **retryable** error backs off and retries — delay is
    `base_delay * 2**n`, capped at `max_delay`, with ±15% jitter to avoid a
    thundering herd;
  * an **unknown** error (matching neither list) is treated as permanent and
    re-raised — we never hammer an endpoint over a failure we don't understand;
  * exhausting the budget raises a `RuntimeError` chained to the last error.

### The local Whisper backend (`local_whisper.py`)

Runs `large-v3` via Faster-Whisper / CTranslate2. Two things matter:

1. **CUDA setup before import.** `_setup_cuda()` scans site-packages for the
   NVIDIA cuDNN/cuBLAS and CTranslate2 shared libraries shipped as wheels and
   prepends them to `LD_LIBRARY_PATH` *before* `faster_whisper` is imported,
   because the import itself triggers native library loading. Any failure here
   is logged and non-fatal.
2. **Verbatim decoding is pinned.** `temperature=0` and
   `condition_on_previous_text=False` are mandatory and must not change — they
   stop the model "improving" or hallucinating beyond what was spoken.
   `_format_segments` joins segments into text, inserting a paragraph break
   wherever the gap between consecutive segments exceeds 2 seconds.

### Why verbatim, and where it is enforced

OctoScribe's promise is a *word-for-word* record. That promise lives in two
places: the `VERBATIM_PROMPT` string (`prompt.py`) sent to prompt-aware backends,
and `normalize_text` (`normalize.py`), which is allowed to touch **only
whitespace** (normalise line endings, strip trailing spaces, cap blank-line
runs) — never a spoken word.

---

## The data repository: git lifecycle

`src/repository.py` manages the separate git repository that stores audio,
transcripts and the manifest. All git operations shell out via `subprocess`
(no third-party git library), and each returns a `GitResult` dataclass that
captures returncode/stdout/stderr and exposes a `.success` property.

### `ensure_ready()` — called at the start of `run` and `status`

```
if <path>/.git exists:          pull latest  (resume an existing clone)
elif path exists but no .git:   raise DataRepoError  (refuse to clobber a non-repo dir)
elif data_repo.url is set:      git clone --branch <branch> <url> <path>
else:                           git init -b <branch> <path>; set git identity
finally:                        create audio/ and transcriptions/ (+ .gitkeep)
```

The "path exists but isn't a git repo" branch is a deliberate safety stop: rather
than risk initialising over a directory that already holds files, it tells you to
move or remove it.

### `commit_and_push(message)` — called at the end of `run` and by `sync`

```
1. _ensure_git_identity()        # set user.name/email if unset (needed in CI)
2. git add -A
3. git commit -m <message>
      └─ "nothing to commit" → return a SUCCESS no-op (don't push an empty commit)
4. if auto_push and origin exists → git push origin <branch>
```

### `pull()` and `status()`

`pull()` is a successful no-op when there is no remote configured (local-only
mode). `status()` returns a dict (`is_git_repo`, `has_remote`, `branch`,
`uncommitted_changes`, `ahead_count`, `path`) used by the `status` command.

### Why a separate repo at all?

Audio files are large binaries that do not belong in a source tree. Keeping them
in their own repo means the code repo stays small and fast, every pipeline run
produces an auditable commit, the remote doubles as off-site backup, and the data
can be cloned independently of the code.

---

## The other commands and what they trigger

| Command      | Handler          | What it actually does |
| ------------ | ---------------- | --------------------- |
| `run`        | `cmd_run`        | The full four-stage pipeline above. |
| `download`   | `cmd_download`   | Builds a `Manifest`, calls `acquire_audio` (Telegram or folder), prints stats. No transcription, no git. |
| `transcribe` | `cmd_transcribe` | Builds a `Manifest`, runs `Transcriber.run()` over everything `pending_transcription()` returns. Useful after switching backends. |
| `sync`       | `cmd_sync`       | `DataRepository.pull()` then `commit_and_push()`. `--pull-only`/`--push-only` skip one half. |
| `status`     | `cmd_status`     | `DataRepository.status()` + `Manifest.stats()`; prints source, repo/branch/remote, uncommitted flag, and download/transcribe/pending counts. Read-only. |
| `debug`      | `cmd_debug`      | Runs `DebugInspector`: connects to Telegram and dumps full metadata for the first N audio messages (`--scan-limit`). For diagnosing connectivity and finding a private group's numeric id. |
| `session`    | `cmd_session`    | `export` prints the base64-encoded `.session` file; `check` (default) reports whether one exists and its size/mtime. |
| `ci-export`  | `cmd_ci_export`  | Prints every secret + variable you need to configure CI. **Refuses to run inside a CI environment** (guards on `CI`, `GITHUB_ACTIONS`, etc.) so secrets never leak into logs. |

`debug` and `session` are diagnostics/setup helpers, not part of the pipeline;
they exist to make first-time setup and CI configuration painless.

---

## Command-line options and their outcomes

Global options (before the command) apply everywhere:

| Option         | Effect |
| -------------- | ------ |
| `--config PATH`| Use a specific INI file (else `$OCTOSCRIBE_CONFIG`, else defaults). |
| `--env PATH`   | Use a specific `.env` file (else `$OCTOSCRIBE_ENV`, else `./.env`). |
| `--data-repo PATH` | Override `data_repo.path`. In CI this points at the pre-cloned `./data`. |
| `--verbose`    | DEBUG logging (and full tracebacks on error) to stderr. |

Per-command options and the behaviour they change:

| Command      | Option            | Outcome |
| ------------ | ----------------- | ------- |
| `run`/`download` | `--source telegram\|folder` | Force the audio source for this run. |
| `run`/`download` | `--folder PATH`   | Import from this folder; implies `--source folder`. |
| `run`/`download` | `--group GROUP`   | Override the Telegram group (`@username` or numeric id). |
| `run`/`transcribe` | `--backend openai\|local` | Override the transcription backend. |
| `run`/`transcribe` | `--dry-run`     | Print the files *pending transcription* and exit — no download, no API calls, no commits. |
| `run`        | `--no-push`       | Run everything but skip the final `git push` (still commits locally). |
| `sync`       | `--pull-only`     | Pull from remote, don't commit/push. |
| `sync`       | `--push-only`     | Commit/push, don't pull first. (Mutually exclusive with `--pull-only`.) |
| `debug`      | `--scan-limit N`  | Inspect N audio messages (default 3). |

**Example — the same audio through two completely different paths:**

```bash
# Telegram source + cloud backend (the default production path)
python octoscribe.py run

# Local folder source + local GPU backend, preview only, no push
python octoscribe.py run --folder ~/sermons/incoming --backend local --dry-run
```

The first resolves to `source.mode=telegram` / `transcribe.backend=openai`,
constructs a `TelegramDownloader` and an `OpenAIBackend`, and pushes at the end.
The second sets `source.mode=folder` (because `--folder` was given) and
`transcribe.backend=local`, so it would construct a `FolderImporter` and a
`LocalWhisperBackend` — except `--dry-run` short-circuits after printing the
pending list, so nothing is downloaded, transcribed, or committed.

---

## Durability: atomic writes and periodic saves

`src/persistence.py` holds two small but important guarantees, shared by every
component that writes to disk:

* **Atomic writes.** `atomic_write_bytes`/`atomic_write_text` write to a
  temporary sibling file (`<name>.tmp`) and then `os.replace` it into position.
  `os.replace` is atomic on every supported platform, so a reader never sees a
  half-written file — it sees either the old contents or the complete new ones.
  On failure the temp file is cleaned up and the original exception re-raised.
  Both the manifest and every transcript are written this way.
* **Periodic saving.** `PeriodicSaver` wraps any object with a `save()` method
  and flushes it every N successful items (`tick()` returns `True` on the
  boundary). The downloader, importer and transcriber all use it so that an
  interrupted long run only loses the last few items of progress, not the whole
  batch. A final authoritative `save()` always runs at the end.

These two helpers exist because the same logic was previously copy-pasted across
three modules; centralising it means every writer shares identical crash-safety.

---

## Concurrency model

OctoScribe uses two different concurrency mechanisms, each chosen to match its
workload:

| Stage | Mechanism | Why |
| ----- | --------- | --- |
| Telegram download | `asyncio` + `Semaphore(download.workers)` | Telethon is async; downloads are I/O-bound. The semaphore bounds concurrent transfers to avoid Telegram rate limits (`FloodWaitError`). |
| Folder import | Sequential | Local file copies are fast and disk-bound; parallelism buys nothing. |
| OpenAI transcription | `ThreadPoolExecutor(transcribe.workers)` | Each request is an independent blocking HTTP call; threads overlap the network latency. |
| Local transcription | Sequential | One GPU; threading would only add contention. |

Because the OpenAI path is multi-threaded, the `Manifest` is fully lock-guarded —
that is the one place where shared mutable state is touched from multiple threads
at once.

---

## Running under GitHub Actions

OctoScribe ships as a **composite GitHub Action** (`action.yml`) plus a reference
**scheduled workflow** (`.github/workflows/pipeline.yml`). The README covers how
to *configure* the secrets and variables; this section explains what actually
*executes*.

### The workflow's three steps (and why the order is fixed)

```
1. octoleo/git-user@v2        Installs the SSH key, GPG key and git identity.
        │                     Without this, the clone and push below cannot authenticate.
        ▼
2. git clone … ./data         Clones the DATA repository to ./data using that SSH key.
        │                     OctoScribe expects the data repo to already exist on disk.
        ▼
3. octoleo/octoscribe@v1      Runs the pipeline against ./data and pushes results
                              using the identity from step 1.
```

Reordering breaks it: no SSH key → clone fails; no identity → commit fails; no
pre-cloned data repo → OctoScribe has nowhere to write.

### What the action does internally

`action.yml` is a composite action that:

1. Sets up Python 3.11.
2. `pip install -r requirements.txt` (from the action's own checkout).
3. Builds and runs a command line, roughly:
   ```
   python3 octoscribe.py [--config PATH] [--verbose] \
       --data-repo <data_repo_path> <command> --group <telegram_group> \
       [--backend <transcribe_backend>] [--no-push]
   ```
   with `TELEGRAM_API_ID/HASH/PHONE`, `TELEGRAM_SESSION_B64` and
   `OPENAI_API_KEY` passed through as environment variables.

Note the `--data-repo ./data` override: in CI the data repo is cloned by the
workflow (step 2), so OctoScribe is told to use that path rather than its default
`~/.octoscribe/data`. Inside `ensure_ready()`, the `<path>/.git exists` branch
fires and it simply pulls — it does not re-clone.

### How non-interactive auth works in CI

The interactive Telegram phone-code login is impossible in CI. The escape hatch
is the **`TELEGRAM_SESSION_B64`** secret: you authenticate once locally, run
`octoscribe.py session export` to get the base64 blob, and store it as a secret.
On every CI run, `restore_session_from_env()` (called in the downloader/inspector
constructor, before the client connects) decodes that blob back into a
`.session` file, so Telethon starts already authenticated.

### The CI workflow for the project's own tests

`.github/workflows/ci.yml` is separate: it runs the **unit test suite** on every
push/PR to `main`/`v1` across Python 3.11 and 3.12, with coverage. See
[`docs/TESTING.md`](TESTING.md).

---

## Why the code is shaped this way

A few decisions explain most of the structure. They are listed here as
*trade-offs*, not rules:

* **One pluggable seam per axis of change.** The two things most likely to change
  are *where audio comes from* and *how it gets transcribed*. Each is isolated
  behind a small seam (`acquire_audio` dispatch; the `TranscriptionBackend`
  interface) so adding a new source or backend touches one place, not the whole
  pipeline. The orchestrators (`cmd_run`, `Transcriber`) depend on the
  abstraction, never on a concrete OpenAI/Telegram type.
* **The manifest is the only state.** Making one file the single source of truth
  is what makes every command idempotent and re-runnable. There is no hidden
  state in memory that a crash could lose.
* **Crash-safety is shared, not per-module.** Atomic writes and periodic saves
  live in one module so every writer has identical guarantees, and we have one
  place to reason about durability.
* **Resilience is a testable unit.** The retry/backoff/classification logic is
  its own class (`RetryPolicy`) rather than being entangled in the API call, so
  it can be unit-tested deterministically without any network.
* **Secrets are environment-only.** Credentials never come from the INI file,
  which keeps the committed configuration safe and the secret surface small.
* **Lazy, local imports.** Heavy/optional dependencies (Telethon, OpenAI,
  Faster-Whisper) are imported inside the functions that use them, so unrelated
  commands stay fast and a missing optional dependency only fails the feature
  that needs it.

These add up to the broader patterns you'll recognise — a Strategy seam for
sources and backends, dependency inversion in the orchestrators, and a strict
one-way dependency graph — but the patterns are a *consequence* of these
decisions, not the goal.

---

## Extending OctoScribe

### Add a new transcription backend

1. Create `src/transcribe/backends/<name>.py` with a class that subclasses
   `TranscriptionBackend` and implements `transcribe(self, audio_path) -> str`
   and the `name` property.
2. Register it in `Transcriber.create_backend` (one new `if` branch).
3. Add `<name>` to the `valid_backends` set in `loader._validate`.
4. Add a unit test that injects a mock and asserts the orchestrator path, plus
   one that exercises the new backend's `transcribe` with its client mocked.

Nothing in `Transcriber.run`, the CLI, or the manifest changes.

### Add a new audio source

1. Create a class with a `run()` method that returns a stats object exposing a
   `.summary()` and, for each item, calls `manifest.mark_downloaded(key, meta)`.
2. Add a branch to `acquire_audio()` in `octoscribe.py` dispatching on a new
   `source.mode`.
3. Extend `SourceConfig`/`_build_source`/`_validate` for any new settings.

The transcriber and repository stages need no changes — they only ever read the
manifest.

### Reuse the shared helpers

When writing either of the above, lean on the existing leaves: `audio.py`
(`sha256_file`, `sanitize_filename`, `unique_filepath`, `format_duration`),
`persistence.py` (`atomic_write_text`, `PeriodicSaver`), and `telegram_client.py`
(session/entity helpers). They exist precisely so new sources and backends don't
re-implement durability, hashing, or session handling.
