# OctoScribe

OctoScribe is a Python 3.11+ CLI pipeline that monitors a Telegram group, downloads audio files (sermons, devotions, or any voice recordings) that it has not seen before, transcribes them verbatim using AI, and stores both the audio and the transcriptions in a dedicated, version-controlled data repository. Every run ends with an automatic git commit and push so your archive is always up to date.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Commands Reference](#commands-reference)
5. [Data Repository](#data-repository)
6. [Transcription Backends](#transcription-backends)
7. [Testing](#testing)
8. [Project Structure](#project-structure)

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Earlier versions are not supported. |
| git 2.x | Required for data repository management. |
| Telegram account | You need a real phone number. Bots cannot join groups as members. |
| OpenAI API key | Required for the `openai` transcription backend. |
| NVIDIA GPU (optional) | Required for the `local` Faster-Whisper backend at reasonable speed. |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/octoleo/octoscribe.git
cd octoscribe
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain Telegram API credentials

1. Visit [https://my.telegram.org](https://my.telegram.org) and log in with your phone number.
2. Click **API development tools**.
3. Create a new application. The name and platform do not affect functionality.
4. Note your **App api_id** (an integer) and **App api_hash** (a hex string).

### 5. Configure secrets

```bash
cp conf/.env.example .env
```

Open `.env` and fill in at minimum:

```dotenv
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_PHONE=+27821234567
OPENAI_API_KEY=sk-proj-...          # omit if using the local backend
DATA_REPO_URL=git@github.com:you/sermons-data.git   # optional but recommended
```

### 6. Configure runtime settings

```bash
cp conf/octoscribe.ini.example conf/octoscribe.ini
```

Open `conf/octoscribe.ini` and set at minimum:

```ini
[telegram]
group = @your_church_group
```

### 7. Authenticate with Telegram

The first run opens an interactive prompt asking for the verification code sent to your phone (standard Telegram login). The session is saved in `.session/` and reused on all subsequent runs.

```bash
python octoscribe.py status
```

### 8. Run the full pipeline

```bash
python octoscribe.py run
```

This will download any new audio files from the configured group, transcribe each one, write the results to the data repository, and push to the remote if `auto_push = true`.

---

## Configuration

OctoScribe separates secrets from settings deliberately.

### `.env` — secrets only

The `.env` file (copied from `conf/.env.example`) holds every credential:

- Telegram API ID and hash
- Telegram phone number
- OpenAI API key
- Anthropic API key (future use)
- Data repository remote URL

This file is listed in `.gitignore` and must never be committed. See `conf/.env.example` for full documentation of every variable.

You can also export variables in your shell or pass them inline:

```bash
TELEGRAM_API_ID=12345678 TELEGRAM_API_HASH=abc123 python octoscribe.py run
```

### `conf/octoscribe.ini` — non-secret settings

The INI file (copied from `conf/octoscribe.ini.example`) holds all non-sensitive runtime settings: which group to monitor, worker counts, retry behaviour, transcription model, data repository path, and directory layout. This file is also in `.gitignore` so that accidental system paths or group names are not leaked.

Override either file's path at runtime:

```bash
OCTOSCRIBE_CONFIG=/path/to/other.ini OCTOSCRIBE_ENV=/path/to/other.env python octoscribe.py run
```

### Settings precedence

Environment variables always win over the INI file. Explicitly set environment variables override `.env` file values.

---

## Commands Reference

All commands are subcommands of the single entry point `octoscribe.py`.

```
python octoscribe.py <command> [options]
```

### `run` — Full pipeline

Download new audio, transcribe it, commit and push the data repository.

```bash
python octoscribe.py run
```

This is the command you will schedule in cron or run manually after a service.

### `download` — Download only

Fetch new audio files into the data repository without transcribing.

```bash
python octoscribe.py download
```

Useful for bulk downloading when you want to transcribe later, or to test connectivity.

### `transcribe` — Transcribe only

Transcribe audio files that are in the data repository but do not yet have a transcription.

```bash
python octoscribe.py transcribe
```

Useful after switching backends (e.g. re-transcribing existing audio with a better model).

### `sync` — Commit and push data repository

Commit any uncommitted changes in the data repository and push to the remote.

```bash
python octoscribe.py sync
```

### `status` — Pipeline status

Print a summary of the data repository state: total files, transcribed, pending, last run timestamp, and remote sync status.

```bash
python octoscribe.py status
```

Also triggers Telegram authentication if no session file exists yet, making it a safe first command to run.

### `debug` — Diagnostic information

Print detailed diagnostics: resolved config values (with secrets redacted), Telegram connection details, group membership, recent messages, and data repository git log.

```bash
python octoscribe.py debug
```

Use this to find a group's numeric chat ID or to troubleshoot connection issues.

### `session` — Manage Telegram session files

#### `session check` — Inspect local session

```bash
python octoscribe.py session check
```

Shows whether a session file exists, its size, and last-modified time.

#### `session export` — Export session for CI/CD

```bash
python octoscribe.py session export
```

Prints the base64-encoded session file to stdout. Use the output to populate the `TELEGRAM_SESSION_B64` GitHub Secret so GitHub Actions can authenticate without interactive prompts.

---

## CI/CD: Setting up TELEGRAM_SESSION_B64

GitHub Actions cannot prompt for a Telegram verification code. The solution is to authenticate once on your local machine, export the session file, and store it as a repository secret.

**Step 1 — Authenticate locally (once)**

```bash
python octoscribe.py download
```

Telethon will prompt for the verification code sent to your phone. Complete the login. The session is saved to the path shown in `session check`.

**Step 2 — Export the session**

```bash
python octoscribe.py session export
```

Copy the entire output string (it will be a long base64 line with no spaces or newlines).

**Step 3 — Store as a GitHub Secret**

In your repository go to **Settings → Secrets and variables → Actions → New repository secret**:

- Name: `TELEGRAM_SESSION_B64`
- Value: paste the output from Step 2

**Step 4 — Done**

On every workflow run `_restore_session_from_env()` decodes the secret and writes the session file before any Telegram requests are made. No interactive authentication is needed.

> **Important:** If you ever regenerate your Telegram session (e.g. after logging out or revoking API access), repeat Steps 1–3 to update the secret.

---

## Data Repository

### What it is

The data repository is a plain git repository that holds:

```
~/.octoscribe/data/       (default location)
  audio/                  downloaded .ogg / .mp3 / .m4a files
  transcriptions/         one .txt file per audio file (verbatim transcript)
  manifest.json           index mapping Telegram message IDs to files and metadata
```

It is intentionally separate from the OctoScribe source code repository. Audio files are large binary assets that do not belong in a source tree, and keeping them separate means the code repository stays small and fast.

### Setting it up

**Option A — Use an existing remote repository**

Create an empty repository on GitHub, GitLab, or any git host, then set `DATA_REPO_URL` in your `.env`. OctoScribe will initialise the local repository at the configured `path` (default `~/.octoscribe/data`) on first run and wire up the remote automatically.

**Option B — Local only**

Leave `DATA_REPO_URL` unset and set `auto_push = false` in `conf/octoscribe.ini`. The data repository will still be a git repo (for local history), it just will not push anywhere.

### Why git for data?

- Every run creates an auditable commit showing exactly what was added.
- Audio and transcriptions can be recovered from any point in history.
- The remote acts as an off-site backup with zero extra infrastructure.
- Team members can clone the data repo independently of the code repo.

---

## Transcription Backends

OctoScribe transcribes verbatim — no summarisation, no paraphrasing, no correction of perceived errors. The transcript is a faithful record of exactly what was said.

### OpenAI (`backend = openai`)

Uses the `gpt-4o-transcribe` model via the OpenAI API.

**Pros:** Excellent accuracy, handles accents and domain-specific vocabulary well, no local hardware required, parallelises easily.

**Cons:** Costs money (billed per minute of audio), requires an internet connection and an OpenAI account, audio is sent to OpenAI's servers.

**Required:** `OPENAI_API_KEY` in `.env`.

**Recommended for:** Most users. The cost per sermon is typically a few cents.

### Local Faster-Whisper (`backend = local`)

Runs the `large-v3` Whisper model locally using the CTranslate2 runtime.

**Pros:** Free after hardware cost, audio never leaves your machine, works offline.

**Cons:** Requires a CUDA-capable NVIDIA GPU for practical throughput (CPU inference is extremely slow for `large-v3`). Requires CUDA drivers and libraries to be installed correctly.

**Recommended for:** Users with a suitable GPU who have privacy concerns or process very large volumes.

### Choosing a backend

| | OpenAI | Local |
|---|---|---|
| Accuracy | Excellent | Excellent (large-v3) |
| Cost | ~$0.006/min | Hardware only |
| Speed | Fast (API, parallelised) | Fast with GPU, slow on CPU |
| Privacy | Audio sent to OpenAI | Audio stays local |
| Setup complexity | Low | Medium–High |

---

## Testing

OctoScribe uses [pytest](https://pytest.org) with asyncio and mock support.

### Run the full test suite

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Run a specific test file or test

```bash
pytest tests/test_downloader.py
pytest tests/test_transcriber.py::test_openai_retry_on_rate_limit
```

### Run only fast tests (skip slow integration tests)

```bash
pytest -m "not slow"
```

---

## Project Structure

```
octoscribe/
|
|-- octoscribe.py              Entry point. Parses subcommands and wires the pipeline.
|
|-- conf/
|   |-- .env.example           Template for secrets (.env). Copy to project root.
|   |-- octoscribe.ini.example Template for runtime config. Copy to conf/octoscribe.ini.
|
|-- src/
|   |-- config.py              Loads and validates .env + INI; exposes a typed Config object.
|   |-- telegram_client.py     Telethon wrapper: connects, lists messages, downloads audio.
|   |-- downloader.py          Orchestrates parallel audio downloads; checks manifest.
|   |-- transcriber.py         Abstract base + OpenAI and Faster-Whisper implementations.
|   |-- data_repo.py           Manages the data git repository: init, commit, push.
|   |-- manifest.py            Reads and writes manifest.json; tracks processing state.
|   |-- pipeline.py            Composes downloader + transcriber + data_repo into a run.
|   |-- cli.py                 argparse definitions for all subcommands.
|
|-- tests/
|   |-- conftest.py            Shared fixtures: mock config, mock Telegram client, tmp repos.
|   |-- test_config.py         Config loading, env override, missing-required-var errors.
|   |-- test_downloader.py     Download logic, deduplication, resume, parallel workers.
|   |-- test_transcriber.py    OpenAI and local backends, retry logic, error handling.
|   |-- test_manifest.py       Manifest read/write, state transitions, JSON integrity.
|   |-- test_data_repo.py      Git init, commit, push, branch management.
|   |-- test_pipeline.py       End-to-end pipeline integration tests with mocked I/O.
|
|-- requirements.txt           All Python dependencies with minimum version pins.
|-- .gitignore                 Excludes secrets, session files, data, and build artefacts.
|-- LICENSE                    The LICENSE of this project.
|-- README.md                  This file.
```

---

## Security Notes

- **Never commit `.env`** — it is in `.gitignore`, but be careful with `git add -A`.
- **Never commit `conf/octoscribe.ini`** — it is also in `.gitignore`.
- **Session files** in `.session/` are equivalent to your Telegram password. Back them up securely and never share them.
- **The data repository** may contain sensitive audio recordings. Consider whether the remote should be private.
- If your `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` are ever compromised, revoke them immediately at [https://my.telegram.org](https://my.telegram.org) under **API development tools**.

## License

```text
Copyright (C) 2021-2026
Llewellyn van der Merwe

Licensed under the GNU General Public License v2 (GPLv2)
See LICENSE for details.
```
