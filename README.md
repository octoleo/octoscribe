# OctoScribe

OctoScribe is a Python 3.11+ CLI pipeline that collects audio files (sermons, devotions, or any voice recordings) that it has not seen before, transcribes them verbatim using AI, and stores both the audio and the transcriptions in a dedicated, version-controlled data repository. Every run ends with an automatic git commit and push so your archive is always up to date.

Audio can come from two sources: a **Telegram group** (the default) or a **local folder** of files you already have on disk. In folder mode no Telegram credentials are required — see [Audio sources](#audio-sources).

This README is a **getting-started guide** — everything you need to install, configure and run OctoScribe. If you want to understand *how it works internally* or *contribute to the code*, see the [Documentation](#documentation) section below.

---

## Table of Contents

1. [Documentation](#documentation)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Commands Reference](#commands-reference)
6. [Data Repository](#data-repository)
7. [Transcription Backends](#transcription-backends)
8. [GitHub Actions Workflow](#github-actions-workflow)
9. [Testing](#testing)
10. [Project Structure](#project-structure)

---

## Documentation

| Document | What it is for |
| --- | --- |
| **README** (this file) | How to **install, configure and run** OctoScribe. Start here. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How OctoScribe **works under the hood** — the full execution path from command to commit, which classes and functions run at each stage, the different ways it can run, and the reasoning behind the design. Read this before changing the code. |
| [`docs/TESTING.md`](docs/TESTING.md) | How the **test suite** is structured and run, and how to add to it. |

New documentation belongs in the [`docs/`](docs/) folder; link to it from this table so it stays easy to find.

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

### Audio sources

OctoScribe can acquire audio from one of two sources, selected by the
`[source] mode` setting in `conf/octoscribe.ini` (or `--source` on the command
line):

| Mode | Where audio comes from | Telegram credentials |
| --- | --- | --- |
| `telegram` (default) | New audio messages in a Telegram group | Required |
| `folder` | Audio files in a local folder | Not required |

**Folder mode** is the simplest way to process sermons you already have on
disk. Configure it in the INI file:

```ini
[source]
mode = folder
folder = ~/sermons/incoming
recursive = true
```

…or entirely from the command line (passing `--folder` implies
`--source folder`):

```bash
python octoscribe.py download --folder ~/sermons/incoming
python octoscribe.py run --folder ~/sermons/incoming --backend local
```

In folder mode the importer copies every recognised audio file (`.mp3`, `.wav`,
`.flac`, `.m4a`, `.aac`, `.ogg`, `.oga`, `.opus`) into the data repository's
audio directory and queues it for transcription. Files are deduplicated by
SHA-256 content hash, so re-running over the same folder will not import the
same recording twice. The original files in your source folder are never
modified or moved. Because no Telegram connection is involved, none of the
`TELEGRAM_*` variables are needed — only an `OPENAI_API_KEY` (when
`transcribe.backend = openai`).

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

Download (or import) new audio, transcribe it, then commit and push the data repository.

```bash
python octoscribe.py run
```

This is the command you will schedule in cron or run manually after a service.

Useful options:

| Option | Effect |
| --- | --- |
| `--source telegram\|folder` | Force the audio source for this run. |
| `--folder PATH` | Import from a local folder (implies `--source folder`). |
| `--group GROUP` | Override the Telegram group (`@username` or numeric chat ID). |
| `--backend openai\|local` | Override the transcription backend. |
| `--dry-run` | List the files pending transcription and exit — no downloads, no API calls, no commits. |
| `--no-push` | Run everything but skip the final `git push` (changes are still committed locally). |

### `download` — Acquire audio only

Fetch new audio files into the data repository without transcribing. The source
depends on `[source] mode` — a Telegram group by default, or a local folder:

```bash
# From the configured Telegram group
python octoscribe.py download

# From a local folder (implies --source folder)
python octoscribe.py download --folder ~/sermons/incoming
```

Useful for bulk acquisition when you want to transcribe later, or to test
connectivity.

### `transcribe` — Transcribe only

Transcribe audio files that are in the data repository but do not yet have a transcription.

```bash
python octoscribe.py transcribe

# Preview what would be transcribed, without calling any backend
python octoscribe.py transcribe --dry-run

# Re-transcribe with a different backend
python octoscribe.py transcribe --backend local
```

Useful after switching backends (e.g. re-transcribing existing audio with a better model).

### `sync` — Commit and push data repository

Pull the latest data repository state, then commit any changes and push to the remote.

```bash
python octoscribe.py sync

# Pull only (don't commit or push)
python octoscribe.py sync --pull-only

# Commit and push only (don't pull first)
python octoscribe.py sync --push-only
```

### `status` — Pipeline status

Print a read-only summary: the active audio source, the data repository path/branch/remote, whether there are uncommitted changes, and manifest counts (total, downloaded, transcribed, failed, pending transcription).

```bash
python octoscribe.py status
```

### `debug` — Telegram diagnostics

Connect to Telegram and print full metadata for the first few audio messages in the configured group (message id, MIME type, attributes, and the values OctoScribe extracts from them).

```bash
python octoscribe.py debug

# Inspect more messages (default is 3)
python octoscribe.py debug --scan-limit 10
```

Use this to find a private group's numeric chat ID or to troubleshoot why a message is or isn't detected as audio.

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

### `ci-export` — Print all CI/CD secrets and variables

```bash
python octoscribe.py ci-export
```

Prints every secret and variable you need to configure a CI/CD pipeline (including the base64 session), formatted for copying into your repository's Secrets and Variables. **Local use only** — it refuses to run inside a CI environment so secrets never end up in build logs.

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

On every workflow run, OctoScribe decodes this secret back into a `.session` file before any Telegram requests are made, so Telethon starts already authenticated. (See [the architecture doc](docs/ARCHITECTURE.md#how-non-interactive-auth-works-in-ci) for exactly when and where this happens.) No interactive authentication is needed.

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

## GitHub Actions Workflow

OctoScribe ships with a composite GitHub Action (`action.yml` at the repository root) that you can drop into any workflow as `octoleo/octoscribe@v1`. The action installs Python, installs OctoScribe's dependencies, and runs the pipeline against a pre-cloned data repository.

A complete reference workflow lives at [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml) — copy it into your own repository, replace the secrets and variables, and you have a fully scheduled archive pipeline.

### Required companion action: `octoleo/git-user@v2`

The OctoScribe pipeline writes commits and pushes them to the data repository. Before `octoleo/octoscribe@v1` runs, the workflow **must** configure a git identity, an SSH key (so the data-repo clone and push succeed over `git@github.com`), and optionally a GPG key for signed commits. That configuration is provided by [`octoleo/git-user@v2`](https://github.com/octoleo/git-user).

`octoleo/git-user@v2` is what makes `git clone git@github.com:...` and `git push` work non-interactively inside the runner — without it, OctoScribe will fail to clone the data repository and will not be able to push the new audio and transcriptions back.

### Required secrets and variables

Configure these in **Settings → Secrets and variables → Actions** in the repository that hosts your workflow.

**Secrets** (under *Repository secrets*):

| Secret | Used by | Description |
|---|---|---|
| `GPG_KEY` | `octoleo/git-user@v2` | ASCII-armoured private GPG key for signed commits. |
| `GPG_USER` | `octoleo/git-user@v2` | Email address associated with the GPG key. |
| `SSH_KEY` | `octoleo/git-user@v2` | Private SSH key authorised to push to the data repository. |
| `SSH_PUB` | `octoleo/git-user@v2` | Matching public SSH key. |
| `GIT_USER` | `octoleo/git-user@v2` | Display name for git commits (e.g. `OctoScribe Bot`). |
| `GIT_EMAIL` | `octoleo/git-user@v2` | Email address for git commits. |
| `TELEGRAM_API_ID` | `octoleo/octoscribe@v1` | Telegram API ID from [my.telegram.org](https://my.telegram.org). |
| `TELEGRAM_API_HASH` | `octoleo/octoscribe@v1` | Telegram API hash from [my.telegram.org](https://my.telegram.org). |
| `TELEGRAM_PHONE` | `octoleo/octoscribe@v1` | Phone number for the Telegram account, with country code. |
| `TELEGRAM_SESSION_B64` | `octoleo/octoscribe@v1` | Base64-encoded session file produced by `python octoscribe.py session export`. See [CI/CD: Setting up TELEGRAM_SESSION_B64](#cicd-setting-up-telegram_session_b64). |
| `OPENAI_API_KEY` | `octoleo/octoscribe@v1` | OpenAI API key. Required only when `transcribe_backend` is `openai`. |

**Variables** (under *Repository variables*):

| Variable | Description |
|---|---|
| `DATA_REPO_ORG` | GitHub org/user that owns the data repository (e.g. `your-org`). |
| `DATA_REPO_NAME` | Name of the data repository (e.g. `sermons-data`). |
| `TELEGRAM_GROUP` | Group to archive — `@username` or numeric chat ID. |
| `TRANSCRIBE_BACKEND` | Optional — `openai` (default) or `local`. |

### Minimal example workflow

```yaml
name: OctoScribe Pipeline

on:
  schedule:
    - cron: '0 6 * * 1'   # Mondays at 06:00 UTC
  workflow_dispatch:

jobs:
  build:
    name: Run Pipeline
    runs-on: ubuntu-latest

    steps:
      - name: Setup GitHub User Details
        uses: octoleo/git-user@v2
        with:
          gpg-key:   ${{ secrets.GPG_KEY }}
          gpg-user:  ${{ secrets.GPG_USER }}
          ssh-key:   ${{ secrets.SSH_KEY }}
          ssh-pub:   ${{ secrets.SSH_PUB }}
          git-user:  ${{ secrets.GIT_USER }}
          git-email: ${{ secrets.GIT_EMAIL }}

      - name: Clone Data Repository
        run: |
          /bin/git clone git@github.com:${{ vars.DATA_REPO_ORG }}/${{ vars.DATA_REPO_NAME }}.git ./data

      - name: Run OctoScribe
        uses: octoleo/octoscribe@v1
        with:
          telegram_api_id:      ${{ secrets.TELEGRAM_API_ID }}
          telegram_api_hash:    ${{ secrets.TELEGRAM_API_HASH }}
          telegram_phone:       ${{ secrets.TELEGRAM_PHONE }}
          telegram_session_b64: ${{ secrets.TELEGRAM_SESSION_B64 }}
          telegram_group:       ${{ vars.TELEGRAM_GROUP }}
          openai_api_key:       ${{ secrets.OPENAI_API_KEY }}
          data_repo_path:       ./data
          transcribe_backend:   ${{ vars.TRANSCRIBE_BACKEND || 'openai' }}
          command:              run
```

### Action inputs

`octoleo/octoscribe@v1` accepts the following inputs (full schema in [`action.yml`](action.yml)):

| Input | Required | Default | Description |
|---|---|---|---|
| `telegram_api_id` | yes | — | Telegram API ID. |
| `telegram_api_hash` | yes | — | Telegram API hash. |
| `telegram_phone` | yes | — | Telegram phone number with country code. |
| `telegram_session_b64` | yes | — | Base64-encoded `.session` file for non-interactive auth. |
| `telegram_group` | yes | — | `@username` or numeric chat ID of the group to archive. |
| `openai_api_key` | no | `''` | OpenAI key. Required when `transcribe_backend=openai`. |
| `data_repo_path` | no | `./data` | Local path to the pre-cloned data repository. |
| `transcribe_backend` | no | `openai` | `openai` or `local`. |
| `command` | no | `run` | One of `run`, `download`, `transcribe`, `sync`, `status`. |
| `no_push` | no | `false` | `true` to skip the final `git push` (dry-run). |
| `config_path` | no | `''` | Path to a custom `octoscribe.ini`. |
| `verbose` | no | `false` | `true` enables debug logging. |

### Pipeline order matters

The three steps run in a specific order for a reason:

1. **`octoleo/git-user@v2`** — installs the SSH key, GPG key, and `git config` so subsequent git operations succeed.
2. **Clone data repository** — uses the SSH key from step 1 to clone the data repo into `./data`. OctoScribe expects the data repo to already exist on disk.
3. **`octoleo/octoscribe@v1`** — runs the pipeline against the cloned data repo and uses the git identity from step 1 to commit and push results.

Skipping or reordering any of these will cause the pipeline to fail (no SSH key → clone fails; no git identity → commit fails; no cloned data repo → OctoScribe has nowhere to write).

---

## Testing

OctoScribe uses [pytest](https://pytest.org) with asyncio and mock support. The whole suite is hermetic — it mocks Telethon, OpenAI and Faster-Whisper, so it needs no network, credentials or GPU and runs in well under a second.

```bash
# Run the full suite
pytest

# With coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html

# A single file, or a single test
pytest tests/test_transcribe.py
pytest tests/test_transcribe.py::test_openai_backend_retries_on_rate_limit
```

For the full picture — what each test file covers, the shared fixtures, and how each external service is mocked — see **[`docs/TESTING.md`](docs/TESTING.md)**.

---

## Project Structure

At a glance:

```
octoscribe/
|-- octoscribe.py     CLI entry point: parses subcommands and wires the pipeline.
|-- conf/             .env.example and octoscribe.ini.example templates.
|-- src/              All the implementation: config, audio sources, transcription,
|                     the manifest, and the data-repository git wrapper.
|-- tests/            The pytest suite (everything external is mocked).
|-- docs/             ARCHITECTURE.md and TESTING.md.
|-- action.yml        Composite GitHub Action wrapper.
|-- requirements.txt  Python dependencies with minimum version pins.
```

For a full, annotated module map — every file in `src/`, what it is responsible for, and how the pieces depend on each other — see the [**Module map** in `docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#module-map--what-lives-where).

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
