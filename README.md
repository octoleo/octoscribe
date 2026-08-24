# OctoScribe

OctoScribe is a fidelity-first audio-to-text pipeline for long recordings such
as sermons. It imports audio from Telegram or a local folder, transcribes the
same deterministic chunks with one to three speech-recognition providers, and
preserves the source, provider candidates, comparison results, and published
text as reviewable evidence.

The default provider is OpenAI `gpt-transcribe`. Add xAI/Grok STT for an
independent cross-check and, optionally, a hosted Meta Omnilingual ASR endpoint
as a third arbiter. Local Faster-Whisper remains available as an optional
provider but is not installed or enabled by default.

> Machine transcription cannot prove word-for-word identity with the audio.
> OctoScribe therefore never labels machine output as human verified. It keeps
> the primary provider's words, flags unresolved differences, and sends those
> results to `transcriptions/needs-review/`.

## Fidelity model

OctoScribe deliberately avoids generative cleanup, proofreading, paraphrasing,
or majority-vote rewriting.

| Enabled providers | Automatic behaviour | Final state when successful |
| --- | --- | --- |
| One | Transcribe once with that provider. | `machine_transcribed` |
| Two | Transcribe independently and compare. If they disagree, retry both once. | `cross_checked` only if their normalized word sequences agree; otherwise `needs_review` |
| Three | Use the first two as above. If the checker is unavailable, the third becomes its replacement; after a normal unresolved retry, the third arbitrates once. | `cross_checked` only when an independent provider agrees with the primary; otherwise `needs_review` |

The first configured provider is canonical unless `primary_provider` is set.
If a discrepancy retry produces a new successful primary result, that latest
primary text is canonical. A failed secondary may be replaced by the third
provider, but failure to obtain the initial primary candidate always fails the
transcription. If a later primary retry fails, its earlier successful candidate
remains canonical and may still be verified by an independent peer. Even when
the two non-primary providers agree, they never outvote or rewrite the primary.

Comparison ignores case, punctuation, Unicode compatibility differences, and
whitespace so superficial formatting does not create false alarms. It still
reports additions, deletions, and substitutions and gives extra priority to
negations, numbers, and Scripture references. Raw provider output is retained
unchanged in candidate evidence files.

Long recordings target eight-minute core chunks with a twelve-second audio
overlap and a hard ten-minute request limit. Boundaries move to nearby silence
when possible. Every enabled provider receives the same materialized chunk
bytes. Adjacent text is joined deterministically: only a duplicated prefix
supported by at least six exact normalized overlap tokens is removed. If that
conservative seam cannot be proven, both sides are retained and the result is
marked `needs_review`.

See [Architecture](docs/ARCHITECTURE.md) for the exact state machine and
evidence layout, and [Testing](docs/TESTING.md) for validation coverage.

## Requirements

- Python 3.11 or 3.12
- `ffmpeg` and `ffprobe`
- git 2.x
- at least one configured ASR provider
- Telegram credentials only when using the Telegram source
- an NVIDIA GPU is recommended only for optional local Faster-Whisper

## Install

```bash
git clone https://github.com/octoleo/octoscribe.git
cd octoscribe
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp conf/.env.example .env
cp conf/octoscribe.ini.example conf/octoscribe.ini
```

Install the large local model runtime only if `whisper` is explicitly enabled:

```bash
pip install -r requirements-local.txt
```

Keep secrets in `.env`, exported environment variables, or a CI secret store.
Do not put API keys, repository credentials, or Telegram sessions in the INI
file or either evidence repository.

## Configure providers

Provider discovery is credential-driven and deterministic:

1. `OPENAI_API_KEY` enables `openai`.
2. `XAI_API_KEY` enables `xai`.
3. `META_ASR_URL` enables `meta`.

OpenAI is therefore primary when its key is configured. Set an explicit list
when exact provider selection matters:

```dotenv
OPENAI_API_KEY=sk-proj-...
XAI_API_KEY=xai-...

OCTOSCRIBE_ASR_PROVIDERS=openai,xai
OCTOSCRIBE_PRIMARY_ASR=openai
```

For the optional third provider:

```dotenv
OCTOSCRIBE_ASR_PROVIDERS=openai,xai,meta
META_ASR_URL=http://127.0.0.1:8000
META_ASR_MODEL=omniASR_LLM_Unlimited_7B_v2
META_ASR_LANGUAGE=eng_Latn
# META_ASR_API_KEY=optional-private-service-token
```

`META_ASR_URL` must point to a real OpenAI-compatible audio transcription
service. OctoScribe accepts a base URL, a `/v1` URL, or a complete
`/v1/audio/transcriptions` URL. Plain HTTP is allowed only on loopback; use
HTTPS remotely. A normal text-only Llama or Ollama endpoint cannot transcribe
audio and is not a substitute for Meta Omnilingual ASR.

To use one provider only:

```dotenv
OCTOSCRIBE_ASR_PROVIDERS=openai
```

To opt into local Faster-Whisper:

```dotenv
OCTOSCRIBE_ASR_PROVIDERS=whisper
OCTOSCRIBE_PRIMARY_ASR=whisper
```

The legacy `TRANSCRIBE_BACKEND=openai|local` and `--backend` interface remains
supported for single-provider deployments. Do not set the legacy backend and
an explicit provider list together.

The main INI defaults are:

```ini
[transcribe]
model = gpt-transcribe
language = en
workers = 4
retry_attempts = 1
provider_timeout_seconds = 900

[chunking]
target_seconds = 480
max_seconds = 600
overlap_seconds = 12
silence_search_seconds = 45
max_chunk_megabytes = 24

[quality]
disagreement_retry_limit = 1
arbitration_limit = 1
```

Both quality limits are restricted to zero or one. This makes the automatic
loop finite and its maximum semantic work predictable: one request per chunk
with one provider, at most four base-provider requests after a two-provider
disagreement, and at most one additional arbiter request with three providers.
Transport retries for transient failures are separately bounded by
`retry_attempts` (one retry by default). Thus the five-listen worst semantic
path can make at most ten transport attempts with default settings.

## Configure the audio source

Telegram and folder ingestion enter the same hash, manifest, transcription,
logging, and repository pipeline.

Telegram is the default:

```ini
[source]
mode = telegram

[telegram]
group = @your_group_or_chat_id
```

It requires `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and `TELEGRAM_PHONE`. The
first local connection prompts for Telegram authentication and saves the
session outside the evidence repositories by default.

Folder mode needs no Telegram credentials:

```ini
[source]
mode = folder
folder = ~/sermons/incoming
recursive = true
```

Or select it per command:

```bash
python octoscribe.py run --folder ~/sermons/incoming
```

Recognized audio files are copied, never moved. The copy is SHA-256 verified,
and content-hash deduplication makes rescans resumable even when filenames or
directories change.

## Keep audio and text in separate repositories

Separate repositories are the recommended default:

```dotenv
AUDIO_REPO_URL=git@github.com:your-org/sermons-audio.git
TRANSCRIPT_REPO_URL=git@github.com:your-org/sermons-transcripts.git
# AUDIO_REPO_PATH=~/.octoscribe/audio-data
# TRANSCRIPT_REPO_PATH=~/.octoscribe/transcript-data
```

The layouts are intentionally different:

```text
sermons-audio/
└── audio/                     immutable source recordings

sermons-transcripts/
├── manifest.json              identity, state, hashes, and provenance
├── transcriptions/            machine or cross-checked canonical text
│   └── needs-review/          unresolved results, never silently accepted
├── candidates/                append-only raw provider/chunk attempts
└── reports/                   comparisons, seams, final hash, audio revision
```

The production `run` command commits the audio first, proves that each manifest
audio file is tracked and clean, records the audio git revision, and only then
transcribes and commits text evidence. Source audio is checked by SHA-256 again
before any provider call. A changed source is rejected.

Legacy shared repositories remain supported with `DATA_REPO_URL` and
`DATA_REPO_PATH`; do not mix legacy and split repository variables.

## Commands

Global options such as repository paths go before the command:

```bash
# Full integrity-gated production path
python octoscribe.py run

# Folder source with an explicit two-provider policy
python octoscribe.py run \
  --folder ~/sermons/incoming \
  --providers openai,xai \
  --primary-provider openai

# Preview pending entries; no provider calls or transcript writes. Repository
# setup and integrity checks may still initialize, clone, or pull.
python octoscribe.py run --dry-run
python octoscribe.py transcribe --dry-run

# Source-only and transcription-only operations
python octoscribe.py download --folder ~/sermons/incoming
python octoscribe.py transcribe --providers openai,xai

# Repository and manifest information
python octoscribe.py status
python octoscribe.py sync
```

Use `run` for the strongest audio-before-text publication guarantee. `download`
and `transcribe` are useful maintenance stages; `sync` performs repository git
operations. `--no-push` is available on `run` for local commits only.

Other commands:

- `debug --scan-limit N`: inspect Telegram message metadata.
- `session check|export`: inspect or export the Telegram session for CI.
- `ci-export`: print the currently resolved CI secrets/variables locally; it
  refuses to run inside CI.

Run `python octoscribe.py --help` or a subcommand with `--help` for the complete
CLI surface.

## GitHub Actions

The composite action supports Telegram or folder input, split repositories,
and all four provider names (`openai`, `xai`, `meta`, `whisper`). The workflow
must prepare the evidence repositories before invoking the action and configure
authentication appropriate to their remotes; SSH is needed only for SSH URLs,
and GPG signing is optional. For Telegram, authenticate locally and store the
output of
`python octoscribe.py session export` as the `TELEGRAM_SESSION_B64` secret.

A minimal split-repository step looks like:

```yaml
- name: Run OctoScribe
  uses: octoleo/octoscribe@v1
  with:
    command: run
    source: telegram
    telegram_api_id: ${{ secrets.TELEGRAM_API_ID }}
    telegram_api_hash: ${{ secrets.TELEGRAM_API_HASH }}
    telegram_phone: ${{ secrets.TELEGRAM_PHONE }}
    telegram_session_b64: ${{ secrets.TELEGRAM_SESSION_B64 }}
    telegram_group: ${{ vars.TELEGRAM_GROUP }}
    openai_api_key: ${{ secrets.OPENAI_API_KEY }}
    xai_api_key: ${{ secrets.XAI_API_KEY }}
    providers: openai,xai
    primary_provider: openai
    audio_repo_path: ./audio-data
    transcript_repo_path: ./transcript-data
```

See [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml) for the
full clone, provider, source, and legacy-layout example.

## Validation

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers deterministic chunk planning, real ffmpeg materialization,
conservative seams, bounded provider resolution, OpenAI/xAI/Meta adapters,
evidence conflicts and hashes, truthful manifest states, folder/Telegram
dispatch, repository ordering, secret redaction, and CLI/action configuration.

## Security and operational notes

- Treat Telegram session files like passwords. Keep them outside evidence
  repositories and rotate them if exposed.
- Keep audio and transcript remotes private when sermon recordings or metadata
  are sensitive.
- Provider credentials are loaded from environment variables, not the INI.
- The xAI destination is pinned to `https://api.x.ai/v1/stt`.
- A remote Meta ASR endpoint must use HTTPS; credentials embedded in URLs,
  query strings, and fragments are rejected.
- Review `needs-review` text against the original audio before promoting it to
  `human_verified`. Automation narrows the review surface; it does not replace
  a listening human when exactness is required.

## Documentation

- [Architecture and evidence model](docs/ARCHITECTURE.md)
- [Testing guide](docs/TESTING.md)
- [Configuration template](conf/octoscribe.ini.example)
- [Environment template](conf/.env.example)

## License

See [LICENSE](LICENSE).
