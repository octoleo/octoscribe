# OctoScribe

OctoScribe is a GitHub Action for fidelity-first transcription of long-form
audio such as sermons. It accepts audio from Telegram or an already-prepared
folder, splits long recordings into deterministic overlapping chunks, runs one
to three independent speech-to-text providers, and writes transcripts,
comparison reports, raw candidates, and a persistent manifest to caller-owned
workspace paths.

OctoScribe does **not** clone, commit, pull, or push repositories. The calling
workflow owns Git authentication and publication. The recommended setup uses
[`octoleo/git-user@v2`](https://github.com/octoleo/git-user) before cloning the
audio and transcript repositories, then commits the action's output afterward.
This keeps OctoScribe focused on one job: audio-to-text conversion.

## Start with a workflow

Choose one of the complete templates:

- [`examples/full.yml`](examples/full.yml) is the recommended production
  workflow. It supports split or shared repositories, Telegram or a prepared
  folder, one to three providers, branch-aware clones, explicit audio revision
  provenance, bounded concurrency, and verbose CI logs.
- [`examples/minimal.yml`](examples/minimal.yml) is the smallest useful
  one-provider workflow. It uses a shared data repository and one `run` action
  call. It is simpler, but its transcript evidence cannot name a separately
  committed source-audio revision; use the full workflow when provenance is
  critical.
- [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml) is the
  repository's operational reference and follows the full two-stage pattern.
  Its `validate` operation runs the same Python matrix and real composite-action
  smoke tests used by pull-request CI, without requiring provider credentials.

The production sequence is deliberately split between the action and its
caller:

```text
caller: configure Git identity and clone worktrees
action: download/import and hash audio; update manifest
caller: commit and push exact audio bytes; capture revision and branch
action: transcribe those bytes with the captured provenance
caller: commit and push manifest, candidates, reports, and text
```

In a shared repository, the first caller commit stages `audio/` together with
the manifest checkpoint; the final commit stages transcript evidence and the
updated manifest. In split mode, the caller checkpoints audio first, then the
download-state manifest, and finally transcript evidence.

## GitHub configuration

Store credentials in repository or organization **Secrets**. Store non-secret
policy and repository names in **Variables**. The full example recognizes all
of the following placeholders; only values needed by the selected source,
layout, and providers are required.

### Secrets

| Name | Used for |
| --- | --- |
| `SSH_KEY`, `SSH_PUB` | Cloning and pushing caller-owned repositories through `octoleo/git-user`. |
| `GPG_KEY`, `GPG_USER` | Signing identity required by `octoleo/git-user@v2`. |
| `GIT_USER`, `GIT_EMAIL` | Commit author configured by `octoleo/git-user`. |
| `GIT_TOKEN` | Optional personal access token exposed by `octoleo/git-user` for caller-side GitHub CLI use. |
| `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_PHONE`, `TELEGRAM_SESSION_B64` | Telegram source only. Treat the saved session like a password. |
| `OPENAI_API_KEY` | Enables OpenAI transcription. This is the recommended primary/default provider. |
| `XAI_API_KEY` | Enables xAI/Grok as an independent provider. |
| `META_ASR_API_KEY` | Optional bearer token for a hosted Meta-compatible ASR endpoint. |

### Variables

| Name | Example | Purpose |
| --- | --- | --- |
| `REPOSITORY_LAYOUT` | `split` | `split` for separate audio/transcript repositories, or `shared`. |
| `DATA_REPO_OWNER`, `DATA_REPO_NAME`, `DATA_REPO_BRANCH` | `your-org`, `sermon-data`, `main` | Shared-layout repository. |
| `AUDIO_REPO_OWNER`, `AUDIO_REPO_NAME`, `AUDIO_REPO_BRANCH` | `your-org`, `sermon-audio`, `main` | Split-layout immutable audio repository. |
| `TRANSCRIPT_REPO_OWNER`, `TRANSCRIPT_REPO_NAME`, `TRANSCRIPT_REPO_BRANCH` | `your-org`, `sermon-text`, `main` | Split-layout manifest, evidence, and text repository. |
| `SOURCE_MODE` | `telegram` | `telegram` or `folder`. |
| `TELEGRAM_GROUP` | `@sermon_archive` | Telegram group, channel, or numeric chat ID. |
| `SOURCE_REPO_OWNER`, `SOURCE_REPO_NAME`, `SOURCE_REPO_BRANCH` | `your-org`, `incoming-audio`, `main` | Optional repository cloned by the full example when `SOURCE_MODE=folder`. |
| `SOURCE_FOLDER` | `./prepared-source/audio` | Prepared folder passed to OctoScribe. |
| `SSH_TYPE`, `SSH_HOST`, `GIT_USER_FORCE` | `ed25519`, `github.com`, `false` | Optional `octoleo/git-user` SSH and replacement policy. |
| `OCTOSCRIBE_ASR_PROVIDERS` | `openai,xai,meta` | Ordered list of one to three providers. |
| `OCTOSCRIBE_PRIMARY_ASR` | `openai` | Provider whose exact word surface is canonical. |
| `TRANSCRIBE_MODEL`, `TRANSCRIBE_LANGUAGE` | `gpt-transcribe`, `en` | Primary transcription model and language hint. |
| `XAI_BASE_URL` | `https://api.x.ai/v1/stt` | xAI STT URL; retain the official TLS endpoint. |
| `META_ASR_URL`, `META_ASR_MODEL`, `META_ASR_LANGUAGE` | service URL, model ID, `eng_Latn` | Hosted Meta Omnilingual-compatible ASR service. |
| `LOCAL_MODEL`, `LOCAL_DEVICE`, `LOCAL_COMPUTE_TYPE` | `large-v3`, `cpu`, `int8` | Hosted-runner Faster-Whisper defaults; a self-hosted GPU can override device/compute type. |
| `OCTOSCRIBE_CONFIG_PATH`, `OCTOSCRIBE_VERBOSE` | `conf/octoscribe.ini`, `true` | Optional checked-out INI policy and debug-level logging. |

The GitHub UI may leave optional values unset. The full example supplies
documented defaults without placing tokens in the workflow file.

For Telegram CI, authenticate once in a trusted local environment and run
`python octoscribe.py session export`. Store the resulting base64 value as the
`TELEGRAM_SESSION_B64` secret. Do not paste session bytes into workflow YAML or
commit the generated Telethon database.

## Composite action inputs

The action consumes existing local paths. It never interprets repository names
or remote URLs.

| Input | Meaning |
| --- | --- |
| `command` | `run`, `download`, `transcribe`, or `status`. Production provenance uses `download` followed by `transcribe`. |
| `source` | `telegram` or `folder` for `run`/`download`. |
| `source_folder` | Prepared input directory when `source=folder`. |
| `telegram_api_id`, `telegram_api_hash`, `telegram_phone`, `telegram_session_b64`, `telegram_group` | Telegram source credentials and target. |
| `openai_api_key`, `xai_api_key` | Provider credentials. |
| `meta_asr_url`, `meta_asr_api_key`, `meta_asr_model`, `meta_asr_language` | Hosted Meta-compatible ASR configuration. |
| `providers` | Ordered provider list: `openai`, `xai`, `meta`, and/or `whisper` (maximum three). |
| `primary_provider` | Canonical provider; otherwise the first enabled provider. |
| `transcribe_model`, `transcribe_language` | Main model and spoken-language hint. |
| `xai_base_url` | xAI STT endpoint. |
| `local_model`, `local_device`, `local_compute_type` | Optional Faster-Whisper runtime policy. |
| `audio_repo_path`, `transcript_repo_path` | Existing caller worktrees for split layout; provide both. |
| `data_repo_path` | Existing caller worktree for shared layout; do not combine with split paths. |
| `audio_revision`, `audio_repository_branch` | Caller-captured source commit and branch added to transcript provenance. |
| `config_path` | Optional INI file for advanced non-secret policy. |
| `verbose` | `true` for debug-level CI logging. Secrets remain redacted. |

The action also exposes `audio_dir`, `manifest_file`, `transcriptions_dir`,
`candidates_dir`, and `reports_dir` outputs so a caller can archive, inspect, or
selectively publish the generated paths without guessing the configured
layout.

All work is reported to standard output so GitHub Actions logs show source
discovery, skips, hashes, chunk planning, provider attempts, comparisons,
seams, output locations, and final state. Credentials and Telegram session
bytes are never intentionally logged.

## Persistent state and idempotence

The caller must preserve the text/evidence worktree between runs by committing
its output. Its `manifest.json` is the index that records source identity,
SHA-256 hashes, acquisition state, transcript state, providers, output hashes,
and quality results. On the next checkout, OctoScribe reads that index and
skips content already completed. Folder imports use content hashes, so renaming
the same source file does not make it new work.

Recommended split layout:

```text
sermon-audio/
└── audio/                         immutable source recordings

sermon-text/
├── manifest.json                  persistent index and provenance
├── transcriptions/                canonical output
│   └── needs-review/              unresolved output quarantine
├── candidates/                    raw provider/chunk attempts
└── reports/                       comparisons, seams, hashes, revision
```

Shared layout uses the same paths under one worktree. Never delete the manifest
between runs unless intentionally rebuilding the index.

## Fidelity model

OctoScribe does not proofread, paraphrase, or generatively “clean up” sermons.
The primary provider owns the canonical words; independent providers validate
or dispute them but never silently replace them.

| Providers | Bounded behavior | Successful machine state |
| --- | --- | --- |
| One | One transcription. | `machine_transcribed` |
| Two | Independent transcription and comparison; retry both once on disagreement. | `cross_checked` only when normalized words agree; otherwise `needs_review`. |
| Three | The first two compare and retry; the third is a one-time fallback checker or arbiter. | `cross_checked` only when an independent provider supports the primary; otherwise `needs_review`. |

The automatic loop is finite. Comparison ignores superficial casing,
punctuation, Unicode compatibility, and whitespace, while retaining original
provider text unchanged in evidence. Negations, numbers, biblical names, and
Scripture references remain visible in discrepancy reports.

Long recordings—including 90-minute sermons—target eight-minute core chunks,
with twelve seconds of audio overlap and a hard ten-minute request window.
Boundaries move to nearby silence where possible. The same materialized chunk
bytes go to every enabled provider. The overlapping ASR text is joined by a
deterministic token alignment, not a generative editor, so seaming adds no
provider cost and cannot silently rewrite wording. Only a proven duplicate
prefix of at least six exact normalized tokens is removed; otherwise both sides
remain and the recording becomes `needs_review`.

Machine transcription cannot prove word-for-word identity with audio. Only a
human listening comparison can produce `human_verified`; automation makes that
review smaller and evidence-based.

See [Architecture](docs/ARCHITECTURE.md) for the state machine and evidence
model, and [Testing](docs/TESTING.md) for the validation contract.

## Provider notes

OpenAI `gpt-transcribe` is the recommended default. When `providers` is blank,
credentials enable providers in OpenAI, xAI, then Meta order; the first enabled
provider becomes primary. An explicit list remains available when exact
selection matters. `META_ASR_URL` must serve an OpenAI-compatible transcription
API; a text-only Llama or Ollama endpoint cannot transcribe audio. Local
Faster-Whisper is optional, is not installed unless selected, and is generally
less accurate than the recommended hosted primary for this use case.

## Local use

Local operation follows the same path semantics and does no Git publication.

```bash
git clone https://github.com/octoleo/octoscribe.git
cd octoscribe
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp conf/.env.example .env
cp conf/octoscribe.ini.example conf/octoscribe.ini
```

Prepare existing directories, then run either a shared or split layout:

```bash
# Shared worktree, prepared folder, one provider
python octoscribe.py --data-repo ./sermon-data \
  run --folder ./incoming-audio --providers openai

# Split production stages; capture the revision with your own Git tooling
python octoscribe.py --audio-repo ./sermon-audio \
  --transcript-repo ./sermon-text \
  download --folder ./incoming-audio

python octoscribe.py --audio-repo ./sermon-audio \
  --transcript-repo ./sermon-text \
  transcribe --providers openai,xai \
  --audio-revision "$AUDIO_REVISION" \
  --audio-repository-branch main
```

Install `requirements-local.txt` only when selecting the `whisper` provider.
Run `python -m pytest` after installing `requirements-dev.txt` to validate the
project.

## Security

- Put all tokens and session data in GitHub Secrets or local environment
  variables, never in INI files, manifests, or evidence repositories.
- Keep sensitive sermon audio and transcripts in appropriately protected
  repositories.
- Use HTTPS for remote Meta endpoints; loopback HTTP is intended only for local
  services.
- Review every `needs_review` transcript against the original audio before
  promoting it to a human-verified artifact.

## License

See [LICENSE](LICENSE).
