# OctoScribe

OctoScribe is a GitHub Action for fidelity-first transcription of long-form
audio such as sermons. It accepts audio from Telegram or an already-prepared
folder, splits long recordings into deterministic overlapping chunks, runs one
to three independent speech-to-text providers, and writes transcripts,
comparison reports, raw candidates, and a persistent manifest to caller-owned
workspace paths. Published transcript text is presentation-normalized so each
recognized sentence occupies its own line, without changing the provider's
words or punctuation.

## Three paths are the contract

Every normal run needs only these locations:

| Environment variable | Action input | Default | Purpose |
| --- | --- | --- | --- |
| `AUDIO_PATH` | `audio_path` | `./audio` | Acquisition destination and transcription input directory. |
| `TRANSCRIPT_PATH` | `transcript_path` | `./transcriptions` | Every completed transcript is written here. |
| `MANIFEST_PATH` | `manifest_path` | `./manifest.json` | Persistent identity, hashes, completion state, provenance, and output links. |

Resolution is deliberately simple: a non-empty action input overrides the
inherited environment variable, which overrides the effective default shown
above.

Relative paths resolve from `GITHUB_WORKSPACE` in GitHub Actions and from the
current working directory locally. Absolute paths are also accepted. The paths
do not have to share a repository:

```yaml
env:
  AUDIO_PATH: ./audio
  TRANSCRIPT_PATH: ./transcriptions
  MANIFEST_PATH: ./manifest.json

steps:
  - uses: octoleo/octoscribe@v1
    with:
      command: transcribe
      providers: openai
      primary_provider: openai
      openai_api_key: ${{ secrets.OPENAI_API_KEY }}
```

For one checked-out data repository, prefix all three paths with that checkout.
For separate repositories, point `audio_path` into the audio checkout and point
`transcript_path` and `manifest_path` into the text checkout. OctoScribe does
not need a repository-layout mode; it only uses the three resolved paths.

```yaml
- uses: octoleo/octoscribe@v1
  with:
    audio_path: ./sermon-audio/audio
    transcript_path: ./sermon-text/transcriptions
    manifest_path: ./sermon-text/manifest.json
    openai_api_key: ${{ secrets.OPENAI_API_KEY }}
```

Raw candidates and evidence reports are automatically derived as `candidates/`
and `reports/` beside `TRANSCRIPT_PATH`. Verification-only references and
comparison reports similarly default to `reference-transcripts/` and
`comparison-reports/` beside it. Advanced overrides remain available, but a
normal user does not configure those directories.

OctoScribe does **not** clone, commit, pull, or push repositories. The calling
workflow owns Git authentication and publication. The recommended setup uses
[`octoleo/git-user@v2`](https://github.com/octoleo/git-user) before cloning the
audio and transcript repositories, then commits the action's output afterward.
This keeps OctoScribe focused on one job: audio-to-text conversion.

## Start with a workflow

Choose one of the complete templates:

- [`examples/in-repository.yml`](examples/in-repository.yml) is the recommended
  workflow when audio and text live in the repository where the workflow runs.
  It uses the built-in `GITHUB_TOKEN`, stores `audio/`, `manifest.json`,
  `transcriptions/`, `candidates/`, and `reports/` together, and requires no
  `git-user` step or second data-repository checkout.
- [`examples/full.yml`](examples/full.yml) is the recommended production
  workflow. It shows caller-owned checkouts, Telegram or folder input, explicit
  three-path placement, one to three providers, revision provenance, bounded
  concurrency, and verbose CI logs.
- [`examples/minimal.yml`](examples/minimal.yml) is the smallest useful
  one-provider workflow using the three default paths and one action call.
- [`examples/pipeline.yml`](examples/pipeline.yml) is a complete scheduled and
  manually dispatched operational workflow to copy into a consuming repository.
  It follows the full two-stage pattern without activating that credentialed
  pipeline inside the OctoScribe source repository.

The production sequence is deliberately split between the action and its
caller:

```text
caller: check out whichever repository or repositories own the three paths
action: Telegram or folder acquisition writes AUDIO_PATH; update MANIFEST_PATH
caller: preserve exact audio bytes and manifest; optionally capture the audio revision
action: transcribe to TRANSCRIPT_PATH and update MANIFEST_PATH
caller: preserve manifest, text, candidates, and reports
```

With all three defaults, a completed run leaves this durable layout:

```text
audio/             original source recordings
manifest.json      persistent index; completed items are skipped on later runs
transcriptions/    authoritative machine transcript text
candidates/        raw responses retained from each provider
reports/           comparison, discrepancy, hash, and provenance evidence
```

## GitHub configuration

Store credentials in repository or organization **Secrets**. Store non-secret
policy and repository names in **Variables**. The full example recognizes all
of the following placeholders; only values needed by the selected source,
three paths, caller-owned checkouts, and providers are required.

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
| `AUDIO_PATH` | `./audio` | Acquisition destination and transcription input directory. |
| `TRANSCRIPT_PATH` | `./transcriptions` | Completed transcript directory. |
| `MANIFEST_PATH` | `./manifest.json` | Persistent index file. |
| `SOURCE_MODE` | `telegram` | `telegram` or `folder`. |
| `SOURCE_FOLDER` | `./incoming-audio` | Folder-acquisition source copied into `AUDIO_PATH`. |
| `TELEGRAM_GROUP` | `@sermon_archive` | Telegram group, channel, or numeric chat ID. |
| `DATA_REPO_OWNER`, `DATA_REPO_NAME`, `DATA_REPO_BRANCH` | `your-org`, `sermon-data`, `main` | Optional caller-workflow coordinates when all three paths live in another repository. |
| `AUDIO_REPO_OWNER`, `AUDIO_REPO_NAME`, `AUDIO_REPO_BRANCH` | `your-org`, `sermon-audio`, `main` | Optional caller-workflow coordinates for a separate audio checkout. |
| `TRANSCRIPT_REPO_OWNER`, `TRANSCRIPT_REPO_NAME`, `TRANSCRIPT_REPO_BRANCH` | `your-org`, `sermon-text`, `main` | Optional caller-workflow coordinates for a separate text/manifest checkout. |
| `SSH_TYPE`, `SSH_HOST`, `GIT_USER_FORCE` | `ed25519`, `github.com`, `false` | Optional `octoleo/git-user` SSH and replacement policy. |
| `OCTOSCRIBE_ASR_PROVIDERS` | `openai,xai,meta` | Ordered list of one to three providers. |
| `OCTOSCRIBE_PRIMARY_ASR` | `openai` | Provider whose exact word surface is canonical. |
| `TRANSCRIBE_MODEL`, `TRANSCRIBE_LANGUAGE` | `gpt-transcribe`, `en` | Primary transcription model and language hint. |
| `XAI_BASE_URL` | `https://api.x.ai/v1/stt` | xAI STT URL; retain the official TLS endpoint. |
| `META_ASR_URL`, `META_ASR_MODEL`, `META_ASR_LANGUAGE` | service URL, model ID, `eng_Latn` | Hosted Meta Omnilingual-compatible ASR service. |
| `LOCAL_MODEL`, `LOCAL_DEVICE`, `LOCAL_COMPUTE_TYPE` | `large-v3`, `cpu`, `int8` | Faster-Whisper fallback/explicit-provider settings; a self-hosted GPU can override device/compute type. |
| `OCTOSCRIBE_CONFIG_PATH`, `OCTOSCRIBE_VERBOSE` | `conf/octoscribe.ini`, `true` | Optional checked-out INI policy and debug-level logging. |

The first three variables are OctoScribe's placement contract. Repository-owner
variables belong only to caller workflows that choose to clone other
repositories; OctoScribe never consumes repository names or URLs. The GitHub UI
may leave optional values unset.

For Telegram CI, authenticate once in a trusted local environment and run
`python octoscribe.py session export`. Store the resulting base64 value as the
`TELEGRAM_SESSION_B64` secret. Do not paste session bytes into workflow YAML or
commit the generated Telethon database.

## Composite action inputs

The action consumes existing local paths. It never interprets repository names
or remote URLs.

| Input | Meaning |
| --- | --- |
| `command` | `run`, `download`, `transcribe`, `verify`, or `status`. |
| `source` | `telegram` or `folder` for `run`/`download`. |
| `source_folder` | Source directory for folder `run`/`download`; recognized files are copied and verified into `audio_path`. |
| `audio_path` | Acquisition destination and transcription input; default `./audio`. |
| `transcript_path` | Normal completed-text directory; default `./transcriptions`. |
| `manifest_path` | Persistent index file; default `./manifest.json`. |
| `telegram_api_id`, `telegram_api_hash`, `telegram_phone`, `telegram_session_b64`, `telegram_group` | Telegram source credentials and target. |
| `openai_api_key`, `xai_api_key` | Provider credentials. |
| `meta_asr_url`, `meta_asr_api_key`, `meta_asr_model`, `meta_asr_language` | Hosted Meta-compatible ASR configuration. |
| `providers` | Ordered provider list: `openai`, `xai`, `meta`, and/or `whisper` (maximum three). |
| `primary_provider` | Canonical provider; otherwise the first enabled provider. |
| `transcribe_model`, `transcribe_language` | Main model and spoken-language hint. |
| `xai_base_url` | xAI STT endpoint. |
| `local_model`, `local_device`, `local_compute_type` | Optional Faster-Whisper runtime policy. |
| `reference_path`, `comparison_report_path` | Verify-only overrides; by default they are siblings of `transcript_path`. |
| `audio_revision`, `audio_repository_branch` | Caller-captured source commit and branch added to transcript provenance. |
| `config_path` | Optional INI file for advanced non-secret policy. |
| `verbose` | `true` for debug-level CI logging. Secrets remain redacted. |

The action exposes resolved path outputs so a caller can archive or publish the
generated files without guessing. Legacy `data_repo_path`, `audio_repo_path`,
and `transcript_repo_path` root inputs remain an advanced compatibility layer
where supported; new workflows should use the three explicit path inputs.

All work is reported to standard output so GitHub Actions logs show source
discovery, skips, hashes, chunk planning, provider attempts, comparisons,
seams, output locations, and final state. Credentials and Telegram session
bytes are never intentionally logged.

## Persistent state and idempotence

The caller must preserve `MANIFEST_PATH` and completed outputs between runs.
The manifest is the index that records source identity,
SHA-256 hashes, acquisition state, transcript state, providers, output hashes,
and quality results. On the next checkout, OctoScribe reads that index and
skips content already completed. A terminal manifest state is skipped only when
its recorded transcript is still a safe regular file and its SHA-256 still
matches the manifest. A missing or changed output is re-queued, while an intact
output never incurs another provider call. Folder imports use content hashes,
so renaming the same source file does not make it new work.

Telegram entries retain the established message/audio fields: message ID,
date, title, performer, duration, formatted duration, extension, sanitized and
original filenames, and the exact source SHA-256. Folder entries record the
same common fields where available and use mutagen to collect safe embedded
tags plus technical properties such as album, artist, composer, genre, track,
bitrate, sample rate, channels, codec, MIME type, file size, and source modified
time. Missing metadata remains `null`; it never blocks import or alters audio.

After transcription, each entry's `transcription` object retains the legacy
`output_file` and also records `audio_path` and `output_path`, the transcript
SHA-256, provider/model provenance, quality state,
evidence-report link, and structured `integrity_warnings` for provider
disagreement, an unaligned seam, or a provider failure. The manifest can
therefore be used directly as the catalog joining every preserved recording to
its published text and evidence.

The same three inputs cover every placement without a layout switch:

```text
# Same workflow repository
AUDIO_PATH=./audio
TRANSCRIPT_PATH=./transcriptions
MANIFEST_PATH=./manifest.json

# One separately checked-out data repository
AUDIO_PATH=./sermon-data/audio
TRANSCRIPT_PATH=./sermon-data/transcriptions
MANIFEST_PATH=./sermon-data/manifest.json

# Separate checked-out repositories
AUDIO_PATH=./sermon-audio/audio
TRANSCRIPT_PATH=./sermon-text/transcriptions
MANIFEST_PATH=./sermon-text/manifest.json
```

Never delete the manifest between runs unless intentionally rebuilding the
index. The caller decides which repositories contain these paths and owns all
checkout and publication steps.

## Fidelity model

OctoScribe does not proofread, paraphrase, or generatively “clean up” sermons.
The primary provider owns the canonical words; independent providers validate
or dispute them but never silently replace them.

OctoScribe does not perform theology checks, religious-content classification,
content moderation, censorship, or approval. Provider disagreement and seam
alignment are technical transcription signals only. They never suppress,
relocate, or withhold a transcript because of what the speaker said.

| Providers | Bounded behavior | Completed machine state |
| --- | --- | --- |
| One | One transcription. | `machine_transcribed` |
| Two | Independent transcription and comparison; retry both once on disagreement. | `cross_checked` when normalized words agree; otherwise `completed_with_warnings`. |
| Three | The first two compare and retry; the third is a one-time fallback checker or arbiter. | `cross_checked` when an independent provider supports the primary; otherwise `completed_with_warnings`. |

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
remain, the transcript is published normally, and its state becomes
`completed_with_warnings` so the technical seam uncertainty remains visible.

Sentence-per-line formatting runs only after all chunks have been stitched.
It changes whitespace at recognized sentence boundaries and never invents,
removes, or corrects words or punctuation. Common abbreviations, initials, and
Scripture-reference abbreviations are kept within their sentence. Raw provider
candidates remain unchanged. The formatted publication surface is established
before the evidence report and manifest calculate the final transcript
SHA-256, so both hashes identify the exact bytes written to
`TRANSCRIPT_PATH`.

Machine transcription cannot prove word-for-word identity with audio. A
machine-generated reference transcript—including one produced by the real-audio
workflow—is therefore not human-verified ground truth. `human_verified` means a
person explicitly compared the transcript with the source audio; it is an
additional provenance fact, not permission to publish and not a content gate.

See [Architecture](docs/ARCHITECTURE.md) for the state machine and evidence
model, and [Testing](docs/TESTING.md) for the validation contract.

## Real-audio workflow evidence

Ordinary CI remains credential-free. The separate
`.github/workflows/openai-real-audio.yml` workflow is both a real integration
test and a clean consumer example: it checks out the requested revision,
transcribes the two owner-supplied Telegram OGG recordings through `uses: ./`,
runs the action's `command: verify` path, and uploads the generated manifest,
transcripts, candidates, evidence reports, reference-comparison reports, and
reference files as GitHub Actions artifacts. The workflow itself contains no
inline Python implementation of the verifier.

Manual capture mode is the only operation that produces bootstrap machine
reference files; its artifact can be inspected and deliberately committed.
Runs after a pull request is merged into `v1` require those committed references
and compare generated text word-for-word after normalizing only case,
punctuation, and whitespace. Every added, deleted, or substituted spoken word
is reported; a merge-triggered run never rewrites its reference.

The repository workflow resolves references under
`tests/fixtures/telegram/reference-transcripts/` and writes comparison JSON
under `tests/fixtures/telegram/comparison-reports/`. The equivalent direct CLI
uses `--transcript-path`, `--reference-path`, and `--comparison-report-path`
before the `verify` command. `verify --allow-missing-references
--capture-reference` is reserved for intentional manual bootstrap; normal and
merge-triggered verification requires every reference.

Those uploaded transcripts are machine reference outputs. They prove that the
real transport and pipeline ran and give future runs a concrete result to
inspect; they are not asserted as word-error ground truth. Only a separately
prepared transcript that a person checked while listening to the complete
recording may be described as human-verified ground truth.

## Provider notes

OpenAI `gpt-transcribe` is the recommended default. When `providers` is blank,
configured hosted providers are discovered in OpenAI, xAI, then Meta order;
the first enabled provider becomes primary. If any hosted provider is
available, OctoScribe uses the API path and does not install Faster-Whisper.
When no hosted provider is configured, a transcription run falls back to local
Whisper. An explicit `providers: whisper` (or legacy local-backend selection)
also requests that fallback directly. Only those two cases install the optional
runtime from `requirements-local.txt`. An explicit provider list remains
available when exact selection matters. `META_ASR_URL` must serve an
OpenAI-compatible transcription API; a text-only Llama or Ollama endpoint
cannot transcribe audio. Local Faster-Whisper is generally less accurate than
the recommended hosted primary for this use case.

## Local use

Local operation follows the same path semantics and does no Git publication.
Python 3.11 through 3.14 are supported and tested; the composite action itself
sets up Python 3.14. The local example uses that same runtime.

```bash
git clone https://github.com/octoleo/octoscribe.git
cd octoscribe
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp conf/.env.example .env
cp conf/octoscribe.ini.example conf/octoscribe.ini
```

Prepare the three paths, then run the same command regardless of which checked
out repositories contain them:

```bash
# Transcribe files already acquired or placed in ./audio
python octoscribe.py \
  --audio-path ./audio \
  --transcript-path ./transcriptions \
  --manifest-path ./manifest.json \
  transcribe --providers openai --primary-provider openai

# The same paths may point into separate checked-out repositories
python octoscribe.py \
  --audio-path ./sermon-audio/audio \
  --transcript-path ./sermon-text/transcriptions \
  --manifest-path ./sermon-text/manifest.json \
  transcribe --providers openai,xai \
  --audio-revision "$AUDIO_REVISION" \
  --audio-repository-branch main
```

The equivalent environment variables are `AUDIO_PATH`, `TRANSCRIPT_PATH`, and
`MANIFEST_PATH`. Legacy `--data-repo`, `--audio-repo`, and
`--transcript-repo` root options are compatibility conveniences, not the
recommended interface.

Install `requirements-local.txt` only when explicitly selecting the `whisper`
provider or when intentionally running without any configured hosted provider.
Run `python -m pytest` after installing `requirements-dev.txt` to validate the
project.

## Security

- Put all tokens and session data in GitHub Secrets or local environment
  variables, never in INI files, manifests, or evidence repositories.
- Keep sensitive sermon audio and transcripts in appropriately protected
  repositories.
- Use HTTPS for remote Meta endpoints; loopback HTTP is intended only for local
  services.
- Treat `completed_with_warnings` as a completed, normally published transcript
  whose report retains technical fidelity warnings. It is never a content or
  religious-material restriction.

## License

See [LICENSE](LICENSE).
