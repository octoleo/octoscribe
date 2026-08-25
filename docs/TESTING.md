# OctoScribe Testing Guide

The test suite treats fidelity, bounded execution, and evidence preservation as
contracts rather than prompt aspirations. Provider and Telegram tests do not
make live network calls or spend API credits. Filesystem tests use temporary
directories, and the one real audio-tool integration test generates its own WAV
file locally.

For the implementation model, see [Architecture](ARCHITECTURE.md).

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

The supported/tested minor versions are Python 3.11, 3.12, 3.13, and 3.14. The
composite action uses Python 3.14.

`requirements-dev.txt` includes the core runtime and pytest tooling. It does
not install the optional Faster-Whisper runtime; local backend tests replace
that dependency with a test double.

The real chunk materialization test runs when `ffmpeg` and `ffprobe` are on
`PATH` and skips cleanly otherwise.

All integration tests use the public three-path contract:
`AUDIO_PATH`/`audio_path` (default `./audio`),
`TRANSCRIPT_PATH`/`transcript_path` (default `./transcriptions`), and
`MANIFEST_PATH`/`manifest_path` (default `./manifest.json`). Tests cover relative
resolution from the workspace/current directory, absolute paths, and paths in
the same or independently checked-out repositories. Legacy root inputs are
compatibility cases, not the primary test setup.
Action-contract tests enforce the precedence `non-empty action input > inherited
environment variable > effective default`; empty input metadata must not mask a
caller-provided environment value.

## Run tests

```bash
# Whole suite
python -m pytest

# Detailed names and short tracebacks
python -m pytest tests/ -v --tb=short

# Fidelity-critical areas
python -m pytest \
  tests/test_chunking.py \
  tests/test_consensus.py \
  tests/test_ensemble.py \
  tests/test_evidence.py

# A single behaviour
python -m pytest \
  tests/test_ensemble.py::test_third_provider_runs_only_after_retry_and_loop_stops

# Keyword selection
python -m pytest -k "hash or completed_with_warnings"
```

Coverage:

```bash
python -m pytest --cov=src --cov-report=term-missing
python -m pytest --cov=src --cov-report=html
```

## Coverage map

| Test file | Primary contract |
| --- | --- |
| `test_audio_chunks.py` | ffprobe/ffmpeg command construction, silence parsing, deterministic WAV extraction, size limits, cleanup, and tool timeouts. |
| `test_audio_chunks_ffmpeg_integration.py` | Real generated audio is probed, silence-detected, extracted twice, hashed identically, and verified as mono 16 kHz PCM. |
| `test_chunking.py` | Gapless silence-aware planning, eight-minute/12-second defaults, hard limits, deterministic ties, exact conservative seams, and randomized invariants. |
| `test_consensus.py` | Comparison-only normalization, additions/deletions/substitutions, high-fidelity-priority negation/number/Scripture differences, and bounded policy decisions. |
| `test_ensemble.py` | One/two/three-provider flows, primary ownership, one retry, one arbiter, provider outages, seam downgrade, source hashes, and evidence persistence. |
| `test_evidence.py` | Schema validation, source/chunk/transcript hashes, deterministic JSON, append-only conflicts, safe filenames, reports, seams, and audio revisions. |
| `test_provider.py` | Provider transcript and timed-word validation plus detailed-backend adaptation. |
| `test_backend_registry.py` | Canonical provider names, ordered lazy construction, model provenance, duplicates, and bad configuration. |
| `test_xai_backend.py` | Pinned endpoint, multipart fidelity options, response parsing, word timestamps, retries, size checks, and secret-safe errors. |
| `test_meta_backend.py` | Explicit ASR URL resolution, HTTPS/loopback rules, multipart model/language, optional bearer token, response formats, retries, and validation. |
| `test_transcribe.py` | OpenAI and optional local adapter contracts, transport retries, empty output, backend compatibility, and batch orchestration. |
| `test_transcriber_ensemble.py` | Batch integration with the ensemble, normally published `completed_with_warnings` results, hashes, provider failures, and manifest/report links. |
| `test_transcribe_cmd.py` | Caller-supplied audio revision/branch provenance and Git-independent transcription. |
| `test_transcribe_hardening.py` | Collision-safe output paths and the empty-result failure guard. |
| `test_retry.py` | Transient/permanent error classification, backoff, exhaustion, and hard retry bounds. |
| `test_manifest.py` | Source-hash immutability, quality states, pending queries, human verification, stats, atomic JSON, and thread safety. |
| `test_folder.py` | Recursive/flat `source_folder` scans, verified copies into `AUDIO_PATH`, supported formats, embedded/technical metadata, content-hash dedup, resume, and source preservation. |
| `test_telegram.py`, `test_telegram_client.py` | Audio selection, the historical OGG manifest contract, metadata, download behaviour, entity resolution, session handling, and Telegram retries with Telethon mocked. |
| `test_config.py` | Precedence, provider discovery/order, command-aware secrets, the three explicit paths and defaults, endpoint rules, path resolution, and numeric bounds. |
| `test_source_cmd.py` | CLI overrides and folder/Telegram source dispatch. |
| `test_ci_export_cmd.py`, `test_session_cmd.py`, `test_debug.py` | CI export guard/redaction, Telegram session commands, and diagnostic behaviour. |
| `test_persistence.py` | Atomic text writes and periodic manifest saves. |
| `test_action_contract.py` | `audio_path`/`transcript_path`/`manifest_path` defaults and mapping, workspace-relative and absolute placement, no-Git boundary, compatibility roots, and caller-supplied provenance. |

## Fidelity properties under test

### Source identity

Tests prove that a stored audio digest cannot be replaced by a different
SHA-256 value, that folder sources are copied and hash-verified into
`AUDIO_PATH` without modifying the source, that the ensemble rejects an
unexpected source hash before provider work, and that evidence writing rechecks
source and chunk bytes.

When modifying acquisition or workspace-path code, include at least these cases:

- identical bytes resume without duplication;
- changed bytes under an existing identity fail;
- a corrupt or unreadable source does not become accepted audio;
- caller-supplied revision and branch are recorded without running Git.

### Chunk planning and seams

Chunk tests use both examples and seeded randomized inputs. Every non-empty plan
must satisfy:

- cores begin at zero and end at the recording duration;
- adjacent cores meet exactly, with no gap or double ownership;
- context windows remain within the hard maximum;
- adjacent contexts share exactly the configured overlap;
- reversing or duplicating silence inputs does not change the plan.

Production seam defaults require six exact normalized matching tokens and
similarity `1.0`. Stitching is a deterministic comparison of overlapping ASR
text, never a generative edit or additional provider call. Tests specifically
guard against deleting text when there is a substitution, insertion, deletion,
short common phrase, or unrelated seam. If no seam is accepted, both surfaces
remain, normal transcript publication continues, and the overall result becomes
`completed_with_warnings`.

Include long-duration plans (including 90-minute inputs) to prove that the
recording is partitioned into bounded requests with the configured overlap and
without gaps. Such tests operate on duration/boundary policy and do not require
a 90-minute fixture file.

### Provider independence and hard stop

The ensemble suite injects simple backends with scripted results and call
counts. It proves the maximum semantic call pattern per chunk:

| Providers | Maximum semantic calls after disagreement |
| --- | --- |
| One | one primary call |
| Two | initial pair plus one retry of the pair |
| Three, normal disagreement | initial pair, one retry of the pair, then one arbiter call |
| Three, checker outage | initial pair and one fallback call; then at most one retry of primary + fallback, or of the original pair if fallback also fails |

Transport-level retries inside a backend have their own bounded unit tests;
the configured default is one transient retry and the OpenAI SDK's internal
retry layer is disabled.
Tests must never replace this finite state machine with a loop whose bound
depends on provider text.

The canonical result is asserted separately from agreement. Tests prove that a
third provider or majority cannot silently rewrite the primary, and that an
unresolved result remains completed output with the
`completed_with_warnings` technical-fidelity state.

### Evidence preservation

Candidate assertions should inspect persisted JSON, not just call counts. A
candidate must identify the source, exact chunk, provider/model/attempt, raw
transcript, transcript hash, and run, plus optional word timings, confidence,
language, and duration when supplied by a provider. Aggregate reports must
identify the canonical attempt and the exact attempts used by every initial,
retry, fallback, availability, or arbitration comparison.

Include failure-path tests where useful. Successful paid candidates must remain
available even when a later chunk or the primary provider fails.

### Truthful states

Do not assert a generic `completed` state for new ensemble output. Assert one
of:

- `machine_transcribed` for a one-provider result;
- `cross_checked` only after independent normalized-word agreement and proven
  seams;
- `completed_with_warnings` for a completed, normally published transcript
  with unresolved provider disagreement, unavailable independent verification,
  or an unresolved seam;
- `human_verified` only after a person explicitly compared the transcript with
  the source audio.

All completed states use the normal `transcriptions/` output directory.
Technical warning states must never route religious or any other material to a
different directory, suppress output, or introduce content moderation.
Tests also assert that `integrity_warnings` identifies each applicable
provider disagreement, unaligned seam, or provider failure so the state is
machine-readable and never opaque.

## Test-double patterns

### Providers

Prefer injecting a small `TranscriptionBackend` implementation into the
ensemble or transcriber. Script text and exceptions per call, then assert both
the call bound and persisted outcome. Test a concrete adapter separately at its
HTTP/SDK seam.

- OpenAI uses a mocked `client.audio.transcriptions.create` method.
- xAI and Meta inject a fake urllib-compatible transport.
- Faster-Whisper replaces `faster_whisper.WhisperModel` with fake segments.

Never use real provider keys in tests, fixtures, snapshots, or CI logs.

### Audio tools

Pure planning tests do not call ffmpeg. Unit tests inject a subprocess runner
and assert arguments/results. The integration test is intentionally narrow: it
generates PCM samples, uses local ffmpeg/ffprobe, and performs no network I/O.

### Telegram and workflow Git boundary

Telegram tests replace asynchronous client methods with `AsyncMock`. Action and
CLI contract tests assert that OctoScribe itself never runs Git. Workflow
templates may be statically checked for the expected caller-owned sequence:
checkout, supply the three paths, acquire or scan audio, checkpoint the manifest,
transcribe, and preserve text/evidence with failure propagation. Temporary Git
repositories used to test workflow helper behavior must remain inside
`tmp_path` and require no remote.

## Adding or changing functionality

For a new provider, add tests for:

1. configuration and canonical registry name;
2. missing credential/endpoint rejection before a request;
3. exact request contract and unchanged transcript extraction;
4. empty/malformed response rejection;
5. transient retry and permanent failure;
6. timeout and upload limits;
7. secret redaction;
8. participation in one-, two-, and three-provider ensemble roles;
9. provenance model name in candidate/report evidence.

For a chunking or comparison change, add deterministic regression examples and
seeded property-style cases. Any change that can remove a word needs a direct
test proving what exact evidence authorizes that removal.

For a manifest or evidence schema change, test older accepted input where
compatibility is intended, deterministic serialization, idempotent identical
writes, and rejection of conflicting replacement.

## CI

`.github/workflows/ci.yml` is the credential-free validation workflow. It runs
for every pull-request update targeting `v1` and for pushes to that integration
branch, and it can also be started manually on any selected branch through
`workflow_dispatch`. CI tests Python 3.11, 3.12, 3.13, and 3.14, installs
`requirements-dev.txt`, runs the full suite with coverage, and uploads the
Python 3.14 coverage file to Codecov without making Codecov availability a test
gate. A separate explicit job runs the complete suite on Ubuntu 26.04 with
Python 3.14, matching the newest supported deployment target rather than relying
only on `ubuntu-latest`.

`.github/workflows/openai-real-audio.yml` is the separate paid real-audio
workflow and a clean consumer example. It checks out the revision under test,
invokes OctoScribe through `uses: ./`, runs OctoScribe's own reference
verification through `command: verify`, and uploads the manifest, generated
transcripts, candidates, evidence reports, comparison reports, and reference
files as GitHub Actions artifacts. Its workflow file contains no inline Python
implementation of the verifier.

The real-audio workflow can be triggered manually and after a pull request is
merged into `v1`. Manual capture mode is the only operation allowed to produce
bootstrap machine reference files; the capture artifact can then be inspected
and deliberately committed. Merge-triggered validation requires those
references to be committed already and never silently replaces them.
Validation compares each generated transcript with its committed reference
word-for-word after normalizing only case, punctuation, and whitespace, and
reports every added, deleted, or substituted spoken word.

`command: verify` is exact by default: `max_word_error_rate`, the
`MAX_WORD_ERROR_RATE` environment variable, and the
`--max-word-error-rate` CLI option all default to `0`. An explicit action input
takes precedence over the environment variable. A nonzero value does not hide
or rewrite any difference: each JSON report and standard-output record retains
the exact-match flag, word counts, additions, deletions, substitutions, WER,
and the complete word-level diff. Each referenced transcript must independently
meet the configured rate. Missing generated or reference files always fail.
Differences involving a numeric token or a common negation always fail even
when the overall WER is below the configured tolerance.

The paid fixture workflow uses the deliberately narrow value `0.0025` (0.25%)
to detect material model drift while permitting the few known nondeterministic
word variations in the machine reference corpus. Its reports still distinguish
`exact_match`, `mismatch_within_tolerance`, and `mismatch`; passing within
tolerance is never presented as an exact match.

The workflow uses `tests/fixtures/telegram/reference-transcripts/` for committed
references and `tests/fixtures/telegram/comparison-reports/` for machine-readable
diffs. The direct command supplies `--transcript-path`, `--reference-path`, and
`--comparison-report-path` before `verify`; only an intentional manual bootstrap
may add `--allow-missing-references --capture-reference`.

The credentialed operational workflow for consumers remains at
`examples/pipeline.yml`.
`examples/in-repository.yml` covers the common case where the workflow checkout
itself owns preserved audio, the persistent manifest, transcripts, raw
candidates, and reports; it publishes through the built-in `GITHUB_TOKEN`.
`examples/minimal.yml`, `examples/full.yml`, and `examples/pipeline.yml` show
the same three inputs pointing either into the workflow repository or into
caller-managed external checkouts.
The caller workflow owns all Git operations; the OctoScribe action only
acquires, hashes, transcribes, records evidence, and reports progress to
standard output.

The source repository also contains two owner-supplied Telegram OGG/Opus
fixtures under `tests/fixtures/telegram/`. Ordinary CI verifies their historical
metadata and SHA-256 identities, probes their complete durations, derives the
production silence-aware chunk plans, and decodes short materialized windows
without a network call. With the repository secret `OPENAI_API_KEY` present,
the separate real-audio workflow transcribes both complete recordings through
the composite action and validates transcript/evidence links, hashes, and every
recorded chunk seam. Fully aligned seams produce `machine_transcribed`; an
unaligned seam produces the normally published `completed_with_warnings` state.
Neither state withholds the transcript.

Committed reference transcripts and uploaded run outputs are machine reference
material: they make API or model drift visible, but are not automatically
human-verified ground truth. A ground-truth corpus requires a person to listen
to the complete source audio and verify the corresponding text. Accordingly,
the real-audio comparison is a paid transport, repeatability, and regression
test; it must not be represented as measured word-error accuracy unless its
references have separately acquired that human-verified provenance.

CI validates provider contracts offline. For measured word-error evaluation,
separately use a human-verified sermon set covering the actual speakers,
microphones, accents, biblical names, references, and recording conditions.
That optional accuracy evaluation is about audio/text fidelity only; OctoScribe
does not perform theological approval or content moderation. Never commit a
private evaluation corpus or its credentials to this code repository.
