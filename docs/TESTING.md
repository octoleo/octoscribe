# OctoScribe Testing Guide

The test suite treats fidelity, bounded execution, and evidence preservation as
contracts rather than prompt aspirations. Provider and Telegram tests do not
make live network calls or spend API credits. Filesystem tests use temporary
directories, and the one real audio-tool integration test generates its own WAV
file locally.

For the implementation model, see [Architecture](ARCHITECTURE.md).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the core runtime and pytest tooling. It does
not install the optional Faster-Whisper runtime; local backend tests replace
that dependency with a test double.

The real chunk materialization test runs when `ffmpeg` and `ffprobe` are on
`PATH` and skips cleanly otherwise.

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
python -m pytest -k "hash or needs_review"
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
| `test_consensus.py` | Comparison-only normalization, additions/deletions/substitutions, critical negation/number/Scripture detection, and bounded policy decisions. |
| `test_ensemble.py` | One/two/three-provider flows, primary ownership, one retry, one arbiter, provider outages, seam downgrade, source hashes, and evidence persistence. |
| `test_evidence.py` | Schema validation, source/chunk/transcript hashes, deterministic JSON, append-only conflicts, safe filenames, reports, seams, and audio revisions. |
| `test_provider.py` | Provider transcript and timed-word validation plus detailed-backend adaptation. |
| `test_backend_registry.py` | Canonical provider names, ordered lazy construction, model provenance, duplicates, and bad configuration. |
| `test_xai_backend.py` | Pinned endpoint, multipart fidelity options, response parsing, word timestamps, retries, size checks, and secret-safe errors. |
| `test_meta_backend.py` | Explicit ASR URL resolution, HTTPS/loopback rules, multipart model/language, optional bearer token, response formats, retries, and validation. |
| `test_transcribe.py` | OpenAI and optional local adapter contracts, transport retries, empty output, backend compatibility, and batch orchestration. |
| `test_transcriber_ensemble.py` | Batch integration with the ensemble, `needs-review` quarantine, hashes, provider failures, and manifest/report links. |
| `test_transcribe_cmd.py` | Caller-supplied audio revision/branch provenance and Git-independent transcription. |
| `test_transcribe_hardening.py` | Collision-safe output paths and the empty-result failure guard. |
| `test_retry.py` | Transient/permanent error classification, backoff, exhaustion, and hard retry bounds. |
| `test_manifest.py` | Source-hash immutability, quality states, pending queries, human verification, stats, atomic JSON, and thread safety. |
| `test_folder.py` | Recursive/flat scans, supported formats, embedded/technical metadata, content-hash dedup, resume, copy verification, and source preservation. |
| `test_telegram.py`, `test_telegram_client.py` | Audio selection, the historical OGG manifest contract, metadata, download behaviour, entity resolution, session handling, and Telegram retries with Telethon mocked. |
| `test_config.py` | Precedence, provider discovery/order, command-aware secrets, split/shared workspace paths, endpoint rules, path resolution, and numeric bounds. |
| `test_source_cmd.py` | CLI overrides and folder/Telegram source dispatch. |
| `test_ci_export_cmd.py`, `test_session_cmd.py`, `test_debug.py` | CI export guard/redaction, Telegram session commands, and diagnostic behaviour. |
| `test_persistence.py` | Atomic text writes and periodic manifest saves. |
| `test_action_contract.py` | Composite-action input mapping, no-Git boundary, shared/split paths, and caller-supplied provenance. |

## Fidelity properties under test

### Source identity

Tests prove that a stored audio digest cannot be replaced by a different
SHA-256 value, that folder copies are verified, that the ensemble rejects an
unexpected source hash before provider work, and that evidence writing rechecks
source and chunk bytes.

When modifying acquisition or workspace-path code, include at least these cases:

- identical bytes resume without duplication;
- changed bytes under an existing identity fail;
- a failed or corrupt copy does not remain as accepted audio;
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
remain and the overall result becomes `needs_review`.

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
unresolved result remains `needs_review`.

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
- `needs_review` for unresolved disagreement, unavailable independent
  verification, or an unresolved seam;
- `human_verified` only after an explicit human-verification operation.

Also assert the output location: unresolved text belongs below
`transcriptions/needs-review/`.

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
clone, acquire, audio/index checkpoint, revision capture, transcribe, evidence
checkpoint, and failure propagation. Temporary Git repositories used to test
workflow helper behavior must remain inside `tmp_path` and require no remote.

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

`.github/workflows/ci.yml` runs for every pull-request update targeting `v1`,
and for pushes to that integration branch. It can also be started manually on
any selected branch through `workflow_dispatch`. CI uses Python 3.11 and 3.12,
installs `requirements-dev.txt`, runs the full suite with coverage, and uploads
the Python 3.12 coverage file to Codecov without making Codecov availability a
test gate.

CI is the only active workflow in this source repository. The credentialed
operational reference lives at `examples/pipeline.yml` for consumers to copy.
`examples/in-repository.yml` covers the common case where the workflow checkout
itself owns preserved audio, the persistent manifest, transcripts, raw
candidates, and reports; it publishes through the built-in `GITHUB_TOKEN`.
`examples/minimal.yml`, `examples/full.yml`, and `examples/pipeline.yml` retain
the caller-managed external shared/split repository patterns.
The caller workflow owns all Git operations; the OctoScribe action only
acquires, hashes, transcribes, records evidence, and reports progress to
standard output.

The source repository also contains two owner-supplied Telegram OGG/Opus
fixtures under `tests/fixtures/telegram/`. Ordinary CI verifies their historical
metadata and SHA-256 identities, probes their complete durations, derives the
production silence-aware chunk plans, and decodes short materialized windows
without a network call. A separate `openai-live` job is eligible only when a
pull request is merged into `v1`; it is skipped for pull-request updates,
direct pushes, and manual branch runs. With the repository secret
`OPENAI_API_KEY` present, that job transcribes both complete
recordings through the composite action and validates transcript/evidence links,
hashes, and every recorded chunk seam. The published state must match that
evidence: fully aligned seams require `machine_transcribed`, while any unaligned
seam requires `needs_review` and quarantine under `transcriptions/needs-review/`.
The job then invokes the action again and proves that no output changed. This is
a paid transport and end-to-end regression, not a claim of human verification
or measured word-error accuracy.

CI validates provider contracts offline. Before promoting a model or provider
configuration, separately evaluate it on a private, human-verified sermon set
covering the actual speakers, microphones, accents, biblical names, references,
and recording conditions. Never commit that private evaluation audio or its
credentials to this code repository.
