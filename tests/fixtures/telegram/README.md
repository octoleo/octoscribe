# Telegram OGG integration fixtures

These two OGG/Opus recordings and their acquisition manifest were supplied by
the repository owner for OctoScribe testing. They are retained byte-for-byte;
tests verify their SHA-256 values before any processing.

| File | SHA-256 | Container duration |
| --- | --- | --- |
| `1 Timothy 15-6.ogg` | `7b2bc9c89b96cc528cd3f63a88e63710e2128516dae1079f4fabf8414cd8b060` | 1411.093 s |
| `1 John 17-8.ogg` | `698056c50804e1033b1c68adcbf7d4064c32f112aaa6aba23f0fbe524472849a` | 1849.253 s |

The Telegram durations in `manifest.json` are preserved exactly as reported by
Telegram and may differ by about one second from the container duration that
ffprobe reports.

The fixture workflow uses the normal three-path contract:

```text
AUDIO_PATH=tests/fixtures/telegram/audio
TRANSCRIPT_PATH=tests/fixtures/telegram/transcriptions
MANIFEST_PATH=tests/fixtures/telegram/manifest.json
```

The reference and comparison paths are the derived siblings
`reference-transcripts/` and `comparison-reports/` under the same fixture root.

Normal CI probes the complete durations, derives the real silence-aware chunk
plans, and decodes short windows without sending audio anywhere. The separate
`.github/workflows/openai-real-audio.yml` workflow requires the repository
secret `OPENAI_API_KEY`, transcribes both complete recordings through
`uses: ./`, runs OctoScribe's `command: verify` path, and uploads the generated
manifest, transcripts, candidates, evidence reports, comparison reports, and
reference files as workflow artifacts. Committed references live under
`reference-transcripts/`; machine-readable added/deleted/substituted-word
reports are written under `comparison-reports/`.

Generated and committed reference transcripts use OctoScribe's
sentence-per-line publication format. That formatter runs after deterministic
chunk stitching and changes whitespace only; it does not add, remove, correct,
or reinterpret any spoken words or provider punctuation. Formatting occurs
before the final transcript SHA-256 is written to the evidence report and
manifest, so those hashes match the exact published bytes. Raw provider
candidates remain unchanged.

Manual capture mode is the only way that workflow produces bootstrap machine
reference files; its artifact can be inspected and deliberately committed. A
run after merge into `v1` requires those committed references and compares
generated text with them word-for-word, normalizing only case, punctuation, and
whitespace. Added, deleted, and substituted spoken words are reported
explicitly. Fully aligned seams publish as
`machine_transcribed`; any unaligned seam publishes normally as
`completed_with_warnings`. Both results remain under `transcriptions/`; neither
is withheld or moved to a separate output area.

The verifier itself remains strict by default (`max_word_error_rate: 0`). This
paid regression workflow explicitly uses a narrow `0.0025` (0.25%) ceiling for
each transcript because API output can vary slightly from its machine-generated
reference. Reports still record `exact_spoken_word_match`, exact word/error
counts, the WER, and every individual difference. Numeric-token and common
negation differences fail regardless of the configured ceiling, as do missing
generated or reference files. Tolerance changes only the workflow pass/fail
gate; it never changes, suppresses, or rewrites transcript text.

The direct verifier command supplies `--transcript-path`, `--reference-path`,
and `--comparison-report-path` before `verify`.
`--allow-missing-references --capture-reference` is limited to intentional
manual bootstrap and is not used by merge-triggered validation.

The committed references and uploaded outputs are machine reference
transcripts, not automatically human-verified ground truth. They prove real
transport, chunking, conservative seaming, publication, evidence, drift
detection, and repeat-run consistency. Word-error accuracy can be claimed only
for a reference that a person separately checked while listening to the
complete source audio. No quality state performs theological judgment, content
moderation, or approval of the Christian material.

The real-audio fixture configures OpenAI and therefore follows the API-only
path; it must not install the optional Faster-Whisper runtime. In general that
runtime is installed only when `whisper` is explicitly selected or when a
transcription run has no configured hosted provider and uses the local
fallback.
