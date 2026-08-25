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

Normal CI probes the complete durations, derives the real silence-aware chunk
plans, and decodes short windows without sending audio anywhere. The paid
OpenAI integration job is eligible only after a pull request is merged into
`v1`, and requires the repository secret `OPENAI_API_KEY`. It transcribes both
complete recordings and then reruns the same action to prove idempotence. The
post-merge release regression requires each result to agree with its recorded
seam evidence: fully aligned seams publish as `machine_transcribed`; any
unaligned seam must publish as `needs_review` under the quarantine directory.
The fixtures do not include human-verified reference transcripts, so that live
job proves transport, chunking, conservative seaming, publication, evidence,
and repeat-run consistency—not word-error accuracy.
