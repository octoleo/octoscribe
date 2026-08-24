"""Bounded, evidence-preserving multi-provider transcription engine."""

from __future__ import annotations

import tempfile
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.audio import sha256_file
from src.config import TranscribeConfig
from src.transcribe.audio_chunks import FFmpegAudioTools, MaterializedChunk
from src.transcribe.backends.base import TranscriptionBackend
from src.transcribe.chunking import (
    ChunkMetadata,
    OverlapAlignment,
    plan_chunks,
    stitch_with_alignment,
)
from src.transcribe.consensus import (
    ConsensusReport,
    QualityState,
    compare_transcripts,
)
from src.transcribe.evidence import (
    AudioEvidence,
    ChunkEvidence,
    ComparisonSummary,
    DiscrepancySummary,
    EvidenceReport,
    EvidenceStore,
    ProviderAttemptEvidence,
    ProviderFailureSummary,
    SeamSummary,
    TimedWordEvidence,
)
from src.transcribe.normalize import normalize_text
from src.transcribe.provider import ProviderTranscript, run_backend


@dataclass(frozen=True, slots=True)
class ProviderAttempt:
    """One independent provider listening pass for a chunk."""

    provider: str
    attempt: int
    role: str
    transcript: ProviderTranscript


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """A provider failure retained in the quality report."""

    provider: str
    attempt: int
    role: str
    error: str


class PrimaryProviderError(RuntimeError):
    """Initial primary failure with any successful peer evidence attached."""

    def __init__(
        self,
        provider: str,
        attempts: tuple[ProviderAttempt, ...],
        failures: tuple[ProviderFailure, ...],
    ) -> None:
        self.provider = provider
        self.attempts = attempts
        self.failures = failures
        failure = next(
            (item for item in failures if item.provider == provider),
            None,
        )
        detail = failure.error if failure else "produced no transcript"
        super().__init__(f"primary provider {provider!r} failed: {detail}")


@dataclass(frozen=True, slots=True)
class AttemptComparison:
    """One comparison tied to the exact attempts that produced it."""

    attempts: tuple[ProviderAttempt, ...]
    report: ConsensusReport
    stage: str


@dataclass(frozen=True, slots=True)
class ChunkOutcome:
    """Canonical primary text plus every independent piece of chunk evidence."""

    chunk: MaterializedChunk
    canonical_text: str
    quality_state: QualityState
    attempts: tuple[ProviderAttempt, ...]
    failures: tuple[ProviderFailure, ...]
    comparison: ConsensusReport | None
    comparison_history: tuple[AttemptComparison, ...] = ()


@dataclass(frozen=True, slots=True)
class SeamEvidence:
    """Deterministic alignment evidence for one adjacent chunk seam."""

    left_chunk: int
    right_chunk: int
    alignment: OverlapAlignment | None


@dataclass(frozen=True, slots=True)
class EnsembleOutcome:
    """Completed machine transcript and its truthful quality classification."""

    text: str
    quality_state: QualityState
    audio_sha256: str
    duration_ms: int
    chunks: tuple[ChunkOutcome, ...]
    seams: tuple[SeamEvidence, ...]
    evidence_report_path: Path | None = None
    candidate_paths: tuple[Path, ...] = ()

    @property
    def unresolved_discrepancies(self) -> int:
        return sum(
            len(chunk.comparison.discrepancies)
            for chunk in self.chunks
            if (
                chunk.quality_state is QualityState.NEEDS_REVIEW
                and chunk.comparison is not None
                and not chunk.comparison.all_agree
            )
        )

    @property
    def provider_failures(self) -> tuple[ProviderFailure, ...]:
        """Return every bounded provider failure in chunk order."""
        return tuple(failure for chunk in self.chunks for failure in chunk.failures)


class EnsembleEngine:
    """Transcribe long audio with one to three ASR providers and a hard stop."""

    def __init__(
        self,
        config: TranscribeConfig,
        backends: Mapping[str, TranscriptionBackend],
        *,
        audio_tools: FFmpegAudioTools | None = None,
        model_names: Mapping[str, str] | None = None,
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        providers = tuple(config.providers) or tuple(backends)
        if not 1 <= len(providers) <= 3:
            raise ValueError("the ensemble requires one, two, or three providers")
        if len(set(providers)) != len(providers):
            raise ValueError("ensemble providers must be unique")
        missing = [name for name in providers if name not in backends]
        if missing:
            raise ValueError(f"missing backend(s): {', '.join(missing)}")
        primary = config.primary_provider or providers[0]
        if primary not in providers:
            raise ValueError("primary provider must be enabled")

        # Keep the configured order, but primary evidence is always the
        # canonical surface and therefore leads comparisons.
        self._providers = (primary,) + tuple(p for p in providers if p != primary)
        self._primary = primary
        self._config = config
        self._backends = dict(backends)
        self._audio_tools = audio_tools or FFmpegAudioTools(
            timeout_seconds=config.provider_timeout_seconds
        )
        self._model_names = dict(model_names or {})
        self._evidence_store = evidence_store

    def transcribe(
        self,
        audio_path: Path,
        *,
        expected_sha256: str | None = None,
        logical_audio_path: str | Path | None = None,
        audio_revision: str | None = None,
        audio_repository_branch: str | None = None,
    ) -> EnsembleOutcome:
        """Transcribe a recording without allowing an unbounded correction loop."""
        path = Path(audio_path)
        actual_hash = sha256_file(path)
        if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
            raise ValueError(
                "audio SHA-256 mismatch; refusing to transcribe changed evidence"
            )

        duration_ms = self._audio_tools.probe_duration_ms(path)
        silences = self._audio_tools.detect_silences(
            path,
            threshold_db=self._config.silence_threshold_db,
            min_silence_ms=self._config.silence_min_ms,
            duration_ms=duration_ms,
        )
        plan = plan_chunks(
            duration_ms,
            silences,
            target_core_ms=self._config.chunk_target_seconds * 1000,
            overlap_ms=self._config.chunk_overlap_seconds * 1000,
            hard_max_ms=self._config.chunk_max_seconds * 1000,
            silence_search_ms=self._config.silence_search_seconds * 1000,
        )

        with tempfile.TemporaryDirectory(prefix="octoscribe-chunks-") as temp_dir:
            materialized = self._audio_tools.materialize(
                path,
                plan,
                Path(temp_dir),
                max_bytes=self._config.max_chunk_megabytes * 1024 * 1024,
            )
            audio_evidence = AudioEvidence(
                path=str(logical_audio_path or path),
                sha256=actual_hash,
                duration_seconds=duration_ms / 1000,
            )
            run_id = uuid.uuid4().hex
            processed: list[ChunkOutcome] = []
            prewritten_candidates: list[Path] = []
            for chunk in materialized:
                try:
                    outcome = self._process_chunk(chunk)
                except PrimaryProviderError as exc:
                    if self._evidence_store is not None and exc.attempts:
                        prewritten_candidates.extend(
                            self._write_candidates(
                                path,
                                audio_evidence,
                                chunk,
                                exc.attempts,
                                run_id,
                            )
                        )
                    raise
                processed.append(outcome)
                if self._evidence_store is not None:
                    prewritten_candidates.extend(
                        self._write_candidates(
                            path,
                            audio_evidence,
                            chunk,
                            outcome.attempts,
                            run_id,
                        )
                    )
            chunk_outcomes = tuple(processed)
            text, seams = self._stitch(chunk_outcomes)
            # The published transcript uses the same whitespace-only canonical
            # surface whose hash is recorded in the evidence report. Raw
            # provider candidates remain byte-for-byte unchanged.
            text = normalize_text(text)
            quality = self._overall_quality(chunk_outcomes, seams)
            report_path: Path | None = None
            candidate_paths: tuple[Path, ...] = ()
            if self._evidence_store is not None:
                report_path, candidate_paths = self._persist_evidence(
                    path,
                    audio_evidence,
                    chunk_outcomes,
                    seams,
                    quality,
                    run_id,
                    text,
                    audio_revision,
                    audio_repository_branch,
                )
                candidate_paths = tuple(
                    dict.fromkeys((*prewritten_candidates, *candidate_paths))
                )

            return EnsembleOutcome(
                text=text,
                quality_state=quality,
                audio_sha256=actual_hash,
                duration_ms=duration_ms,
                chunks=chunk_outcomes,
                seams=seams,
                evidence_report_path=report_path,
                candidate_paths=candidate_paths,
            )

    def _process_chunk(self, chunk: MaterializedChunk) -> ChunkOutcome:
        base_names = self._providers[:2]
        arbiter = self._providers[2] if len(self._providers) == 3 else None
        attempts: list[ProviderAttempt] = []
        failures: list[ProviderFailure] = []
        comparison_history: list[AttemptComparison] = []
        arbiter_consumed = False

        initial, initial_failures = self._listen(
            base_names, chunk.path, attempt=1, role="initial"
        )
        attempts.extend(initial)
        failures.extend(initial_failures)
        try:
            primary_attempt = self._require_primary(initial)
        except RuntimeError as exc:
            raise PrimaryProviderError(
                self._primary,
                tuple(attempts),
                tuple(failures),
            ) from exc

        # A one-provider configuration has nothing independent to compare.
        if len(base_names) == 1:
            return ChunkOutcome(
                chunk=chunk,
                canonical_text=primary_attempt.transcript.text,
                quality_state=QualityState.MACHINE_TRANSCRIBED,
                attempts=tuple(attempts),
                failures=tuple(failures),
                comparison=None,
                comparison_history=(),
            )

        latest_by_provider = {item.provider: item for item in initial}
        report: ConsensusReport | None = None
        if len(initial) == len(base_names):
            report = self._compare(initial)
            comparison_history.append(
                AttemptComparison(tuple(initial), report, "initial")
            )
            if report.all_agree:
                return self._chunk_result(
                    chunk,
                    primary_attempt,
                    report,
                    attempts,
                    failures,
                    comparison_history,
                    primary_verified=True,
                )

        secondary_available = base_names[1] in latest_by_provider

        # If the configured checker is unavailable, use the third provider as
        # the active checker immediately instead of paying to retry a known
        # outage.  Only this active pair is retried if it actually disagrees.
        if not secondary_available and arbiter and self._config.arbitration_limit:
            # The third provider is a one-shot resource.  Count the fallback
            # invocation even when it fails so a later availability retry
            # cannot silently invoke the same provider again as an arbiter.
            arbiter_consumed = True
            fallback, fallback_failures = self._listen(
                (arbiter,), chunk.path, attempt=1, role="fallback_checker"
            )
            attempts.extend(fallback)
            failures.extend(fallback_failures)
            if fallback:
                active_names = (self._primary, arbiter)
                active_latest = {
                    self._primary: primary_attempt,
                    arbiter: fallback[0],
                }
                active = [active_latest[name] for name in active_names]
                report = self._compare(active)
                comparison_history.append(
                    AttemptComparison(tuple(active), report, "fallback_checker")
                )
                if report.all_agree:
                    return self._chunk_result(
                        chunk,
                        primary_attempt,
                        report,
                        attempts,
                        failures,
                        comparison_history,
                        primary_verified=True,
                    )
                if self._config.disagreement_retry_limit:
                    retried, retry_failures = self._listen(
                        active_names,
                        chunk.path,
                        attempt=2,
                        role="discrepancy_retry",
                    )
                    attempts.extend(retried)
                    failures.extend(retry_failures)
                    active_latest.update(
                        {item.provider: item for item in retried}
                    )
                    if any(item.provider == self._primary for item in retried):
                        primary_attempt = self._require_primary(retried)
                    active = [active_latest[name] for name in active_names]
                    report = self._compare(active)
                    comparison_history.append(
                        AttemptComparison(
                            tuple(active), report, "discrepancy_retry"
                        )
                    )
                    if report.all_agree:
                        return self._chunk_result(
                            chunk,
                            primary_attempt,
                            report,
                            attempts,
                            failures,
                            comparison_history,
                            primary_verified=True,
                        )
                return ChunkOutcome(
                    chunk=chunk,
                    canonical_text=primary_attempt.transcript.text,
                    quality_state=QualityState.NEEDS_REVIEW,
                    attempts=tuple(attempts),
                    failures=tuple(failures),
                    comparison=report,
                    comparison_history=tuple(comparison_history),
                )

        # Normal base-pair disagreement, or availability retry when the third
        # provider could not replace an unavailable checker.
        if self._config.disagreement_retry_limit:
            retry_stage = (
                "discrepancy_retry" if secondary_available else "availability_retry"
            )
            retried, retry_failures = self._listen(
                base_names, chunk.path, attempt=2, role=retry_stage
            )
            attempts.extend(retried)
            failures.extend(retry_failures)
            if any(item.provider == self._primary for item in retried):
                primary_attempt = self._require_primary(retried)
            latest_by_provider.update({item.provider: item for item in retried})
            latest_base = [
                latest_by_provider[name]
                for name in base_names
                if name in latest_by_provider
            ]
            if len(latest_base) == len(base_names):
                report = self._compare(latest_base)
                comparison_history.append(
                    AttemptComparison(tuple(latest_base), report, retry_stage)
                )
                if report.all_agree:
                    return self._chunk_result(
                        chunk,
                        primary_attempt,
                        report,
                        attempts,
                        failures,
                        comparison_history,
                        primary_verified=True,
                    )

        # After the one semantic retry, the third provider is a one-shot
        # arbiter.  A majority validates only when it includes the primary.
        latest_base = [
            latest_by_provider[name]
            for name in base_names
            if name in latest_by_provider
        ]
        if (
            arbiter
            and self._config.arbitration_limit
            and not arbiter_consumed
            and len(latest_base) == len(base_names)
        ):
            arbitrated, arbiter_failures = self._listen(
                (arbiter,), chunk.path, attempt=1, role="arbiter"
            )
            attempts.extend(arbitrated)
            failures.extend(arbiter_failures)
            if arbitrated:
                report = self._compare((*latest_base, *arbitrated))
                comparison_history.append(
                    AttemptComparison(
                        tuple((*latest_base, *arbitrated)),
                        report,
                        "arbitration",
                    )
                )
                if self._primary_has_agreeing_peer(report):
                    return self._chunk_result(
                        chunk,
                        primary_attempt,
                        report,
                        attempts,
                        failures,
                        comparison_history,
                        primary_verified=True,
                    )

        return ChunkOutcome(
            chunk=chunk,
            canonical_text=primary_attempt.transcript.text,
            quality_state=QualityState.NEEDS_REVIEW,
            attempts=tuple(attempts),
            failures=tuple(failures),
            comparison=report,
            comparison_history=tuple(comparison_history),
        )

    def _listen(
        self,
        providers: tuple[str, ...],
        audio_path: Path,
        *,
        attempt: int,
        role: str,
    ) -> tuple[list[ProviderAttempt], list[ProviderFailure]]:
        successes: dict[str, ProviderAttempt] = {}
        failures: dict[str, ProviderFailure] = {}

        def listen(name: str) -> ProviderAttempt:
            transcript = run_backend(
                self._backends[name],
                audio_path,
                model=self._model_names.get(name, ""),
                provider=name,
            )
            return ProviderAttempt(name, attempt, role, transcript)

        workers = max(1, min(len(providers), self._config.workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_names = {executor.submit(listen, name): name for name in providers}
            for future in as_completed(future_names):
                name = future_names[future]
                try:
                    successes[name] = future.result()
                except Exception as exc:
                    failures[name] = ProviderFailure(
                        name,
                        attempt,
                        role,
                        self._safe_failure(exc),
                    )

        ordered_successes = [successes[name] for name in providers if name in successes]
        ordered_failures = [failures[name] for name in providers if name in failures]
        return ordered_successes, ordered_failures

    def _safe_failure(self, exc: Exception) -> str:
        """Return bounded diagnostics with configured credentials redacted."""
        message = f"{type(exc).__name__}: {exc}"
        for secret in (
            getattr(self._config, "api_key", None),
            getattr(self._config, "xai_api_key", None),
            getattr(self._config, "meta_asr_api_key", None),
        ):
            if secret:
                message = message.replace(str(secret), "***")
        return " ".join(message.split())[:500]

    def _require_primary(self, attempts: list[ProviderAttempt]) -> ProviderAttempt:
        for attempt in attempts:
            if attempt.provider == self._primary:
                return attempt
        raise RuntimeError(f"primary provider {self._primary!r} produced no transcript")

    @staticmethod
    def _compare(attempts: tuple[ProviderAttempt, ...] | list[ProviderAttempt]) -> ConsensusReport:
        return compare_transcripts(
            [item.transcript.text for item in attempts],
            labels=[item.provider for item in attempts],
        )

    @staticmethod
    def _chunk_result(
        chunk: MaterializedChunk,
        primary: ProviderAttempt,
        report: ConsensusReport,
        attempts: list[ProviderAttempt],
        failures: list[ProviderFailure],
        comparison_history: list[AttemptComparison],
        *,
        primary_verified: bool,
    ) -> ChunkOutcome:
        return ChunkOutcome(
            chunk=chunk,
            canonical_text=primary.transcript.text,
            # Historical outages remain in evidence but do not invalidate a
            # later independent provider's successful verification.
            quality_state=(
                QualityState.CROSS_CHECKED
                if primary_verified
                else QualityState.NEEDS_REVIEW
            ),
            attempts=tuple(attempts),
            failures=tuple(failures),
            comparison=report,
            comparison_history=tuple(comparison_history),
        )

    @staticmethod
    def _primary_has_agreeing_peer(report: ConsensusReport) -> bool:
        """Return whether the first (canonical primary) transcript has support."""
        for comparison in report.comparisons:
            if comparison.left_index != 0 and comparison.right_index != 0:
                continue
            if comparison.agrees:
                return True
        return False

    def _stitch(
        self, chunks: tuple[ChunkOutcome, ...]
    ) -> tuple[str, tuple[SeamEvidence, ...]]:
        if not chunks:
            raise RuntimeError("chunk planner produced no audio chunks")
        text = chunks[0].canonical_text
        seams: list[SeamEvidence] = []
        for left, right in zip(chunks, chunks[1:]):
            result = stitch_with_alignment(text, right.canonical_text)
            seams.append(
                SeamEvidence(
                    left_chunk=left.chunk.metadata.index,
                    right_chunk=right.chunk.metadata.index,
                    alignment=result.alignment,
                )
            )
            text = result.text
        return text, tuple(seams)

    def _overall_quality(
        self,
        chunks: tuple[ChunkOutcome, ...],
        seams: tuple[SeamEvidence, ...],
    ) -> QualityState:
        if any(chunk.quality_state is QualityState.NEEDS_REVIEW for chunk in chunks):
            return QualityState.NEEDS_REVIEW
        if self._config.chunk_overlap_seconds and any(
            seam.alignment is None for seam in seams
        ):
            return QualityState.NEEDS_REVIEW
        if len(self._providers) == 1:
            return QualityState.MACHINE_TRANSCRIBED
        return QualityState.CROSS_CHECKED

    def _persist_evidence(
        self,
        audio_file: Path,
        audio: AudioEvidence,
        chunks: tuple[ChunkOutcome, ...],
        seams: tuple[SeamEvidence, ...],
        quality: QualityState,
        run_id: str,
        final_text: str,
        audio_revision: str | None,
        audio_repository_branch: str | None,
    ) -> tuple[Path, tuple[Path, ...]]:
        assert self._evidence_store is not None
        chunk_evidence: list[ChunkEvidence] = []
        candidate_paths: list[Path] = []
        comparisons: list[ComparisonSummary] = []
        provider_failures: list[ProviderFailureSummary] = []

        for outcome in chunks:
            chunk = self._chunk_evidence(
                audio,
                outcome.chunk,
                outcome.attempts,
                canonical=max(
                    (
                        item
                        for item in outcome.attempts
                        if item.provider == self._primary
                        and item.transcript.text == outcome.canonical_text
                    ),
                    key=lambda item: item.attempt,
                ),
            )
            attempts = chunk.attempts
            metadata = outcome.chunk.metadata
            chunk_evidence.append(chunk)
            provider_failures.extend(
                ProviderFailureSummary(
                    chunk_index=metadata.index,
                    provider=failure.provider,
                    attempt=failure.attempt,
                    role=failure.role,
                    error=failure.error,
                )
                for failure in outcome.failures
            )
            for attempt in attempts:
                candidate_paths.append(
                    self._evidence_store.write_candidate(
                        audio,
                        chunk,
                        attempt,
                        audio_file=audio_file,
                        chunk_file=outcome.chunk.path,
                        run_id=run_id,
                    )
                )

            for pass_index, comparison_pass in enumerate(
                outcome.comparison_history
            ):
                by_provider = {
                    item.provider: item for item in comparison_pass.attempts
                }
                for pair in comparison_pass.report.comparisons:
                    left_provider = comparison_pass.report.transcripts[
                        pair.left_index
                    ].label
                    right_provider = comparison_pass.report.transcripts[
                        pair.right_index
                    ].label
                    left = by_provider[left_provider]
                    right = by_provider[right_provider]
                    kinds = [item.kind.value for item in pair.discrepancies]
                    comparisons.append(
                        ComparisonSummary(
                            chunk_index=metadata.index,
                            left_provider=left_provider,
                            left_model=(
                                left.transcript.model
                                or self._model_names.get(left_provider)
                                or left_provider
                            ),
                            left_attempt=left.attempt,
                            right_provider=right_provider,
                            right_model=(
                                right.transcript.model
                                or self._model_names.get(right_provider)
                                or right_provider
                            ),
                            right_attempt=right.attempt,
                            agrees=pair.agrees,
                            discrepancy_count=len(pair.discrepancies),
                            critical_discrepancy_count=sum(
                                item.is_critical for item in pair.discrepancies
                            ),
                            additions=kinds.count("addition"),
                            deletions=kinds.count("deletion"),
                            substitutions=kinds.count("substitution"),
                            stage=comparison_pass.stage,
                            pass_index=pass_index,
                            discrepancies=tuple(
                                DiscrepancySummary(
                                    kind=item.kind.value,
                                    priority=item.priority.value,
                                    left_start=item.left_span.start,
                                    left_end=item.left_span.end,
                                    right_start=item.right_span.start,
                                    right_end=item.right_span.end,
                                    left_tokens=item.left_tokens,
                                    right_tokens=item.right_tokens,
                                    critical_categories=tuple(
                                        sorted(
                                            {
                                                hit.category.value
                                                for hit in item.critical_hits
                                            }
                                        )
                                    ),
                                )
                                for item in pair.discrepancies
                            ),
                        )
                    )

        seam_summaries = tuple(
            SeamSummary(
                left_chunk=seam.left_chunk,
                right_chunk=seam.right_chunk,
                aligned=seam.alignment is not None,
                duplicate_tokens_removed=(
                    seam.alignment.continuation_tokens if seam.alignment else 0
                ),
                exact_matches=(seam.alignment.exact_matches if seam.alignment else 0),
                edits=(seam.alignment.edits if seam.alignment else 0),
                similarity=(seam.alignment.similarity if seam.alignment else None),
            )
            for seam in seams
        )
        report = EvidenceReport(
            audio=audio,
            chunks=tuple(chunk_evidence),
            comparisons=tuple(comparisons),
            final_quality_state=quality,
            seams=seam_summaries,
            failures=tuple(provider_failures),
            primary_provider=self._primary,
            final_transcript_sha256=hashlib.sha256(
                final_text.encode("utf-8")
            ).hexdigest(),
            run_id=run_id,
            audio_revision=audio_revision,
            audio_repository_branch=audio_repository_branch,
        )
        report_path = self._evidence_store.write_report(
            report,
            audio_file=audio_file,
            chunk_files={item.chunk.metadata.index: item.chunk.path for item in chunks},
        )
        return report_path, tuple(candidate_paths)

    def _chunk_evidence(
        self,
        audio: AudioEvidence,
        chunk: MaterializedChunk,
        attempts: tuple[ProviderAttempt, ...],
        canonical: ProviderAttempt | None = None,
    ) -> ChunkEvidence:
        """Build immutable evidence for the exact bytes all providers heard."""
        metadata = chunk.metadata
        attempt_evidence = tuple(
            ProviderAttemptEvidence(
                provider=item.provider,
                model=(
                    item.transcript.model
                    or self._model_names.get(item.provider)
                    or item.provider
                ),
                attempt=item.attempt,
                raw_transcript=item.transcript.text,
                words=tuple(
                    TimedWordEvidence(
                        text=word.text,
                        start_seconds=word.start_seconds,
                        end_seconds=word.end_seconds,
                        confidence=word.confidence,
                        speaker=word.speaker,
                    )
                    for word in item.transcript.words
                ),
                language=item.transcript.language,
                duration_seconds=item.transcript.duration_seconds,
            )
            for item in attempts
        )
        return ChunkEvidence(
            index=metadata.index,
            path=f"derived/{audio.sha256}/chunk-{metadata.index:04d}.wav",
            start_seconds=metadata.context_start_ms / 1000,
            end_seconds=metadata.context_end_ms / 1000,
            sha256=chunk.sha256,
            attempts=attempt_evidence,
            canonical_provider=(canonical.provider if canonical else None),
            canonical_attempt=(canonical.attempt if canonical else None),
        )

    def _write_candidates(
        self,
        audio_file: Path,
        audio: AudioEvidence,
        materialized: MaterializedChunk,
        attempts: tuple[ProviderAttempt, ...],
        run_id: str,
    ) -> tuple[Path, ...]:
        """Persist paid provider outputs immediately, before later chunks run."""
        assert self._evidence_store is not None
        chunk = self._chunk_evidence(audio, materialized, attempts)
        return tuple(
            self._evidence_store.write_candidate(
                audio,
                chunk,
                attempt,
                audio_file=audio_file,
                chunk_file=materialized.path,
                run_id=run_id,
            )
            for attempt in chunk.attempts
        )


__all__ = [
    "ChunkOutcome",
    "AttemptComparison",
    "EnsembleEngine",
    "EnsembleOutcome",
    "ProviderAttempt",
    "ProviderFailure",
    "PrimaryProviderError",
    "SeamEvidence",
]
