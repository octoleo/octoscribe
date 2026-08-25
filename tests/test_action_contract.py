"""Static contract checks for the workflow-facing composite action."""

from __future__ import annotations

import hashlib
import re
import runpy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ACTION = (ROOT / "action.yml").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
LIVE_VERIFIER = ROOT / "tests" / "openai_live_verifier.py"
PIPELINE = (ROOT / "examples" / "pipeline.yml").read_text(
    encoding="utf-8"
)
EXAMPLES = tuple(sorted((ROOT / "examples").glob("*.yml")))


def _top_level_section(document: str, name: str, next_name: str) -> str:
    return document.split(f"\n{name}:\n", 1)[1].split(f"\n{next_name}:\n", 1)[0]


def _input_stanza(name: str) -> str:
    inputs = _top_level_section("\n" + ACTION, "inputs", "outputs")
    marker = f"  {name}:\n"
    assert marker in inputs, f"missing action input: {name}"
    remainder = inputs.split(marker, 1)[1]
    next_input = re.search(r"\n  [a-z][a-z0-9_]*:\n", remainder)
    return remainder[: next_input.start()] if next_input else remainder


def test_action_path_layout_is_explicit_and_unambiguous() -> None:
    for name in ("audio_repo_path", "transcript_repo_path", "data_repo_path"):
        assert "default: ''" in _input_stanza(name)

    assert "Supply data_repo_path, or supply both audio_repo_path" in ACTION
    assert "Use data_repo_path alone" in ACTION
    assert "Path(audio_path).resolve() == Path(text_path).resolve()" in ACTION
    assert "use data_repo_path for a shared layout" in ACTION


def test_action_exposes_complete_provider_and_provenance_controls() -> None:
    expected = {
        "openai_api_key",
        "xai_api_key",
        "xai_base_url",
        "meta_asr_url",
        "meta_asr_api_key",
        "meta_asr_model",
        "meta_asr_language",
        "providers",
        "primary_provider",
        "transcribe_model",
        "transcribe_language",
        "local_model",
        "local_device",
        "local_compute_type",
        "audio_revision",
        "audio_repository_branch",
    }
    for name in expected:
        _input_stanza(name)

    assert "XAI_STT_URL: ${{ inputs.xai_base_url }}" in ACTION
    assert "TRANSCRIBE_MODEL: ${{ inputs.transcribe_model }}" in ACTION
    assert "TRANSCRIBE_LANGUAGE: ${{ inputs.transcribe_language }}" in ACTION
    assert "--audio-revision" in ACTION
    assert "--audio-repository-branch" in ACTION


def test_action_supports_only_processing_commands_and_performs_no_git() -> None:
    assert "run|download|transcribe|status" in ACTION
    assert "run|download|transcribe|sync|status" not in ACTION
    assert "no_push" not in ACTION
    assert "--no-push" not in ACTION

    # A whole-word match avoids the harmless ``github.action_path`` context.
    assert re.search(r"(?i)\bgit\b", ACTION) is None
    for operation in ("rev-parse", "ls-files", "protect_evidence_repo"):
        assert operation not in ACTION


def test_action_does_not_print_credentials_or_sessions() -> None:
    logging_lines = [
        line
        for line in ACTION.splitlines()
        if re.search(r"\b(?:echo|printf)\b", line)
    ]
    rendered_logging = "\n".join(logging_lines)
    for secret_variable in (
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "META_ASR_API_KEY",
        "INPUT_TELEGRAM_API_HASH",
        "INPUT_TELEGRAM_PHONE",
        "INPUT_TELEGRAM_SESSION_B64",
    ):
        assert secret_variable not in rendered_logging


def test_action_exposes_resolved_artifact_paths() -> None:
    outputs = _top_level_section("\n" + ACTION, "outputs", "runs")
    for name in (
        "audio_dir",
        "manifest_file",
        "transcriptions_dir",
        "candidates_dir",
        "reports_dir",
    ):
        assert f"  {name}:\n" in outputs
        assert f"steps.process.outputs.{name}" in outputs
    assert 'handle.write(f"{name}={rendered}' in ACTION


def test_ci_executes_the_real_action_offline_in_both_layouts() -> None:
    assert "layout: [split, shared]" in CI
    assert CI.count("uses: ./") == 4
    assert "tests/action_smoke_server.py" in CI
    assert "command: download" in CI
    assert "command: transcribe" in CI
    assert "providers: meta" in CI
    assert "octoscribe-command-shim" in CI
    assert "exit 97" in CI
    assert 'assert not list(base.rglob(".git"))' in CI
    assert 'result["audio_revision"]' in CI
    assert 'report["final_transcript_sha256"]' in CI


def test_paid_openai_test_runs_only_after_a_pr_merge_to_v1() -> None:
    assert "types: [opened, synchronize, reopened, closed]" in CI
    live_job = CI.split("\n  openai-live:\n", 1)[1]
    assert "github.event_name == 'pull_request'" in live_job
    assert "github.event.action == 'closed'" in live_job
    assert "github.event.pull_request.merged == true" in live_job
    assert "github.event.pull_request.base.ref == 'v1'" in live_job
    assert "github.event_name == 'push'" not in live_job
    assert "needs: [test, action-smoke]" in live_job
    assert "secrets.OPENAI_API_KEY" in live_job
    assert "OPENAI_API_KEY must be configured as a repository secret" in live_job
    assert live_job.count("uses: ./") == 2
    assert "tests/openai_live_verifier.py prepare" in live_job
    assert "tests/openai_live_verifier.py verify" in live_job
    assert "tests/openai_live_verifier.py assert-idempotent" in live_job
    assert "tests/fixtures/telegram" in live_job
    assert "providers: openai" in live_job
    assert "primary_provider: openai" in live_job
    assert "transcribe_model: gpt-transcribe" in live_job
    assert "ref: ${{ github.event.pull_request.merge_commit_sha }}" in live_job
    assert "audio_revision: ${{ github.event.pull_request.merge_commit_sha }}" in live_job
    assert "audio_repository_branch: ${{ github.event.pull_request.base.ref }}" in live_job
    assert "Require machine-transcribed outputs and aligned seams" in live_job


def test_paid_openai_verifier_rejects_review_state_and_unaligned_seams() -> None:
    verifier = runpy.run_path(str(LIVE_VERIFIER))

    verifier["_require_release_quality_state"]("856", "machine_transcribed")
    with pytest.raises(AssertionError, match="must finish 'machine_transcribed'"):
        verifier["_require_release_quality_state"]("856", "needs_review")

    raw = "the exact overlap words are retained here"
    transcript_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    expected = {
        "filename": "clear-fixture.ogg",
        "hash": "a" * 64,
        "container_duration_ms": 100_000,
        "expected_chunks": 2,
    }
    chunks = []
    for index, (start, end) in enumerate(((0.0, 60.0), (48.0, 100.0))):
        chunks.append(
            {
                "index": index,
                "start_seconds": start,
                "end_seconds": end,
                "sha256": str(index) * 64,
                "attempts": [
                    {
                        "provider": "openai",
                        "model": "gpt-transcribe",
                        "attempt": 1,
                        "raw_transcript": raw,
                        "transcript_sha256": transcript_hash,
                    }
                ],
                "canonical": {"provider": "openai", "attempt": 1},
            }
        )
    report = {
        "chunks": chunks,
        "seams": [{"left_chunk": 0, "right_chunk": 1, "aligned": True}],
    }
    verifier["_validate_report_chunks"](report, expected)

    report["seams"][0]["aligned"] = False
    with pytest.raises(AssertionError, match="every overlap to align"):
        verifier["_validate_report_chunks"](report, expected)


def test_paid_openai_idempotence_pass_cannot_reach_the_cloud() -> None:
    live_job = CI.split("\n  openai-live:\n", 1)[1]
    rerun = live_job.split(
        "Re-run with cloud access disabled (completed work must be skipped)", 1
    )[1]
    assert "OPENAI_BASE_URL: http://127.0.0.1:9/v1" in rerun
    assert "openai_api_key: second-pass-must-not-call-openai" in rerun
    assert "Prove the rerun preserved every persistent byte" in rerun


def test_live_verifier_prepares_exact_owner_supplied_fixture(tmp_path: Path) -> None:
    verifier = runpy.run_path(str(LIVE_VERIFIER))
    fixture = ROOT / "tests" / "fixtures" / "telegram"
    workspace = tmp_path / "data"
    verifier["prepare_workspace"](fixture, workspace)

    source_snapshot = verifier["content_snapshot"](fixture)
    prepared_snapshot = verifier["content_snapshot"](workspace)
    assert prepared_snapshot == {
        path: details
        for path, details in source_snapshot.items()
        if path == "manifest.json" or path.startswith("audio/")
    }

    snapshot_path = tmp_path / "snapshot.json"
    verifier["write_snapshot"](workspace, snapshot_path)
    verifier["assert_idempotent"](workspace, snapshot_path)
    (workspace / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="idempotent rerun changed"):
        verifier["assert_idempotent"](workspace, snapshot_path)


def test_ci_is_the_only_active_workflow_and_has_complete_triggers() -> None:
    active = tuple(sorted((ROOT / ".github" / "workflows").glob("*.yml")))
    assert [path.name for path in active] == ["ci.yml"]
    assert "workflow_dispatch:" in CI
    assert "pull_request:" in CI
    assert CI.count("branches: [v1]") == 2
    assert "permissions:\n  contents: read" in CI
    assert "branches: [main" not in CI
    assert "feature/**" not in CI


def test_operational_pipeline_is_an_inactive_consumer_example() -> None:
    assert "name: OctoScribe Pipeline" in PIPELINE
    assert "workflow_dispatch:" in PIPELINE
    assert "schedule:" in PIPELINE
    assert "uses: octoleo/octoscribe@v1" in PIPELINE
    assert "uses: ./.github/workflows/ci.yml" not in PIPELINE


def test_workflows_and_examples_are_valid_yaml() -> None:
    documents = (
        ROOT / ".github" / "workflows" / "ci.yml",
        *EXAMPLES,
    )
    assert {path.name for path in EXAMPLES} == {
        "full.yml",
        "in-repository.yml",
        "minimal.yml",
        "pipeline.yml",
    }
    for path in documents:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), f"{path} must contain one YAML mapping"


def test_examples_pass_only_declared_octoscribe_inputs() -> None:
    inputs = _top_level_section("\n" + ACTION, "inputs", "outputs")
    declared = set(re.findall(r"^  ([a-z][a-z0-9_]*):$", inputs, re.MULTILINE))

    for path in EXAMPLES:
        lines = path.read_text(encoding="utf-8").splitlines()
        calls = 0
        for index, line in enumerate(lines):
            if "uses: octoleo/octoscribe@" not in line:
                continue
            calls += 1
            assert index + 1 < len(lines) and lines[index + 1].strip() == "with:"
            supplied: set[str] = set()
            for candidate in lines[index + 2 :]:
                if not candidate.startswith("          "):
                    break
                match = re.match(r"^          ([a-z][a-z0-9_]*):", candidate)
                if match:
                    supplied.add(match.group(1))
            assert supplied <= declared, (
                f"{path} supplies undeclared inputs: {sorted(supplied - declared)}"
            )
        assert calls, f"{path} must invoke OctoScribe"


def test_full_example_preserves_the_fidelity_checkpoint_order() -> None:
    full = (ROOT / "examples" / "full.yml").read_text(encoding="utf-8")
    ordered_markers = (
        "uses: octoleo/git-user@v2",
        "command: download",
        "Publish split audio checkpoint",
        "Capture exact audio provenance",
        "Publish split manifest checkpoint",
        "command: transcribe",
        "audio_revision: ${{ steps.audio.outputs.revision }}",
        "Publish transcript and evidence results",
    )
    offsets = [full.index(marker) for marker in ordered_markers]
    assert offsets == sorted(offsets)
    assert "continue-on-error: true" in full
    assert "steps.transcribe.outcome" in full


def test_in_repository_example_uses_one_checkout_and_builtin_auth() -> None:
    example = (ROOT / "examples" / "in-repository.yml").read_text(
        encoding="utf-8"
    )
    assert "permissions:\n  contents: write" in example
    assert "uses: actions/checkout@v4" in example
    assert "uses: octoleo/git-user" not in example
    assert "GITHUB_TOKEN" in example
    assert "github-actions[bot]" in example
    assert example.count("data_repo_path: .") == 2
    assert "audio_repo_path:" not in example
    assert "transcript_repo_path:" not in example
    for evidence_path in (
        "audio/",
        "manifest.json",
        "transcriptions/",
        "candidates/",
        "reports/",
    ):
        assert evidence_path in example
    ordered_markers = (
        "Download or import audio into this repository",
        "Publish audio and index checkpoint",
        "Capture exact audio provenance",
        "Transcribe committed audio in this repository",
        "Publish transcript, evidence, and updated index",
    )
    offsets = [example.index(marker) for marker in ordered_markers]
    assert offsets == sorted(offsets)


def test_minimal_example_uses_read_only_checkout_credentials() -> None:
    example = (ROOT / "examples" / "minimal.yml").read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in example
    assert "persist-credentials: false" in example
