"""Static contract checks for the workflow-facing composite action."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ACTION = (ROOT / "action.yml").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
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
    assert CI.count("uses: ./") == 2
    assert "tests/action_smoke_server.py" in CI
    assert "command: download" in CI
    assert "command: transcribe" in CI
    assert "providers: meta" in CI
    assert "octoscribe-command-shim" in CI
    assert "exit 97" in CI
    assert 'assert not list(base.rglob(".git"))' in CI
    assert 'result["audio_revision"]' in CI
    assert 'report["final_transcript_sha256"]' in CI


def test_workflows_and_examples_are_valid_yaml() -> None:
    documents = (
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / ".github" / "workflows" / "pipeline.yml",
        *EXAMPLES,
    )
    assert {path.name for path in EXAMPLES} == {"full.yml", "minimal.yml"}
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
