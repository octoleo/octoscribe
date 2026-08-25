"""Static contract checks for the workflow-facing composite action."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ACTION = (ROOT / "action.yml").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
REAL_AUDIO = (ROOT / ".github" / "workflows" / "openai-real-audio.yml").read_text(
    encoding="utf-8"
)
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


def _action_steps() -> list[dict]:
    document = yaml.safe_load(ACTION)
    steps = document["runs"]["steps"]
    assert isinstance(steps, list)
    return steps


def _step_with_script(marker: str) -> dict:
    matches = [
        step
        for step in _action_steps()
        if marker in str(step.get("run", ""))
    ]
    assert len(matches) == 1, f"expected one action step containing {marker!r}"
    return matches[0]


def test_action_path_layout_is_explicit_and_unambiguous() -> None:
    for name in (
        "audio_path",
        "transcript_path",
        "manifest_path",
        "audio_repo_path",
        "transcript_repo_path",
        "data_repo_path",
    ):
        assert "default: ''" in _input_stanza(name)
    assert '${INPUT_AUDIO_PATH:-${AUDIO_PATH:-./audio}}' in ACTION
    assert '${INPUT_TRANSCRIPT_PATH:-${TRANSCRIPT_PATH:-./transcriptions}}' in ACTION
    assert '${INPUT_MANIFEST_PATH:-${MANIFEST_PATH:-./manifest.json}}' in ACTION

    assert "Supply data_repo_path, or supply both audio_repo_path" in ACTION
    assert "Use data_repo_path alone" in ACTION
    assert "Path(audio_repo_path).resolve() == Path(text_repo_path).resolve()" in ACTION
    assert "use data_repo_path for a shared layout" in ACTION


def test_action_exposes_strict_by_default_verification_tolerance() -> None:
    assert "default: ''" in _input_stanza("max_word_error_rate")
    assert (
        '${INPUT_MAX_WORD_ERROR_RATE:-${MAX_WORD_ERROR_RATE:-0}}' in ACTION
    )
    assert 'CMD+=(--max-word-error-rate "$EFFECTIVE_MAX_WORD_ERROR_RATE")' in ACTION


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


def test_action_installs_local_whisper_only_for_effective_processing_provider() -> None:
    plan = next(step for step in _action_steps() if step.get("id") == "provider_plan")
    install = _step_with_script("requirements-local.txt")
    plan_script = str(plan["run"])

    assert plan.get("if") == (
        "${{ inputs.command == 'run' || inputs.command == 'transcribe' }}"
    )
    assert 'validation_profile="transcribe"' in plan_script
    assert '"whisper" in config.transcribe.providers' in plan_script
    assert "inputs.command == 'run'" in str(install.get("if"))
    assert "inputs.command == 'transcribe'" in str(install.get("if"))
    assert "steps.provider_plan.outputs.local_requested == 'true'" in str(
        install.get("if")
    )
    assert "requirements-local.txt" in str(install["run"])


def test_action_provider_inputs_preserve_inherited_environment() -> None:
    provider_inputs = {
        "OPENAI_API_KEY": "openai_api_key",
        "XAI_API_KEY": "xai_api_key",
        "XAI_STT_URL": "xai_base_url",
        "META_ASR_URL": "meta_asr_url",
        "META_ASR_API_KEY": "meta_asr_api_key",
    }

    for step in (
        next(step for step in _action_steps() if step.get("id") == "provider_plan"),
        next(step for step in _action_steps() if step.get("id") == "process"),
    ):
        environment = step.get("env", {})
        script = str(step["run"])
        for canonical_name, input_name in provider_inputs.items():
            input_variable = f"INPUT_{canonical_name}"
            assert environment.get(input_variable) == (
                f"${{{{ inputs.{input_name} }}}}"
            )
            assert environment.get(canonical_name) != (
                f"${{{{ inputs.{input_name} }}}}"
            )
            assert f"export {canonical_name}" in script


def test_action_supports_only_processing_commands_and_performs_no_git() -> None:
    assert "run|download|transcribe|verify|status" in ACTION
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


def test_real_audio_workflow_is_a_pure_action_consumer() -> None:
    assert "workflow_dispatch:" in REAL_AUDIO
    assert "branches: [v1]" in REAL_AUDIO
    assert "types: [opened, synchronize, reopened, closed]" in REAL_AUDIO
    assert (
        "github.event_name == 'workflow_dispatch' || "
        "github.event.action != 'closed' || "
        "github.event.pull_request.merged == true"
    ) in REAL_AUDIO
    assert REAL_AUDIO.count("uses: ./") == 2
    assert "command: transcribe" in REAL_AUDIO
    assert "command: verify" in REAL_AUDIO
    assert "max_word_error_rate: 0.005" in REAL_AUDIO
    assert "secrets.OPENAI_API_KEY" in REAL_AUDIO
    assert "AUDIO_PATH: tests/fixtures/telegram/audio" in REAL_AUDIO
    assert "TRANSCRIPT_PATH: tests/fixtures/telegram/transcriptions" in REAL_AUDIO
    assert "MANIFEST_PATH: tests/fixtures/telegram/manifest.json" in REAL_AUDIO
    assert "audio_path:" not in REAL_AUDIO
    assert "transcript_path:" not in REAL_AUDIO
    assert "manifest_path:" not in REAL_AUDIO
    assert "actions/upload-artifact@v4" in REAL_AUDIO
    assert "python" not in REAL_AUDIO.lower()
    assert "run:" not in REAL_AUDIO


def test_ci_covers_python_314_and_ubuntu_2604() -> None:
    assert "python-version: ['3.11', '3.12', '3.13', '3.14']" in CI
    assert "if: matrix.python-version == '3.14'" in CI
    assert "runs-on: ubuntu-26.04" in CI
    assert "python -VV" in CI
    assert "python -m pytest tests/ -v --tb=short" in CI
    assert "Set up Python 3.14" in ACTION
    assert "python-version: '3.14'" in ACTION


def test_active_workflows_have_complete_v1_triggers() -> None:
    active = tuple(sorted((ROOT / ".github" / "workflows").glob("*.yml")))
    active_names = {path.name for path in active} - {"pipeline.yml", "validation.yml"}
    assert active_names == {"ci.yml", "openai-real-audio.yml"}
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
        ROOT / ".github" / "workflows" / "openai-real-audio.yml",
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


def test_examples_use_only_the_three_path_contract() -> None:
    legacy_inputs = ("data_repo_path:", "audio_repo_path:", "transcript_repo_path:")
    for path in EXAMPLES:
        example = path.read_text(encoding="utf-8")
        for legacy_input in legacy_inputs:
            assert legacy_input not in example

    for name in ("full.yml", "pipeline.yml"):
        example = (ROOT / "examples" / name).read_text(encoding="utf-8")
        assert "AUDIO_PATH:" in example
        assert "TRANSCRIPT_PATH:" in example
        assert "MANIFEST_PATH:" in example
        assert "'./audio-data/audio' || './data/audio'" in example
        assert "'./transcript-data/transcriptions' || './data/transcriptions'" in example
        assert "'./transcript-data/manifest.json' || './data/manifest.json'" in example


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
    assert "AUDIO_PATH: ./audio" in example
    assert "TRANSCRIPT_PATH: ./transcriptions" in example
    assert "MANIFEST_PATH: ./manifest.json" in example
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
    assert example.count("uses: octoleo/octoscribe@v1") == 1
    assert "command: transcribe" in example
    assert "uses: actions/upload-artifact@v4" in example
    assert "uses: octoleo/git-user" not in example
    assert "/bin/git" not in example
    assert "AUDIO_PATH:" not in example
    assert "TRANSCRIPT_PATH:" not in example
    assert "MANIFEST_PATH:" not in example
    assert "audio_path:" not in example
    assert "transcript_path:" not in example
    assert "manifest_path:" not in example
