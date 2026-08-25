"""
tests/test_manifest.py — Comprehensive pytest tests for Manifest.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.manifest import Manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_metadata(msg_id: int | str = 1) -> dict:
    return {
        "filename": f"Sermon_{msg_id}.mp3",
        "title": f"Sermon {msg_id}",
        "performer": "Speaker Name",
        "date": "2024-01-15",
        "duration": "45:30",
        "duration_seconds": 2730,
        "extension": ".mp3",
        "hash": "abc123def456",
    }


def _sample_transcription_result() -> dict:
    return {
        "output_file": "Sermon_1.txt",
        "output_path": "transcriptions/Sermon_1.txt",
        "audio_path": "audio/Sermon_1.mp3",
        "model": "gpt-4o-transcribe",
        "completed_at": "2024-01-15T10:30:00Z",
    }


# ---------------------------------------------------------------------------
# 1. New manifest starts empty
# ---------------------------------------------------------------------------

def test_new_manifest_is_empty(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    assert m.all_entries() == {}


# ---------------------------------------------------------------------------
# 2. mark_downloaded sets correct fields
# ---------------------------------------------------------------------------

def test_mark_downloaded_sets_fields(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    meta = _sample_metadata(42)
    m.mark_downloaded(42, meta)
    entry = m.get_entry(42)
    assert entry is not None
    assert entry["downloaded"] is True
    assert entry["filename"] == "Sermon_42.mp3"
    assert entry["title"] == "Sermon 42"
    assert entry["telegram_msg_id"] == 42


# ---------------------------------------------------------------------------
# 3. is_downloaded returns False for unknown id
# ---------------------------------------------------------------------------

def test_is_downloaded_unknown_id(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    assert m.is_downloaded(9999) is False


# ---------------------------------------------------------------------------
# 4. is_downloaded returns True after mark_downloaded
# ---------------------------------------------------------------------------

def test_is_downloaded_true_after_mark(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(1, _sample_metadata(1))
    assert m.is_downloaded(1) is True


# ---------------------------------------------------------------------------
# 5. is_transcribed returns False until mark_transcribed with status=completed
# ---------------------------------------------------------------------------

def test_is_transcribed_false_before_transcription(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(1, _sample_metadata(1))
    assert m.is_transcribed(1) is False


def test_is_transcribed_false_for_unknown_id(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    assert m.is_transcribed(9999) is False


# ---------------------------------------------------------------------------
# 6. mark_transcribed updates existing entry
# ---------------------------------------------------------------------------

def test_mark_transcribed_updates_entry(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(1, _sample_metadata(1))
    m.mark_transcribed(1, _sample_transcription_result())

    assert m.is_transcribed(1) is True
    entry = m.get_entry(1)
    assert entry is not None
    transcription = entry["transcription"]
    assert transcription["status"] == "completed"
    assert transcription["output_file"] == "Sermon_1.txt"
    assert transcription["output_path"] == "transcriptions/Sermon_1.txt"
    assert transcription["audio_path"] == "audio/Sermon_1.mp3"
    assert transcription["model"] == "gpt-4o-transcribe"
    # The original download data must still be intact.
    assert entry["filename"] == "Sermon_1.mp3"


# ---------------------------------------------------------------------------
# 7. save() creates file; reload() reads it back correctly
# ---------------------------------------------------------------------------

def test_save_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    m = Manifest(path)
    m.mark_downloaded(1, _sample_metadata(1))
    m.save()
    assert path.exists()


def test_reload_reads_back_correctly(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    m = Manifest(path)
    m.mark_downloaded(7, _sample_metadata(7))
    m.save()

    m2 = Manifest(path)
    assert m2.is_downloaded(7) is True
    entry = m2.get_entry(7)
    assert entry is not None
    assert entry["title"] == "Sermon 7"


# ---------------------------------------------------------------------------
# 8. Atomic write — .tmp file does not persist after save()
# ---------------------------------------------------------------------------

def test_atomic_write_no_tmp_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    m = Manifest(path)
    m.mark_downloaded(1, _sample_metadata(1))
    m.save()
    tmp_file = path.with_suffix(".tmp")
    assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# 9. pending_transcription() only returns downloaded-but-not-transcribed entries
# ---------------------------------------------------------------------------

def test_pending_transcription_correct_subset(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(1, _sample_metadata(1))  # downloaded, not transcribed
    m.mark_downloaded(2, _sample_metadata(2))  # downloaded and transcribed
    m.mark_transcribed(2, _sample_transcription_result())

    pending = m.pending_transcription()
    ids = {str(e["telegram_msg_id"]) for e in pending}
    assert "1" in ids
    assert "2" not in ids


def test_pending_transcription_empty_when_all_transcribed(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(1, _sample_metadata(1))
    m.mark_transcribed(1, _sample_transcription_result())
    assert m.pending_transcription() == []


@pytest.mark.parametrize(
    "quality_state",
    ["machine_transcribed", "cross_checked", "needs_review", "human_verified"],
)
def test_quality_states_are_terminal_without_claiming_completed(
    tmp_path: Path, quality_state: str
) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(1, _sample_metadata(1))
    m.mark_transcribed(
        1,
        {
            **_sample_transcription_result(),
            "quality_state": quality_state,
        },
    )
    assert m.is_transcribed(1)
    assert m.pending_transcription() == []
    assert m.get_entry(1)["transcription"]["status"] == quality_state


def test_human_verification_is_explicit_and_auditable(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(1, _sample_metadata(1))
    m.mark_transcribed(
        1,
        {**_sample_transcription_result(), "quality_state": "needs_review"},
    )
    m.mark_human_verified(1, reviewer="reviewer@example.org")
    transcription = m.get_entry(1)["transcription"]
    assert transcription["status"] == "human_verified"
    assert transcription["human_verified_by"] == "reviewer@example.org"
    assert transcription["human_verified_at"].endswith("Z")


# ---------------------------------------------------------------------------
# 10. stats() returns correct counts
# ---------------------------------------------------------------------------

def test_stats_correct_counts(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    # 3 downloaded, 2 transcribed, 1 failed
    for i in range(1, 4):
        m.mark_downloaded(i, _sample_metadata(i))
    m.mark_transcribed(1, _sample_transcription_result())
    m.mark_transcribed(2, _sample_transcription_result())
    m.mark_failed(3, "transcription", "timeout")

    s = m.stats()
    assert s["total"] == 3
    assert s["downloaded"] == 3
    assert s["transcribed"] == 2
    assert s["failed"] == 1


def test_stats_empty_manifest(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    s = m.stats()
    assert s == {"total": 0, "downloaded": 0, "transcribed": 0, "failed": 0}


# ---------------------------------------------------------------------------
# 11. Context manager saves on exit
# ---------------------------------------------------------------------------

def test_context_manager_saves_on_exit(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    with Manifest(path) as m:
        m.mark_downloaded(99, _sample_metadata(99))
    # File must have been written by __exit__.
    assert path.exists()
    reloaded = Manifest(path)
    assert reloaded.is_downloaded(99) is True


def test_context_manager_saves_on_exit_even_after_exception(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    try:
        with Manifest(path) as m:
            m.mark_downloaded(50, _sample_metadata(50))
            raise RuntimeError("deliberate test error")
    except RuntimeError:
        pass
    assert path.exists()
    reloaded = Manifest(path)
    assert reloaded.is_downloaded(50) is True


# ---------------------------------------------------------------------------
# 12. Thread safety — 50 concurrent mark_downloaded calls, no data loss
# ---------------------------------------------------------------------------

def test_thread_safety_concurrent_mark_downloaded(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    num_threads = 50
    barrier = threading.Barrier(num_threads)

    def worker(msg_id: int) -> None:
        barrier.wait()  # All threads start at the same time.
        m.mark_downloaded(msg_id, _sample_metadata(msg_id))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entries = m.all_entries()
    assert len(entries) == num_threads
    for i in range(num_threads):
        assert str(i) in entries
        assert entries[str(i)]["downloaded"] is True


# ---------------------------------------------------------------------------
# 13. mark_failed records error correctly
# ---------------------------------------------------------------------------

def test_mark_failed_records_error(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(1, _sample_metadata(1))
    m.mark_failed(1, "transcription", "API timeout after 30s")

    entry = m.get_entry(1)
    assert entry is not None
    assert entry["failed_stage"] == "transcription"
    assert entry["failed_error"] == "API timeout after 30s"
    assert "failed_at" in entry
    # Transcription sub-dict must also reflect the failure.
    assert entry["transcription"]["status"] == "failed"
    assert entry["transcription"]["error"] == "API timeout after 30s"


def test_mark_failed_download_stage(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_failed(2, "download", "network error")
    entry = m.get_entry(2)
    assert entry is not None
    assert entry["failed_stage"] == "download"
    assert entry["failed_error"] == "network error"


@pytest.mark.parametrize("stage", ["download", "import"])
def test_mark_downloaded_clears_recovered_acquisition_failure(
    tmp_path: Path, stage: str
) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_failed(2, stage, "temporary source failure")

    m.mark_downloaded(2, _sample_metadata(2))

    entry = m.get_entry(2)
    assert entry is not None
    assert entry["downloaded"] is True
    assert "failed_stage" not in entry
    assert "failed_error" not in entry
    assert "failed_at" not in entry
    assert m.stats()["failed"] == 0


def test_mark_downloaded_clears_legacy_acquisition_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "2": {
                    "telegram_msg_id": 2,
                    "error": "legacy download error",
                    "failed_error": "legacy download error",
                    "failed_at": "2024-01-01T00:00:00Z",
                }
            }
        ),
        encoding="utf-8",
    )
    m = Manifest(path)

    m.mark_downloaded(2, _sample_metadata(2))

    entry = m.get_entry(2)
    assert entry is not None
    assert "error" not in entry
    assert "failed_error" not in entry
    assert "failed_at" not in entry


def test_mark_downloaded_preserves_unresolved_transcription_failure(
    tmp_path: Path,
) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(2, _sample_metadata(2))
    m.mark_failed(2, "transcription", "provider unavailable")

    m.mark_downloaded(2, _sample_metadata(2))

    entry = m.get_entry(2)
    assert entry is not None
    assert entry["failed_stage"] == "transcription"
    assert entry["transcription"]["status"] == "failed"


def test_mark_transcribed_clears_stale_failure_markers(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(2, _sample_metadata(2))
    m.mark_failed(2, "transcription", "temporary provider failure")

    m.mark_transcribed(2, _sample_transcription_result())

    entry = m.get_entry(2)
    assert entry is not None
    assert entry["transcription"]["status"] == "completed"
    assert "error" not in entry["transcription"]
    assert "failed_stage" not in entry
    assert "failed_error" not in entry
    assert "failed_at" not in entry
    assert m.stats()["failed"] == 0


# ---------------------------------------------------------------------------
# 14. Manifest survives being loaded from an existing file
# ---------------------------------------------------------------------------

def test_manifest_loads_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    existing = {
        "100": {
            "downloaded": True,
            "filename": "Old Sermon.mp3",
            "title": "Old Sermon",
            "performer": "Pastor X",
            "date": "2023-06-01",
            "duration": "30:00",
            "duration_seconds": 1800,
            "extension": ".mp3",
            "hash": "deadbeef",
            "telegram_msg_id": 100,
            "transcription": {
                "status": "completed",
                "output_file": "Old Sermon.txt",
                "model": "gpt-4o-transcribe",
                "completed_at": "2023-06-01T09:00:00Z",
            },
        }
    }
    path.write_text(json.dumps(existing), encoding="utf-8")

    m = Manifest(path)
    assert m.is_downloaded(100) is True
    assert m.is_transcribed(100) is True
    entry = m.get_entry(100)
    assert entry is not None
    assert entry["title"] == "Old Sermon"


# ---------------------------------------------------------------------------
# 15. Manifest keys are strings — int input is normalised to str
# ---------------------------------------------------------------------------

def test_keys_are_strings_after_int_input(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(42, _sample_metadata(42))

    entries = m.all_entries()
    assert "42" in entries
    # Querying with either int or str must work.
    assert m.is_downloaded(42) is True
    assert m.is_downloaded("42") is True


def test_keys_are_strings_in_saved_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    m = Manifest(path)
    m.mark_downloaded(7, _sample_metadata(7))
    m.save()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "7" in raw
    # No integer keys must appear.
    assert all(isinstance(k, str) for k in raw)


# ---------------------------------------------------------------------------
# 16. save() creates parent directory if missing
# ---------------------------------------------------------------------------

def test_save_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "subdir" / "nested" / "manifest.json"
    m = Manifest(path)
    m.mark_downloaded(1, _sample_metadata(1))
    m.save()
    assert path.exists()


# ---------------------------------------------------------------------------
# 17. get_entry returns None for missing id
# ---------------------------------------------------------------------------

def test_get_entry_returns_none_for_missing(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    assert m.get_entry(9999) is None


# ---------------------------------------------------------------------------
# 18. mark_transcribed adds completed_at automatically when not supplied
# ---------------------------------------------------------------------------

def test_mark_transcribed_adds_completed_at_automatically(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.mark_downloaded(1, _sample_metadata(1))
    # Deliberately omit completed_at from the result dict.
    m.mark_transcribed(1, {"output_file": "Sermon_1.txt", "model": "gpt-4o-transcribe"})
    entry = m.get_entry(1)
    assert entry is not None
    assert "completed_at" in entry["transcription"]


# ---------------------------------------------------------------------------
# 19. sorted keys in saved JSON for stable git diffs
# ---------------------------------------------------------------------------

def test_save_sorts_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    m = Manifest(path)
    # Add entries out of numeric order.
    for msg_id in [30, 10, 20]:
        m.mark_downloaded(msg_id, _sample_metadata(msg_id))
    m.save()

    raw_text = path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    keys = list(data.keys())
    assert keys == sorted(keys)


def test_record_audio_hash_backfills_then_rejects_changed_evidence(
    tmp_path: Path,
) -> None:
    m = Manifest(tmp_path / "manifest.json")
    metadata = _sample_metadata(1)
    metadata.pop("hash", None)
    m.mark_downloaded(1, metadata)
    expected = "a" * 64
    m.record_audio_hash(1, expected)
    assert m.get_entry(1)["hash"] == expected

    with pytest.raises(ValueError, match="source evidence changed"):
        m.record_audio_hash(1, "b" * 64)
