from pathlib import Path

from mlx_video.webui.job_store import (
    JobStore,
    build_job_paths,
    latest_jobs,
    read_log_tail,
)


def test_latest_jobs_keeps_latest_record_per_job(tmp_path: Path):
    store = JobStore(tmp_path)
    store.append({"id": "job-a", "status": "queued", "updated_at": "2026-05-24T10:00:00Z"})
    store.append({"id": "job-a", "status": "running", "updated_at": "2026-05-24T10:01:00Z"})
    store.append({"id": "job-a", "status": "done", "updated_at": "2026-05-24T10:02:00Z"})

    jobs = latest_jobs(store.read_all())

    assert len(jobs) == 1
    assert jobs[0]["id"] == "job-a"
    assert jobs[0]["status"] == "done"


def test_latest_jobs_sorts_newest_first(tmp_path: Path):
    store = JobStore(tmp_path)
    store.append({"id": "older", "status": "done", "updated_at": "2026-05-24T10:00:00Z"})
    store.append({"id": "newer", "status": "queued", "updated_at": "2026-05-24T10:05:00Z"})

    jobs = latest_jobs(store.read_all())

    assert [job["id"] for job in jobs] == ["newer", "older"]


def test_build_job_paths_uses_outputs_subdirectories(tmp_path: Path):
    paths = build_job_paths(tmp_path, "abc123")

    assert paths.video == tmp_path / "videos" / "abc123.mp4"
    assert paths.log == tmp_path / "logs" / "abc123.log"
    assert paths.video.parent.exists()
    assert paths.log.parent.exists()


def test_read_log_tail_returns_last_lines(tmp_path: Path):
    log = tmp_path / "job.log"
    log.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    assert read_log_tail(log, lines=2) == "three\nfour"
