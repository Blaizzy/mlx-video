from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JobPaths:
    video: Path
    log: Path


class JobStore:
    def __init__(self, output_root: Path | str = "outputs") -> None:
        self.output_root = Path(output_root)
        self.jobs_file = self.output_root / "jobs.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        with self.jobs_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.jobs_file.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.jobs_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def build_job_paths(output_root: Path | str, job_id: str) -> JobPaths:
    root = Path(output_root)
    video_dir = root / "videos"
    log_dir = root / "logs"
    video_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return JobPaths(video=video_dir / f"{job_id}.mp4", log=log_dir / f"{job_id}.log")


def latest_jobs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[record["id"]] = record
    return sorted(latest.values(), key=lambda record: record.get("updated_at", ""), reverse=True)


def read_log_tail(path: Path | str, lines: int = 80) -> str:
    log_path = Path(path)
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
