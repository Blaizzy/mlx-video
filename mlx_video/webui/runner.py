from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlx_video.webui.job_store import JobStore, build_job_paths, latest_jobs


def entrypoint_path(name: str) -> str:
    return str(Path(sys.executable).with_name(name))


@dataclass
class GenerationRequest:
    engine: str
    prompt: str
    width: int
    height: int
    num_frames: int
    seed: int
    output_path: Path
    negative_prompt: str = ""
    steps: int | None = None
    cfg_scale: float | None = None
    image_path: Path | None = None
    audio_path: Path | None = None
    ltx_pipeline: str = "distilled"
    ltx_model_repo: str = "Lightricks/LTX-2"
    fps: int = 24
    wan_model_dir: Path | None = None
    guide_scale: str | None = None
    shift: float | None = None
    scheduler: str = "unipc"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_request(request: GenerationRequest) -> None:
    if not request.prompt.strip():
        raise ValueError("Prompt is required")
    if request.width <= 0 or request.height <= 0:
        raise ValueError("Width and height must be positive")
    if request.num_frames <= 0:
        raise ValueError("Number of frames must be positive")
    if request.image_path and not request.image_path.exists():
        raise ValueError(f"Image file does not exist: {request.image_path}")
    if request.audio_path and not request.audio_path.exists():
        raise ValueError(f"Audio file does not exist: {request.audio_path}")
    if request.engine == "wan" and (not request.wan_model_dir or not request.wan_model_dir.exists()):
        raise ValueError("Wan model directory does not exist")
    if request.engine not in {"ltx", "wan"}:
        raise ValueError(f"Unsupported engine: {request.engine}")


def build_command(request: GenerationRequest) -> list[str]:
    validate_request(request)
    if request.engine == "ltx":
        command = [
            entrypoint_path("mlx_video.ltx_2.generate"),
            "--prompt",
            request.prompt,
            "--pipeline",
            request.ltx_pipeline,
            "--height",
            str(request.height),
            "--width",
            str(request.width),
            "--num-frames",
            str(request.num_frames),
            "--seed",
            str(request.seed),
            "--fps",
            str(request.fps),
            "--model-repo",
            request.ltx_model_repo,
        ]
        if request.negative_prompt:
            command.extend(["--negative-prompt", request.negative_prompt])
        if request.steps is not None:
            command.extend(["--steps", str(request.steps)])
        if request.cfg_scale is not None:
            command.extend(["--cfg-scale", str(request.cfg_scale)])
        if request.image_path:
            command.extend(["--image", str(request.image_path)])
        if request.audio_path:
            command.extend(["--audio-file", str(request.audio_path)])
        command.extend(["--output-path", str(request.output_path)])
        return command

    command = [
        entrypoint_path("mlx_video.wan_2.generate"),
        "--model-dir",
        str(request.wan_model_dir),
        "--prompt",
        request.prompt,
        "--width",
        str(request.width),
        "--height",
        str(request.height),
        "--num-frames",
        str(request.num_frames),
        "--seed",
        str(request.seed),
        "--scheduler",
        request.scheduler,
    ]
    if request.negative_prompt:
        command.extend(["--negative-prompt", request.negative_prompt])
    if request.image_path:
        command.extend(["--image", str(request.image_path)])
    if request.steps is not None:
        command.extend(["--steps", str(request.steps)])
    if request.guide_scale:
        command.extend(["--guide-scale", request.guide_scale])
    if request.shift is not None:
        command.extend(["--shift", str(request.shift)])
    command.extend(["--output-path", str(request.output_path)])
    return command


def request_to_params(request: GenerationRequest) -> dict[str, Any]:
    params = asdict(request)
    params.pop("output_path", None)
    for key, value in list(params.items()):
        if isinstance(value, Path):
            params[key] = str(value)
    return params


def request_from_params(params: dict[str, Any], output_path: Path) -> GenerationRequest:
    data = dict(params)
    for key in ("image_path", "audio_path", "wan_model_dir"):
        if data.get(key):
            data[key] = Path(data[key])
    data["output_path"] = output_path
    return GenerationRequest(**data)


def clone_for_retry(job: dict[str, Any], new_id: str, output_root: Path | str) -> dict[str, Any]:
    paths = build_job_paths(output_root, new_id)
    now = utc_now()
    return {
        "id": new_id,
        "engine": job["engine"],
        "prompt": job["prompt"],
        "params": job.get("params", {}),
        "status": "queued",
        "output_path": str(paths.video),
        "log_path": str(paths.log),
        "created_at": now,
        "updated_at": now,
    }


class JobManager:
    def __init__(self, output_root: Path | str = "outputs") -> None:
        self.output_root = Path(output_root)
        self.store = JobStore(self.output_root)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def enqueue(self, request: GenerationRequest) -> dict[str, Any]:
        validate_request(request)
        job_id = uuid.uuid4().hex[:12]
        paths = build_job_paths(self.output_root, job_id)
        request.output_path = paths.video
        now = utc_now()
        job = {
            "id": job_id,
            "engine": request.engine,
            "prompt": request.prompt,
            "params": request_to_params(request),
            "status": "queued",
            "output_path": str(paths.video),
            "log_path": str(paths.log),
            "created_at": now,
            "updated_at": now,
        }
        self.store.append(job)
        self.start_worker()
        return job

    def retry(self, job_id: str) -> dict[str, Any]:
        jobs = {job["id"]: job for job in latest_jobs(self.store.read_all())}
        if job_id not in jobs:
            raise ValueError(f"Unknown job id: {job_id}")
        new_job = clone_for_retry(jobs[job_id], uuid.uuid4().hex[:12], self.output_root)
        self.store.append(new_job)
        self.start_worker()
        return new_job

    def start_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run_pending_jobs, daemon=True)
            self._worker.start()

    def _run_pending_jobs(self) -> None:
        while True:
            job = self._next_pending_job()
            if not job:
                return
            self._run_job(job)

    def _next_pending_job(self) -> dict[str, Any] | None:
        for job in reversed(latest_jobs(self.store.read_all())):
            if job.get("status") == "queued":
                return job
        return None

    def _run_job(self, job: dict[str, Any]) -> None:
        now = utc_now()
        running = dict(job, status="running", updated_at=now)
        self.store.append(running)
        log_path = Path(job["log_path"])
        request = request_from_params(job["params"], Path(job["output_path"]))
        command = build_command(request)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n\n")
            log.flush()
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
                returncode = process.returncode
            except OSError as exc:
                log.write(f"\nFailed to start command: {exc}\n")
                returncode = 127
        status = "done" if returncode == 0 else "failed"
        finished = dict(
            running,
            status=status,
            returncode=returncode,
            updated_at=utc_now(),
        )
        if status == "failed":
            finished["error"] = f"Command exited with status {returncode}"
        self.store.append(finished)
