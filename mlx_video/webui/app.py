from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from mlx_video.webui.job_store import build_job_paths, latest_jobs, read_log_tail
from mlx_video.webui.runner import GenerationRequest, JobManager


DEFAULT_OUTPUT_ROOT = Path("outputs")
MANAGER = JobManager(DEFAULT_OUTPUT_ROOT)


def _path_or_none(value: Any) -> Path | None:
    if not value:
        return None
    if isinstance(value, str):
        return Path(value)
    name = getattr(value, "name", None)
    return Path(name) if name else None


def _jobs_table() -> list[list[str]]:
    rows: list[list[str]] = []
    for job in latest_jobs(MANAGER.store.read_all()):
        rows.append(
            [
                job.get("id", ""),
                job.get("status", ""),
                job.get("engine", ""),
                job.get("prompt", "")[:96],
                job.get("output_path", ""),
                job.get("updated_at", ""),
            ]
        )
    return rows


def _latest_video_path() -> str | None:
    for job in latest_jobs(MANAGER.store.read_all()):
        output = Path(job.get("output_path", ""))
        if job.get("status") == "done" and output.exists():
            return str(output)
    return None


def _latest_log_tail() -> str:
    jobs = latest_jobs(MANAGER.store.read_all())
    if not jobs:
        return ""
    return read_log_tail(jobs[0].get("log_path", ""), lines=120)


def enqueue_ltx(
    prompt: str,
    negative_prompt: str,
    pipeline: str,
    model_repo: str,
    image_file: Any,
    audio_file: Any,
    width: int,
    height: int,
    num_frames: int,
    seed: int,
    fps: int,
    steps: int,
    cfg_scale: float,
) -> tuple[str, list[list[str]], str | None, str]:
    paths = build_job_paths(DEFAULT_OUTPUT_ROOT, "preview")
    request = GenerationRequest(
        engine="ltx",
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        width=int(width),
        height=int(height),
        num_frames=int(num_frames),
        seed=int(seed),
        output_path=paths.video,
        ltx_pipeline=pipeline,
        ltx_model_repo=model_repo or "Lightricks/LTX-2",
        image_path=_path_or_none(image_file),
        audio_path=_path_or_none(audio_file),
        fps=int(fps),
        steps=int(steps) if steps else None,
        cfg_scale=float(cfg_scale) if cfg_scale else None,
    )
    try:
        job = MANAGER.enqueue(request)
    except Exception as exc:
        return f"Error: {exc}", _jobs_table(), _latest_video_path(), _latest_log_tail()
    return f"Queued LTX job {job['id']}", _jobs_table(), _latest_video_path(), _latest_log_tail()


def enqueue_wan(
    prompt: str,
    negative_prompt: str,
    model_dir: str,
    image_file: Any,
    width: int,
    height: int,
    num_frames: int,
    seed: int,
    steps: int,
    guide_scale: str,
    shift: float,
    scheduler: str,
) -> tuple[str, list[list[str]], str | None, str]:
    paths = build_job_paths(DEFAULT_OUTPUT_ROOT, "preview")
    request = GenerationRequest(
        engine="wan",
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        width=int(width),
        height=int(height),
        num_frames=int(num_frames),
        seed=int(seed),
        output_path=paths.video,
        wan_model_dir=Path(model_dir).expanduser() if model_dir else None,
        image_path=_path_or_none(image_file),
        steps=int(steps) if steps else None,
        guide_scale=guide_scale or None,
        shift=float(shift) if shift else None,
        scheduler=scheduler,
    )
    try:
        job = MANAGER.enqueue(request)
    except Exception as exc:
        return f"Error: {exc}", _jobs_table(), _latest_video_path(), _latest_log_tail()
    return f"Queued Wan job {job['id']}", _jobs_table(), _latest_video_path(), _latest_log_tail()


def refresh_queue() -> tuple[list[list[str]], str | None, str]:
    return _jobs_table(), _latest_video_path(), _latest_log_tail()


def retry_job(job_id: str) -> tuple[str, list[list[str]], str | None, str]:
    try:
        job = MANAGER.retry(job_id.strip())
    except Exception as exc:
        return f"Error: {exc}", _jobs_table(), _latest_video_path(), _latest_log_tail()
    return f"Queued retry job {job['id']}", _jobs_table(), _latest_video_path(), _latest_log_tail()


def gallery_choices() -> list[str]:
    video_dir = DEFAULT_OUTPUT_ROOT / "videos"
    if not video_dir.exists():
        return []
    return [str(path) for path in sorted(video_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)]


def refresh_gallery() -> tuple[gr.Dropdown, str | None]:
    choices = gallery_choices()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None), choices[0] if choices else None


def select_gallery_video(path: str | None) -> str | None:
    return path or None


def check_models(ltx_repo: str, wan_dir: str) -> str:
    lines = [
        f"LTX model repo: {ltx_repo or 'Lightricks/LTX-2'}",
        "LTX weights will be requested by the CLI/Hugging Face when generation starts.",
    ]
    if wan_dir:
        path = Path(wan_dir).expanduser()
        lines.append(f"Wan model dir exists: {path.exists()} ({path})")
    else:
        lines.append("Wan model dir is empty. Wan generation needs a converted MLX model directory.")
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="mlx-video Mini Studio") as demo:
        gr.Markdown("# mlx-video Mini Studio")
        gr.Markdown("Local Gradio UI for persistent mlx-video jobs.")

        status = gr.Textbox(label="Status", interactive=False)
        latest_video = gr.Video(label="Latest video", interactive=False)
        log_tail = gr.Textbox(label="Latest log tail", lines=12, interactive=False)

        with gr.Tab("Generate"):
            with gr.Tab("LTX-2"):
                ltx_prompt = gr.Textbox(label="Prompt", lines=3)
                ltx_negative = gr.Textbox(label="Negative prompt")
                with gr.Row():
                    ltx_pipeline = gr.Dropdown(
                        ["distilled", "dev", "dev-two-stage", "dev-two-stage-hq"],
                        value="distilled",
                        label="Pipeline",
                    )
                    ltx_model_repo = gr.Textbox(value="Lightricks/LTX-2", label="Model repo")
                with gr.Row():
                    ltx_width = gr.Number(value=512, label="Width", precision=0)
                    ltx_height = gr.Number(value=512, label="Height", precision=0)
                    ltx_frames = gr.Number(value=97, label="Frames", precision=0)
                with gr.Row():
                    ltx_seed = gr.Number(value=42, label="Seed", precision=0)
                    ltx_fps = gr.Number(value=24, label="FPS", precision=0)
                    ltx_steps = gr.Number(value=30, label="Steps", precision=0)
                    ltx_cfg = gr.Number(value=3.0, label="CFG scale")
                with gr.Row():
                    ltx_image = gr.File(label="Optional image")
                    ltx_audio = gr.File(label="Optional audio")
                ltx_button = gr.Button("Add LTX job to persistent queue", variant="primary")

            with gr.Tab("Wan 2.x"):
                wan_prompt = gr.Textbox(label="Prompt", lines=3)
                wan_negative = gr.Textbox(label="Negative prompt")
                wan_model_dir = gr.Textbox(label="Converted MLX model directory")
                wan_image = gr.File(label="Optional image")
                with gr.Row():
                    wan_width = gr.Number(value=1280, label="Width", precision=0)
                    wan_height = gr.Number(value=704, label="Height", precision=0)
                    wan_frames = gr.Number(value=81, label="Frames", precision=0)
                with gr.Row():
                    wan_seed = gr.Number(value=-1, label="Seed", precision=0)
                    wan_steps = gr.Number(value=40, label="Steps", precision=0)
                    wan_guide = gr.Textbox(value="", label="Guide scale")
                    wan_shift = gr.Number(value=0, label="Shift")
                    wan_scheduler = gr.Dropdown(["euler", "dpm++", "unipc"], value="unipc", label="Scheduler")
                wan_button = gr.Button("Add Wan job to persistent queue", variant="primary")

        with gr.Tab("Queue"):
            queue_table = gr.Dataframe(
                headers=["id", "status", "engine", "prompt", "output", "updated"],
                value=_jobs_table(),
                interactive=False,
                label="Persistent jobs",
            )
            with gr.Row():
                refresh_button = gr.Button("Refresh")
                retry_id = gr.Textbox(label="Job id to retry")
                retry_button = gr.Button("Retry job")

        with gr.Tab("Gallery"):
            gallery_select = gr.Dropdown(choices=gallery_choices(), label="Generated videos")
            gallery_video = gr.Video(label="Selected video", interactive=False)
            gallery_refresh = gr.Button("Refresh gallery")

        with gr.Tab("Models / Settings"):
            settings_ltx_repo = gr.Textbox(value="Lightricks/LTX-2", label="Default LTX repo")
            settings_wan_dir = gr.Textbox(label="Wan model directory to check")
            settings_button = gr.Button("Check")
            settings_output = gr.Textbox(label="Model hints", lines=8, interactive=False)

        ltx_button.click(
            enqueue_ltx,
            inputs=[
                ltx_prompt,
                ltx_negative,
                ltx_pipeline,
                ltx_model_repo,
                ltx_image,
                ltx_audio,
                ltx_width,
                ltx_height,
                ltx_frames,
                ltx_seed,
                ltx_fps,
                ltx_steps,
                ltx_cfg,
            ],
            outputs=[status, queue_table, latest_video, log_tail],
        )
        wan_button.click(
            enqueue_wan,
            inputs=[
                wan_prompt,
                wan_negative,
                wan_model_dir,
                wan_image,
                wan_width,
                wan_height,
                wan_frames,
                wan_seed,
                wan_steps,
                wan_guide,
                wan_shift,
                wan_scheduler,
            ],
            outputs=[status, queue_table, latest_video, log_tail],
        )
        refresh_button.click(refresh_queue, outputs=[queue_table, latest_video, log_tail])
        retry_button.click(retry_job, inputs=[retry_id], outputs=[status, queue_table, latest_video, log_tail])
        gallery_refresh.click(refresh_gallery, outputs=[gallery_select, gallery_video])
        gallery_select.change(select_gallery_video, inputs=[gallery_select], outputs=[gallery_video])
        settings_button.click(check_models, inputs=[settings_ltx_repo, settings_wan_dir], outputs=[settings_output])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch mlx-video Mini Studio")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().queue().launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
