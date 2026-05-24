# mlx-video Mini Studio Guide

Mini Studio is a local Gradio UI for running `mlx-video` generation jobs from a browser while keeping persistent job history, logs, and generated videos.

## Start

```bash
cd /path/to/mlx-video
source .venv/bin/activate
mlx_video.webui --server-port 7861
```

Open:

```text
http://127.0.0.1:7861
```

## Where State Is Stored

Mini Studio writes runtime state under `outputs/`:

- `outputs/jobs.jsonl` — persistent job history.
- `outputs/logs/<job-id>.log` — stdout/stderr for each generation command.
- `outputs/videos/<job-id>.mp4` — generated videos.

`outputs/` is ignored by git because it contains local runtime artifacts and large media files.

## Basic LTX-2 Flow

1. Open `Generate -> LTX-2`.
2. Fill `Prompt`.
3. Optionally upload an image for image-to-video.
4. Start with conservative settings:
   - `Pipeline`: `distilled`
   - `Width`: `512`
   - `Height`: `512`
   - `Frames`: `97`
   - `Steps`: `30`
5. Click `Add LTX job to persistent queue` once.
6. Open `Queue` and click `Refresh`.
7. Watch `status` and `Latest log tail`.
8. When the job is `done`, open `Gallery` and click `Refresh gallery`.

## Status Meanings

- `queued` — the job was created and is waiting.
- `running` — a subprocess is currently executing.
- `done` — the command exited successfully and should have produced an MP4.
- `failed` — the command exited with an error; inspect the job log.

## How To Understand What Is Happening

The most useful signal is the log file for the current job:

```bash
tail -f outputs/logs/<job-id>.log
```

Typical stages:

1. Model download from Hugging Face.
2. Text encoder loading.
3. Transformer/model loading.
4. Denoising/generation.
5. Video writing.

The first run can spend a long time downloading model weights. If Hugging Face warns about unauthenticated requests, setting `HF_TOKEN` can improve rate limits.

## Wan 2.x Notes

Wan generation requires an already converted MLX model directory. Mini Studio does not download or convert Wan weights in the first version.

Use the `Models / Settings` tab to check whether a local Wan model directory exists.

## Troubleshooting

If a job appears stuck:

1. Click `Queue -> Refresh`.
2. Check the newest job status.
3. Read the log:

   ```bash
   tail -n 120 outputs/logs/<job-id>.log
   ```

4. Confirm whether a generation process is alive:

   ```bash
   ps -axo pid,ppid,stat,etime,%cpu,%mem,command | rg "mlx_video|ltx_2|wan_2|webui"
   ```

If a job fails after downloading LTX-2 with:

```text
TypeError: LTXModelConfig.__init__() got an unexpected keyword argument '_class_name'
```

the local fix is to load LTX config through `LTXModelConfig.from_dict()`, which ignores Hugging Face metadata keys that are not dataclass fields.
