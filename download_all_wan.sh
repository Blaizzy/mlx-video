#!/bin/bash
# Sequential downloader + converter for all 4 Wan variants (mlx-video).
# Runs unattended in background; logs to ~/mlx-video/logs/download_all.log.

set -uo pipefail  # unbound var + fail on pipe error; NO -e (continue past failed models)

cd "$HOME/mlx-video"
source venv/bin/activate

LOG="$HOME/mlx-video/logs/download_all.log"
STATE="$HOME/mlx-video/logs/state.json"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

update_state() {
  python - "$1" "$2" "$3" "$STATE" <<'PY'
import json, sys, os, time
model, phase, status, path = sys.argv[1:5]
data = {}
if os.path.exists(path):
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
data.setdefault(model, {})[phase] = {"status": status, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
data["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(data, open(path, "w"), indent=2)
PY
}

download_and_convert() {
  local repo="$1"
  local short="$2"
  local ckpt="$HOME/mlx-video/checkpoints/$short"
  local out="$HOME/mlx-video/mlx-models/${short}-MLX"

  log "=========================================="
  log "START $short"
  log "  repo:   $repo"
  log "  ckpt:   $ckpt"
  log "  output: $out"
  log "=========================================="

  if [ -d "$out" ] && [ -f "$out/config.json" ]; then
    log "SKIP $short - MLX output already exists"
    update_state "$short" "convert" "already_done"
    return 0
  fi

  update_state "$short" "download" "in_progress"
  log "Downloading $repo ..."
  set +e
  hf download "$repo" --local-dir "$ckpt" --max-workers 8 >>"$LOG" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    update_state "$short" "download" "ok"
    log "Download OK: $(du -sh "$ckpt" | cut -f1)"
  else
    update_state "$short" "download" "failed_rc${rc}"
    log "DOWNLOAD FAILED for $short (rc=$rc) - skipping convert"
    return 1
  fi

  update_state "$short" "convert" "in_progress"
  log "Converting $short -> MLX bf16 ..."
  set +e
  python -m mlx_video.models.wan_2.convert \
        --checkpoint-dir "$ckpt" \
        --output-dir "$out" \
        --dtype bfloat16 >>"$LOG" 2>&1
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    update_state "$short" "convert" "ok"
    log "Convert OK: $(du -sh "$out" | cut -f1)"
  else
    update_state "$short" "convert" "failed_rc${rc}"
    log "CONVERT FAILED for $short (rc=$rc)"
    return 1
  fi

  log "DONE $short"
  echo "" >>"$LOG"
  return 0
}

log "======= Wan MLX bulk downloader starting (PID=$$) ======="
log "Free disk before: $(df -h ~ | tail -1 | awk '{print $4}')"

download_and_convert "Wan-AI/Wan2.1-T2V-1.3B"  "Wan2.1-T2V-1.3B" || true
download_and_convert "Wan-AI/Wan2.2-TI2V-5B"   "Wan2.2-TI2V-5B"  || true
download_and_convert "Wan-AI/Wan2.2-T2V-A14B"  "Wan2.2-T2V-A14B" || true
download_and_convert "Wan-AI/Wan2.2-I2V-A14B"  "Wan2.2-I2V-A14B" || true

log "======= ALL DONE ======="
log "Free disk after: $(df -h ~ | tail -1 | awk '{print $4}')"
log "Checkpoints total: $(du -sh ~/mlx-video/checkpoints | cut -f1)"
log "MLX models total:  $(du -sh ~/mlx-video/mlx-models | cut -f1)"
