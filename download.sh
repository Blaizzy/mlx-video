#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="Lightricks/LTX-2"
TEXT_ENCODER_REPO="google/gemma-3-12b-it"
SKIP_TEXT_ENCODER=0
HF_HOME_DIR=""
HF_MIRROR="https://hf-mirror.com"

usage() {
  cat <<'USAGE'
Usage: ./download.sh [options]

Options:
  --model-repo REPO          Model repo to download (default: Lightricks/LTX-2)
  --text-encoder-repo REPO   Text encoder repo to download (default: google/gemma-3-12b-it)
  --skip-text-encoder        Skip downloading the text encoder repo
  --hf-home DIR              Set HF_HOME for Hugging Face cache location
  -h, --help                 Show this help

Notes:
  - Set HF_TOKEN in your environment if the repo is gated or requires auth.
  - The script uses HF mirror by default and automatically falls back to
    https://huggingface.co when the mirror returns HTTP 403.
  - Downloads go to the Hugging Face cache so the repo can find them.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-repo)
      MODEL_REPO="$2"
      shift 2
      ;;
    --text-encoder-repo)
      TEXT_ENCODER_REPO="$2"
      shift 2
      ;;
    --skip-text-encoder)
      SKIP_TEXT_ENCODER=1
      shift
      ;;
    --hf-home)
      HF_HOME_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

export HF_ENDPOINT="$HF_MIRROR"
if [[ -n "$HF_HOME_DIR" ]]; then
  export HF_HOME="$HF_HOME_DIR"
fi

download_repo() {
  local repo="$1"
  shift 1
  local patterns_str
  patterns_str=$(IFS='|'; echo "$*")

  REPO="$repo" ALLOW_PATTERNS="$patterns_str" "$PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ["REPO"]
patterns = os.environ.get("ALLOW_PATTERNS")
allow_patterns = patterns.split("|") if patterns else None
mirror = os.environ.get("HF_ENDPOINT")

def run_download(endpoint):
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        label = endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)
        label = "https://huggingface.co"
    print(f"==> Downloading {repo} via {label}")
    return snapshot_download(
        repo_id=repo,
        resume_download=True,
        allow_patterns=allow_patterns,
    )

try:
    run_download(mirror)
except Exception as err:  # pragma: no cover - shell script helper
    status = getattr(getattr(err, "response", None), "status_code", None)
    is_403 = status == 403 or "403" in str(err)
    if is_403:
        print("Mirror returned 403; retrying with https://huggingface.co")
        run_download("https://huggingface.co")
    else:
        raise
PY
}

download_repo "$MODEL_REPO" \
  "*.safetensors" \
  "*.json" \
  "tokenizer/**" \
  "tokenizer.*" \
  "special_tokens_map.json" \
  "tokenizer_config.json" \
  "tokenizer.json" \
  "vocab.json" \
  "merges.txt"

if [[ "$SKIP_TEXT_ENCODER" -eq 0 ]]; then
  download_repo "$TEXT_ENCODER_REPO" \
    "*.safetensors" \
    "*.json" \
    "tokenizer/**" \
    "tokenizer.*" \
    "special_tokens_map.json" \
    "tokenizer_config.json" \
    "tokenizer.json" \
    "tokenizer.model" \
    "vocab.json" \
    "merges.txt"
fi

echo "Done. Downloaded models are in the Hugging Face cache."
