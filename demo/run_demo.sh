#!/usr/bin/env bash
# StreamingVLM Real-Time Demo Launcher
# Usage: bash demo/run_demo.sh [path/to/video.mp4] [model_path]
#
# Defaults:
#   video  = NHL Blackhawks vs Capitals (Inf-Stream-Eval dataset)
#   model  = mit-han-lab/StreamingVLM (or local checkpoint)

set -e
cd "$(dirname "$0")/.."   # repo root

# ── Paths ────────────────────────────────────────────────────────────────────
DEFAULT_VIDEO="/workspace/storage_nassim/datasets/Inf-Stream-Eval/Baidu_NHL/NHL_Chicago Blackhawks vs Washington Capitals January 1st 2015.mp4"
DEFAULT_MODEL="mit-han-lab/StreamingVLM"

VIDEO_PATH="${1:-$DEFAULT_VIDEO}"
MODEL_PATH="${2:-$DEFAULT_MODEL}"
PORT=8765

# ── Generate unique VTT filename: <video_basename>_<YYYYMMDD_HHMMSS>.vtt ────
VIDEO_BASENAME=$(basename "$VIDEO_PATH" .mp4)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VTT_DIR="$(dirname "$0")/subtitles"
mkdir -p "$VTT_DIR"
VTT_PATH="$VTT_DIR/${VIDEO_BASENAME}_${TIMESTAMP}.vtt"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -f "$VIDEO_PATH" ]; then
    echo "ERROR: Video file not found: $VIDEO_PATH"
    echo "Usage: bash demo/run_demo.sh [video_path] [model_path]"
    exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  StreamingVLM Live Demo"
echo "  Video : $VIDEO_PATH"
echo "  Model : $MODEL_PATH"
echo "  VTT   : $VTT_PATH"
echo "  Open  : http://localhost:$PORT"
echo "========================================"
echo ""
echo "  On your Mac, open: http://<server-ip>:$PORT"
echo "  Then click 'Start Inference' in the browser."
echo ""

export PYTHONPATH="$(pwd):$PYTHONPATH"

python demo/server.py \
    --video "$VIDEO_PATH" \
    --vtt   "$VTT_PATH" \
    --model-path "$MODEL_PATH" \
    --port  "$PORT"
