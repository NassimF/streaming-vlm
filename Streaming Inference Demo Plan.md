# Streaming Inference Demo Plan

## Context
The user wants a real-time demo for their advisor showing the model generating live commentary subtitles as a video plays in the browser. `inference.py` already writes the `.vtt` file chunk by chunk (~0.2–0.6s per chunk, faster than real-time). The demo server bridges inference and the browser.

---

## Architecture

```
[run_demo.sh]
    └── launches demo/server.py (Python stdlib HTTP server, port 8765)
              ├── GET  /               → serve index.html
              ├── GET  /video          → serve video file (with HTTP range support)
              ├── GET  /subtitles.vtt  → return live .vtt content (fresh read, no-cache)
              ├── GET  /status         → {"running": bool}
              └── POST /start          → spawn inference.py subprocess

[Browser (Mac)]
    └── index.html
              ├── <video src="/video">
              ├── Start button → POST /start → play video → poll subtitles every 500ms
              └── JS: swap Blob URL on each poll (forces browser to reload track)
```

---

## Files to Create

| File | Purpose |
|---|---|
| `demo/server.py` | Python stdlib HTTP server |
| `demo/index.html` | Self-contained browser UI |
| `demo/run_demo.sh` | One-command launcher |

---

## Key Design Decisions

1. **Python stdlib only** — no Flask/FastAPI needed, zero extra dependencies
2. **HTTP range requests** — required for Chrome/Safari `<video>` seeking (206 Partial Content)
3. **Blob URL swap** — browsers cache `<track>` aggressively; swapping Blob URLs forces reload
4. **Poll every 500ms** — inference writes one cue/second, polling is indistinguishable from live
5. **VTT at `/tmp/demo_live.vtt`** — temp path, cleaned up on each run

---

## How Real-Time Sync Works

- Inference writes one subtitle cue per second of video in ~0.2–0.6s (faster than real-time)
- Browser polls `/subtitles.vtt` every 500ms and refreshes the `<track>` element
- The HTML5 `<track>` element only displays a cue when the video clock reaches its timestamp
- Result: subtitles appear in sync with the video, even if they arrive slightly early

---

## Launch Command

```bash
conda activate streamingvlm-infer
cd /workspace/storage_nassim/StreamingVLM
export PYTHONPATH=/workspace/storage_nassim/StreamingVLM:$PYTHONPATH
bash demo/run_demo.sh
```

Then open `http://<server-ip>:8765` on your Mac and click **Start Inference**.

---

## Key Files Referenced

- `streaming_vlm/inference/inference.py` — CLI args: `--video_path`, `--output_dir`, `--model_path`, `--model_base Qwen2_5`
- `streaming_vlm/utils/vtt_utils.py` — `open_vtt()` appends one cue at a time, safe to poll
- `demo/server.py` — HTTP server (created)
- `demo/index.html` — Browser UI (created)
- `demo/run_demo.sh` — Launch script (created)
