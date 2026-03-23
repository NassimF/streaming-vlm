#!/usr/bin/env python3
"""
StreamingVLM Demo Server
Serves the demo HTML page, video file, and live .vtt subtitles.
Usage: python demo/server.py --video /path/to/video.mp4 --vtt /tmp/demo_live.vtt --model-path mit-han-lab/StreamingVLM
"""

import argparse
import http.server
import json
import os
import subprocess
import sys
from pathlib import Path

# Globals set from CLI args
VIDEO_PATH = None
VTT_PATH = None
MODEL_PATH = None
REPO_ROOT = Path(__file__).parent.parent
HTML_PATH = Path(__file__).parent / "index.html"

inference_proc = None


class DemoHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Suppress per-request logs except errors
        if args and str(args[1]) >= "400":
            super().log_message(format, *args)

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query string (cache-busting param)

        if path == "/":
            self._serve_file(HTML_PATH, "text/html")

        elif path == "/video":
            self._serve_video()

        elif path == "/subtitles.vtt":
            self._serve_vtt()

        elif path == "/status":
            self._serve_status()

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/start":
            self._start_inference()
        else:
            self.send_error(404)

    # -------------------------------------------------------------------------

    def _serve_file(self, path, content_type):
        try:
            data = Path(path).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)

    def _serve_video(self):
        if not VIDEO_PATH or not os.path.exists(VIDEO_PATH):
            self.send_error(404, "Video file not found")
            return

        file_size = os.path.getsize(VIDEO_PATH)
        range_header = self.headers.get("Range")

        if range_header:
            # Parse "bytes=start-end"
            range_val = range_header.strip().replace("bytes=", "")
            parts = range_val.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", length)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(VIDEO_PATH, "rb") as f:
                f.seek(start)
                remaining = length
                chunk = 64 * 1024  # 64KB chunks
                while remaining > 0:
                    data = f.read(min(chunk, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", file_size)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(VIDEO_PATH, "rb") as f:
                while True:
                    data = f.read(64 * 1024)
                    if not data:
                        break
                    self.wfile.write(data)

    def _serve_vtt(self):
        if VTT_PATH and os.path.exists(VTT_PATH):
            content = Path(VTT_PATH).read_text(encoding="utf-8")
        else:
            content = "WEBVTT\n\n"

        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/vtt; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_status(self):
        global inference_proc
        running = inference_proc is not None and inference_proc.poll() is None
        cues = 0
        if VTT_PATH and os.path.exists(VTT_PATH):
            content = Path(VTT_PATH).read_text(encoding="utf-8")
            cues = content.count("-->")

        body = json.dumps({"running": running, "cues": cues}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _start_inference(self):
        global inference_proc

        # Idempotent — don't restart if already running
        if inference_proc is not None and inference_proc.poll() is None:
            body = json.dumps({"status": "already_running"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
            return

        # Remove stale VTT from a previous run
        if VTT_PATH and os.path.exists(VTT_PATH):
            os.remove(VTT_PATH)

        cmd = [
            sys.executable, "-m", "streaming_vlm.inference.inference",
            "--video_path", VIDEO_PATH,
            "--output_dir", VTT_PATH,
            "--model_path", MODEL_PATH,
            "--model_base", "Qwen2_5",
            "--chunk_duration", "1",
        ]

        print(f"[server] Starting inference: {' '.join(cmd)}", flush=True)
        inference_proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )

        body = json.dumps({"status": "started"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


def main():
    global VIDEO_PATH, VTT_PATH, MODEL_PATH

    parser = argparse.ArgumentParser(description="StreamingVLM Demo Server")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--vtt", required=True, help="Path for live .vtt output")
    parser.add_argument("--model-path", default="mit-han-lab/StreamingVLM", help="Model path or HF repo")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on")
    args = parser.parse_args()

    VIDEO_PATH = args.video
    VTT_PATH = args.vtt
    MODEL_PATH = args.model_path

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), DemoHandler)
    print(f"[server] StreamingVLM Demo running at http://0.0.0.0:{args.port}")
    print(f"[server] Video : {VIDEO_PATH}")
    print(f"[server] VTT   : {VTT_PATH}")
    print(f"[server] Model : {MODEL_PATH}")
    print(f"[server] Open  : http://localhost:{args.port}  (or your server IP)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down...")
        if inference_proc and inference_proc.poll() is None:
            inference_proc.terminate()


if __name__ == "__main__":
    main()
