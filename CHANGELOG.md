# Changelog

All code and configuration changes made to this repo are logged here.

## Format
Each entry should include:
- **Date**
- **What changed** (file, line, description)
- **Why** (reason for the change)
- **Result** (did it work, did it break something)

---

## Entries

### 2026-03-13
**File:** `streaming_vlm/eval/efficiency/efficiency_test.py` line 45
**Change:** Fixed hardcoded default `--video_path` — changed from Chinese-character filename to the ASCII-transliterated name that was actually downloaded (`FIFA_Worldcup_20180623_2018.RussiaWorldCupFGroup2Round_KoreaVSMexico_1080I_ITV_EN_2nd.ts.mp4`).
**Why:** HuggingFace Hub downloaded the file with a transliterated filename; the original Chinese name in the script caused `FileNotFoundError`.
**Result:** Benchmark runs without needing an explicit `--video_path` argument.

### 2026-03-13
**File:** `streaming_vlm/eval/efficiency/efficiency_test.py` line 74
**Change:** Fixed unpack bug — `streaming_inference` returns only `time_results` when `time_test=True`, but the script tried to unpack two values `(time_result, token_decoded_num)`. Fixed by assigning separately and defaulting `token_decoded_num` to zeros.
**Why:** `inference.py` doesn't track per-chunk token counts. `gen_time_per_token` will be None in output JSON (minor — the primary metric `gen_time_sec` is unaffected).
**Result:** Allows efficiency_test.py to run.

### 2026-03-12
**File:** `infer_requirements.txt`
**Change:** Using `av==12.0.0` instead of `av==14.4.0` during env setup
**Why:** `av==14.4.0` has no pre-built wheel and requires ffmpeg 7 dev libraries to build from source. `av==12.0.0` has pre-built wheels and is already the version specified in `env_infer.sh`. Inconsistency between the two files.
**Result:** Worked. `av==12.0.0` installed via pre-built wheel. Final env verified: torch 2.7.1, transformers 4.51.3, flash_attn 2.8.3.

### 2026-03-13
**File:** `streamingvlm-infer` env
**Change:** Manually installed `ffmpeg-python==0.2.0`
**Why:** It's in `infer_requirements.txt` but was silently dropped during installation (pip resolved `ffmpy` instead, which is a different package). `import ffmpeg` in `qwen2/patch_model.py` requires `ffmpeg-python`.
**Result:** Fixed ModuleNotFoundError for `ffmpeg`.

### 2026-03-12
**File:** `env_infer.sh` vs `infer_requirements.txt`
**Change:** Upgraded `transformers` from 4.51.3 to 4.52.4
**Why:** `env_infer.sh` pins 4.51.3 but the code imports `make_flex_block_causal_mask` which only exists in 4.52.x+. `infer_requirements.txt` correctly pins 4.52.4. The two files contradict each other.
**Result:** Fixed ImportError on `language_forward.py` line 7.

### 2026-03-12
**File:** `ovo_requirements.txt` / `scripts/env_ovo.sh`
**Change:** Installing `torch==2.7.1` before running `--no-deps` install, and using pre-built `flash_attn` wheel instead of building from source
**Why:** `--no-deps` skips dependency installs, so `flash_attn` fails to build because `torch` isn't present. Pre-built wheel avoids the build entirely.
**Result:** Worked. Verified: torch 2.7.1+cu126, flash_attn 2.8.3.

### 2026-03-14
**File:** `streaming_vlm/inference/inference.py`, `streaming_vlm/eval/efficiency/efficiency_test.py`
**Change:** Added `decoded_tokens` tracking per chunk. `inference.py` now stores `section_time['decoded_tokens'] = newly_generated_ids.shape[-1]` after generation. `efficiency_test.py` reads this to compute `gen_time_per_token` in the output JSON.
**Why:** Token counts were hardcoded to zero (workaround from the unpack bug fix). Without real token counts, `gen_time_per_token` is None — making the Y-axis incompatible with the paper's Figure 7.
**Result:** Output JSONs now include per-chunk token counts and valid `gen_time_per_token` values.

### 2026-03-14
**File:** `streaming_vlm/eval/efficiency/efficiency_test.py`
**Change:** Wrapped inference loop in `try/except` with `save_results(partial=True)` in the except block. Extracted save logic into a reusable `save_results()` helper.
**Why:** Mode a (Full Attention) OOMs at chunk ~698, crashing before the JSON save at the end of the script. All timing data was lost on crash.
**Result:** On OOM or any other exception, a `*_partial.json` file is saved with all chunks collected up to the crash point.

### 2026-03-14
**File:** `streaming_vlm/inference/inference.py`
**Change:** Added `hard_reset_interval=None` parameter. When set to an integer N, the KV cache, conversation history, and video window are fully cleared every N chunks.
**Why:** The paper's mode b (Sliding Window w/o Overlapping) uses non-overlapping windows with a hard cache reset between windows, producing the sawtooth latency pattern in Figure 7. The original implementation used continuous sliding eviction, producing a flat line instead.
**Result:** `baseline_b_config` in `efficiency_test.py` now sets `hard_reset_interval=100` to reproduce the paper's sawtooth.

### 2026-03-14
**File:** `streaming_vlm/eval/efficiency/plot_efficiency.py`
**Change:** Capped Y-axis at 0.7 s/token to match the paper's Figure 7 scale.
**Why:** Without the cap, partial mode a data (high raw gen_time_sec values with low token counts) dominated the Y-axis scale and compressed the b/c/d lines into noise.
**Result:** Plot now matches the paper's Y-axis range.

### 2026-03-20
**File:** `AGENTS.md` (new file)
**Change:** Created `AGENTS.md` with project goal, repo structure, key paths, conda environments, environment variables, and git reminder instruction.
**Why:** Provides persistent context for AI agents working in this repo across sessions.
**Result:** Future sessions start with full project context.

### 2026-03-20
**File:** `scripts/sft_stage_1.sh`, `scripts/sft_stage_2.sh`
**Change:** Adapted both scripts for 2-GPU setup: `--nproc_per_node=8` → `2`, `gradient_accumulation_steps=64` → `256`, `DATASET_PATH` set to actual path, `--dataloader_num_workers 32` → `4`, switched from WandB to `--report_to tensorboard`, removed WandB env vars from torchrun command.
**Why:** Paper uses 8× H100; we have 2× A100. Accumulation steps scaled to preserve effective batch size of 512. WandB personal entity disabled on this account.
**Result:** Scripts runnable on 2-GPU setup.

### 2026-03-20
**File:** `scripts/sft_stage_1.sh`
**Change:** Reduced `FPS_MAX_FRAMES` from 480 to 64, reduced `VIDEO_MAX_PIXELS` from 19267584 to 6422528.
**Why:** Original values caused CUDA OOM during first batch — too many frames × too high resolution = activations exceeded 80 GB VRAM.
**Result:** Forward pass completes without OOM.

### 2026-03-20
**File:** `streaming_vlm/utils/patch_liger_kernel.py` lines 83, 155-167, 173, 256
**Change:** Updated attribute paths for transformers 4.52.x API change in Qwen2.5-VL: `self.model.embed_tokens` → `self.model.language_model.embed_tokens`; `self.rope_deltas` → `self.model.rope_deltas`; `self.get_rope_index(...)` → `self.model.get_rope_index(...)`.
**Why:** In transformers 4.52, `Qwen2_5_VLModel` was refactored — the language model part moved into `self.model.language_model` (a `Qwen2_5_VLTextModel`), so `embed_tokens` and `rope_deltas` are no longer directly on `self.model`.
**Result:** `AttributeError: 'Qwen2_5_VLModel' object has no attribute 'embed_tokens'` fixed.

### 2026-03-20
**File:** `scripts/zero3.json`
**Change:** Added `"offload_optimizer": {"device": "cpu", "pin_memory": true}` to the ZeRO-3 config.
**Why:** Adam optimizer states for 8.29B params require ~33 GB per GPU (sharded across 2). Combined with model activations this exceeds 80 GB A100 VRAM, causing OOM at the first optimizer step.
**Result:** Optimizer states offloaded to CPU RAM; training fits within VRAM at cost of some speed.

---

## Progress Log

### 2026-03-13 — Inference Sanity Check ✅
**Status:** PASSED. Interrupted manually after ~20 chunks (sufficient for verification).
**Video tested:** `Inf-Stream-Eval/Baidu_NHL/NHL_Chicago Blackhawks vs Washington Capitals January 1st 2015.mp4`
**Model:** `mit-han-lab/StreamingVLM` (local copy at `/workspace/storage_nassim/models/StreamingVLM`)

**Sample output (first 5 chunks):**
```
Time=00:00:00-00:00:01:  ...                          past_key_values: 806
Time=00:00:04-00:00:05:  tonight's Bridgestone Winter Classic between the ...   past_key_values: 3790
Time=00:00:05-00:00:06:  Blackhawks and the Capitals in Washington, D.C. ...    past_key_values: 4543
Time=00:00:16-00:00:17:  this year. ...               past_key_values: 12011   ← KV cache stabilizes here
Time=00:00:18-00:00:19:  captain Patrick Kane will get back out there ...       past_key_values: 12021
```

**Per-chunk timing breakdown (Loop N):**
- `total`: wall-clock time for the entire chunk (target: ≤ chunk_duration = 1s for real-time)
- `PKV`: time to prune/reorganize the KV cache (sink + sliding window eviction)
- `CHECK`: sanity checks on cache consistency
- `VIDEO`: time to decode and resize the video frame(s) for this chunk
- `INPUT`: time to tokenize and build the model input tensor
- `GEN`: time for the model to generate tokens (the actual forward pass + sampling)
- `POST`: time to decode generated token IDs back to text and write the .vtt file

**Key observations:**
- Loop 0 is slow (2.68s) — first chunk triggers model warm-up and full KV cache build
- Loops 1+ settle to ~0.2-0.6s per 1-second chunk — faster than real-time ✅
- `past_key_values` grows 806→12011 over 16 chunks then stabilizes → sink+sliding window working ✅
- PKV cost jumps to ~0.117s at loop 16 (when eviction starts) — expected behavior ✅
- Commentary is semantically correct (correctly identifies teams, venue, player names) ✅

**Next step:** Efficiency benchmark (Option B — skipping full video inference for now)

---

## TODO — Next Steps (pick up from here)

### Completed ✅
- Inference sanity check (NHL video, ~20 chunks)
- Efficiency benchmark: modes b, c, d (full 1000s runs with token counts)
- Efficiency benchmark: mode a (partial ~698 chunks, OOM confirmed)
- Efficiency plot: `gen_time_per_token` vs video length (matches paper Figure 7 qualitatively)
- SFT env setup (`streamingvlm-sft`: torch 2.7.1, transformers 4.52.4, deepspeed 0.17.1, flash_attn 2.8.3)
- Adapted `sft_stage_1.sh` and `sft_stage_2.sh` for 2 GPUs (nproc=2, accum=256, tensorboard)
- Downloaded all `.jsonl` annotation files from `mit-han-lab/Inf-Stream-Train`
- Created filtered training set: `train_s12w24_local_with_seeks.jsonl` (19,409 samples, local videos only)
- **SFT Stage 1 training — COMPLETED ✅** (2026-03-20, 38 optimizer steps, loss 2.76→1.96)
  - Checkpoint: `./checkpoints/StreamingVLM_SFT_stage_1_e1_lr1e-5_ps512_pw512_20260320_083527/checkpoint-38/`

### Up Next — Stage 2 (GPU temporarily unavailable as of 2026-03-20)

1. **Filter fg/ dataset to local videos** (same issue as Stage 1 — many videos missing locally):
   ```bash
   /root/miniconda3/envs/streamingvlm-sft/bin/python3 -c "
   import json, os, random, pathlib
   dataset_root = '/workspace/storage_nassim/datasets/Inf-Stream-Train'
   for split in ['train', 'valid']:
       src = f'{dataset_root}/fg/{split}_fg_with_seeks.jsonl'
       lines = pathlib.Path(src).read_text().splitlines()
       kept = [l for l in lines if os.path.exists(os.path.join(dataset_root, json.loads(l)[0]['content'][0]['video']))]
       out = f'{dataset_root}/fg/{split}_fg_local_with_seeks.jsonl'
       pathlib.Path(out).write_text('\n'.join(kept) + '\n')
       seeks = []
       with open(out, 'rb') as f:
           while True:
               pos = f.tell(); line = f.readline()
               if not line: break
               seeks.append(pos)
       json.dump(seeks, open(f'{dataset_root}/fg/{split}_fg_local_seeks.jsonl', 'w'))
       print(f'{split}: {len(kept)}/{len(lines)} kept')
   "
   ```

2. **Update Stage 2 script** — set `model_name` and dataset names:
   - In `scripts/sft_stage_2.sh`:
     - `model_name` → `"/workspace/storage_nassim/StreamingVLM/checkpoints/StreamingVLM_SFT_stage_1_e1_lr1e-5_ps512_pw512_20260320_083527/checkpoint-38"`
     - Change `TRAIN_DATASET_NAMES` to `"fg/train_fg_local_with_seeks.jsonl"`
     - Change `VALID_DATASET_NAMES` to `"fg/valid_fg_local_with_seeks.jsonl"`
     - Add `FPS_MAX_FRAMES=64` and `VIDEO_MAX_PIXELS=6422528` (same as Stage 1 to avoid OOM)
     - Change `--report_to wandb` → `--report_to tensorboard`

3. **Run Stage 2**:
   ```bash
   cd /workspace/storage_nassim/StreamingVLM
   export PYTHONPATH=/workspace/storage_nassim/StreamingVLM:$PYTHONPATH
   conda activate streamingvlm-sft
   bash scripts/sft_stage_2.sh
   ```

4. **Remaining inference evals** (deferred):
   - Inf-Stream-Eval (GPT-4o-mini judge, OpenAI API key ready)
   - VQA evaluation (VLMEvalKit)
   - OVOBench (`streamingvlm-ovo` env)

### Known Bugs Fixed (2026-03-20 session)
- `train_s12w24_10pct_seeks.jsonl` missing → generated seeks for subsampled jsonl
- `liger_kernel 0.7.0` requires `transformers>=4.52` → upgraded from 4.51.3
- WandB personal entity disabled → switched to `--report_to tensorboard`
- Missing videos caused infinite retry recursion in dataloader → filtered to local-only videos
- CUDA OOM on first batch → reduced `FPS_MAX_FRAMES` 480→64, `VIDEO_MAX_PIXELS` 19267584→6422528
- `Qwen2_5_VLModel` no longer has `embed_tokens` in transformers 4.52 → patched `patch_liger_kernel.py` to use `self.model.language_model.embed_tokens`, `self.model.rope_deltas`, `self.model.get_rope_index`

---

## New Features

### 2026-03-26 — Feature 2: Live Camera Feed
**Branch:** `feat/demo-camera`
**Files changed/created:**
| File | Change |
|---|---|
| `demo/camera_inference.py` | New — reads JPEG frames from a directory, runs same StreamingVLM pipeline as inference.py, writes VTT cues |
| `demo/server.py` | Added `POST /frame` (saves browser JPEG frames), `POST /start-camera` (spawns camera_inference.py), `GET /camera-subtitles.vtt` |
| `demo/index.html` | Added "Use Camera" toggle button; `getUserMedia` camera capture; posts one JPEG frame/second to `/frame`; switches VTT polling endpoint to `/camera-subtitles.vtt` in camera mode |

**How it works:**
1. User clicks "Use Camera" → browser asks for camera permission → live feed appears in video element
2. User clicks "Start Inference" → server spawns `camera_inference.py`, browser starts posting frames every 1s
3. `camera_inference.py` watches `/tmp/camera_frames/` for new frames, runs the model, writes commentary to `/tmp/camera_live.vtt`
4. Commentary text box updates every 500ms as in video mode
5. Video mode (pre-recorded files) is unchanged — both modes coexist

**Bugs fixed before/during testing:**
- Race condition: server writing JPEG while inference read it → `PIL.UnidentifiedImageError` crash at frame 152. Fixed with atomic write (`os.replace` from `.tmp`).
- Added blank-frame fallback in `_load_frame()` so a single unreadable frame doesn't crash the loop.
- Tab switching pauses browser `setInterval` (Chrome throttles background tabs) → frame posting stops → inference times out. Fixed with Page Visibility API warning banner that appears when tab loses focus in camera mode.

**Test results (2026-03-26, ~6.7 min run, 403+ loops):**

*Latency:*
- Steady-state: ~0.9–1.1s per chunk (slightly over 1s budget but acceptable)
- Loop 383: **30.5s spike** — caused by KV cache eviction at the 16-second visual window boundary; when the oldest visual tokens are pruned and the cache is reorganized, it occasionally stalls. Not a crash — inference resumed normally on the next loop.

*Commentary quality:*
- The model hallucinated an entire fictional YouTube product review (vitamin D3 by "Nisha") and sustained it for the full 6+ minutes, largely ignoring the actual camera frames after the first few loops.
- Root cause: once the KV cache accumulates a few hallucinated tokens, the model conditions all future generation on that fictional context rather than re-anchoring to the visual input. The model was also trained exclusively on sports commentary, so a person sitting still in front of a camera is far out-of-distribution.
- Camera mode works best when pointed at something active and sports-like. For a static subject, commentary quality degrades quickly.
- Potential fix: use a more constrained system prompt or reset the KV cache more frequently to force visual re-anchoring.

### 2026-03-26 — Feature 1: Commentary Text Box Below Video
**Branch:** `feat/demo-textbox`
**File:** `demo/index.html`
**Change:** Replaced the HTML5 `<track>`/blob-URL subtitle overlay with a styled scrollable text box below the video. Commentary now accumulates as a flowing paragraph with no timestamps visible. Removed `swapTrack()`, `currentBlobUrl`, blob URL logic, and `video::cue` CSS. Updated `pollSubtitles()` to parse VTT cue texts, append only new cues since last poll, and auto-scroll the box to the bottom.

**Additional fixes after testing:**
- Stripped trailing ` ...` from model cue output (model appends this to most cues)
- Increased box height from 140px → 260px
- Auto-scroll only triggers when user is within 60px of the bottom (allows scrolling up freely)
**Result:** Tested and working. Merged into `feat/token-count-tracking`.

---

## Streaming Log

### 2026-03-23 — Real-Time Demo Created ✅



**Files created:**

| File | Purpose |
|---|---|
| `demo/server.py` | Python stdlib HTTP server — no extra dependencies |
| `demo/index.html` | Self-contained browser UI with live subtitle polling |
| `demo/run_demo.sh` | One-command launcher |
| `Streaming Inference Demo Plan.md` | Architecture and design notes |

**Architecture:**
```
bash demo/run_demo.sh
  └── demo/server.py (ThreadingHTTPServer, port 8765)
        ├── GET  /               → index.html
        ├── GET  /video          → video file with HTTP 206 range support
        ├── GET  /subtitles.vtt  → live VTT (fresh read, Cache-Control: no-cache)
        ├── GET  /status         → {"running": bool, "cues": int}
        └── POST /start          → spawn inference.py subprocess

Browser (Mac)
  └── index.html
        ├── <video src="/video">
        ├── Start button → POST /start → poll subtitles every 500ms
        └── Blob URL swap on each poll (forces browser to reload <track>)
```

**Key design decisions:**
- Python stdlib only — zero extra dependencies
- HTTP range requests (206 Partial Content) required for Chrome/Safari `<video>` seeking
- Blob URL swap: browsers cache `<track src="...">` aggressively; swapping Blob URLs forces reload
- Poll every 500ms: inference writes one cue/second at ~0.2–0.6s/chunk (faster than real-time)
- `waitForFirstCue()` waits for first subtitle before starting video playback for A/V sync

**Launch command:**
```bash
conda activate streamingvlm-infer
cd /workspace/storage_nassim/StreamingVLM
bash demo/run_demo.sh
# Open http://<server-ip>:8765 on Mac, click Start Inference

# 10-minute demo clips (trimmed to reduce buffering and start at action-packed segments)
# NHL — Blackhawks vs Capitals (starts at 28:00)
bash demo/run_demo.sh /workspace/storage_nassim/StreamingVLM/demo/nhl_demo_10min.mp4

# NBA — Pelicans vs Warriors (starts at 20:00)
bash demo/run_demo.sh /workspace/storage_nassim/StreamingVLM/demo/nba_demo_10min.mp4

# FIFA — Argentina vs Belgium, World Cup 2014 (starts at 20:00)
bash demo/run_demo.sh /workspace/storage_nassim/StreamingVLM/demo/fifa_demo_10min.mp4
```

**Status:** Files created and tested successfully end-to-end.

### 2026-03-23 — Increased chunk duration to 2 seconds (temporary, revert after testing)
**File:** `demo/server.py` line 170
**Change:** Added `--chunk_duration 2` to the inference subprocess command (default is 1).
**Why:** 1-second chunks produce very short subtitle text. Testing whether 2-second chunks generate longer, more descriptive commentary.
**Note:** Model was trained on 1-second chunks so this is out-of-distribution — revert to default if quality degrades.
**Result:** Real-time confirmed. Per-chunk timing at chunk_duration=2: `total≈0.65s`, `PKV≈0.23s`, `GEN≈0.38s` — processing each 2-second chunk in ~0.65s (~3× faster than real-time). Commentary quality appears reasonable for action scenes (e.g. "Puck battle along...").

### 2026-03-23 — Chunk duration comparison: reverted to 1 second (default)
**File:** `demo/server.py`
**Summary of all three chunk durations tested:**

| chunk_duration | Avg latency | Real-time? | Commentary quality |
|---|---|---|---|
| 1s (default) | ~0.39s | ✅ (~2.5× faster) | Best — full coherent sentences flow across consecutive cues |
| 2s | ~0.65s | ✅ (~3× faster) | Shorter fragments per cue; slightly less coherent |
| 4s | ~1.5s | ✅ (~2.7× faster) | Worst — model generates 1–2 word fragments ("with ...", "and ..."); out-of-distribution |

**Conclusion:** 1-second chunks are optimal. The model was trained on 1-second chunks so it produces the best commentary at that duration. Reverted `--chunk_duration` to `1` (the default).

### 2026-03-23 — Tested chunk_duration=4, reverted to 2
**File:** `demo/server.py`
**Change:** Tested `--chunk_duration 4`, then reverted to `--chunk_duration 2`.
**Result:** chunk_duration=4 produced worse commentary ("with ...", "your puck, ...", "and ...") despite having more visual context. Timing was `total≈1.5s` per 4-second chunk (still real-time). Root cause: model was trained on 1-second chunks, so 4-second chunks are out-of-distribution — it generates a fragment and stops instead of a full sentence. chunk_duration=2 remains the best tradeoff between commentary length and training distribution.

### 2026-03-23 — Removed timing metadata from VTT output
**File:** `streaming_vlm/inference/inference.py` line 527
**Change:** Removed `Infer Time: {loop_total:.3f}s\n` prefix from each VTT cue.
**Why:** Timing metadata was being written into the subtitle text and displayed on screen during the browser demo instead of clean commentary.
**Result:** Subtitles now show only the model's commentary text.

---
### Window Size

_Consistent with 3.4.2 in paper_

- Visual window:
*  DEFAULT_WINDOW_SIZE = 16 — inference.py:36
*  visual_round=window_size passed to process_past_kv() — inference.py:329
*  Meaning: the model keeps the most recent 16 seconds of video frames in the KV cache; older visual tokens are evicted.


- Text Sink:
*  DEFAULT_TEXT_SINK = 512 — inference.py:38
*   Used in process_past_kv() at line 165 as cut_start_idx = previous_text_start_idx + text_sink
*   Meaning: the first 512 generated text tokens are always kept ("anchor" tokens that stabilize attention).


- Text Sliding Window:
*  DEFAULT_TEXT_SLIDING_WINDOW = 512 — inference.py:39
*  Used at line 166 as cut_end_idx = previous_text_end_idx - text_sliding_window
*  Meaning: the most recent 512 generated text tokens are always kept. Together with the sink, total text context is capped at 1024 tokens regardless of video length.



### Inference Call Chain

Ordered list of files invoked during a single inference run, from browser click to subtitle file:

| # | File | Called | Does |
|---|---|---|---|
| 1 | `demo/server.py` | Once | Receives browser Start click → spawns inference subprocess |
| 2 | `streaming_vlm/inference/inference.py` | Once | Main orchestrator — loads model, loops through chunks |
| 3 | `streaming_vlm/inference/qwen2_5/patch_model.py` | Once at startup | Replaces standard Qwen2.5-VL forward methods with streaming versions |
| 4 | `streaming_vlm/livecc_utils/src/livecc_utils/video_process_patch.py` | Every chunk | Reads next chunk of video frames via decord and resizes to pixel budget |
| 5 | `streaming_vlm/inference/generate/streaming_generate_qwen.py` | Every chunk | Runs token generation loop until `<|im_end|>` is produced |
| 6 | `streaming_vlm/inference/qwen2_5/model_forward.py` | Every token | Combines video and text embeddings into one sequence, passes to LM |
| 7 | `streaming_vlm/inference/qwen2_5/vision_forward.py` | Every chunk | Runs frozen vision encoder on video frames → produces visual tokens |
| 8 | `streaming_vlm/inference/qwen2_5/language_forward.py` | Every token | Runs LM layers with flash attention, updates KV cache |
| 9 | `streaming_vlm/inference/qwen2_5/pos_emb.py` | Every token | Computes rotary position embeddings for combined video+text sequence |
| 10 | `streaming_vlm/utils/get_qwen_range.py` | Every chunk | Identifies token ranges (visual/sink/window) in cache for eviction |
| 11 | `streaming_vlm/inference/generate/streaming_cache.py` | Passive | Holds KV cache state between chunks; grows each chunk then gets pruned |
| 12 | `streaming_vlm/utils/vtt_utils.py` | Every chunk | Converts generated text to a timestamped VTT cue, appends to `.vtt` file |

**Note:** Files 6–10 run once per generated token (5–15 times per chunk). Files 4, 7, and 12 run once per chunk (once per second of video).

