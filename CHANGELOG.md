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

### In Progress / Up Next
1. **SFT Training** (current focus):
   - Download `.jsonl` annotation files from `mit-han-lab/Inf-Stream-Train`
   - Download LiveCC dataset (`chenjoya/Live-WhisperX-526K`) + flatten
   - Adapt `scripts/sft_stage_1.sh` and `scripts/sft_stage_2.sh` for 2 GPUs
   - Run Stage 1, then Stage 2

2. **Remaining inference evals** (deferred):
   - Inf-Stream-Eval (GPT-4o-mini judge, OpenAI API key ready)
   - VQA evaluation (VLMEvalKit)
   - OVOBench (`streamingvlm-ovo` env)
