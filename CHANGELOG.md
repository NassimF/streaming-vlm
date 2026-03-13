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

1. **Verify downloads completed** — check that these exist:
   - `/workspace/storage_nassim/models/StreamingVLM`
   - `/workspace/storage_nassim/datasets/Inf-Stream-Eval`
   - `/workspace/storage_nassim/datasets/Inf-Stream-Train`

2. **Run inference test** (needs GPUs + downloads done):
   ```bash
   conda activate streamingvlm-infer
   cd /workspace/storage_nassim/StreamingVLM
   export EVAL_DATASET_PATH=/workspace/storage_nassim/datasets/Inf-Stream-Eval
   python streaming_vlm/inference/inference.py \
     --model_path /workspace/storage_nassim/models/StreamingVLM
   ```
   Output: `.vtt` subtitle file in `output/`

3. **Run efficiency benchmark**:
   ```bash
   python streaming_vlm/eval/efficiency/efficiency_test.py --baseline_mode d
   ```

4. **Run Inf-Stream-Eval** (needs OpenAI API key):
   ```bash
   ./streaming_vlm/eval/model_compete/generate.sh -m mit-han-lab/StreamingVLM -b Qwen
   ./streaming_vlm/eval/model_compete/merge.sh mit-han-lab/StreamingVLM
   ./streaming_vlm/eval/model_compete/score.sh --model1 "mit-han-lab/StreamingVLM" --model2 "gpt-4o-mini"
   ```

5. **Set dataset paths** in SFT scripts before training:
   - `scripts/sft_stage_1.sh` → set `DATASET_PATH`
   - `scripts/sft_stage_2.sh` → set `DATASET_PATH` and `model_name` (Stage 1 checkpoint)

6. **SFT Stage 1** → **SFT Stage 2** (needs 2x A100s + full dataset)

<!-- Example:
### 2026-03-12
**File:** `scripts/sft_stage_1.sh`
**Change:** Set `DATASET_PATH` to `/workspace/storage_nassim/datasets/Inf-Stream-Train`
**Why:** Placeholder path needed to be set for local setup
**Result:** Training launched successfully
-->
