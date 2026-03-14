# Notes

Personal learning log for understanding StreamingVLM and related concepts.

---

## Paper: StreamingVLM — Real-Time Understanding for Infinite Video Streams

### Core Idea
- Standard VLMs fail on long/infinite video because the KV cache grows quadratically with video length
- StreamingVLM fixes this by keeping a **compact KV cache**: sink tokens (first N) + sliding window (last M)
- The key contribution: **training-inference alignment** — SFT data is preprocessed to simulate the same truncation that happens at inference, eliminating distribution shift

### Architecture
- Built on **Qwen2.5-VL-7B-Instruct** (base model)
- Vision encoder is **frozen** during SFT (only the LLM is trained)
- Two position modes at inference:
  - `append`: position IDs grow indefinitely (like full attention but KV is evicted)
  - `shrink`: position IDs stay continuous (KV cache positions are remapped)

### SFT Pipeline
- **Stage 1**: trains on streaming captions (s12w24 format) + LiveCC commentary data
  - text_sink=512, text_sliding_window=512
- **Stage 2**: high-quality annealing on fine-grained QA data (`fg/` split)
- Training simulates streaming: video → multi-turn dialogue, one chunk of frames per turn
- `preprocess_conversation_stream()` in `lmm_dataset.py` handles this conversion

### Key Parameters
- `text_sink`: number of sink tokens kept from beginning of previous context
- `text_sliding_window`: number of recent tokens kept from end of previous context
- `FPS_MAX_FRAMES=480`: max frames per video (480 / 2 FPS = 4 min)
- `VIDEO_MAX_PIXELS=19267584`: max total video tokens (24k, leaving 8k for language)

### Benchmarks
- **Inf-Stream-Eval**: in-house benchmark, 66.18% win rate vs GPT-4o-mini (scored by GPT-4o-mini as judge)
- **OVOBench**: online video understanding benchmark
- **VQA**: general video QA via VLMEvalKit
- **LiveSports3k-cc**: live sports commentary evaluation
- **Efficiency**: measures inference time per chunk vs. cumulative video length across 4 modes:
  - **(a) FullAttention**: KV cache grows forever (`window_size=100000`), slows down quadratically
  - **(b) Sliding window w/o overlap**: fixed window, reuses KV cache, constant time but limited context. Paper shows a sawtooth pattern (cache fills then hard-resets every 100s). **Our implementation does NOT hard-reset** — it uses continuous selective pruning, so our mode b produces a flat noisy line instead.
  - **(c) Sliding window w/ overlap**: fixed window but recomputes from scratch each chunk (`recompute=True`), constant but slow
  - **(d) StreamingVLM**: sink + sliding window, no recompute (`text_sink=512`, `text_sliding_window=512`) — constant time AND fast
  - Output: JSON with per-chunk `gen_time_sec` vs `video_len_sec` → plot to reproduce paper figure
  - This is the most hardware-agnostic result (1 GPU sufficient) and good for advisor demos
  - **Output JSON metrics per chunk:**

    | Metric | What it measures |
    |---|---|
    | `gen_time_sec` | Wall-clock seconds the model spent generating tokens for this 1-second chunk (GEN phase only) |
    | `video_len_sec` | Cumulative video processed so far (1s, 2s, ..., 1000s) |
    | `avg_gen_time_sec` | Mean `gen_time_sec` across all chunks (summary field) |
    | `gen_time_per_token` | `gen_time_sec / decoded_tokens` — `null` in our run (token counts not tracked by inference.py) |

  - **Per-chunk terminal output fields** (printed as `[Loop N] total=Xs | PKV=Xs | ...`):

    | Field | What it measures |
    |---|---|
    | `total` | Wall-clock time for the entire chunk pipeline (target: ≤ `chunk_duration` = 1s for real-time) |
    | `PKV` | Time to prune/evict the KV cache — keeps first `text_sink` + last `text_sliding_window` tokens, drops the rest. Only costly once cache hits its size limit |
    | `CHECK` | Time for internal sanity checks on KV cache consistency |
    | `VIDEO` | Time to decode and resize the video frame(s) for this chunk |
    | `INPUT` | Time to tokenize and build the model input tensor (text + visual tokens) |
    | `GEN` | Time for the LLM to generate output tokens (forward pass + sampling) — this is `gen_time_sec` in the JSON. **GPU-bound.** |
    | `POST` | Takes the raw token IDs from GEN and converts them back to text (tokenizer decode), then formats and writes the `.vtt` subtitle file. **CPU-bound, essentially free.** |

  - **Key plot:** `gen_time_sec` vs `video_len_sec` across all 4 modes — mode (a) curves up quadratically, modes (b/c/d) stay flat, mode (d) has the lowest flat line

  - **Our plot vs paper plot — important difference:**
    - The paper's Y-axis is `gen_time_per_token` (seconds **per token generated**)
    - Our plot uses `gen_time_sec` (seconds **per chunk**) — a different metric
    - **Why we used per-chunk instead of per-token:** when we fixed the unpack bug in `efficiency_test.py` (line 74), `streaming_inference()` only returns timing data, not token counts. We had to default `token_decoded_num = [0] * len(time_result)`, making per-token latency impossible to compute. Per-chunk was the only available metric.
    - **Why this matters:** `gen_time_per_token` normalizes out variable caption length. A chunk that generates 20 tokens in 2s and one that generates 2 tokens in 0.2s both give 0.1 s/token. Without this normalization, noisy caption lengths dominate the plot, masking patterns like mode b's sawtooth (which is caused by the visual window resetting every 100 frames).
    - **Fix needed:** patch `streaming_inference()` to also return decoded token counts per chunk, then re-run all 4 modes to reproduce the paper figure exactly.

  - **KV cache eviction — how it actually works in this codebase:**
    - There is **no hard reset** (fully emptying the cache). All modes use `prune_id_and_kv_cache()` which surgically removes specific token ranges.
    - Eviction is **continuous and selective**: each chunk, the oldest visual frame tokens and/or oldest text round tokens are removed one at a time.
    - The paper's mode b sawtooth implies a full cache clear every N chunks — this behavior **does not exist in the codebase** and would require adding a `cache.reset()` call every `window_size` chunks in `streaming_inference()`.
    - What each mode actually prunes per chunk:
      - **(a)**: nothing (window so large it never triggers eviction)
      - **(b)**: oldest visual frame once >100 frames accumulated; no text eviction (`text_sink=None`, `text_sliding_window=None`)
      - **(c)**: nothing pruned — full recompute from scratch (`recompute=True`)
      - **(d)**: oldest visual frame + middle text tokens (keeps first `text_sink=512` + last `text_sliding_window=512`, drops middle)

---

## Setup & Environment

### Environments
- `streamingvlm-infer`: inference + all evals except OVOBench
- `streamingvlm-ovo`: OVOBench only (pins numpy==1.24.4, uses --no-deps)
- `streamingvlm-sft`: SFT training (needs deepspeed)

### Hardware
- Paper uses H100s, achieves up to 8 FPS
- Our setup: 2x A100s (temporarily unavailable as of 2026-03-12)

---

## Questions / Things to Look Up

- [ ] How exactly does `shrink` mode remap position IDs? See `pos_emb.py`
- [ ] What is the `s12w24` naming convention in dataset files? (sink=12? window=24?)
- [ ] How does liger kernel patch interact with the model forward pass?

---

## Resources
- Paper: `/workspace/storage_nassim/StreamingVLM/STREAMINGVLM.pdf`
- Repo: `https://github.com/NassimF/streaming-vlm`
- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Pretrained checkpoint: `mit-han-lab/StreamingVLM`
- Train dataset: `mit-han-lab/Inf-Stream-Train`
- Eval dataset: `mit-han-lab/Inf-Stream-Eval`
