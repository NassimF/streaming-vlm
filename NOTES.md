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
  - `s12w24` = visual sink of 12 frames + visual window of 24 frames used when generating the annotation data
  - Teaches the model to do continuous streaming commentary over long videos (coarse-grained)
- **Stage 2**: high-quality annealing on fine-grained QA data (`fg/` split)
  - Fine-tunes Stage 1 checkpoint on precise Q&A about events in the video
  - Sharpens factual accuracy on top of Stage 1's streaming capability
- Training simulates streaming: video → multi-turn dialogue, one chunk of frames per turn
- `preprocess_conversation_stream()` in `lmm_dataset.py` handles this conversion

### GPU Adaptation (8 → 2 GPUs)
- The paper and training scripts assume 8 GPUs (8× H100s)
- We have 2× A100 80GB
- To maintain the same effective batch size of 512: `8 GPUs × 1 batch × 64 accum = 512`
- With 2 GPUs: `2 × 1 × 256 accum = 512` — so gradient_accumulation_steps must increase to 256
- This keeps the training mathematics identical but makes each optimizer step 4× slower → ~4× longer total training time

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

  - **Decoded token count per chunk:**
    - Token count is **not fixed** — it varies per chunk depending on how much the model has to say about that second of video
    - The model generates tokens until it emits `<|im_end|>` (token ID `151645` in Qwen's vocabulary — the "end of turn" special token) or hits `MAX_TOKEN_PER_DURATION` (hard cap)
    - Quiet/uneventful seconds → few tokens (e.g. `"..."` = ~3–5 tokens)
    - Action-packed seconds → more tokens (e.g. `"He puts his hand up deliberately."` = ~9 tokens)
    - This variability is why `gen_time_per_token` is a better metric than `gen_time_sec` — it normalizes out caption length and isolates the attention cost

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
- Our setup: 2x A100s 






---

## Presentation Notes

### Efficiency Benchmark Plots

- **Qestion:** Does this mode slow down as the video gets longer?

**Plot 1 — Generation time per chunk**

![efficiency_plot](output/efficiency/efficiency_plot.png)

**Plot 2 — Generation time per token**

![efficiency_plot_per_token](output/efficiency/efficiency_plot_per_token.png)

#### Difference between the two plots



Both plots show the same 4 inference modes (a/b/c/d) over 1000 seconds of video. The only difference is what is on the Y-axis:

| | Plot 1 | Plot 2 |
|---|---|---|
| Y-axis | `gen_time_sec` — total seconds the model spent generating text for that 1-second chunk | `gen_time_per_token` — generation time divided by the number of tokens produced |
| Real-time threshold | 1.0s (one chunk = one second of video) | 0.1s/token (paper's threshold) |
| Matches paper | No | Yes |

**Note on mode b (Sliding Window w/o Overlapping):** In the paper, mode b shows a sawtooth pattern which means the KV cache accumulates for 100 seconds, then gets fully wiped, causing latency to drop and rise again periodically. The released GitHub code does not include this hard-reset functionality, so my reproduction produces a flat cumulative line instead. 
#### Why per-token is the better metric

Each 1-second video chunk does not produce the same number of tokens. A quiet moment (no action) might produce 3–4 tokens (`"..."`), while an eventful second generates 12–15 tokens (`"He puts his hand up deliberately."`).

Dividing by token count normalizes for this variability and isolates the **attention cost** — the part that actually differs between modes. 

The paper uses `gen_time_per_token` to demonstrate that StreamingVLM maintains **constant-time inference** regardless of video length, staying below the 0.1s/token real-time threshold — unlike full attention which degrades and eventually OOMs. 

#### Why there is a spike in the second graph at the beginning for StreamingVLM?

The initial spike above 0.1s/token for mode d is caused by CUDA warm-up on the first chunk. 
- Chunk 0 is the first ever forward pass — CUDA kernels need to be compiled and loaded into GPU memory for the first time. This is a one-time fixed cost that has nothing to do with the attention mechanism.

- With smooth=10, this warm-up cost gets averaged into chunks 1–9 as well, stretching the elevated region further.
After ~50 seconds, mode d settles to ~0.05s/token and stays flat below the threshold — consistent with the paper.

The authors probably Discarded the first N chunks as a warm-up period before recording timing

### Inference Process

#### Are there two separate KV caches?

**No — there is only ONE KV cache**, belonging to the language model. Here's the full picture:

**Vision Encoder — No KV cache**
The vision encoder is stateless. Each chunk of frames is processed fresh through `streaming_visual_encoder_forward()` ([vision_forward.py:104](streaming_vlm/inference/qwen2_5/vision_forward.py#L104)), which returns only visual **embeddings** — no KV states are stored or reused.

**Language Model — One unified KV cache**
The `StreamingCache` class ([streaming_cache.py](streaming_vlm/inference/generate/streaming_cache.py)) holds a single unified cache with `key_cache[layer]` and `value_cache[layer]` for every LM layer. This cache stores **both** visual tokens and text tokens together — they are merged into one `inputs_embeds` sequence ([model_forward.py:93](streaming_vlm/inference/qwen2_5/model_forward.py#L93)) before the LM ever sees them.

**How eviction works on the single cache**
`process_past_kv()` ([inference.py:87-172](streaming_vlm/inference/inference.py#L87)) manages everything by token position:

| What gets evicted | When | Parameters |
|---|---|---|
| Old **visual** tokens | Every chunk after `visual_round` | `Vwindow = 16s` |
| Old **text** tokens (middle) | Every chunk after `text_round` | `Tsink=512, Twindow=512` |

Both use the same `prune_id_and_kv_cache()` function ([inference.py:50-61](streaming_vlm/inference/inference.py#L50)).

**Paper confirmation (page 3):**
> *"we reuse the states of (i) a set of sink text tokens of length Tsink; (ii) a long window of the most recent text tokens of length Twindow; and (iii) a short window of the most recent vision tokens of length Vwindow."*

All three refer to regions of the **same** KV cache — they are just different logical sections of it.

### Sample video Inference

- Saved at:/workspace/storage_nassim/StreamingVLM/output/_workspace_storage_nassim_models_StreamingVLM_viswin16_txtwin16_prvsink512_prvwin512_tprt0.9.vtt

- viswin16 — visual sliding window = 16. The seconds of visual context kept in the visual KV cache. Older frames beyond the last 16 get evicted.

- txtwin16 — text sliding window = 16 tokens kept from recent text context in the visual KV cache side.

- prvsink512 — previous text sink = 512. The number of tokens from the very beginning of the generated text that are always kept (the "anchor" tokens that stabilize attention).

- prvwin512 — previous text sliding window = 512. The most recent 512 generated text tokens kept in the rolling context. Together with prvsink512, the total text context is capped at 1024 tokens no matter how long the video is.

- tprt0.9 — temperature = 0.9. Controls randomness in token sampling during generation. 0.9 is slightly below 1.0 (fully random), making the output a bit more focused while still allowing variation. 



### Training

#### Changes I made
- Sports SFT set: In paper->	525K samples	Mine: 19,409 (local videos only, ~3.7%) because it would take days on my weaker gpu. (Paper took 128 H100 days for full training)
- Did not use the LiveCC dataset because it was too large. In total paper used 1M samples but I used 19k
-  Hit OOM so reduced FPS_MAX_FRAMES from 480 → 64 (fewer frames per video clip) and Reduced VIDEO_MAX_PIXELS from 19M → 6.4M (lower resolution per frame). This shrunk the activation tensors that had to be held during the forward pass
-  Adam optimizer states for 8.29B params require ~33 GB per GPU (sharded across 2). Combined with model activations this exceeds 80 GB A100 VRAM, causing OOM at the first optimizer step. Solution:Optimizer states offloaded to CPU RAM; training fits within VRAM at cost of some speed.
- gradient_accumulation_steps: 64 → 256 (to compensate for 2 GPUs vs 8)

#### Result
- 'train_runtime': 52039.5793, 'train_samples_per_second': 0.373, 'train_steps_per_second': 0.001, 'train_loss': 2.0986071323093616, 'epoch': 1.0

- Visualize :conda activate streamingvlm-sft
tensorboard --logdir /workspace/storage_nassim/StreamingVLM/checkpoints/ --port 6006

- train/epoch — Progress through the dataset. Goes 0→1 linearly over 38 steps, confirming 1 full epoch completed.

- train/loss — Per-step training loss. Dropped from ~2.76→~1.96. The sharp drop in the first ~10 steps is the model quickly adapting from the base Qwen2.5-VL weights to the sports commentary task. Flattening after step 15 means it's converging — good sign.

- train/grad_norm — Magnitude of the gradients before the weight update. Starts high (~22) and drops to ~1.5. High early grad_norm is normal at the start of fine-tuning. The steady decrease means training is stable (no exploding gradients).

- train/learning_rate — The cosine decay schedule. Ramps up for the first ~3 steps (warmup), peaks at 1e-5, then decays to ~0 by step 38. This is exactly what --warmup_ratio 0.03 --lr_scheduler_type cosine produces.

- train/train_loss — Same as train/loss but logged once at the end (epoch-level average = 2.0986). Single dot because it's a summary metric.

- train/total_flos — Total floating point operations performed (~5.7×10¹⁷). A measure of total compute used. Single dot, logged at end only.

- train/train_runtime — Total training time in seconds (~52,039s = 14.5 hours). Single dot.

- train/train_samples_per_second — Throughput: 0.373 samples/sec. Slow due to CPU optimizer offloading, but expected given the workaround.

---

## Resources
- Paper: `/workspace/storage_nassim/StreamingVLM/STREAMINGVLM.pdf`
- Repo: `https://github.com/NassimF/streaming-vlm`
- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Pretrained checkpoint: `mit-han-lab/StreamingVLM`
- Train dataset: `mit-han-lab/Inf-Stream-Train`
- Eval dataset: `mit-han-lab/Inf-Stream-Eval`
