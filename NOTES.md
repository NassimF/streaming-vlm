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
  - **(b) Sliding window w/o overlap**: fixed window, reuses KV cache, constant time but limited context
  - **(c) Sliding window w/ overlap**: fixed window but recomputes from scratch each chunk (`recompute=True`), constant but slow
  - **(d) StreamingVLM**: sink + sliding window, no recompute (`text_sink=512`, `text_sliding_window=512`) — constant time AND fast
  - Output: JSON with per-chunk `gen_time_sec` vs `video_len_sec` → plot to reproduce paper figure
  - This is the most hardware-agnostic result (1 GPU sufficient) and good for advisor demos

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
