# Agent Instructions

## Project Goal
Reproduce all results from the StreamingVLM paper ("StreamingVLM: Real-Time Understanding for Infinite Video Streams"). This includes:
1. Inference benchmarks (efficiency, Inf-Stream-Eval, VQA, OVOBench)
2. Full SFT training pipeline (Stage 1 + Stage 2)

## Repo Structure
```
StreamingVLM/
├── train.py                        # SFT training entry point
├── streaming_vlm/
│   ├── inference/
│   │   ├── inference.py            # Core streaming inference loop
│   │   ├── streaming_args.py       # StreamingArgs (pos_mode, all_text)
│   │   └── generate/
│   │       └── streaming_cache.py  # Custom KV cache (sink + sliding window)
│   ├── data/
│   │   └── lmm_dataset.py          # SFT data pipeline
│   └── eval/
│       └── efficiency/
│           ├── efficiency_test.py  # Efficiency benchmark (4 modes)
│           ├── plot_efficiency.py  # Plot gen_time_per_token vs video length
│           └── parse_efficiency_log.py  # Parse terminal log → JSON
├── scripts/
│   ├── sft_stage_1.sh              # Stage 1 training script (8 GPUs → adapted to 2)
│   ├── sft_stage_2.sh              # Stage 2 training script
│   ├── zero3.json                  # DeepSpeed ZeRO-3 config
│   └── env_sft.sh                  # SFT environment setup
├── output/
│   └── efficiency/                 # Benchmark JSON results + plots
├── CHANGELOG.md                    # Log of all code changes made
└── NOTES.md                        # Learning log and paper concepts
```

## Key Paths
- **Models**: `/workspace/storage_nassim/models/StreamingVLM`
- **Inference dataset**: `/workspace/storage_nassim/datasets/Inf-Stream-Eval`
- **Training dataset**: `/workspace/storage_nassim/datasets/Inf-Stream-Train`

## Conda Environments
- `streamingvlm-infer` — inference, efficiency benchmarks, Inf-Stream-Eval
- `streamingvlm-sft` — SFT training
- `streamingvlm-ovo` — OVOBench evaluation

## Key Environment Variables
```bash
export PYTHONPATH=/workspace/storage_nassim/StreamingVLM:$PYTHONPATH
export DATASET_PATH=/workspace/storage_nassim/datasets/Inf-Stream-Eval   # for inference
export DATASET_PATH=/workspace/storage_nassim/datasets/Inf-Stream-Train  # for training
```

## Git
- Active branch for efficiency/token-count work: `feat/token-count-tracking`
- At every major checkpoint, remind the user to commit changes to git.
