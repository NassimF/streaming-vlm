"""
Plot gen_time_sec vs video_len_sec for efficiency benchmark modes.
Reproduces the efficiency figure from the StreamingVLM paper.

Usage:
    python streaming_vlm/eval/efficiency/plot_efficiency.py \
        --output_dir output/efficiency \
        --out_path output/efficiency/efficiency_plot.png
"""

import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Map JSON "mode" field (or filename prefix) to display label and style
MODE_STYLES = {
    "baseline_a": dict(label="(a) Full Attention",          color="#d62728", linestyle="-",  linewidth=1.5),
    "baseline_b": dict(label="(b) Sliding w/o Overlap",     color="#ff7f0e", linestyle="--", linewidth=1.5),
    "baseline_c": dict(label="(c) Sliding w/ Recompute",    color="#2ca02c", linestyle="-.", linewidth=1.5),
    "streaming":  dict(label="(d) StreamingVLM (ours)",     color="#1f77b4", linestyle="-",  linewidth=2.0),
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def infer_mode_key(filename):
    base = os.path.basename(filename)
    for key in MODE_STYLES:
        if base.startswith(key):
            return key
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output/efficiency",
                        help="Directory containing efficiency JSON files")
    parser.add_argument("--out_path", default="output/efficiency/efficiency_plot.png",
                        help="Where to save the plot")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Rolling average window for gen_time_sec (1=no smoothing)")
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.output_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {args.output_dir}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    loaded = {}
    for path in json_files:
        mode_key = infer_mode_key(path)
        if mode_key is None:
            print(f"  [skip] unrecognised file: {os.path.basename(path)}")
            continue
        if mode_key in loaded:
            print(f"  [skip] duplicate mode {mode_key}, keeping first")
            continue
        data = load_json(path)
        chunks = data["per_chunk"]
        x = [c["video_len_sec"] for c in chunks]
        y = [c["gen_time_sec"]  for c in chunks]

        # Optional rolling average
        if args.smooth > 1:
            y = np.convolve(y, np.ones(args.smooth) / args.smooth, mode="valid")
            x = x[len(x) - len(y):]

        style = MODE_STYLES[mode_key]
        ax.plot(x, y, label=style["label"], color=style["color"],
                linestyle=style["linestyle"], linewidth=style["linewidth"], alpha=0.85)
        loaded[mode_key] = True
        print(f"  [loaded] {mode_key}: {len(chunks)} chunks, "
              f"avg gen={np.mean([c['gen_time_sec'] for c in chunks]):.3f}s")

    # Draw real-time line (gen_time == video_time, i.e. y=1 since each chunk = 1s video)
    xlim = ax.get_xlim()
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.2, label="Real-time threshold (1s/chunk)")

    ax.set_xlabel("Video length processed (seconds)", fontsize=12)
    ax.set_ylabel("Generation time per chunk (seconds)", fontsize=12)
    ax.set_title("Streaming Inference Efficiency: gen time vs video length", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_path, dpi=150)
    print(f"\nSaved plot -> {args.out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
