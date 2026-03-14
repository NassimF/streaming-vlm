"""
Parse terminal output from efficiency_test.py into a JSON file
compatible with plot_efficiency.py.

Usage:
    # 1. Save your terminal output to a text file, e.g. mode_a_log.txt
    # 2. Run:
    python streaming_vlm/eval/efficiency/parse_efficiency_log.py \
        --log mode_a_log.txt \
        --mode baseline_a \
        --out output/efficiency/baseline_a_parsed.json
"""

import argparse
import json
import re
import os


LOOP_RE = re.compile(
    r"\[Loop\s+(\d+)\]\s+total=([\d.]+)s\s*\|"
    r"\s*PKV=([\d.]+)s\s*\|"
    r"\s*CHECK=([\d.]+)s\s*\|"
    r"\s*VIDEO=([\d.]+)s\s*\|"
    r"\s*INPUT=([\d.]+)s\s*\|"
    r"\s*GEN=([\d.]+)s\s*\|"
    r"\s*POST=([\d.]+)s"
)


def parse_log(log_path):
    chunks = []
    with open(log_path) as f:
        for line in f:
            m = LOOP_RE.search(line)
            if m:
                idx = int(m.group(1))
                chunks.append({
                    "chunk_index": idx,
                    "time_start_sec": idx,
                    "video_len_sec": idx + 1,
                    "gen_time_sec": float(m.group(7)),
                    "total_time_sec": float(m.group(2)),
                    "pkv_time_sec": float(m.group(3)),
                    "video_time_sec": float(m.group(5)),
                    "decoded_tokens": 0,
                    "gen_time_per_token": None,
                })
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to terminal log text file")
    parser.add_argument("--mode", default="baseline_a",
                        choices=["baseline_a", "baseline_b", "baseline_c", "streaming"],
                        help="Which benchmark mode this log is from")
    parser.add_argument("--out", default=None, help="Output JSON path (auto-named if omitted)")
    args = parser.parse_args()

    chunks = parse_log(args.log)
    if not chunks:
        print("No [Loop N] lines found — check the log file path/format.")
        return

    if args.out is None:
        os.makedirs("output/efficiency", exist_ok=True)
        args.out = f"output/efficiency/{args.mode}_parsed.json"

    data = {
        "meta": {
            "mode": args.mode,
            "source": "parsed_from_terminal_log",
            "log_file": args.log,
            "chunks_parsed": len(chunks),
            "loop_range": f"{chunks[0]['chunk_index']}–{chunks[-1]['chunk_index']}",
        },
        "per_chunk": chunks,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Parsed {len(chunks)} loops "
          f"(loop {chunks[0]['chunk_index']}–{chunks[-1]['chunk_index']})")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
