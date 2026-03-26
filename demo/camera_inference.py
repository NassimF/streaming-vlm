#!/usr/bin/env python3
"""
Camera-based streaming inference for StreamingVLM demo.
Reads JPEG frames posted by the browser (via demo/server.py POST /frame)
and generates live commentary written to a .vtt file.

Usage:
    python demo/camera_inference.py \
        --frame_dir /tmp/camera_frames \
        --output_dir /tmp/camera_live.vtt \
        --model_path mit-han-lab/StreamingVLM \
        --model_base Qwen2_5
"""
import argparse, os, time
import torch
import numpy as np
from PIL import Image

from streaming_vlm.inference.inference import (
    load_model_and_processor,
    process_past_kv,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_TEXT_ROUND,
    DEFAULT_TEXT_SINK,
    DEFAULT_TEXT_SLIDING_WINDOW,
    DEFAULT_TEMPERATURE,
    DEFAULT_REPETITION_PENALTY,
    MAX_TOKEN_PER_DURATION,
    SYSTEM_PROMPT_OFFSET,
    TOKEN_IDS,
)
from streaming_vlm.inference.streaming_args import StreamingArgs
from streaming_vlm.utils.vtt_utils import open_vtt, sec2ts
from qwen_vl_utils.vision_process import (
    VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS, FRAME_FACTOR, FPS
)
from qwen_vl_utils.vision_process import smart_resize

CHUNK_DURATION = 1  # seconds per chunk — must match frame posting interval in browser


def _wait_for_frame(frame_dir, frame_idx, timeout=60):
    """Block until frame_{frame_idx:05d}.jpg appears in frame_dir."""
    path = os.path.join(frame_dir, f"frame_{frame_idx:05d}.jpg")
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() > deadline:
            return None
        time.sleep(0.05)
    return path


def _load_frame(path, target_h, target_w):
    """Load a JPEG and return uint8 numpy array of shape [1, H, W, 3]."""
    img = Image.open(path).convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)[np.newaxis]  # [1, H, W, 3]


def camera_inference(frame_dir, output_dir, model_path, model_base="Qwen2_5",
                     window_size=DEFAULT_WINDOW_SIZE, text_round=DEFAULT_TEXT_ROUND,
                     text_sink=DEFAULT_TEXT_SINK, text_sliding_window=DEFAULT_TEXT_SLIDING_WINDOW,
                     temperature=DEFAULT_TEMPERATURE, repetition_penalty=DEFAULT_REPETITION_PENALTY,
                     max_chunks=3600):

    os.makedirs(frame_dir, exist_ok=True)

    model, processor = load_model_and_processor(model_path, model_base)
    device = model.device
    streaming_args = StreamingArgs(pos_mode="shrink", all_text=False)

    assistant_start_bias = len(processor(text="<|im_start|>assistant\n")['input_ids'][0])
    assistant_end_bias = len(processor(text=" ...<|im_end|>")['input_ids'][0])

    # Compute frame resize target (same logic as inference.py)
    NFRAMES = FPS * window_size
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / NFRAMES * FRAME_FACTOR),
        int(VIDEO_MIN_PIXELS * 1.05)
    )
    resized_height, resized_width = smart_resize(
        360, 640, factor=28, min_pixels=VIDEO_MIN_PIXELS, max_pixels=max_pixels
    )

    # Init VTT file
    if output_dir and os.path.exists(output_dir):
        os.remove(output_dir)
    with open_vtt(output_dir):
        pass

    past_key_values = None
    full_conversation_history = []
    prev_generated_ids = None
    recent_video_window_clips = []
    recent_pixel_values_videos = []
    query = "Commentate on what you see"

    print(f"[camera_inference] Model loaded. Waiting for frames in {frame_dir}...", flush=True)

    for i in range(max_chunks):
        loop_start = time.perf_counter()
        start_time = float(i * CHUNK_DURATION)

        frame_path = _wait_for_frame(frame_dir, i, timeout=60)
        if frame_path is None:
            print(f"[camera_inference] Timeout waiting for frame {i}. Stopping.", flush=True)
            break

        current_video_chunk = _load_frame(frame_path, resized_height, resized_width)

        # KV cache pruning (identical to inference.py)
        past_key_values, prev_generated_ids, recent_video_window_clips, recent_pixel_values_videos = process_past_kv(
            past_key_values, i,
            text_round=text_round, visual_round=window_size,
            full_conversation_history=full_conversation_history,
            prev_generated_ids=prev_generated_ids,
            assistant_start_bias=assistant_start_bias,
            assistant_end_bias=assistant_end_bias,
            recent_video_window_clips=recent_video_window_clips,
            recent_pixel_values_videos=recent_pixel_values_videos,
            text_sink=text_sink,
            text_sliding_window=text_sliding_window,
        )
        recent_video_window_clips.append(current_video_chunk)

        # Build conversation
        prompt = f'Time={start_time:.1f}-{start_time + CHUNK_DURATION:.1f}s'
        if i == 0:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "video", "video": frame_path},
                {"type": "text", "text": query},
            ]
            full_conversation_history = [
                {"role": "previous text", "content": ""},
                {"role": "user", "content": user_content},
            ]
            text = processor.apply_chat_template(
                full_conversation_history, tokenize=False, add_generation_prompt=True
            )
        else:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "video", "video": frame_path},
            ]
            full_conversation_history.append({"role": "user", "content": user_content})
            text = processor.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False, add_generation_prompt=True
            )
            text = '\n' + text[SYSTEM_PROMPT_OFFSET:]

        inputs = processor(
            text=[text],
            videos=recent_video_window_clips[-1],
            padding=True,
            return_tensors="pt",
        ).to(device)

        if prev_generated_ids is not None:
            if prev_generated_ids[:, -1].item() != TOKEN_IDS["\n"]:
                inputs['input_ids'] = torch.cat([prev_generated_ids, inputs['input_ids']], dim=1)
            else:
                inputs['input_ids'] = torch.cat([prev_generated_ids, inputs['input_ids'][:, 1:]], dim=1)
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

        recent_pixel_values_videos.append(inputs['pixel_values_videos'])
        streaming_args.input_ids = inputs['input_ids']
        if i == 0:
            streaming_args.video_grid_thw = inputs['video_grid_thw']
            streaming_args.second_per_grid_ts = inputs.get('second_per_grid_ts')
        else:
            streaming_args.video_grid_thw = torch.cat(
                [streaming_args.video_grid_thw, inputs['video_grid_thw']], dim=0
            )
            if streaming_args.second_per_grid_ts is not None and 'second_per_grid_ts' in inputs:
                streaming_args.second_per_grid_ts = torch.cat(
                    [streaming_args.second_per_grid_ts, inputs['second_per_grid_ts']], dim=0
                )

        current_input_len = inputs['input_ids'].shape[1]

        outputs = model.generate(
            **inputs,
            past_key_values=past_key_values,
            max_new_tokens=MAX_TOKEN_PER_DURATION,
            use_cache=True,
            return_dict_in_generate=True,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            streaming_args=streaming_args,
            pad_token_id=151645,
            temperature=temperature,
        )

        generated_ids = outputs.sequences
        if generated_ids[0, -1].item() != 151645:
            generated_ids = torch.cat(
                [generated_ids, torch.tensor([[151645]], device=device)], dim=1
            )
        newly_generated_ids = generated_ids[:, current_input_len:]
        response = processor.batch_decode(newly_generated_ids, skip_special_tokens=True)[0]

        past_key_values = outputs.past_key_values
        prev_generated_ids = generated_ids.clone()
        full_conversation_history.append({"role": "assistant", "content": response})

        loop_total = time.perf_counter() - loop_start
        hms = time.strftime('%H:%M:%S', time.gmtime(int(start_time)))
        print(f"[Loop {i}] total={loop_total:.3f}s | {hms}: {response}", flush=True)

        ts_start = sec2ts(start_time)
        ts_end = sec2ts(start_time + CHUNK_DURATION)
        with open_vtt(output_dir) as vf:
            vf.write(f"{ts_start} --> {ts_end}\n{response}\n\n")

    print("[camera_inference] Done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_path", default="mit-han-lab/StreamingVLM")
    ap.add_argument("--model_base", default="Qwen2_5", choices=["Qwen2_5", "Qwen2"])
    ap.add_argument("--text_sink", type=int, default=DEFAULT_TEXT_SINK)
    ap.add_argument("--text_sliding_window", type=int, default=DEFAULT_TEXT_SLIDING_WINDOW)
    ap.add_argument("--max_chunks", type=int, default=3600)
    args = ap.parse_args()
    camera_inference(**vars(args))
