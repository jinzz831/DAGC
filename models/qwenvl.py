import time
import json
import os
import gc
from datetime import datetime

from transformers import AutoProcessor
import torch
from models.utils import LazySampledVideo, fetch_video, resize_video
import numpy as np


def load_video(video_path, args):
    raw_video, frame_idx, fps = fetch_video({"video": video_path, "fps": args.fps}, resize=False)
    video, fps = resize_video(
        raw_video,
        fps,
        total_pixels=args.total_pixels*max(1, int(round(np.ceil(len(raw_video) / args.chunk_size))))*28*28,
        resize_batch_frames=args.chunk_size,
    )
    lazy_raw_video = LazySampledVideo(video_path, frame_idx)
    del raw_video
    gc.collect()
    return [lazy_raw_video], None, None, frame_idx, fps, [video], None

def load_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    if "Qwen2.5" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        video_llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration
        video_llm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    processor = AutoProcessor.from_pretrained(model_name)
    video_llm.to("cuda")
    return None, video_llm, processor, None

# def mllm_response(video_llm, tokenizer, processor, text, image_inputs, video, max_new_tokens=512, size_list=None, fps=None):
#     if video is not None:
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "video",
#                         "video": "test.mp4",
#                         "max_pixels": 360 * 420,
#                         "fps": 1.0,
#                     },
#                     {"type": "text", "text": text},
#                 ],
#             }
#         ]
#     else:
#         messages = [
#             {
#                 "role": "user",
#                 "content": [{"type": "text", "text": text}],
#             }
#         ]
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     outputs = video_llm.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_logits=True)
#     generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
#     output_text = processor.decode(
#         generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     return output_text

def _append_perf_log(log_path, record):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def mllm_response(
    video_llm,
    tokenizer,
    processor,
    text,
    image_inputs,
    video,
    max_new_tokens=512,
    size_list=None,
    fps=None,
    perf_log_path="logs/flashvid_perf.jsonl",
    tag="default",
):
    if video is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "test.mp4",
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 统计输入长度
    input_token_length = int(inputs.input_ids.shape[1])

    # 清空并重置显存统计
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    outputs = video_llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_logits=True,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
    output_text = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # 时间统计
    elapsed_time = end_time - start_time
    generated_token_length = int(generated_tokens.shape[0])
    total_output_seq_len = int(outputs.sequences.shape[1])

    # 显存统计
    if torch.cuda.is_available():
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    else:
        peak_memory_bytes = 0
        peak_memory_mb = 0.0

    # FlashVID token 压缩统计
    original_visual_tokens = None
    compressed_visual_tokens = None
    token_reduction_ratio = None

    if hasattr(video_llm, "flashvid_config"):
        cfg = video_llm.flashvid_config
        original_visual_tokens = getattr(cfg, "original_visual_token_length", None)
        compressed_visual_tokens = getattr(cfg, "compressed_visual_token_length", None)
        token_reduction_ratio = getattr(cfg, "token_reduction_ratio", None)

    record = {
        "timestamp": datetime.now().isoformat(),
        "tag": tag,
        "model_name": str(video_llm.__class__.__name__),
        "input_token_length": input_token_length,
        "generated_token_length": generated_token_length,
        "total_output_seq_len": total_output_seq_len,
        "elapsed_time_sec": round(elapsed_time, 4),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "original_visual_tokens": original_visual_tokens,
        "compressed_visual_tokens": compressed_visual_tokens,
        "token_reduction_ratio": round(token_reduction_ratio, 6) if token_reduction_ratio is not None else None,
        "flashvid_enabled": hasattr(video_llm, "flashvid_config"),
    }

    _append_perf_log(perf_log_path, record)

    # 终端也打印一份，便于调试
    # print(
    #     f"[Perf] tag={tag} time={elapsed_time:.3f}s "
    #     f"peak_mem={peak_memory_mb:.1f}MB "
    #     f"visual_tokens={original_visual_tokens}->{compressed_visual_tokens} "
    #     f"reduction={token_reduction_ratio}"
    # )

    return output_text
