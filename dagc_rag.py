import math
import torch
import os
import json
from transformers.trainer_pt_utils import IterableDatasetShard
import datetime
from tqdm import tqdm
import pysubs2
import argparse
import random
import time
import numpy as np
import hashlib
import shlex
import sys
import traceback
from itertools import chain
from utils.data import EvalDatasetMLVU, EvalDatasetVideoMME, EvalDatasetLongVideoBench, get_subtitles
from utils.dagc import DAGC
import pickle
from torch import distributed as dist
from utils.config import get_args

run_wall_start = time.perf_counter()
run_started_at = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
args = get_args()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8*24))
torch.distributed.barrier(device_ids=[local_rank])
world_size = torch.distributed.get_world_size()
world_rank = torch.distributed.get_rank()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if world_rank == 0:
    effective_keys = [
        "model_name", "model_path", "task", "data_path", "output_path", "graph_path",
        "uniform_frame", "chunk_size", "n_retrieval", "n_refine", "fps",
        "duration",
        "mlvu_subset", "mlvu_task_type", "mlvu_task", "debug_video_names", "graph_only", "graph_min_frames",
        "semantic_merge_threshold", "adjacent_sim_threshold",
        "subtitle_merge_threshold", "max_supernode_span",
        "supernode_target_frames", "top_seed_k", "original_expand_hop",
        "original_rerank_topk", "supernode_expand_hop", "subtitle_novelty_threshold",
        "keep_rank_checkpoints", "max_samples", "max_videos", "experiment_dir",
        "boundary_aware_merge", "scene_boundary_threshold", "event_boundary_threshold",
        "boundary_visual_weight", "boundary_motion_weight", "boundary_cross_motion_weight",
        "boundary_subtitle_weight", "boundary_frame_window", "boundary_spatial_size",
        "boundary_motion_scale", "boundary_cross_motion_scale",
        "boundary_debug", "boundary_diagnostics_dir", "boundary_visualization_dir",
        "seed",
    ]
    effective_args = {k: getattr(args, k, None) for k in effective_keys}
    effective_args["supernode_merge_rule"] = (
        "visual redundancy + max span + scene/event hard guards"
        if args.boundary_aware_merge
        else "visual_only: adjacent visual similarity >= adjacent_sim_threshold and group_len < max_supernode_span"
    )
    print("Effective args: " + json.dumps(effective_args, sort_keys=True), flush=True)
checkpoint_dir = args.experiment_dir or os.path.join(f"{args.output_path}/{args.model_name}/{args.task}")
os.makedirs(checkpoint_dir, exist_ok=True)
if args.boundary_diagnostics_dir is None:
    args.boundary_diagnostics_dir = os.path.join(checkpoint_dir, "boundary_diagnostics")
if args.boundary_visualization_dir is None:
    args.boundary_visualization_dir = os.path.join(checkpoint_dir, "boundary_visualizations")
if world_rank == 0:
    config_payload = vars(args).copy()
    with open(os.path.join(checkpoint_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    command_path = os.path.join(checkpoint_dir, "command.txt")
    if not os.path.exists(command_path):
        with open(command_path, "w", encoding="utf-8") as f:
            f.write(" ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]) + "\n")
checkpoint_file = os.path.join(checkpoint_dir, f"cuda:{world_rank}.json")

dagc = DAGC(args)


processed_identifiers = set() 
output = []

all_graph_stats_local = []
failed_records_local = []
rank_peak_memory_mb = 0.0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
        output = checkpoint_data.get('output', [])
        for item in output:
            if 'video_name' in item and 'question' in item:
                identifier = (item['video_name'], item['question']) 
                processed_identifiers.add(identifier)
    print(f"Rank {world_rank}: Resuming with {len(output)} already processed question-video pairs.")

def correct_tool_for_videomme(question, candidates, llm_info):
    q = question.lower()

    if llm_info is None:
        llm_info = {}

    # 真正的顺序题
    order_keywords = [
        "chronological order",
        "correct order",
        "sequential order",
        "sequence of",
        "in which order",
        "first",
        "second",
        "third",
        "after",
        "before",
        "next",
        "then",
    ]

    # 明确计数题
    count_keywords = [
        "how many",
        "number of",
        "times does",
        "times do",
        "exact number",
    ]

    # 普通属性/关系/识别题，不要强行 state/order/action
    normal_prefixes = (
        "which activity",
        "which skill",
        "which player",
        "which country",
        "which color",
        "what color",
        "what clothes",
        "what is the",
        "what are the",
        "who ",
        "where ",
    )

    if any(k in q for k in count_keywords):
        # 这里可以先统一 object counting；action counting 只在明确动作次数时使用
        if any(k in q for k in ["squat", "jump", "goal", "appear", "cross", "perform", "die", "action"]):
            llm_info["tool"] = "action counting"
        else:
            llm_info["tool"] = "object counting"
        llm_info["multiple"] = "yes"

    elif any(k in q for k in order_keywords):
        llm_info["tool"] = "order"
        llm_info["multiple"] = "yes"

    elif q.startswith(normal_prefixes):
        llm_info["tool"] = "none"

    # state change 只保留真正 before/after/change 类问题
    if llm_info.get("tool") == "state change":
        if not any(k in q for k in ["before", "after", "change", "changed", "turns into", "becomes"]):
            llm_info["tool"] = "none"

    return llm_info

def shard_dataset_by_video(dataset, world_size, world_rank, graph_min_duration):
    """Assign every video's questions to exactly one rank.

    Returning a rank bucket directly avoids the cache race caused by interleaving
    uneven buckets and then applying a second strided shard.
    """
    if not hasattr(dataset, "data"):
        return list(dataset)
    if world_size <= 1:
        return list(dataset.data)

    grouped = []
    group_by_video = {}
    for item in dataset.data:
        video_name = item.get("video_name")
        if video_name not in group_by_video:
            group_by_video[video_name] = []
            grouped.append(group_by_video[video_name])
        group_by_video[video_name].append(item)

    # Long videos pay the graph-generation cost once plus QA cost per question;
    # short videos pay only the QA cost. Greedy LPT assignment removes the very
    # large tail imbalance caused by round-robin placement of the long videos.
    weighted_groups = []
    for original_order, group in enumerate(grouped):
        duration = float(group[0].get("duration", 0.0) or 0.0)
        question_count = len(group)
        if duration >= graph_min_duration:
            estimated_cost = duration * (0.30 + 0.015 * question_count) + 80.0 * question_count
        else:
            estimated_cost = question_count * (20.0 + 0.04 * duration)
        weighted_groups.append((estimated_cost, original_order, group))

    bucket_groups = [[] for _ in range(world_size)]
    bucket_costs = [0.0] * world_size
    for estimated_cost, original_order, group in sorted(
        weighted_groups, key=lambda item: (-item[0], item[1])
    ):
        rank_idx = min(range(world_size), key=lambda idx: (bucket_costs[idx], idx))
        bucket_groups[rank_idx].append((original_order, group))
        bucket_costs[rank_idx] += estimated_cost

    rank_groups = sorted(bucket_groups[world_rank], key=lambda item: item[0])
    return list(chain(*(group for _, group in rank_groups)))

if args.task == "mlvu":
    mlvu_filter = (
        getattr(args, "mlvu_task", None)
        or getattr(args, "mlvu_task_type", None)
        or args.mlvu_subset
    )
    dataset = EvalDatasetMLVU(data_path=args.data_path, mlvu_subset=mlvu_filter)
    if world_rank == 0:
        print(f"MLVU task filter: {mlvu_filter or 'ALL'}", flush=True)
        print(
            f"MLVU filter samples: {dataset.filter_before_count} -> "
            f"{dataset.filter_after_count}; task_types={dataset.selected_task_types}",
            flush=True,
        )

    if getattr(args, "graph_only", False):
        graph_min_frames = args.graph_min_frames or (args.chunk_size * 20)
        min_duration = graph_min_frames / max(float(args.fps), 1e-8)
        before_graph_filter = len(dataset)
        dataset.data = [
            item for item in dataset.data
            if float(item.get("duration", 0) or 0) >= min_duration
        ]
        if world_rank == 0:
            print(
                f"Graph-only filter: duration >= {min_duration:.3f}s "
                f"(graph_min_frames={graph_min_frames}, fps={args.fps})",
                flush=True,
            )
            print(
                f"num_graph_only_samples: {len(dataset)} / {before_graph_filter}",
                flush=True,
            )
elif args.task == "videomme":
    dataset = EvalDatasetVideoMME(data_path=args.data_path)
    if getattr(args, "duration", None):
        duration_filter = {str(x).lower() for x in args.duration}
        before_duration_filter = len(dataset)
        dataset.data = [
            item for item in dataset.data
            if str(item.get("duration", "")).lower() in duration_filter
        ]
        if world_rank == 0:
            unique_long_videos = len({item.get("video_name") for item in dataset.data})
            if duration_filter == {"long"}:
                print("VideoMME subset: Long", flush=True)
                print(f"Total Long questions: {len(dataset)}", flush=True)
                print(f"Total Long videos: {unique_long_videos}", flush=True)
            print(
                f"VideoMME duration filter: {sorted(duration_filter)}",
                flush=True,
            )
            print(
                f"num_videomme_duration_samples: {len(dataset)} / {before_duration_filter}",
                flush=True,
            )
    
elif args.task == "lvb":
    dataset = EvalDatasetLongVideoBench(data_path=args.data_path)

if getattr(args, "debug_video_names", None):
    debug_video_names = set(args.debug_video_names)
    before_debug_filter = len(dataset)
    if hasattr(dataset, "data"):
        dataset.data = [item for item in dataset.data if item.get("video_name") in debug_video_names]
    else:
        dataset = [item for item in dataset if item.get("video_name") in debug_video_names]
    if world_rank == 0:
        print(f"Debug video filter: {sorted(debug_video_names)}", flush=True)
        print(f"num_debug_video_samples: {len(dataset)} / {before_debug_filter}", flush=True)
if args.max_samples is not None:
    if hasattr(dataset, "data"):
        dataset.data = dataset.data[:args.max_samples]
    else:
        dataset = list(dataset)[:args.max_samples]

if args.max_videos is not None and hasattr(dataset, "data"):
    selected_video_names = []
    for item in dataset.data:
        video_name = item.get("video_name")
        if video_name not in selected_video_names:
            selected_video_names.append(video_name)
        if len(selected_video_names) >= args.max_videos:
            break
    selected_video_names = set(selected_video_names)
    before_video_limit = len(dataset.data)
    dataset.data = [
        item for item in dataset.data if item.get("video_name") in selected_video_names
    ]
    if world_rank == 0:
        print(
            f"Max-video filter: {len(selected_video_names)} videos, "
            f"{before_video_limit} -> {len(dataset.data)} samples",
            flush=True,
        )

if args.task in {"videomme", "mlvu"}:
    graph_min_duration = (args.graph_min_frames or (args.chunk_size * 20)) / max(float(args.fps), 1e-8)
    rank_dataset = shard_dataset_by_video(
        dataset, world_size, world_rank, graph_min_duration
    )
    if world_rank == 0:
        print(f"{args.task} shard assignment: one video maps to exactly one rank", flush=True)
else:
    shard_dataset = IterableDatasetShard(
        dataset,
        batch_size=1,
        num_processes=world_size,
        process_index=world_rank,
    )
    rank_dataset = list(shard_dataset)

cache_identity = {
    "method": (
        "uncompressed_graph_rag"
        if args.direct_baseline
        else ("boundary_dagc" if args.boundary_aware_merge else "dagc")
    ),
    "task": args.task,
    "mlvu_task": (
        getattr(args, "mlvu_task", None)
        or getattr(args, "mlvu_task_type", None)
        or args.mlvu_subset
    ),
    "fps": float(args.fps),
    "chunk_size": int(args.chunk_size),
    "adjacent_sim_threshold": float(args.adjacent_sim_threshold),
    "max_supernode_span": int(args.max_supernode_span),
    "scene_boundary_threshold": float(args.scene_boundary_threshold),
    "event_boundary_threshold": float(args.event_boundary_threshold),
    "boundary_visual_weight": float(args.boundary_visual_weight),
    "boundary_motion_weight": float(args.boundary_motion_weight),
    "boundary_cross_motion_weight": float(args.boundary_cross_motion_weight),
    "boundary_subtitle_weight": float(args.boundary_subtitle_weight),
    "boundary_frame_window": int(args.boundary_frame_window),
    "boundary_spatial_size": int(args.boundary_spatial_size),
    "boundary_motion_scale": float(args.boundary_motion_scale),
    "boundary_cross_motion_scale": float(args.boundary_cross_motion_scale),
}
cache_hash = hashlib.sha256(
    json.dumps(cache_identity, sort_keys=True).encode("utf-8")
).hexdigest()[:12]
graph_tag = f"{args.task}_{cache_identity['method']}_{cache_hash}"
graph_base_dir = os.path.join(args.graph_path, graph_tag)
graph_stats_dir = os.path.join(graph_base_dir, "graph_stats")
os.makedirs(graph_stats_dir, exist_ok=True)
if world_rank == 0:
    with open(os.path.join(graph_base_dir, "cache_config.json"), "w", encoding="utf-8") as f:
        json.dump(cache_identity, f, ensure_ascii=False, indent=2, sort_keys=True)

torch.distributed.barrier(device_ids=[local_rank])
total_videos_for_rank = len(rank_dataset)
rank_record_counts = [None] * world_size
dist.all_gather_object(rank_record_counts, total_videos_for_rank)
if world_rank == 0:
    print(f"Per-rank record counts: {rank_record_counts}; total={sum(rank_record_counts)}", flush=True)
pbar = tqdm(rank_dataset, total=total_videos_for_rank, desc=f"Rank {world_rank} Processing Videos")

for line in pbar:
    question_start_time = time.perf_counter()
    video_name = line.get("video_name", None)
    answer = line.get("answer", None)
    prompt = line.get("prompt", None)
    question = line.get("question", None)
    task_type = line.get("task_type", None)
    video_path = line.get("video_path", None)
    candidates = line.get("candidates", None)
    subtitle_path = line.get("subtitle", None)
    duration = line.get("duration", None)

    current_identifier = (video_name, question)
    if current_identifier in processed_identifiers:
        continue

    if not os.path.exists(video_path):
        print(video_path)
        continue
    video_load_start_time = time.perf_counter()
    try:
        raw_video, _, _, frame_idx, fps, video_inputs, size_list = dagc.load_video(video_path, args)
        if "llava_video" in args.model_name:
            video = dagc.image_processor.preprocess(raw_video, return_tensors="pt")["pixel_values"].cuda().to(dtype=torch.bfloat16)
            video_inputs = [video]
        if type(video_inputs) is not list:
            video_inputs = [video_inputs]
    except Exception as exc:
        failure = {
            "video_name": video_name,
            "question": question,
            "stage": "load_video",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        failed_records_local.append(failure)
        print("[Failure] " + json.dumps(failure, ensure_ascii=False), flush=True)
        continue
    video_load_runtime_seconds = time.perf_counter() - video_load_start_time

    # subtitles = get_subtitles(subtitle_path, len(video_inputs[0]), fps=args.fps, data=line)
    subtitles = get_subtitles(subtitle_path, len(video_inputs[0]), fps=args.fps, data=line)

    graph_pkl_path = f"{graph_base_dir}/{video_name.split('.')[0]}.pkl"
    graph_stats_path = f"{graph_stats_dir}/{video_name.split('.')[0]}.json"

    graph_stats = None
    boundary_diagnostics = None
    original_chunk_count = int(math.ceil(len(video_inputs[0]) / args.chunk_size))

    graph_cache_hit = False
    graph_cache_saved = False
    graph_construction_runtime_seconds = 0.0

    # 只保留这一套 graph 逻辑，避免重复 load / build
    # if len(video_inputs[0]) < args.chunk_size * args.n_retrieval:
    if len(video_inputs[0]) < args.chunk_size * 20:
        video_graph, entity_graph = (None, None)
        graph_stats = {
            "video_name": video_name,
            "graph_built": False,
            "reason": "video_too_short_for_graph",
            "original_chunk_count": original_chunk_count,
            "supernode_count": None,
            "compression_ratio": None,
            "retained_node_ratio": None,
            "avg_chunks_per_supernode": None,
            "max_chunks_per_supernode": None,
            "merged_supernode_count": None,
            "scene_boundary_count": None,
            "event_boundary_count": None,
            "blocked_by_scene_count": None,
            "blocked_by_event_count": None,
            "blocked_by_max_span_count": None,
            "blocked_by_low_similarity_count": None,
            "avg_scene_change_score": None,
            "avg_event_boundary_score": None,
            "chunk_size": int(args.chunk_size),
            "fps": float(args.fps),
        }

    elif os.path.exists(graph_pkl_path):
        saved_graph = pickle.load(open(graph_pkl_path, "rb"))
        video_graph, entity_graph = saved_graph["video_graph"], saved_graph["entity_graph"]
        graph_cache_hit = video_graph is not None

        graph_stats = saved_graph.get("graph_stats", None)
        if graph_stats is None and video_graph is not None:
            graph_stats = video_graph.graph.get("stats", None)
        if video_graph is not None:
            boundary_diagnostics = video_graph.graph.get("boundary_diagnostics", None)

        if graph_stats is None:
            graph_stats = {
                "video_name": video_name,
                "graph_built": True,
                "original_chunk_count": original_chunk_count,
                "supernode_count": len(video_graph.nodes) if video_graph is not None else None,
                "compression_ratio": (len(video_graph.nodes) / original_chunk_count) if (video_graph is not None and original_chunk_count > 0) else None,
                "retained_node_ratio": (len(video_graph.nodes) / original_chunk_count) if (video_graph is not None and original_chunk_count > 0) else None,
                "avg_chunks_per_supernode": (original_chunk_count / len(video_graph.nodes)) if (video_graph is not None and len(video_graph.nodes) > 0) else None,
                "max_chunks_per_supernode": None,
                "merged_supernode_count": None,
                "chunk_size": int(args.chunk_size),
                "fps": float(args.fps),
            }

    else:
        graph_construction_start_time = time.perf_counter()
        video_graph, entity_graph = dagc.construct_graph(
            video_inputs,
            subtitles,
            video_name=video_name,
            effective_fps=fps,
        )
        graph_construction_runtime_seconds = time.perf_counter() - graph_construction_start_time

        graph_stats = video_graph.graph.get("stats", {})
        boundary_diagnostics = getattr(dagc, "last_boundary_diagnostics", None)
        graph_stats["video_name"] = video_name
        graph_stats["construction_runtime_seconds"] = graph_construction_runtime_seconds

        pickle.dump(
            {
                "video_graph": video_graph,
                "entity_graph": entity_graph,
                "graph_stats": graph_stats
            },
            open(graph_pkl_path, "wb")
        )
        graph_cache_saved = video_graph is not None

    # 每视频单独保存一份 stats json
    if graph_stats is not None:
        with open(graph_stats_path, "w", encoding="utf-8") as f:
            json.dump(graph_stats, f, ensure_ascii=False, indent=2)

        print(
            f"[Graph Stats] {video_name} | "
            f"original_chunks={graph_stats.get('original_chunk_count')} | "
            f"supernodes={graph_stats.get('supernode_count')} | "
            f"compression_ratio={graph_stats.get('compression_ratio')}",
            flush=True
        )

        # 只汇总真正建图的视频
        if graph_stats.get("graph_built", False):
            all_graph_stats_local.append(graph_stats)

    if boundary_diagnostics is not None and args.boundary_aware_merge:
        os.makedirs(args.boundary_diagnostics_dir, exist_ok=True)
        diagnostics_path = os.path.join(
            args.boundary_diagnostics_dir,
            f"{video_name.split('.')[0]}.json",
        )
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(boundary_diagnostics, f, ensure_ascii=False, indent=2)

    online_inference_start_time = time.perf_counter()
    query_list, llm_info = dagc.extract_keywords(question, candidates, video_inputs)

    # 先做 task_type 级别纠偏
    if llm_info is None:
        llm_info = {}

    q_lower = question.lower()

    if task_type == "findNeedle":
        llm_info["multiple"] = "no"
        if llm_info.get("tool") in ["action counting", "object counting", "order"]:
            llm_info["tool"] = "none"
        if "candidates_necessary" not in llm_info:
            llm_info["candidates_necessary"] = "no"

    # if args.task == "mlvu" and task_type == "count":
    #     if "action" in q_lower or "action scene" in q_lower:
    #         llm_info["tool"] = "action counting"
    #         llm_info["multiple"] = "yes"

    if args.task == "mlvu" and task_type == "count" and video_graph is not None:
        if "action" in q_lower or "action scene" in q_lower:
            llm_info["tool"] = "action counting"
            llm_info["multiple"] = "yes"


    if task_type == "order":
        llm_info["tool"] = "order"
        llm_info["multiple"] = "yes"
        llm_info["candidates_necessary"] = "yes"

    if args.task == "mlvu" and video_graph is None and task_type in ["order"]:
        llm_info["force_all_chunks_in_no_graph"] = True

    # anomaly_reco:
    # - no-graph 样本走原版式直答路径，避免误触发 action/order/state-change 检索
    # - graph 样本保留 supernode graph 流程
    if args.task == "mlvu" and task_type == "anomaly_reco" and video_graph is None:
        llm_info["tool"] = "none"
        llm_info["multiple"] = "no"
        llm_info["time"] = "none"
        llm_info["global"] = "yes"
        llm_info["candidates_necessary"] = "no"
        llm_info["states"] = []
        llm_info["temporal_keywords"] = []
    if args.task == "videomme":
        llm_info = correct_tool_for_videomme(question, candidates, llm_info)
    # 关键：纠偏后重新构建 query_list
    query_list = dagc.build_query_list_from_llm_info(question, candidates, llm_info)



    retrieved_node_list = dagc.retrieve_nodes(
        question, query_list, video_inputs, candidates,
        video_graph, entity_graph, subtitles, llm_info
    )
    retrieved_node_count = len(retrieved_node_list.get("nodes", [])) if isinstance(retrieved_node_list, dict) else None
    refined_node_list, sql_check, check_result = dagc.refine_nodes(
        retrieved_node_list,
        question,
        llm_info,
        candidates,
        video_inputs,
        subtitles,
        video_graph,
        size_list
    )
    # refined_node_list, sql_check, check_result = dagc.refine_nodes(retrieved_node_list, question, llm_info, candidates, video_inputs, subtitles, size_list)
    refined_node_count = len(refined_node_list.get("nodes", [])) if isinstance(refined_node_list, dict) else None
    pred = dagc.aggregate_nodes(refined_node_list, llm_info, video_inputs, raw_video, size_list, subtitles, prompt, line, video_graph, sql_check, check_result, fps)
    online_inference_runtime_seconds = time.perf_counter() - online_inference_start_time
    if torch.cuda.is_available():
        rank_peak_memory_mb = max(
            rank_peak_memory_mb,
            float(torch.cuda.max_memory_allocated()) / (1024 ** 2),
            float(torch.cuda.memory_allocated()) / (1024 ** 2),
        )

    graph_node_count = len(video_graph.nodes) if video_graph is not None else 0
    graph_edge_count = len(video_graph.edges) if video_graph is not None else 0
    graph_entity_nonempty_count = 0
    if video_graph is not None:
        for _, node_data in video_graph.nodes(data=True):
            if node_data.get("entities") or node_data.get("actions") or node_data.get("scenes"):
                graph_entity_nonempty_count += 1
    aggregate_debug = getattr(dagc, "last_aggregate_debug", {})

    output.append(
        {
            "question": question,
            "candidates": candidates,
            "task_type": duration if args.task == "videomme" else task_type,
            "video_name": video_name,
            "duration": len(video_inputs[0]),
            "domain": line.get("domain", None),
            "sub_category": line.get("sub_category", None),
            "video_id": line.get("video_id", None),
            "node_list": refined_node_list["nodes"][:args.n_refine],
            "info": llm_info,
            "sql_check": sql_check,
            "check_result": check_result,
            "pred": pred,
            "answer": answer,
            "graph_built": video_graph is not None,
            "graph_node_count": graph_node_count,
            "graph_edge_count": graph_edge_count,
            "graph_cache_path": graph_pkl_path if video_graph is not None else None,
            "graph_cache_hit": graph_cache_hit,
            "graph_cache_saved": graph_cache_saved,
            "video_input_len": len(video_inputs[0]),
            "n_chunks": original_chunk_count,
            "chunk_size": args.chunk_size,
            "n_retrieval": args.n_retrieval,
            "raw_final_output": aggregate_debug.get("raw_final_output"),
            "cleaned_final_output": aggregate_debug.get("cleaned_final_output"),
            "parse_status": aggregate_debug.get("parse_status"),
            "graph_entity_nonempty_count": graph_entity_nonempty_count,
            "retrieved_node_count": retrieved_node_count,
            "refined_node_count": refined_node_count,
            "selection_mode": aggregate_debug.get("selection_mode"),
            "final_indices_len": aggregate_debug.get("final_indices_len"),
            "final_indices_len_before_cap": aggregate_debug.get("final_indices_len_before_cap"),
            "final_indices_capped": aggregate_debug.get("final_indices_capped"),
            "actual_num_frames_seen_by_model": aggregate_debug.get("actual_num_frames_seen_by_model"),
            "adapter_media_type": aggregate_debug.get("adapter_media_type"),
            "video_load_runtime_seconds": video_load_runtime_seconds,
            "graph_construction_runtime_seconds": graph_construction_runtime_seconds,
            "online_inference_runtime_seconds": online_inference_runtime_seconds,
            "question_end_to_end_runtime_seconds": time.perf_counter() - question_start_time,
            "retained_node_ratio": graph_stats.get("retained_node_ratio") if graph_stats else None,
            "scene_boundary_count": graph_stats.get("scene_boundary_count") if graph_stats else None,
            "event_boundary_count": graph_stats.get("event_boundary_count") if graph_stats else None,
            "blocked_by_scene_count": graph_stats.get("blocked_by_scene_count") if graph_stats else None,
            "blocked_by_event_count": graph_stats.get("blocked_by_event_count") if graph_stats else None,
        }
    )
    print(output[-1], flush=True)

    processed_identifiers.add(current_identifier)
    with open(checkpoint_file, 'w') as f:
        json.dump({'output': output, 'processed_identifiers': list(processed_identifiers)}, f)
    
    print(f"Rank {world_rank} Output for {video_name[:8]}... - {question[:20]}...: {output[-1]['pred']}, answer: {answer}", flush=True)
    
dist.barrier()
    
final_output = [None] * world_size
dist.all_gather_object(
    final_output,
    output,
)
all_output = list(chain(*final_output))
final_graph_stats = [None] * world_size
dist.all_gather_object(final_graph_stats, all_graph_stats_local)
all_graph_stats = list(chain(*final_graph_stats))
final_failures = [None] * world_size
dist.all_gather_object(final_failures, failed_records_local)
all_failures = list(chain(*final_failures))
peak_memory_by_rank = [None] * world_size
dist.all_gather_object(peak_memory_by_rank, rank_peak_memory_mb)

global_rank = dist.get_rank()
if global_rank == 0:
    output_filename = os.path.join(checkpoint_dir, f"output.json")
    with open(output_filename, "w") as f:
        json.dump(all_output, f)
    
    result = {}
    task_types = set([item['task_type'] for item in all_output])
    for task_type in task_types:
        task_type_output = [item for item in all_output if item['task_type'] == task_type]
        accuracy = sum(1 for item in task_type_output if item['answer'] in item['pred'] or item['pred'] in item['answer']) / len(task_type_output)
        result[task_type] = accuracy
    result["overall"] = (
        sum(1 for item in all_output if item['answer'] in item['pred'] or item['pred'] in item['answer']) / len(all_output)
        if all_output else None
    )
    result["correct"] = sum(
        1 for item in all_output
        if item['answer'] in item['pred'] or item['pred'] in item['answer']
    )
    result["total"] = len(all_output)
    print(result)
    
    result_filename = os.path.join(checkpoint_dir, f"result.json")
    with open(result_filename, "w") as f:
        json.dump(result, f)
    with open(os.path.join(checkpoint_dir, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(all_failures, f, ensure_ascii=False, indent=2)
    # 汇总所有“真正建图”的视频 stats，输出到一个总 json
    graph_stats_all_path = os.path.join(
        graph_base_dir,
        "graph_stats_all.json"
    )

    # 按 video_name 去重，避免断点重跑造成重复
    dedup_graph_stats = {}
    for item in all_graph_stats:
        if item is None:
            continue
        video_key = item.get("video_name")
        if video_key is not None:
            dedup_graph_stats[video_key] = item

    merged_graph_stats = sorted(
        dedup_graph_stats.values(),
        key=lambda x: x.get("video_name", "")
    )

    with open(graph_stats_all_path, "w", encoding="utf-8") as f:
        json.dump(merged_graph_stats, f, ensure_ascii=False, indent=2)

    # 再额外输出一个 summary
    if len(merged_graph_stats) > 0:
        valid_ratios = [
            x["compression_ratio"]
            for x in merged_graph_stats
            if x.get("compression_ratio") is not None
        ]
        summary = {
            "num_graph_videos": len(merged_graph_stats),
            "original_chunk_count": sum(x["original_chunk_count"] for x in merged_graph_stats),
            "supernode_count": sum(x["supernode_count"] for x in merged_graph_stats),
            "avg_original_chunk_count": sum(x["original_chunk_count"] for x in merged_graph_stats) / len(merged_graph_stats),
            "avg_supernode_count": sum(x["supernode_count"] for x in merged_graph_stats) / len(merged_graph_stats),
            "avg_compression_ratio": sum(valid_ratios) / len(valid_ratios) if len(valid_ratios) > 0 else None,
            "min_compression_ratio": min(valid_ratios) if len(valid_ratios) > 0 else None,
            "max_compression_ratio": max(valid_ratios) if len(valid_ratios) > 0 else None,
            "global_weighted_compression_ratio": (
                sum(x["supernode_count"] for x in merged_graph_stats) /
                sum(x["original_chunk_count"] for x in merged_graph_stats)
            ),
            "retained_node_ratio": (
                sum(x["supernode_count"] for x in merged_graph_stats) /
                sum(x["original_chunk_count"] for x in merged_graph_stats)
            ),
            "avg_chunks_per_supernode": (
                sum(x["original_chunk_count"] for x in merged_graph_stats) /
                sum(x["supernode_count"] for x in merged_graph_stats)
            ),
            "max_chunks_per_supernode": max(x.get("max_chunks_per_supernode", 0) for x in merged_graph_stats),
            "merged_supernode_count": sum(x.get("merged_supernode_count", 0) for x in merged_graph_stats),
            "scene_boundary_count": sum(x.get("scene_boundary_count", 0) or 0 for x in merged_graph_stats),
            "event_boundary_count": sum(x.get("event_boundary_count", 0) or 0 for x in merged_graph_stats),
            "blocked_by_scene_count": sum(x.get("blocked_by_scene_count", 0) or 0 for x in merged_graph_stats),
            "blocked_by_event_count": sum(x.get("blocked_by_event_count", 0) or 0 for x in merged_graph_stats),
            "blocked_by_max_span_count": sum(x.get("blocked_by_max_span_count", 0) or 0 for x in merged_graph_stats),
            "blocked_by_low_similarity_count": sum(x.get("blocked_by_low_similarity_count", 0) or 0 for x in merged_graph_stats),
            "avg_scene_change_score": float(np.mean([
                x["avg_scene_change_score"] for x in merged_graph_stats
                if x.get("avg_scene_change_score") is not None
            ])) if any(x.get("avg_scene_change_score") is not None for x in merged_graph_stats) else None,
            "avg_event_boundary_score": float(np.mean([
                x["avg_event_boundary_score"] for x in merged_graph_stats
                if x.get("avg_event_boundary_score") is not None
            ])) if any(x.get("avg_event_boundary_score") is not None for x in merged_graph_stats) else None,
            "graph_construction_runtime_seconds": sum(
                float(x.get("construction_runtime_seconds", 0.0) or 0.0)
                for x in merged_graph_stats
            ),
        }
    else:
        summary = {
            "num_graph_videos": 0,
            "avg_original_chunk_count": None,
            "avg_supernode_count": None,
            "avg_compression_ratio": None,
        }

    graph_stats_summary_path = os.path.join(
        graph_base_dir,
        "graph_stats_summary.json"
    )
    with open(graph_stats_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(checkpoint_dir, "graph_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "videos": merged_graph_stats}, f, ensure_ascii=False, indent=2)

    boundary_summary = {
        "method": cache_identity["method"],
        "cache_hash": cache_hash,
        "scene_boundary_count": summary.get("scene_boundary_count"),
        "event_boundary_count": summary.get("event_boundary_count"),
        "blocked_by_scene_count": summary.get("blocked_by_scene_count"),
        "blocked_by_event_count": summary.get("blocked_by_event_count"),
        "avg_scene_change_score": summary.get("avg_scene_change_score"),
        "avg_event_boundary_score": summary.get("avg_event_boundary_score"),
        "diagnostics_dir": args.boundary_diagnostics_dir if args.boundary_aware_merge else None,
        "visualization_dir": args.boundary_visualization_dir if args.boundary_aware_merge else None,
    }
    with open(os.path.join(checkpoint_dir, "boundary_summary.json"), "w", encoding="utf-8") as f:
        json.dump(boundary_summary, f, ensure_ascii=False, indent=2)

    runtime = {
        "started_at": run_started_at,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
        "wall_runtime_seconds": time.perf_counter() - run_wall_start,
        "graph_construction_runtime_seconds": summary.get("graph_construction_runtime_seconds", 0.0),
        "qa_online_runtime_seconds": sum(
            float(item.get("online_inference_runtime_seconds", 0.0) or 0.0)
            for item in all_output
        ),
        "question_end_to_end_runtime_seconds_sum": sum(
            float(item.get("question_end_to_end_runtime_seconds", 0.0) or 0.0)
            for item in all_output
        ),
        "gpu_peak_memory_mb": max(peak_memory_by_rank) if peak_memory_by_rank else None,
        "gpu_peak_memory_mb_by_rank": peak_memory_by_rank,
        "failed_or_skipped_count": len(all_failures),
        "world_size": world_size,
    }
    with open(os.path.join(checkpoint_dir, "runtime.json"), "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)

    if not getattr(args, "keep_rank_checkpoints", False):
        for rank_idx in range(world_size):
            rank_checkpoint_file = os.path.join(checkpoint_dir, f"cuda:{rank_idx}.json")
            if os.path.exists(rank_checkpoint_file):
                os.remove(rank_checkpoint_file)
