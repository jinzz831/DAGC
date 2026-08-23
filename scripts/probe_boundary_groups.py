#!/usr/bin/env python
"""Decode videos and exercise the exact merge grouping without loading the VLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwenvl import load_video
from utils.boundary_segmentation import save_boundary_contact_sheets
from utils.data import EvalDatasetMLVU
from utils.dagc import DAGC


def make_grouping_engine(args):
    engine = DAGC.__new__(DAGC)
    engine.args = SimpleNamespace(chunk_size=args.chunk_size, fps=args.fps)
    engine.max_supernode_span = args.max_supernode_span
    engine.adjacent_sim_threshold = args.adjacent_sim_threshold
    engine.boundary_aware_merge = True
    engine.scene_boundary_threshold = args.scene_boundary_threshold
    engine.event_boundary_threshold = args.event_boundary_threshold
    engine.boundary_visual_weight = args.boundary_visual_weight
    engine.boundary_motion_weight = args.boundary_motion_weight
    engine.boundary_cross_motion_weight = args.boundary_cross_motion_weight
    engine.boundary_subtitle_weight = args.boundary_subtitle_weight
    engine.boundary_frame_window = args.boundary_frame_window
    engine.boundary_spatial_size = args.boundary_spatial_size
    engine.boundary_motion_scale = args.boundary_motion_scale
    engine.boundary_cross_motion_scale = args.boundary_cross_motion_scale
    # No MLVU subtitles are present; the batch encoder returns before touching these.
    engine.embedding_model = None
    engine.embedding_tokenizer = None
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--video_names", required=True)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--total_pixels", type=int, default=16384)
    parser.add_argument("--max_supernode_span", type=int, default=4)
    parser.add_argument("--adjacent_sim_threshold", type=float, default=0.95)
    parser.add_argument("--scene_boundary_threshold", type=float, default=0.45)
    parser.add_argument("--event_boundary_threshold", type=float, default=0.45)
    parser.add_argument("--boundary_visual_weight", type=float, default=0.30)
    parser.add_argument("--boundary_motion_weight", type=float, default=0.25)
    parser.add_argument("--boundary_cross_motion_weight", type=float, default=0.20)
    parser.add_argument("--boundary_subtitle_weight", type=float, default=0.25)
    parser.add_argument("--boundary_frame_window", type=int, default=4)
    parser.add_argument("--boundary_spatial_size", type=int, default=64)
    parser.add_argument("--boundary_motion_scale", type=float, default=0.05)
    parser.add_argument("--boundary_cross_motion_scale", type=float, default=0.40)
    args = parser.parse_args()

    names = {name.strip() for name in args.video_names.split(",") if name.strip()}
    dataset = EvalDatasetMLVU(args.data_path, mlvu_subset="needle")
    records = {}
    for item in dataset.data:
        if item["video_name"] in names:
            records.setdefault(item["video_name"], item)
    missing = sorted(names - set(records))
    if missing:
        raise ValueError(f"Unknown Needle videos: {missing}")

    os.makedirs(args.output_dir, exist_ok=True)
    visualization_dir = os.path.join(args.output_dir, "boundary_visualizations")
    engine = make_grouping_engine(args)
    summaries = []
    for video_name, item in records.items():
        _, _, _, _, effective_fps, video_inputs, _ = load_video(item["video_path"], args)
        chunks = list(torch.split(video_inputs[0], args.chunk_size, dim=0))
        groups, group_meta, boundaries = engine._build_supernode_groups(
            chunks, None, effective_fps=effective_fps
        )
        diagnostic = {
            "video_name": video_name,
            "num_chunks": len(chunks),
            "groups": groups,
            "boundaries": boundaries,
            "stats": {
                "original_chunk_count": len(chunks),
                "supernode_count": len(groups),
                "retained_node_ratio": len(groups) / max(len(chunks), 1),
                "max_chunks_per_supernode": max(map(len, groups), default=0),
                "scene_boundary_count": sum(x["hard_scene_boundary"] for x in boundaries),
                "event_boundary_count": sum(x["hard_event_boundary"] for x in boundaries),
                "blocked_by_scene_count": sum(x["blocked_by_scene"] for x in boundaries),
                "blocked_by_event_count": sum(x["blocked_by_event"] for x in boundaries),
            },
        }
        diagnostic["visualization_files"] = save_boundary_contact_sheets(
            chunks,
            boundaries,
            visualization_dir,
            video_name,
            args.boundary_frame_window,
            args.scene_boundary_threshold,
            args.event_boundary_threshold,
        )
        with open(os.path.join(args.output_dir, os.path.splitext(video_name)[0] + ".json"), "w") as f:
            json.dump(diagnostic, f, indent=2)
        summaries.append(diagnostic["stats"] | {"video_name": video_name})
        del video_inputs, chunks

    total_chunks = sum(x["original_chunk_count"] for x in summaries)
    total_nodes = sum(x["supernode_count"] for x in summaries)
    summary = {
        "videos": summaries,
        "original_chunk_count": total_chunks,
        "supernode_count": total_nodes,
        "retained_node_ratio": total_nodes / max(total_chunks, 1),
        "scene_boundary_count": sum(x["scene_boundary_count"] for x in summaries),
        "event_boundary_count": sum(x["event_boundary_count"] for x in summaries),
        "blocked_by_scene_count": sum(x["blocked_by_scene_count"] for x in summaries),
        "blocked_by_event_count": sum(x["blocked_by_event_count"] for x in summaries),
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
