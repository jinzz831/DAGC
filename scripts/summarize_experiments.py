#!/usr/bin/env python
"""Create the requested Baseline/Boundary-Aware comparison artifacts."""

from __future__ import annotations

import argparse
import json
import os


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run(path):
    output = read_json(os.path.join(path, "output.json"), [])
    result = read_json(os.path.join(path, "result.json"), {})
    graph = read_json(os.path.join(path, "graph_stats.json"), {})
    runtime = read_json(os.path.join(path, "runtime.json"), {})
    failures = read_json(os.path.join(path, "failures.json"), [])
    return {
        "path": path,
        "output": output,
        "result": result,
        "graph": graph.get("summary", {}),
        "runtime": runtime,
        "failures": failures,
    }


def row(run, method):
    graph = run["graph"]
    runtime = run["runtime"]
    return {
        "method": method,
        "needle_accuracy": run["result"].get("overall"),
        "correct": run["result"].get("correct"),
        "total": run["result"].get("total"),
        "original_chunk_count": graph.get("original_chunk_count"),
        "supernode_count": graph.get("supernode_count"),
        "retained_node_ratio": graph.get("retained_node_ratio"),
        "avg_chunks_per_supernode": graph.get("avg_chunks_per_supernode"),
        "max_chunks_per_supernode": graph.get("max_chunks_per_supernode"),
        "scene_boundary_count": graph.get("scene_boundary_count"),
        "event_boundary_count": graph.get("event_boundary_count"),
        "blocked_by_scene_count": graph.get("blocked_by_scene_count"),
        "blocked_by_event_count": graph.get("blocked_by_event_count"),
        "graph_runtime_seconds": runtime.get("graph_construction_runtime_seconds"),
        "qa_runtime_seconds": runtime.get("qa_online_runtime_seconds"),
        "wall_runtime_seconds": runtime.get("wall_runtime_seconds"),
        "gpu_peak_memory_mb": runtime.get("gpu_peak_memory_mb"),
        "failed_or_skipped_count": runtime.get("failed_or_skipped_count"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    baseline = load_run(args.baseline)
    method = load_run(args.method)
    baseline_row = row(baseline, "Existing DAGC")
    method_row = row(method, "Boundary-Aware DAGC")
    for key in (
        "scene_boundary_count",
        "event_boundary_count",
        "blocked_by_scene_count",
        "blocked_by_event_count",
    ):
        baseline_row[key] = None
    baseline_wall = baseline_row["wall_runtime_seconds"] or 0.0
    method_wall = method_row["wall_runtime_seconds"] or 0.0
    baseline_row["speedup"] = 1.0
    method_row["speedup"] = baseline_wall / method_wall if method_wall else None

    baseline_by_key = {
        (item.get("video_name"), item.get("question")): item
        for item in baseline["output"]
    }
    changed = []
    for item in method["output"]:
        key = (item.get("video_name"), item.get("question"))
        old = baseline_by_key.get(key)
        if old is None or old.get("pred") == item.get("pred"):
            continue
        changed.append({
            "video_name": key[0],
            "question": key[1],
            "answer": item.get("answer"),
            "baseline_pred": old.get("pred"),
            "method_pred": item.get("pred"),
            "baseline_correct": old.get("answer") in old.get("pred", "") or old.get("pred", "") in old.get("answer", ""),
            "method_correct": item.get("answer") in item.get("pred", "") or item.get("pred", "") in item.get("answer", ""),
        })

    report = {
        "rows": [baseline_row, method_row],
        "prediction_changes": changed,
        "baseline_failures": baseline["failures"],
        "method_failures": method["failures"],
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "comparison.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    def fmt(value, digits=4):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# MLVU Needle: DAGC vs Boundary-Aware DAGC",
        "",
        "| Method | Needle Accuracy | Retained Nodes | Avg Chunks/Super-node | Scene Boundaries | Event Boundaries | Runtime (s) | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in (baseline_row, method_row):
        lines.append(
            f"| {item['method']} | {fmt(item['needle_accuracy'])} | "
            f"{fmt(item['retained_node_ratio'])} | {fmt(item['avg_chunks_per_supernode'])} | "
            f"{fmt(item['scene_boundary_count'])} | {fmt(item['event_boundary_count'])} | "
            f"{fmt(item['wall_runtime_seconds'], 1)} | {fmt(item['speedup'], 3)}× |"
        )
    lines.extend(["", f"Prediction changes: {len(changed)}", ""])
    for item in changed:
        lines.append(
            f"- {item['video_name']}: {item['baseline_pred']} → {item['method_pred']} "
            f"(answer {item['answer']}; {item['baseline_correct']} → {item['method_correct']})"
        )
    with open(os.path.join(args.output_dir, "comparison.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
