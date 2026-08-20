import json
import re
import ast
import importlib
import numpy as np
import networkx as nx
from collections import defaultdict

import torch
from transformers import AutoModel, AutoTokenizer

from utils.prompts import *
from utils.retrieval import compute_text_similarity, extract_choices, allocate_node, node2indices, count_and_sort_filtered
from utils.boundary_segmentation import (
    appearance_and_motion_scores,
    chunk_motion_magnitude,
    encode_subtitle_chunks,
    save_boundary_contact_sheets,
    subtitle_semantic_scores,
)
from models.utils import resize_video

MODEL_MAP = {
    "qwenvl25_7b": ("models.qwenvl", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "qwenvl25_3b": ("models.qwenvl", "Qwen/Qwen2.5-VL-3B-Instruct"),
    # Backward-compatible aliases. Pass --model_path to use a local checkpoint.
    "qwenvl25_7b_local": ("models.qwenvl", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "qwenvl25_3b_local": ("models.qwenvl", "Qwen/Qwen2.5-VL-3B-Instruct"),
}


def extract_choice_from_response(text, letters):
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("```", "").replace("Answer?", "Answer:").strip()
    valid_letters = [str(letter).upper() for letter in letters]
    letter_class = "".join(re.escape(letter) for letter in valid_letters)
    patterns = [
        rf"(?i)\b(?:answer|final answer|option|choice)\s*[:?]?\s*[\(\[]?([{letter_class}])\b",
        rf"(?im)^\s*[\(\[]?([{letter_class}])\s*[\)\].:?]?\s*$",
        rf"(?i)\b([{letter_class}])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1).upper(), cleaned, "matched"
    return None, cleaned, "unmatched"

class Vgent():
    def __init__(self, args):
        self.args = args

        # model_name = self.args.model_name
        # module_name = "models.qwenvl"
        # # module_name = "models.llavavideo"
        # model_path = model_name
        # module = importlib.import_module(module_name)

        matched = next(
            ((module, model_path) for key, (module, model_path) in MODEL_MAP.items() if key == self.args.model_name),
            None
        )
        if matched is None:
            raise ValueError(f"Unknown model_name: {self.args.model_name}")

        module_name, model_path = matched
        if getattr(self.args, "model_path", None):
            model_path = self.args.model_path
        print(model_path)
        module = importlib.import_module(module_name)

        self.mllm_response, self.load_video, self.load_model = (
            module.mllm_response,
            module.load_video,
            module.load_model,
        )
        self.processor, self.video_llm, self.image_processor, _ = self.load_model(model_path)
        self.last_aggregate_debug = {}

        embedding_path = getattr(self.args, "embedding_path", "BAAI/bge-large-en-v1.5")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_path)

        self.embedding_device = torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        emb_dtype = torch.bfloat16 if self.embedding_device.type == "cuda" else torch.float32

        self.embedding_model = AutoModel.from_pretrained(
            embedding_path,
            torch_dtype=emb_dtype,
        ).to(self.embedding_device).eval()

        for p in self.embedding_model.parameters():
            p.requires_grad = False

        #self.embedding_model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)

        # ====== super-node / graph retrieval 相关超参 ======
        # ====== super-node / graph retrieval 相关超参 ======
        self.semantic_merge_threshold = getattr(self.args, "semantic_merge_threshold", 0.0)
        self.adjacent_sim_threshold = getattr(self.args, "adjacent_sim_threshold", 0.95)
        self.subtitle_merge_threshold = getattr(self.args, "subtitle_merge_threshold", 0.0)
        self.max_supernode_span = getattr(self.args, "max_supernode_span", 4)
        self.supernode_target_frames = getattr(self.args, "supernode_target_frames", None) or self.args.chunk_size

        # 原来默认 4，太窄
        self.top_seed_k = getattr(self.args, "top_seed_k", 12)

        # 新增：supernode 命中后展开 original chunks
        self.original_expand_hop = getattr(self.args, "original_expand_hop", 1)
        self.original_rerank_topk = getattr(
            self.args,
            "original_rerank_topk",
            max(self.args.n_retrieval, 20)
        )

        self.supernode_expand_hop = getattr(self.args, "supernode_expand_hop", 0)
        self.subtitle_novelty_threshold = getattr(self.args, "subtitle_novelty_threshold", 1.0)
        self.boundary_aware_merge = bool(getattr(self.args, "boundary_aware_merge", False))
        self.scene_boundary_threshold = float(getattr(self.args, "scene_boundary_threshold", 0.45))
        self.event_boundary_threshold = float(getattr(self.args, "event_boundary_threshold", 0.45))
        self.boundary_visual_weight = float(getattr(self.args, "boundary_visual_weight", 0.30))
        self.boundary_motion_weight = float(getattr(self.args, "boundary_motion_weight", 0.25))
        self.boundary_cross_motion_weight = float(getattr(self.args, "boundary_cross_motion_weight", 0.20))
        self.boundary_subtitle_weight = float(getattr(self.args, "boundary_subtitle_weight", 0.25))
        self.boundary_frame_window = max(1, int(getattr(self.args, "boundary_frame_window", 4)))
        self.boundary_spatial_size = max(16, int(getattr(self.args, "boundary_spatial_size", 64)))
        self.boundary_motion_scale = float(getattr(self.args, "boundary_motion_scale", 0.05))
        self.boundary_cross_motion_scale = float(getattr(self.args, "boundary_cross_motion_scale", 0.40))
        boundary_weights = (
            self.boundary_visual_weight,
            self.boundary_motion_weight,
            self.boundary_cross_motion_weight,
            self.boundary_subtitle_weight,
        )
        if any(weight < 0.0 for weight in boundary_weights) or sum(boundary_weights) <= 0.0:
            raise ValueError("Boundary weights must be non-negative and have a positive sum")
        if not 0.0 <= self.scene_boundary_threshold <= 1.0:
            raise ValueError("scene_boundary_threshold must be in [0, 1]")
        if not 0.0 <= self.event_boundary_threshold <= 1.0:
            raise ValueError("event_boundary_threshold must be in [0, 1]")
        if self.boundary_motion_scale <= 0.0 or self.boundary_cross_motion_scale <= 0.0:
            raise ValueError("Boundary motion normalization scales must be positive")
        self.last_boundary_diagnostics = None

    # def build_query_list_from_llm_info(self, question, candidates, llm_info):
    #     if llm_info is None:
    #         llm_info = {}

    #     tool = llm_info.get("tool", "none")
    #     need_candidates = llm_info.get("candidates_necessary", "no") == "yes"

    #     query_list = []
    #     query_list.extend(llm_info.get("keywords", []))

    #     if tool in ["order", "state change"]:
    #         query_list.extend(llm_info.get("states", []))
    #         query_list.extend(llm_info.get("temporal_keywords", []))

    #     if need_candidates and tool == "order":
    #         query_list.extend(extract_choices(question, candidates))

    #     query_list = [q for q in query_list if isinstance(q, str) and q.strip()]
    #     query_list = list(dict.fromkeys(query_list))
    #     return query_list

    def _get_original_chunk_count(self, video_graph, video_inputs):
        if video_graph is not None:
            stats = video_graph.graph.get("stats", {})
            if stats.get("original_chunk_count") is not None:
                return int(stats["original_chunk_count"])

        return int(np.ceil(len(video_inputs[0]) / self.args.chunk_size))


    def _build_chunk_to_supernode(self, video_graph):
        mapping = {}

        if video_graph is None:
            return mapping

        for sid, data in video_graph.nodes(data=True):
            original_indices = data.get(
                "original_indices",
                [data.get("original_idx", sid)]
            )
            for idx in original_indices:
                mapping[int(idx)] = int(sid)

        return mapping


    def _expand_supernodes_to_original_chunks(
        self,
        supernode_ids,
        video_graph,
        total_original_chunks,
        hop=1
    ):
        """
        把命中的 supernode 展开回原始 chunk，并补前后邻居。
        """
        chunk_ids = set()

        if video_graph is None:
            return []

        for sid in supernode_ids:
            if sid not in video_graph.nodes:
                continue

            data = video_graph.nodes[sid]
            original_indices = data.get(
                "original_indices",
                [data.get("original_idx", sid)]
            )

            for idx in original_indices:
                idx = int(idx)
                if 0 <= idx < total_original_chunks:
                    chunk_ids.add(idx)

                for h in range(1, hop + 1):
                    if idx - h >= 0:
                        chunk_ids.add(idx - h)
                    if idx + h < total_original_chunks:
                        chunk_ids.add(idx + h)

        return sorted(chunk_ids)


    def _get_original_chunk_text(
        self,
        chunk_id,
        subtitles,
        video_graph,
        chunk_to_supernode
    ):
        """
        用原始 chunk 字幕 + 所属 supernode 的结构化描述做 rerank 文本。
        注意：这是 rerank 用，不是最终视觉输入。
        """
        parts = []

        # 当前原始 chunk 的字幕
        if subtitles is not None:
            parts.extend(self._get_chunk_subtitles(subtitles, chunk_id))

        # 所属 supernode 的粗语义
        sid = chunk_to_supernode.get(int(chunk_id), None)
        if sid is not None and video_graph is not None and sid in video_graph.nodes:
            data = video_graph.nodes[sid]

            parts.extend(data.get("entities", []))
            parts.extend(data.get("actions", []))
            parts.extend(data.get("scenes", []))
            parts.extend(data.get("states", []))

            temporal = data.get("temporal", {})
            if isinstance(temporal, dict):
                if temporal.get("stage"):
                    parts.append(str(temporal["stage"]))
                if temporal.get("evidence"):
                    parts.append(str(temporal["evidence"]))

            if data.get("summary"):
                parts.append(str(data["summary"]))

        text = "; ".join([p for p in parts if isinstance(p, str) and p.strip()])
        return text


    def _rerank_original_chunks(
        self,
        question,
        query_list,
        candidates,
        candidate_chunks,
        subtitles,
        video_graph,
        topk=None
    ):
        """
        supernode 召回后，在 original chunk 粒度上重排。
        """
        if len(candidate_chunks) == 0:
            return []

        if topk is None:
            topk = self.original_rerank_topk

        chunk_to_supernode = self._build_chunk_to_supernode(video_graph)

        key_list = []
        valid_chunks = []

        for cid in sorted(set(candidate_chunks)):
            text = self._get_original_chunk_text(
                cid,
                subtitles,
                video_graph,
                chunk_to_supernode
            )

            if not text.strip():
                text = f"chunk {cid}"

            key_list.append(text)
            valid_chunks.append(int(cid))

        full_query = [q for q in list(query_list) + [question] if isinstance(q, str) and q.strip()]

        if len(full_query) == 0:
            return valid_chunks[:topk]

        sims = compute_text_similarity(
            full_query,
            key_list,
            self.embedding_model,
            self.embedding_tokenizer,
            return_all=True
        )

        scores = torch.mean(sims, dim=0)
        order = torch.argsort(scores, descending=True).tolist()

        reranked = [valid_chunks[i] for i in order]
        return reranked[:topk]
    def build_query_list_from_llm_info(self, question, candidates, llm_info):
        if llm_info is None:
            llm_info = {}

        tool = llm_info.get("tool", "none")
        need_candidates = llm_info.get("candidates_necessary", "no") == "yes"

        query_list = []
        query_list.extend(llm_info.get("keywords", []))

        if tool in ["state change", "order"]:
            query_list.extend(llm_info.get("states", []))
            query_list.extend(llm_info.get("temporal_keywords", []))

        if need_candidates:
            if tool == "order":
                query_list.extend(extract_choices(question, candidates))
            elif tool not in ["object counting", "action counting"]:
                cleaned_candidates = [
                    re.sub(r"^[A-Za-z0-9]+\.\s*", "", c).strip()
                    for c in candidates
                ]
                query_list.extend(cleaned_candidates)

        query_list = [q for q in query_list if isinstance(q, str) and q.strip()]
        query_list = list(dict.fromkeys(query_list))
        return query_list

    def _chunk_to_feature(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk: [T, C, H, W]
        用简单均值特征做相邻 chunk 相似度判断，和 vgent1 的思路一致
        """
        chunk_repr = chunk.float().mean(dim=0)   # [C, H, W]
        feat = chunk_repr.flatten()
        feat = feat / (feat.norm() + 1e-8)
        return feat

    def _adjacent_chunk_similarity(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
        return torch.dot(feat_a, feat_b).item()

    def _get_chunk_subtitles(self, subtitles, original_idx):
        if subtitles is None:
            return []
        start_time = original_idx * self.args.chunk_size // self.args.fps
        end_time = (original_idx + 1) * self.args.chunk_size // self.args.fps
        return [text for time, text in subtitles if time >= start_time and time < end_time]

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _subtitle_similarity(self, subtitle_list_a, subtitle_list_b) -> float:
        """
        简单 token jaccard，相同字幕/近似字幕时更容易硬合并
        """
        text_a = self._normalize_text(" ".join(subtitle_list_a)) if subtitle_list_a else ""
        text_b = self._normalize_text(" ".join(subtitle_list_b)) if subtitle_list_b else ""

        if len(text_a) == 0 and len(text_b) == 0:
            return 1.0
        if len(text_a) == 0 or len(text_b) == 0:
            return 0.0

        set_a = set(text_a.split())
        set_b = set(text_b.split())
        if len(set_a) == 0 and len(set_b) == 0:
            return 1.0
        union = set_a | set_b
        inter = set_a & set_b
        return len(inter) / max(len(union), 1)

    def _subtitle_novelty(self, subtitle_list_a, subtitle_list_b) -> float:
        """
        粗略衡量 cur 相比 prev 带来的新字幕信息比例。
        值越高，越像出现了新的事件边界。
        """
        text_a = self._normalize_text(" ".join(subtitle_list_a)) if subtitle_list_a else ""
        text_b = self._normalize_text(" ".join(subtitle_list_b)) if subtitle_list_b else ""

        if len(text_b) == 0:
            return 0.0
        if len(text_a) == 0:
            return 1.0

        set_a = set(text_a.split())
        set_b = set(text_b.split())
        if len(set_b) == 0:
            return 0.0

        new_tokens = set_b - set_a
        return len(new_tokens) / max(len(set_b), 1)

    def _can_hard_merge(
        self,
        prev_feat,
        cur_feat,
        prev_subtitles=None,
        cur_subtitles=None,
        current_group_len=1,
        boundary=None,
    ):
        """Apply the original DAGC tests, then optional hard boundary guards.

        Subtitle arguments remain accepted for compatibility. The source DAGC
        used visual-only redundancy; subtitle semantics enter through the new
        event score only when boundary-aware merging is enabled.
        """
        if current_group_len >= self.max_supernode_span:
            return False
        visual_sim = self._adjacent_chunk_similarity(prev_feat, cur_feat)
        if visual_sim < self.adjacent_sim_threshold:
            return False
        if self.boundary_aware_merge and boundary is not None:
            if boundary.get("hard_scene_boundary", False):
                return False
            if boundary.get("hard_event_boundary", False):
                return False
        return True

    def _build_supernode_groups(self, split_video_inputs, subtitles, effective_fps=None):
        """Build contiguous super-node groups and adjacent-boundary diagnostics."""
        n = len(split_video_inputs)
        if n == 0:
            return [], [], []

        chunk_feats = [self._chunk_to_feature(chunk) for chunk in split_video_inputs]
        chunk_subtitle_lists = [self._get_chunk_subtitles(subtitles, idx) for idx in range(n)]
        chunk_subtitle_texts = [" ".join(items).strip() for items in chunk_subtitle_lists]
        subtitle_embeddings = [None] * n
        motion_magnitudes = [0.0] * n
        if self.boundary_aware_merge:
            motion_magnitudes = [
                chunk_motion_magnitude(chunk, self.boundary_spatial_size)
                for chunk in split_video_inputs
            ]
            subtitle_embeddings = encode_subtitle_chunks(
                chunk_subtitle_texts,
                self.embedding_model,
                self.embedding_tokenizer,
            )

        groups = []
        group_meta = []
        boundaries = []
        current_group = [0]
        group_sims = []
        current_internal_boundaries = []
        timing_fps = max(float(effective_fps or self.args.fps), 1e-8)

        def finalize_group():
            groups.append(list(current_group))
            group_meta.append({
                "avg_adjacent_sim": float(np.mean(group_sims)) if group_sims else None,
                "num_chunks": len(current_group),
                "internal_boundary_scores": [dict(item) for item in current_internal_boundaries],
                "max_internal_scene_score": max(
                    (float(item["scene_change_score"]) for item in current_internal_boundaries),
                    default=0.0,
                ),
                "max_internal_event_score": max(
                    (float(item["event_boundary_score"]) for item in current_internal_boundaries),
                    default=0.0,
                ),
            })

        for cur_idx in range(1, n):
            prev_idx = cur_idx - 1
            visual_sim = float(self._adjacent_chunk_similarity(
                chunk_feats[prev_idx], chunk_feats[cur_idx]
            ))
            visual_change = min(1.0, max(0.0, 1.0 - visual_sim))
            subtitle_novelty = float(self._subtitle_novelty(
                chunk_subtitle_lists[prev_idx], chunk_subtitle_lists[cur_idx]
            ))

            if self.boundary_aware_merge:
                appearance = appearance_and_motion_scores(
                    split_video_inputs[prev_idx],
                    split_video_inputs[cur_idx],
                    frame_window=self.boundary_frame_window,
                    spatial_size=self.boundary_spatial_size,
                    previous_motion=motion_magnitudes[prev_idx],
                    current_motion=motion_magnitudes[cur_idx],
                    motion_scale=self.boundary_motion_scale,
                    cross_motion_scale=self.boundary_cross_motion_scale,
                )
                subtitle_scores = subtitle_semantic_scores(
                    chunk_subtitle_texts[prev_idx],
                    chunk_subtitle_texts[cur_idx],
                    subtitle_embeddings[prev_idx],
                    subtitle_embeddings[cur_idx],
                )
            else:
                appearance = {
                    "histogram_change_score": 0.0,
                    "edge_change_score": 0.0,
                    "pixel_change_score": 0.0,
                    "scene_change_score": 0.0,
                    "motion_magnitude_prev": 0.0,
                    "motion_magnitude_cur": 0.0,
                    "raw_motion_change": 0.0,
                    "motion_change_score": 0.0,
                    "raw_cross_boundary_motion": 0.0,
                    "cross_boundary_motion_score": 0.0,
                }
                subtitle_scores = {
                    "subtitle_similarity": self._subtitle_similarity(
                        chunk_subtitle_lists[prev_idx], chunk_subtitle_lists[cur_idx]
                    ),
                    "subtitle_change_score": 0.0,
                }

            event_score = (
                self.boundary_visual_weight * visual_change
                + self.boundary_motion_weight * appearance["motion_change_score"]
                + self.boundary_cross_motion_weight * appearance["cross_boundary_motion_score"]
                + self.boundary_subtitle_weight * subtitle_scores["subtitle_change_score"]
            )
            event_score = min(1.0, max(0.0, float(event_score)))
            hard_scene = bool(
                self.boundary_aware_merge
                and appearance["scene_change_score"] >= self.scene_boundary_threshold
            )
            hard_event = bool(
                self.boundary_aware_merge
                and event_score >= self.event_boundary_threshold
            )
            base_span_ok = len(current_group) < self.max_supernode_span
            base_similarity_ok = visual_sim >= self.adjacent_sim_threshold
            boundary = {
                "left_chunk": prev_idx,
                "right_chunk": cur_idx,
                "time_seconds": float(cur_idx * self.args.chunk_size / timing_fps),
                "visual_similarity": visual_sim,
                "visual_change_score": visual_change,
                **appearance,
                **subtitle_scores,
                "subtitle_novelty": subtitle_novelty,
                "event_boundary_score": event_score,
                "hard_scene_boundary": hard_scene,
                "hard_event_boundary": hard_event,
                "blocked_by_scene": bool(base_span_ok and base_similarity_ok and hard_scene),
                "blocked_by_event": bool(base_span_ok and base_similarity_ok and hard_event),
            }
            can_merge = self._can_hard_merge(
                chunk_feats[prev_idx],
                chunk_feats[cur_idx],
                current_group_len=len(current_group),
                boundary=boundary,
            )
            if can_merge:
                decision, reason = "merge", "low_boundary_high_redundancy"
            elif not base_span_ok:
                decision, reason = "split", "max_supernode_span"
            elif not base_similarity_ok:
                decision, reason = "split", "low_visual_similarity"
            elif hard_scene and hard_event:
                decision, reason = "split", "scene_and_event_boundary"
            elif hard_scene:
                decision, reason = "split", "scene_boundary"
            elif hard_event:
                decision, reason = "split", "event_boundary"
            else:
                decision, reason = "split", "subtitle_constraint"
            boundary["decision"] = decision
            boundary["reason"] = reason
            boundaries.append(boundary)

            if can_merge:
                current_group.append(cur_idx)
                group_sims.append(visual_sim)
                current_internal_boundaries.append(boundary)
            else:
                finalize_group()
                current_group = [cur_idx]
                group_sims = []
                current_internal_boundaries = []

        finalize_group()
        return groups, group_meta, boundaries

    def _get_supernode_target_frames(self, num_chunks: int) -> int:
        """
        让 supernode 的采样帧数随跨度轻微增长，避免更长 supernode 被过度压缩。
        """
        return self.supernode_target_frames
        # base = self.args.chunk_size
        # if num_chunks <= 1:
        #     return base
        # return min(int(base * (num_chunks ** 0.5)), base * 2)

    def _sample_frames(self, video_tensor, target_frames):
        """
        video_tensor: [T, C, H, W]
        """
        if len(video_tensor) == 0:
            return video_tensor, None

        if len(video_tensor) <= target_frames:
            keep = np.arange(len(video_tensor))
        else:
            keep = np.linspace(0, len(video_tensor) - 1, target_frames, dtype=int)

        keep = np.asarray(keep, dtype=int)
        sampled = video_tensor[keep]
        return sampled, keep

    def _build_supernode_video_input(self, split_video_inputs, original_indices, split_size_list=None):
        """
        把多个原始 chunk 拼成一个 super-node 的输入，只调一次 generate_entities / refine_nodes
        """
        chunks = [split_video_inputs[i] for i in original_indices if i < len(split_video_inputs)]
        if len(chunks) == 0:
            return None, None

        video_tensor = torch.cat(chunks, dim=0)  # [sumT, C, H, W]
        target_frames = self._get_supernode_target_frames(len(original_indices))
        sampled_video, keep = self._sample_frames(video_tensor, target_frames)

        sampled_size_list = None
        if split_size_list is not None:
            size_chunks = [split_size_list[i] for i in original_indices if i < len(split_size_list)]
            if len(size_chunks) > 0:
                size_tensor = torch.cat(size_chunks, dim=0)
                if keep is not None:
                    sampled_size_list = size_tensor[keep]
                else:
                    sampled_size_list = size_tensor

        return sampled_video, sampled_size_list

    def _collect_subtitles_by_indices(self, subtitles, original_indices):
        if subtitles is None:
            return None

        merged = []
        for original_idx in original_indices:
            merged.extend(self._get_chunk_subtitles(subtitles, original_idx))
        return merged

    def _retrieve_no_graph_by_subtitles(self, query_list, subtitles, total_nodes):
        if subtitles is None or len(subtitles) == 0 or len(query_list) == 0:
            return []

        node_scores = []
        for node_id in range(total_nodes):
            start_time = node_id * self.args.chunk_size // self.args.fps
            end_time = (node_id + 1) * self.args.chunk_size // self.args.fps
            current_subtitles = [
                text for time, text in subtitles
                if time >= start_time and time < end_time
            ]
            if len(current_subtitles) == 0:
                continue

            sim = compute_text_similarity(
                query_list,
                [" ".join(current_subtitles)],
                self.embedding_model,
                self.embedding_tokenizer
            )
            sim = sim.item() if hasattr(sim, "item") else float(sim)
            node_scores.append((node_id, sim))

        node_scores = sorted(node_scores, key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in node_scores[:min(5, len(node_scores))]]

    def _expand_supernode_neighbors(self, seed_nodes, video_graph, hops=1, relation_filter=None):
        """
        默认只沿 temporal_next / temporal_prev 扩展。
        避免 shared_semantic 边把远处语义相似但时序无关的节点拉进来。
        """
        if video_graph is None or len(seed_nodes) == 0:
            return []

        if relation_filter is None:
            relation_filter = {"temporal_next", "temporal_prev"}

        visited = set(seed_nodes)
        frontier = set(seed_nodes)

        for _ in range(hops):
            nxt = set()

            for nid in frontier:
                # 出边
                for nb in video_graph.successors(nid):
                    edge_data = video_graph.get_edge_data(nid, nb, default={})
                    if edge_data.get("relation") in relation_filter:
                        nxt.add(nb)

                # 入边
                for nb in video_graph.predecessors(nid):
                    edge_data = video_graph.get_edge_data(nb, nid, default={})
                    if edge_data.get("relation") in relation_filter:
                        nxt.add(nb)

            nxt = {x for x in nxt if x not in visited}
            if len(nxt) == 0:
                break

            visited.update(nxt)
            frontier = nxt

        return sorted(list(visited))

    def generate_entities(self, prompt, video_input, max_new_tokens=512):
        attempts = 0

        def normalize_info(info):
            # 1. 模型有时返回 [ {...} ]，先拆成 {...}
            if isinstance(info, list):
                if len(info) == 1 and isinstance(info[0], dict):
                    info = info[0]
                else:
                    merged = {}
                    for item in info:
                        if not isinstance(item, dict):
                            continue
                        for k, v in item.items():
                            if k in ["entities", "actions", "scenes", "states"] and isinstance(v, list):
                                merged.setdefault(k, []).extend(v)
                            elif k in ["temporal", "summary"] and k not in merged:
                                merged[k] = v
                    info = merged

            # 2. 仍然不是 dict，就丢弃，避免崩
            if not isinstance(info, dict):
                info = {}

            return info

        def parse_entities(raw_entities):
            out = []
            if not isinstance(raw_entities, list):
                return out

            for entity in raw_entities:
                if isinstance(entity, dict):
                    if "entity name" in entity and "description" in entity:
                        out.append(f"{entity['entity name']}, {entity['description']}")
                    elif len(entity) == 1:
                        k, v = next(iter(entity.items()))
                        out.append(f"{k}, {v}")
                elif isinstance(entity, str):
                    out.append(entity)

            return out

        def parse_actions(raw_actions):
            out = []
            if not isinstance(raw_actions, list):
                return out

            for action in raw_actions:
                if isinstance(action, dict):
                    # 标准格式：{"entity name": "...", "action description": "..."}
                    if "entity name" in action and "action description" in action:
                        out.append(f"{action['entity name']}, {action['action description']}")

                    # 兼容 prompt 里原来的格式：{"man": "walking"}
                    elif len(action) == 1:
                        k, v = next(iter(action.items()))
                        out.append(f"{k}, {v}")

                    # 兼容其他常见字段名
                    else:
                        name = action.get("entity") or action.get("subject") or action.get("name")
                        desc = (
                            action.get("description")
                            or action.get("action")
                            or action.get("action description")
                        )
                        if name is not None and desc is not None:
                            out.append(f"{name}, {desc}")

                elif isinstance(action, str):
                    out.append(action)

            return out

        def parse_scenes(raw_scenes):
            out = []
            if not isinstance(raw_scenes, list):
                return out

            for scene in raw_scenes:
                if isinstance(scene, dict):
                    if "location" in scene:
                        out.append(str(scene["location"]))
                    elif "description" in scene:
                        out.append(str(scene["description"]))
                    elif len(scene) == 1:
                        _, v = next(iter(scene.items()))
                        out.append(str(v))
                elif isinstance(scene, str):
                    out.append(scene)

            return out

        def parse_states(raw_states):
            out = []
            if not isinstance(raw_states, list):
                return out

            for state in raw_states:
                if isinstance(state, dict):
                    if "entity name" in state and "state description" in state:
                        out.append(f"{state['entity name']}, {state['state description']}")
                    elif len(state) == 1:
                        k, v = next(iter(state.items()))
                        out.append(f"{k}, {v}")
                elif isinstance(state, str):
                    out.append(state)

            return out

        while attempts < 5:
            response = None
            try:
                response = self.mllm_response(
                    self.video_llm,
                    self.processor,
                    self.image_processor,
                    prompt,
                    None,
                    video_input,
                    max_new_tokens,
                    tag="generate_entities"
                )

                text = response.replace("```json", "").replace("```", "").strip()
                info = json.loads(text)
                info = normalize_info(info)

                entities = parse_entities(info.get("entities", []))
                actions = parse_actions(info.get("actions", []))
                scenes = parse_scenes(info.get("scenes", []))
                states = parse_states(info.get("states", []))

                temporal = info.get("temporal", {})
                if not isinstance(temporal, dict):
                    temporal = {}

                summary = info.get("summary", "")
                if not isinstance(summary, str):
                    summary = ""

                return entities, actions, scenes, states, temporal, summary

            except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
                attempts += 1
                if attempts >= 5:
                    print(
                        "[generate_entities failed] "
                        f"error={repr(e)} | response={str(response)[:800]}",
                        flush=True
                    )

        return [], [], [], [], {}, ""



    def construct_graph(self, video_inputs, subtitles, video_name=None, effective_fps=None):
        split_video_inputs = list(torch.split(video_inputs[0], self.args.chunk_size, dim=0))

        # ===== Step 1: 先做硬合并分组 =====
        groups, group_meta, boundaries = self._build_supernode_groups(
            split_video_inputs, subtitles, effective_fps=effective_fps
        )

        # print(f"[Hard Merge] original chunks: {len(split_video_inputs)}")
        # print(f"[Hard Merge] super-nodes: {len(groups)}")
        # if len(split_video_inputs) > 0:
        #     print(f"[Hard Merge] compression ratio: {len(groups) / len(split_video_inputs):.4f}")
        original_chunk_count = len(split_video_inputs)

        supernode_count = len(groups)

        graph_stats = {
            "graph_built": True,
            "original_chunk_count": int(original_chunk_count),
            "supernode_count": int(supernode_count),
            "compression_ratio": float(supernode_count / original_chunk_count) if original_chunk_count > 0 else None,
            "retained_node_ratio": float(supernode_count / original_chunk_count) if original_chunk_count > 0 else None,
            "avg_chunks_per_supernode": float(original_chunk_count / supernode_count) if supernode_count > 0 else None,
            "max_chunks_per_supernode": int(max(len(g) for g in groups)) if len(groups) > 0 else 0,
            "merged_supernode_count": int(sum(1 for g in groups if len(g) > 1)),
            "scene_boundary_count": int(sum(bool(item["hard_scene_boundary"]) for item in boundaries)),
            "event_boundary_count": int(sum(bool(item["hard_event_boundary"]) for item in boundaries)),
            "blocked_by_scene_count": int(sum(bool(item["blocked_by_scene"]) for item in boundaries)),
            "blocked_by_event_count": int(sum(bool(item["blocked_by_event"]) for item in boundaries)),
            "blocked_by_max_span_count": int(sum(item["reason"] == "max_supernode_span" for item in boundaries)),
            "blocked_by_low_similarity_count": int(sum(item["reason"] == "low_visual_similarity" for item in boundaries)),
            "avg_scene_change_score": (
                float(np.mean([item["scene_change_score"] for item in boundaries]))
                if boundaries and self.boundary_aware_merge else None
            ),
            "avg_event_boundary_score": (
                float(np.mean([item["event_boundary_score"] for item in boundaries]))
                if boundaries and self.boundary_aware_merge else None
            ),
            "chunk_size": int(self.args.chunk_size),
            "fps": float(effective_fps or self.args.fps),
            "boundary_aware_merge": self.boundary_aware_merge,
        }
        diagnostics = {
            "video_name": video_name,
            "num_chunks": int(original_chunk_count),
            "effective_fps": float(effective_fps or self.args.fps),
            "groups": groups,
            "boundaries": boundaries,
            "stats": graph_stats,
        }
        if (
            self.boundary_aware_merge
            and bool(getattr(self.args, "boundary_debug", False))
            and getattr(self.args, "boundary_visualization_dir", None)
            and video_name
        ):
            diagnostics["visualization_files"] = save_boundary_contact_sheets(
                split_video_inputs,
                boundaries,
                self.args.boundary_visualization_dir,
                video_name,
                self.boundary_frame_window,
                self.scene_boundary_threshold,
                self.event_boundary_threshold,
            )
        self.last_boundary_diagnostics = diagnostics

        video_graph = nx.DiGraph()
        entity_graph = defaultdict(set)

        # ===== Step 2: 每个 super-node 只调一次 generate_entities =====
        for node_idx, original_indices in enumerate(groups):
            super_video_input, _ = self._build_supernode_video_input(
                split_video_inputs,
                original_indices,
                split_size_list=None
            )

            if super_video_input is None or len(super_video_input) == 0:
                entities, actions, scenes, states, temporal, summary = [], [], [], [], {}, ""
            else:
                entities, actions, scenes, states, temporal, summary = self.generate_entities(
                    GRAPH_PROMPT,
                    super_video_input,
                    max_new_tokens=512
                )

            current_subtitles = self._collect_subtitles_by_indices(subtitles, original_indices)

            video_graph.add_node(
                node_idx,
                original_idx=original_indices[0],          # 兼容旧逻辑
                original_indices=original_indices,         # 新增：完整 span
                span_start=original_indices[0],
                span_end=original_indices[-1],
                num_merged=len(original_indices),
                avg_adjacent_sim=group_meta[node_idx]["avg_adjacent_sim"],
                internal_boundary_scores=group_meta[node_idx]["internal_boundary_scores"],
                max_internal_scene_score=group_meta[node_idx]["max_internal_scene_score"],
                max_internal_event_score=group_meta[node_idx]["max_internal_event_score"],
                actions=actions,
                scenes=scenes,
                entities=entities,
                states=states,
                temporal=temporal,
                summary=summary,
                subtitles=current_subtitles
            )

            # ===== temporal edges（super-node 层）=====
            if node_idx > 0:
                video_graph.add_edge(node_idx - 1, node_idx, relation="temporal_next")
                video_graph.add_edge(node_idx, node_idx - 1, relation="temporal_prev")

            # ===== semantic edges =====
            merge_items = entities + actions + scenes + states

            for item in merge_items:
                entity_name = item.split(",")[0].lower().strip()
                if len(entity_name) == 0:
                    continue

                if len(entity_graph) == 0:
                    entity_graph[entity_name].add(node_idx)
                    continue

                graph_keys = list(entity_graph.keys())
                entity_sim = compute_text_similarity(
                    [item], graph_keys,
                    self.embedding_model, self.embedding_tokenizer,
                    return_all=True
                )

                max_sim_idx = torch.argmax(entity_sim[0]).item()
                max_sim = entity_sim[0][max_sim_idx].item()

                if max_sim > self.semantic_merge_threshold:
                    most_similar_entity = graph_keys[max_sim_idx]
                    entity_graph[most_similar_entity].add(node_idx)

                    for i in entity_graph[most_similar_entity]:
                        if i != node_idx:
                            video_graph.add_edge(node_idx, i, relation="shared_semantic", label=most_similar_entity)
                            video_graph.add_edge(i, node_idx, relation="shared_semantic", label=most_similar_entity)
                else:
                    entity_graph[entity_name].add(node_idx)
        video_graph.graph["stats"] = graph_stats
        video_graph.graph["boundary_diagnostics"] = diagnostics
        return video_graph, entity_graph



    # def extract_keywords(self, question, candidates, video_inputs):
    #     reason_prompt = REASONING_PROMPT.format(query=question, candidates=candidates)
    #     flag = True
    #     count = 0
    #     llm_info = None

    #     while flag and count < 5:
    #         try:
    #             response = self.mllm_response(
    #                 self.video_llm,
    #                 self.processor,
    #                 self.image_processor,
    #                 reason_prompt,
    #                 None,
    #                 None,
    #                 max_new_tokens=256,
    #                 tag="extract_keywords"
    #             )
    #             llm_info = json.loads(response.replace("```json", "").replace("```", "").strip())
    #             flag = False
    #         except:
    #             count += 1
    #             continue

    #     if llm_info is None:
    #         llm_info = {}

    #     query_list = []
    #     query_list.extend(llm_info.get("keywords", []))
    #     query_list.extend(llm_info.get("states", []))
    #     query_list.extend(llm_info.get("temporal_keywords", []))
    #     query_list.extend(candidates)
    #     query_list = [q for q in query_list if isinstance(q, str) and len(q.strip()) > 0]
    #     query_list = list(dict.fromkeys(query_list))

    #     return query_list, llm_info
    # def extract_keywords(self, question, candidates, video_inputs):
    #     reason_prompt = REASONING_PROMPT.format(query=question, candidates=candidates)
    #     flag = True
    #     count = 0
    #     llm_info = None

    #     while flag and count < 5:
    #         try:
    #             response = self.mllm_response(
    #                 self.video_llm,
    #                 self.processor,
    #                 self.image_processor,
    #                 reason_prompt,
    #                 None,
    #                 None,
    #                 max_new_tokens=256,
    #                 tag="extract_keywords"
    #             )
    #             llm_info = json.loads(response.replace("```json", "").replace("```", "").strip())
    #             flag = False
    #         except:
    #             count += 1

    #     if llm_info is None:
    #         llm_info = {}

    #     query_list = []
    #     query_list.extend(llm_info.get("keywords", []))
    #     query_list.extend(llm_info.get("states", []))
    #     query_list.extend(llm_info.get("temporal_keywords", []))

    #     need_candidates = llm_info.get("candidates_necessary", "no") == "yes"
    #     tool = llm_info.get("tool", "none")

    #     if need_candidates:
    #         if tool == "order":
    #             query_list.extend(extract_choices(question, candidates))
    #         else:
    #             cleaned_candidates = [
    #                 re.sub(r"^[A-Za-z0-9]+\.\s*", "", c).strip()
    #                 for c in candidates
    #             ]
    #             query_list.extend(cleaned_candidates)

    #     query_list = [q for q in query_list if isinstance(q, str) and q.strip()]
    #     query_list = list(dict.fromkeys(query_list))
    #     return query_list, llm_info

    def extract_keywords(self, question, candidates, video_inputs):
        reason_prompt = REASONING_PROMPT.format(query=question, candidates=candidates)
        flag = True
        count = 0
        llm_info = None

        while flag and count < 5:
            try:
                response = self.mllm_response(
                    self.video_llm,
                    self.processor,
                    self.image_processor,
                    reason_prompt,
                    None,
                    None,
                    max_new_tokens=256,
                    tag="extract_keywords"
                )
                llm_info = json.loads(response.replace("```json", "").replace("```", "").strip())
                flag = False
            except:
                count += 1

        if llm_info is None:
            llm_info = {}

        tool = llm_info.get("tool", "none")
        need_candidates = llm_info.get("candidates_necessary", "no") == "yes"

        # 默认只用 keywords
        query_list = []
        query_list.extend(llm_info.get("keywords", []))

        # 只有真正依赖状态 / 时序的题，才把 states / temporal_keywords 放进检索
        if tool in ["order", "state change"]:
            query_list.extend(llm_info.get("states", []))
            query_list.extend(llm_info.get("temporal_keywords", []))

        # 只有 order 才从 candidates 里抽动作项；其他题不要把候选直接喂给检索
        if need_candidates:
            if tool == "order":
                query_list.extend(extract_choices(question, candidates))
            elif tool not in ["object counting", "action counting"]:
                cleaned_candidates = [
                    re.sub(r"^[A-Za-z0-9]+\.\s*", "", c).strip()
                    for c in candidates
                ]
                query_list.extend(cleaned_candidates)

        query_list = [q for q in query_list if isinstance(q, str) and q.strip()]
        query_list = list(dict.fromkeys(query_list))
        return query_list, llm_info


    # def _need_temporal_expansion(self, question, llm_info):
    #     if llm_info is None:
    #         return False
    #     tool = llm_info.get("tool", "none")
    #     temporal_keywords = llm_info.get("temporal_keywords", [])
    #     multiple = llm_info.get("multiple", "no")

    #     if tool in ["order", "state change", "action counting"]:
    #         return True
    #     if multiple == "yes" and len(temporal_keywords) > 0:
    #         return True
    #     if any(k in question.lower() for k in ["before", "after", "first", "then", "finally"]):
    #         return True
    #     return False
    
    def _need_temporal_expansion(self, question, llm_info):
        if llm_info is None:
            return False

        tool = llm_info.get("tool", "none")
        temporal_keywords = llm_info.get("temporal_keywords", [])
        multiple = llm_info.get("multiple", "no")
        q_lower = question.lower()

        # 只有真正需要前后依赖的题才扩展
        if tool in ["order", "state change"]:
            return True

        # count 不做时序邻居扩展，避免把附近相似动作一起拉进来
        if tool == "action counting":
            return False

        if multiple == "yes" and any(k in q_lower for k in ["before", "after", "first", "then", "finally"]):
            return True

        if len(temporal_keywords) > 0 and any(k in q_lower for k in ["before", "after", "first", "then", "finally"]):
            return True

        return False

    def retrieve_nodes(self, question, query_list, video_inputs, candidates,
                       video_graph, entity_graph, subtitles, llm_info):
        top_seed = None
        nodes_are_original_chunks = False
        indices = None
        time_hint = llm_info.get("time", "none") if llm_info is not None else "none"

        if "subtitle" in question.lower() and subtitles is not None and re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question):
            query_subtitle = re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question)
            indices = []
            for time, text in subtitles:
                if text in query_subtitle:
                    indices.append(time)
            node_list = []

        elif 'beginning' in question.lower() or 'at the start of' in question.lower() or time_hint == "begin":
            if video_graph is not None:
                total_original = self._get_original_chunk_count(video_graph, video_inputs)
                node_list = list(range(min(3, total_original)))
                return {
                    "nodes": node_list,
                    "indices": None,
                    "nodes_are_original_chunks": True,
                    "seed_supernodes": None,
                }
            else:
                total_nodes = round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))
                node_list = [i for i in range(min(3, total_nodes))]

        elif 'at the end of the video' in question.lower() or time_hint == "end":
            if video_graph is not None:
                total_original = self._get_original_chunk_count(video_graph, video_inputs)
                node_list = list(range(max(total_original - 3, 0), total_original))
                return {
                    "nodes": node_list,
                    "indices": None,
                    "nodes_are_original_chunks": True,
                    "seed_supernodes": None,
                }
            else:
                total_nodes = round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))
                node_list = [i for i in range(max(total_nodes - 3, 0), total_nodes)]

        # elif 'beginning' in question.lower() or 'at the start of' in question.lower() or time_hint == "begin":
        #     total_nodes = len(video_graph.nodes) if video_graph is not None else round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))
        #     node_list = [i for i in range(min(3, total_nodes))]

        # elif 'at the end of the video' in question.lower() or time_hint == "end":
        #     total_nodes = len(video_graph.nodes) if video_graph is not None else round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))
        #     node_list = [i for i in range(max(total_nodes - 3, 0), total_nodes)]

        # elif video_graph is None:
        #     total_nodes = round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))
        #     tool = llm_info.get("tool", "none") if llm_info is not None else "none"
        #     global_flag = llm_info.get("global", "no") if llm_info is not None else "no"

        #     if llm_info is not None and llm_info.get("force_all_chunks_in_no_graph", False):
        #         node_list = list(range(total_nodes))
        #     elif (
        #         (tool in ["action counting", "order"] and self.args.task == "mlvu")
        #         or len(video_inputs[0]) <= 128
        #     ):
        #         node_list = list(range(total_nodes))
        #     elif subtitles is not None and global_flag == "no":
        #         node_list = self._retrieve_no_graph_by_subtitles(query_list, subtitles, total_nodes)
        #     else:
        #         node_list = []
        elif video_graph is None:
            total_nodes = round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))

            node_list = (
                list(range(total_nodes))
                if (
                    (
                        llm_info is not None
                        and "tool" in llm_info
                        and llm_info["tool"] in ["action counting", "order"]
                        and self.args.task == "mlvu"
                    )
                    or len(video_inputs[0]) <= 128
                )
                else []
            )
        else:
            if "order" in question.lower():
                query_list = extract_choices(question, candidates)

            full_query_list = list(query_list) + [question]
            seed_nodes = allocate_node(
                self.args,
                video_graph,
                entity_graph,
                full_query_list,
                self.embedding_model,
                self.embedding_tokenizer
            )

            seed_key_list = []
            for node_id in seed_nodes:
                node_data = video_graph.nodes[node_id]
                temporal_text = ""
                if isinstance(node_data.get("temporal"), dict):
                    temporal_text = " ".join([
                        str(node_data["temporal"].get("stage", "")),
                        str(node_data["temporal"].get("evidence", ""))
                    ]).strip()

                segments = []
                segments.extend(node_data.get("entities", []))
                segments.extend(node_data.get("actions", []))
                segments.extend(node_data.get("scenes", []))
                segments.extend(node_data.get("states", []))
                if temporal_text:
                    segments.append(temporal_text)
                if node_data.get("summary"):
                    segments.append(node_data.get("summary"))
                if node_data.get("subtitles") is not None:
                    segments.extend(node_data.get("subtitles", []))
                seed_key_list.append("; ".join(segments))

            if len(seed_nodes) > 0:
                sims = compute_text_similarity(
                    full_query_list,
                    seed_key_list,
                    self.embedding_model,
                    self.embedding_tokenizer,
                    return_all=True
                )
                sorted_idx = torch.argsort(torch.mean(sims, dim=0), descending=True)
                seed_nodes = [seed_nodes[i] for i in sorted_idx]

            seed_k = self.top_seed_k
            tool = llm_info.get("tool", "none") if llm_info is not None else "none"

            if self._need_temporal_expansion(question, llm_info):
                seed_k = self.top_seed_k
            else:
                seed_k = self.args.n_retrieval
            # tool = llm_info.get("tool", "none") if llm_info is not None else "none"
            # if tool == "action counting":
            #     seed_k = max(seed_k, 6)
            # elif tool == "order":
            #     seed_k = max(seed_k, 4)
            top_seed = seed_nodes[:min(seed_k, len(seed_nodes))]

            # 1) supernode 层 temporal expansion
            if self._need_temporal_expansion(question, llm_info):
                supernode_list = self._expand_supernode_neighbors(
                    top_seed,
                    video_graph,
                    hops=self.supernode_expand_hop
                )
            else:
                supernode_list = top_seed

            # 2) supernode -> original chunk
            total_original_chunks = self._get_original_chunk_count(video_graph, video_inputs)

            tool = llm_info.get("tool", "none") if llm_info is not None else "none"

            original_hop = self.original_expand_hop
            if tool in ["order", "state change"]:
                original_hop = max(original_hop, 1)

            candidate_chunks = self._expand_supernodes_to_original_chunks(
                supernode_ids=supernode_list,
                video_graph=video_graph,
                total_original_chunks=total_original_chunks,
                hop=original_hop
            )

            # 3) original chunk rerank
            node_list = self._rerank_original_chunks(
                question=question,
                query_list=query_list,
                candidates=candidates,
                candidate_chunks=candidate_chunks,
                subtitles=subtitles,
                video_graph=video_graph,
                topk=self.original_rerank_topk
            )

# 注意：这里不要再用 video_graph.nodes[node_id] 做二次 rerank

            # 后面 return 前要告诉 refine/aggregate：nodes 已经是 original chunks

        nodes_are_original_chunks = True
        max_retrieval = self.args.n_retrieval
        if llm_info is not None and llm_info.get("tool") == "action counting":
            max_retrieval = max(max_retrieval, 12)
        elif llm_info is not None and llm_info.get("tool") == "order":
            max_retrieval = max(max_retrieval, 12)

        
        return {
            "nodes": node_list[:max_retrieval],
            "indices": indices,
            "nodes_are_original_chunks": nodes_are_original_chunks,
            "seed_supernodes": top_seed ,
        }

    def _build_subqueries(self, question, candidates, llm_info):
        """
        返回:
            info: 子问题字典
            obj_count: 是否走 numeric counting prompt
        关键点：
        - action counting 和 object counting 的分支必须在这里一次性定死
        - 不要再在 refine_nodes 里额外用 "how many" 偷改 obj_count
        """
        question_type = llm_info.get("tool") if llm_info else None
        q_lower = question.lower()

        # 1) order
        if question_type == "order" or "order" in q_lower:
            choices = extract_choices(question, candidates)
            info = {f"Q{i+1}": f"Is '{c.lower()}' shown in video?" for i, c in enumerate(choices)}
            return info, False

        # 2) action counting
        # 只要题目本身是在数 action scene，就保持 yes/no 验证，不走 numeric prompt
        if question_type == "action counting" and "action" in q_lower:
            match = re.search(r"'(.*?)'", question)
            if match:
                extracted_text = match.group(1)
                info = {"Q1": f"Is there a scene featuring the '{extracted_text}' action in the video?"}
            else:
                info = {"Q1": f"Is there evidence of the asked action in the video?"}
            return info, False

        # 3) object counting
        # 只有真正的 object counting，或者 how many 但不是 action scene 计数时，才走 numeric prompt
        if question_type == "object counting" or ("how many" in q_lower and "action" not in q_lower):
            info = {"Q1": question}
            return info, True

        # 4) state change
        if question_type == "state change":
            info = {"Q1": f"Is there evidence relevant to the object state or state change asked in the question: {question}?"}
            return info, False

        # 5) fallback：交给 SQL_PROMPT
        prompt = SQL_PROMPT.format(query=question, candidates=" ".join(candidates))
        for _ in range(5):
            try:
                response = self.mllm_response(
                    self.video_llm, self.processor, self.image_processor,
                    prompt, None, None, 512, tag="build_subqueries"
                )
                parsed = json.loads(response.replace("```json", "").replace("```", "").strip())
                if isinstance(parsed, dict) and len(parsed) > 0:
                    return parsed, False
            except Exception:
                pass

        return {"Q1": f"Is there evidence relevant to answering this question: {question}?"}, False

    def _original_count_and_sort_filtered(self, data):
        """
        完整复刻原版 retrieval.py 的 count_and_sort_filtered：
        只要 value != 'no' and value != '0' and value != 0，就算命中
        """
        count_dict = {}

        for index, answers in data.items():
            if answers is not None and isinstance(answers, dict):
                for _, value in answers.items():
                    if value != 'no' and value != '0' and value != 0:
                        count_dict[index] = count_dict.get(index, 0) + 1

        filtered_dict = {k: v for k, v in count_dict.items() if v > 0}
        sorted_indices = sorted(filtered_dict.keys(), key=lambda k: filtered_dict[k], reverse=True)

        return filtered_dict, sorted_indices

    def _refine_action_count_inside_supernodes(
        self, retrieved_node_list, question, video_inputs, subtitles, video_graph, size_list=None
    ):
        """
        graph 分支下，action counting 在命中的 supernode 内部回退到原始 chunk 做细验证。
        """
        split_video_inputs = list(torch.split(video_inputs[0], self.args.chunk_size, dim=0))
        split_size_list = list(torch.split(size_list, self.args.chunk_size, dim=0)) if size_list is not None else None

        match = re.search(r"'(.*?)'", question)
        if match:
            extracted_text = match.group(1)
            info = {"Q1": f"Is there a scene featuring the '{extracted_text}' action in the video?"}
        else:
            info = {"Q1": "Is there evidence of the asked action in the video?"}

        fine_check_result = {}

        for super_node in retrieved_node_list["nodes"]:
            if super_node not in video_graph.nodes:
                continue

            original_indices = video_graph.nodes[super_node].get(
                "original_indices",
                [video_graph.nodes[super_node].get("original_idx", super_node)]
            )

            for idx in original_indices:
                if idx in fine_check_result or idx >= len(split_video_inputs):
                    continue

                video_input = split_video_inputs[idx]
                size_list_input = split_size_list[idx] if split_size_list is not None else None

                if subtitles is not None:
                    subtitle_prompt = " This video's subtitles are listed below:\n"
                    start_time = idx * self.args.chunk_size // self.args.fps
                    end_time = (idx + 1) * self.args.chunk_size // self.args.fps
                    select_subtitles = [
                        text for time, text in subtitles
                        if time >= start_time and time < end_time
                    ]
                    subtitle_prompt += " ".join(select_subtitles) + "\n"
                else:
                    subtitle_prompt = ""

                instruct = ORIG_SQL_ANSWER_PROMPT.format(questions=info) + subtitle_prompt
                try:
                    output_text = self.mllm_response(
                        self.video_llm, self.processor, self.image_processor,
                        instruct, None, video_input, max_new_tokens=256,
                        size_list=size_list_input,
                        tag="refine_count_inside_supernodes"
                    )
                    pred = json.loads(output_text.replace("```json", "").replace("```", "").strip())
                except Exception:
                    pred = None

                fine_check_result[idx] = pred

        _, sorted_chunks = self._original_count_and_sort_filtered(fine_check_result)
        return info, fine_check_result, sorted_chunks

    def refine_nodes(self, retrieved_node_list, question, llm_info, candidates,
                 video_inputs, subtitles, video_graph, size_list=None):
        if len(retrieved_node_list["nodes"]) == 0:
            return retrieved_node_list, None, None

        question_type = llm_info["tool"] if llm_info is not None and "tool" in llm_info else None

        nodes_are_original_chunks = retrieved_node_list.get("nodes_are_original_chunks", False)

        # 只有当 retrieved nodes 仍然是 supernode 时，才需要进入 supernode 内细化。
        # 如果 retrieve_nodes 已经返回 original chunks，就不要再走这个分支。
        if video_graph is not None and question_type == "action counting" and not nodes_are_original_chunks:
            info, fine_check_result, sorted_chunks = self._refine_action_count_inside_supernodes(
                retrieved_node_list=retrieved_node_list,
                question=question,
                video_inputs=video_inputs,
                subtitles=subtitles,
                video_graph=video_graph,
                size_list=size_list
            )
            retrieved_node_list["nodes"] = sorted_chunks
            retrieved_node_list["nodes_are_original_chunks"] = True
            return retrieved_node_list, info, fine_check_result

        # if video_graph is not None and question_type == "action counting":
        #     info, fine_check_result, sorted_chunks = self._refine_action_count_inside_supernodes(
        #         retrieved_node_list=retrieved_node_list,
        #         question=question,
        #         video_inputs=video_inputs,
        #         subtitles=subtitles,
        #         video_graph=video_graph,
        #         size_list=size_list
        #     )
        #     retrieved_node_list["nodes"] = sorted_chunks
        #     retrieved_node_list["nodes_are_original_chunks"] = True
        #     return retrieved_node_list, info, fine_check_result

        info, obj_count = self._build_subqueries(question, candidates, llm_info)

        if info is None:
            return retrieved_node_list, None, None

        split_video_inputs = list(torch.split(video_inputs[0], self.args.chunk_size, dim=0))
        split_size_list = list(torch.split(size_list, self.args.chunk_size, dim=0)) if size_list is not None else None

        check_result = {}

        for node in retrieved_node_list["nodes"]:
            
            if retrieved_node_list.get("nodes_are_original_chunks", False):
                original_indices = [int(node)]

            elif video_graph is not None:
                if node not in video_graph.nodes:
                    continue
                original_indices = video_graph.nodes[node].get(
                    "original_indices",
                    [video_graph.nodes[node].get("original_idx", node)]
                )

            else:
                original_indices = [int(node)]

            super_video_input, size_list_input = self._build_supernode_video_input(
                split_video_inputs,
                original_indices,
                split_size_list=split_size_list
            )

            if super_video_input is None or len(super_video_input) == 0:
                check_result[node] = None
                continue

            if subtitles is not None:
                subtitle_prompt = " This video's subtitles are listed below:\n"
                select_subtitles = self._collect_subtitles_by_indices(subtitles, original_indices)
                subtitle_prompt += " ".join(select_subtitles) + "\n"
            else:
                subtitle_prompt = ""

            # instruct = (
            #     SQL_ANSWER_COUNT_PROMPT.format(questions=info) + subtitle_prompt
            #     if obj_count
            #     else SQL_ANSWER_PROMPT.format(questions=info) + subtitle_prompt
            # ) 

            question_type = llm_info["tool"] if llm_info is not None and "tool" in llm_info else None

            use_original_non_graph_prompt = (
                video_graph is None and question_type == "action counting"
            )

            answer_prompt = ORIG_SQL_ANSWER_PROMPT if use_original_non_graph_prompt else SQL_ANSWER_PROMPT

            instruct = (
                SQL_ANSWER_COUNT_PROMPT.format(questions=info) + subtitle_prompt
                if obj_count
                else answer_prompt.format(questions=info) + subtitle_prompt
            )

            try:
                output_text = self.mllm_response(
                    self.video_llm, self.processor, self.image_processor,
                    instruct, None, super_video_input, max_new_tokens=256,
                    size_list=size_list_input,
                    tag="refine_nodes"
                )
                pred = json.loads(output_text.replace("```json", "").replace("```", "").strip())
            except:
                pred = None

            check_result[node] = pred

        question_type = llm_info["tool"] if llm_info is not None and "tool" in llm_info else None

        if video_graph is None and question_type == "action counting":
            _, sorted_nodes = self._original_count_and_sort_filtered(check_result)
        else:
            _, sorted_nodes = count_and_sort_filtered(check_result)

        retrieved_node_list["nodes"] = sorted_nodes
        return retrieved_node_list, info, check_result

        # _, sorted_nodes = count_and_sort_filtered(check_result)
        # retrieved_node_list["nodes"] = sorted_nodes

        # return retrieved_node_list, info, check_result


    def _is_positive_action_count_value(self, value) -> bool:
        """
        action counting 只接受非常严格的 positive：
        - "yes"
        - 数值 > 0
        其他字符串（尤其是问题回显）一律不算
        """
        if isinstance(value, (int, float)):
            return value > 0

        if not isinstance(value, str):
            return False

        v = value.strip().lower()
        return v == "yes"

    def _estimate_action_count_from_checks(self, node_list, check_result, video_graph, nodes_are_original_chunks=False):
        """
        用 refine 后的严格命中结果估计动作出现次数。
        核心思路：
        1. 只把严格 positive 的 node 当作命中
        2. 对时间上连续/相邻的命中 node 做一次事件级合并
           避免一个动作跨多个 chunk 时被重复计数

        返回:
            estimated_count: 估计的动作次数（int）
        """
        if node_list is None or check_result is None:
            return 0

        positive_spans = []

        for node in node_list:
            answers = check_result.get(node, None)
            if answers is None or not isinstance(answers, dict):
                continue

            is_positive = any(
                self._is_positive_action_count_value(v)
                for v in answers.values()
            )
            if not is_positive:
                continue

            if nodes_are_original_chunks:
                start_idx = int(node)
                end_idx = int(node)
            elif video_graph is not None and node in video_graph.nodes:
                original_indices = video_graph.nodes[node].get(
                    "original_indices",
                    [video_graph.nodes[node].get("original_idx", node)]
                )
                start_idx = min(original_indices)
                end_idx = max(original_indices)
            else:
                start_idx = int(node)
                end_idx = int(node)

            positive_spans.append((start_idx, end_idx))

        if len(positive_spans) == 0:
            return 0

        positive_spans = sorted(positive_spans, key=lambda x: (x[0], x[1]))

        merged_events = []
        cur_start, cur_end = positive_spans[0]

        for start_idx, end_idx in positive_spans[1:]:
            # 相邻 chunk 视作同一次连续动作，避免重复计数
            if start_idx <= cur_end + 1:
                cur_end = max(cur_end, end_idx)
            else:
                merged_events.append((cur_start, cur_end))
                cur_start, cur_end = start_idx, end_idx

        merged_events.append((cur_start, cur_end))
        return len(merged_events)
    def aggregate_nodes(self, refined_node_list, llm_info, video_inputs, raw_video, size_list,
                        subtitles, prompt, query, video_graph, sql_check, check_result, fps):
        self.last_aggregate_debug = {
            "raw_final_output": None,
            "cleaned_final_output": None,
            "parse_status": None,
            "selection_mode": None,
            "final_indices_len": None,
            "final_indices_len_before_cap": None,
            "final_indices_capped": False,
            "actual_num_frames_seen_by_model": None,
            "adapter_media_type": None,
            "requested_vila_num_video_frames": None,
        }

        def select_size_list(selected_indices):
            if size_list is None or selected_indices is None:
                return None
            selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
            if len(selected_indices) == 0:
                return None
            try:
                size_len = int(len(size_list))
            except TypeError:
                return None
            try:
                max_index = int(selected_indices.max())
            except ValueError:
                return None
            if max_index >= size_len:
                return None
            video_len = int(len(video_inputs[0]))
            if size_len == video_len:
                if torch.is_tensor(size_list):
                    idx = torch.as_tensor(selected_indices, dtype=torch.long, device=size_list.device)
                    return size_list[idx]
                return [size_list[int(i)] for i in selected_indices]
            if size_len == len(selected_indices):
                return size_list
            return None

        question_type = llm_info["tool"] if llm_info is not None and "tool" in llm_info else None
        select_subtitles = None
        node_list = refined_node_list["nodes"]

        if node_list is not None and len(node_list) > 0:
            nodes_are_original_chunks = refined_node_list.get("nodes_are_original_chunks", False)
            if nodes_are_original_chunks:
                original_node_list = sorted(set(node_list))
            elif video_graph is not None:
                original_node_list = []
                for node in node_list:
                    if node in video_graph.nodes:
                        original_indices = video_graph.nodes[node].get(
                            "original_indices",
                            [video_graph.nodes[node].get("original_idx", node)]
                        )
                        original_node_list.extend(original_indices)
                original_node_list = sorted(set(original_node_list))
            else:
                original_node_list = sorted(set(node_list))

            indices, sorted_original_node_list = node2indices(
                original_node_list, question_type, video_inputs, self.args
            )

            input_size_list = select_size_list(indices)
            self.last_aggregate_debug["selection_mode"] = "node_list"

            if subtitles is not None:
                select_subtitles = []
                for original_idx in sorted_original_node_list:
                    start_time = original_idx * self.args.chunk_size // self.args.fps
                    end_time = (original_idx + 1) * self.args.chunk_size // self.args.fps
                    select_subtitles.extend([
                        text for time, text in subtitles
                        if time >= start_time and time < end_time
                    ])

        elif refined_node_list["indices"] is not None:
            indices = refined_node_list["indices"]
            input_size_list = select_size_list(indices)
            self.last_aggregate_debug["selection_mode"] = "subtitle_indices"
            extend_indices = []

            if subtitles is not None:
                select_subtitles = []
                for index in indices:
                    select_subtitles.extend([text for time, text in subtitles if time == index])
                    extend_indices.extend(
                        list(range(max(0, index - 10), min(len(video_inputs[0]), index + 10)))
                    )

            indices = sorted(set(extend_indices)) if len(extend_indices) > 0 else indices
            #video_segments = video_inputs[0][indices]
        else:
            indices = np.linspace(
                0, len(video_inputs[0]) - 1,
                min(self.args.uniform_frame, len(video_inputs[0])),
                dtype=int
            )
            #video_segments = video_inputs[0][indices]
            input_size_list = select_size_list(indices)
            self.last_aggregate_debug["selection_mode"] = "uniform_fallback"

            if subtitles is not None:
                select_subtitles = [text for time, text in subtitles]

        if select_subtitles is not None:
            subtitle_prompt = "This video's subtitles are listed below:\n"
            subtitle_prompt += " ".join(select_subtitles) + "\n"
            input_prompt = subtitle_prompt + prompt
        else:
            input_prompt = prompt


                # ===== 恢复原版 count 出口 =====
        # 对 action counting，不再让模型最后自由选字母；
        # 直接用命中的 node 数映射到最近的数字候选。

                # ===== 更稳健的 count 出口 =====
        # 不再直接用 len(node_list) 当次数；
        # 改成：
        # 1) 先基于 check_result 只保留严格 positive
        # 2) 再把时间上连续的 positive node 合并成一次动作事件
        # 3) 用“事件数”映射到最近候选数字
        # if (
        #     question_type == "action counting"
        #     and check_result is not None
        #     and node_list is not None
        #     and all(re.search(r"\d+", c) is not None for c in query["candidates"])
        # ):
        #     estimated_count = self._estimate_action_count_from_checks(
        #         node_list=node_list,
        #         check_result=check_result,
        #         video_graph=video_graph
        #     )

        #     numbers = [int(re.search(r"\d+", c).group()) for c in query["candidates"]]

        #     # 同距离时优先更小的数字，避免系统性过计数
        #     pred_idx = min(
        #         range(len(numbers)),
        #         key=lambda i: (abs(numbers[i] - estimated_count), numbers[i])
        #     )
        #     pred = query["letters"][pred_idx]
        #     return pred


        # if (
        #     question_type == "action counting"
        #     and check_result is not None
        #     and node_list is not None
        #     and all(re.search(r"\d+", c) is not None for c in query["candidates"])
        # ):
        #     numbers = [int(re.search(r"\d+", c).group()) for c in query["candidates"]]
        #     estimated_count = self._estimate_action_count_from_checks(
        #         node_list=node_list,
        #         check_result=check_result,
        #         video_graph=video_graph,
        #         nodes_are_original_chunks=refined_node_list.get("nodes_are_original_chunks", False)
        #     )
        #     pred_idx = min(
        #         range(len(numbers)),
        #         key=lambda i: (abs(numbers[i] - estimated_count), numbers[i])
        #     )
        #     pred = query["letters"][pred_idx]
        #     return pred

        if (
            question_type == "action counting"
            and node_list is not None
            and all(re.search(r"\d+", c) is not None for c in query["candidates"])
        ):
            numbers = [int(re.search(r"\d+", c).group()) for c in query["candidates"]]

            estimated_count = len(node_list)

            pred_idx = min(
                range(len(numbers)),
                key=lambda i: abs(numbers[i] - estimated_count)
            )
            pred = query["letters"][pred_idx]
            return pred


        # if (
        #     question_type == "action counting"
        #     and node_list is not None
        #     and all(re.search(r"\d+", c) is not None for c in query["candidates"])
        # ):
        #     numbers = [int(re.search(r"\d+", c).group()) for c in query["candidates"]]

        #     # ===== non-graph：回退原版 =====
        #     if video_graph is None:
        #         pred_idx = min(
        #             range(len(numbers)),
        #             key=lambda i: abs(numbers[i] - len(node_list))
        #         )
        #         pred = query["letters"][pred_idx]
        #         return pred

        #     # ===== graph：保留你现在的改版逻辑 =====
        #     estimated_count = self._estimate_action_count_from_checks(
        #         node_list=node_list,
        #         check_result=check_result,
        #         video_graph=video_graph
        #     )

        #     pred_idx = min(
        #         range(len(numbers)),
        #         key=lambda i: (abs(numbers[i] - estimated_count), numbers[i])
        #     )
        #     pred = query["letters"][pred_idx]
        #     return pred
        # if (
        #     question_type == "action counting"
        #     and node_list is not None
        #     and all(re.search(r"\d+", c) is not None for c in query["candidates"])
        # ):
        #     numbers = [int(re.search(r"\d+", c).group()) for c in query["candidates"]]
        #     pred_idx = min(range(len(numbers)), key=lambda i: abs(numbers[i] - len(node_list)))
        #     pred = query["letters"][pred_idx]
        #     return pred
            
        agg_info = None
        multiple = llm_info["multiple"] if llm_info is not None and "multiple" in llm_info else 'no'

        if (
            sql_check is not None
            and check_result is not None
            and node_list is not None
            and len(node_list) > 1
            and multiple == "yes"
            and question_type != "object counting"
        ):
            input_text = ""
            for key in sorted(check_result.keys()):
                input_text += f"video [{key}]:\n"
                for question_id, question_text in sql_check.items():
                    if (
                        key in check_result
                        and check_result[key] is not None
                        and question_id in check_result[key]
                        and check_result[key][question_id] != 'no'
                    ):
                        input_text += f"{question_text}: {check_result[key][question_id]}\n"

            input_candidates = " ".join(query['candidates'])
            input_text = AGGREGATE_PROMPT.format(
                query=query['question'],
                candidates=input_candidates,
                input=input_text
            )

            try:
                agg_info = self.mllm_response(
                    self.video_llm, self.processor, self.image_processor,
                    input_text, None, None, 128
                )
            except:
                agg_info = None

        input_prompt = input_prompt + PRED_PROMPT + agg_info if agg_info is not None else input_prompt + PRED_PROMPT

        proc_len = len(video_inputs[0])
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        indices = indices[(indices >= 0) & (indices < proc_len)]

        if len(indices) == 0 and proc_len > 0:
            indices = np.linspace(
                0,
                proc_len - 1,
                min(self.args.uniform_frame, proc_len),
                dtype=np.int64,
            )
            self.last_aggregate_debug["selection_mode"] = f"{self.last_aggregate_debug.get('selection_mode')}_empty_uniform_fallback"

        self.last_aggregate_debug["final_indices_len_before_cap"] = int(len(indices))
        if str(self.args.model_name).startswith("internvl35"):
            max_frames = int(getattr(self.args, "uniform_frame", 0) or 0)
            if max_frames > 0 and len(indices) > max_frames:
                keep = np.linspace(0, len(indices) - 1, max_frames, dtype=np.int64)
                indices = indices[keep]
                self.last_aggregate_debug["final_indices_capped"] = True

        input_size_list = select_size_list(indices)
        self.last_aggregate_debug["final_indices_len"] = int(len(indices))

        if "internvl" in str(self.args.model_name):
            video_segments = video_inputs[0][indices]
        else:
            raw_tensor = raw_video[0] if raw_video is not None else video_inputs[0]
            raw_len = len(raw_tensor)
            if raw_len != proc_len:
                raw_indices = np.round(indices.astype(np.float64) * (raw_len - 1) / max(proc_len - 1, 1)).astype(np.int64)
            else:
                raw_indices = indices.astype(np.int64)
            raw_indices = np.clip(raw_indices, 0, raw_len - 1)
            raw_indices = sorted(set(raw_indices.tolist()))
            video_segments = raw_tensor[raw_indices] if len(raw_indices) > 0 else raw_tensor
            if len(raw_indices) != len(indices):
                input_size_list = None
            video_segments, fps = resize_video(
                video_segments,
                fps,
                total_pixels=self.args.total_pixels * 28 * 28,
                maximum_frames=512,
            )

        output_text = self.mllm_response(
            self.video_llm, self.processor, self.image_processor,
            input_prompt, None, video_segments, max_new_tokens=10,
            size_list=input_size_list, fps=fps,
            tag="aggregate_nodes",
        )
        model_response_debug = getattr(self.video_llm, "vgent_last_response", {}) if hasattr(self.video_llm, "__dict__") else {}
        if model_response_debug:
            self.last_aggregate_debug["actual_num_frames_seen_by_model"] = model_response_debug.get(
                "actual_num_frames_seen_by_model",
                model_response_debug.get("actual_num_frames_seen_by_vila"),
            )
            self.last_aggregate_debug["adapter_media_type"] = model_response_debug.get("adapter_media_type")
            self.last_aggregate_debug["requested_vila_num_video_frames"] = model_response_debug.get("requested_vila_num_video_frames")
        else:
            self.last_aggregate_debug["actual_num_frames_seen_by_model"] = int(len(indices))

        pred_answer, cleaned_final_output, parse_status = extract_choice_from_response(output_text, query['letters'])
        if pred_answer is not None and pred_answer in query['letters']:
            pred_idx = query['letters'].index(pred_answer)
        else:
            pred_idx = 2
            parse_status = "fallback"
        pred = query['letters'][pred_idx]
        self.last_aggregate_debug.update(
            {
                "raw_final_output": output_text,
                "cleaned_final_output": cleaned_final_output,
                "parse_status": parse_status,
            }
        )

        return pred
