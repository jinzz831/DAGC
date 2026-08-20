"""Training-free boundary signals for adjacent fixed-length video chunks.

The functions in this module operate on small CPU copies of downsampled frames.
They never load a model. Subtitle embeddings are produced in one batch by the
already-loaded text encoder supplied by :class:`Vgent`.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _as_unit_rgb(frames: torch.Tensor, size: int = 64) -> torch.Tensor:
    """Return a small CPU float tensor in ``[T, 3, size, size]`` and ``[0, 1]``."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or len(frames) == 0:
        return torch.empty((0, 3, size, size), dtype=torch.float32)
    x = frames.detach().to(device="cpu", dtype=torch.float32)
    if x.shape[1] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
        x = x.permute(0, 3, 1, 2)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] >= 4:
        x = x[:, :3]
    if x.numel() == 0:
        return torch.empty((0, 3, size, size), dtype=torch.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    if xmin < -0.05 and xmax <= 1.5:
        x = (x + 1.0) * 0.5
    elif xmax > 1.5:
        x = x / 255.0
    x = x.clamp(0.0, 1.0)
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def _sample_temporal(frames: torch.Tensor, maximum: int = 64) -> torch.Tensor:
    if len(frames) <= maximum:
        return frames
    keep = torch.linspace(0, len(frames) - 1, maximum).round().long()
    return frames[keep]


def _rgb_histogram(image: torch.Tensor, bins: int = 16) -> torch.Tensor:
    parts = []
    for channel in image[:3]:
        hist = torch.histc(channel, bins=bins, min=0.0, max=1.0)
        parts.append(hist / hist.sum().clamp_min(1.0))
    return torch.cat(parts)


def _edge_map(gray: torch.Tensor) -> torch.Tensor:
    dx = F.pad(torch.abs(gray[:, 1:] - gray[:, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(gray[1:, :] - gray[:-1, :]), (0, 0, 0, 1))
    return (dx + dy).clamp(0.0, 1.0)


def chunk_motion_magnitude(chunk: torch.Tensor, spatial_size: int = 64) -> float:
    """Mean normalized frame-to-frame absolute difference for one chunk."""
    frames = _sample_temporal(_as_unit_rgb(chunk, spatial_size))
    if len(frames) < 2:
        return 0.0
    return float(torch.abs(frames[1:] - frames[:-1]).mean().clamp(0.0, 1.0))


def appearance_and_motion_scores(
    previous: torch.Tensor,
    current: torch.Tensor,
    frame_window: int = 4,
    spatial_size: int = 64,
    previous_motion: Optional[float] = None,
    current_motion: Optional[float] = None,
    motion_scale: float = 0.05,
    cross_motion_scale: float = 0.40,
) -> Dict[str, float]:
    """Compute scene and motion changes around one adjacent-chunk boundary.

    Histogram, grayscale-edge, and direct pixel changes are each normalized to
    ``[0, 1]``. The scene score is their weighted mean. Motion scores are based
    on normalized RGB frame differences, so source resolution is irrelevant.
    """
    window = max(1, int(frame_window))
    motion_scale = max(float(motion_scale), 1e-8)
    cross_motion_scale = max(float(cross_motion_scale), 1e-8)
    prev = _as_unit_rgb(previous[-window:], spatial_size)
    cur = _as_unit_rgb(current[:window], spatial_size)
    if len(prev) == 0 or len(cur) == 0:
        return {
            "histogram_change_score": 0.0,
            "edge_change_score": 0.0,
            "pixel_change_score": 0.0,
            "scene_change_score": 0.0,
            "motion_magnitude_prev": float(previous_motion or 0.0),
            "motion_magnitude_cur": float(current_motion or 0.0),
            "raw_motion_change": abs(float(previous_motion or 0.0) - float(current_motion or 0.0)),
            "motion_change_score": min(
                1.0,
                abs(float(previous_motion or 0.0) - float(current_motion or 0.0)) / motion_scale,
            ),
            "raw_cross_boundary_motion": 0.0,
            "cross_boundary_motion_score": 0.0,
        }

    prev_mean = prev.mean(dim=0)
    cur_mean = cur.mean(dim=0)
    hist_change = 0.5 * torch.abs(
        _rgb_histogram(prev_mean) - _rgb_histogram(cur_mean)
    ).sum() / 3.0
    prev_gray = 0.299 * prev_mean[0] + 0.587 * prev_mean[1] + 0.114 * prev_mean[2]
    cur_gray = 0.299 * cur_mean[0] + 0.587 * cur_mean[1] + 0.114 * cur_mean[2]
    edge_change = torch.abs(_edge_map(prev_gray) - _edge_map(cur_gray)).mean()
    pixel_change = torch.abs(prev_mean - cur_mean).mean()
    # Color distribution is the most stable shot-cut cue; structure and pixels
    # retain sensitivity to cuts between similarly colored scenes.
    scene_change = 0.40 * hist_change + 0.30 * edge_change + 0.30 * pixel_change

    if previous_motion is None:
        previous_motion = chunk_motion_magnitude(previous, spatial_size)
    if current_motion is None:
        current_motion = chunk_motion_magnitude(current, spatial_size)
    pair_count = min(len(prev), len(cur))
    paired = torch.abs(prev[-pair_count:] - cur[:pair_count]).mean()
    endpoint = torch.abs(prev[-1] - cur[0]).mean()
    raw_cross_motion = 0.5 * paired + 0.5 * endpoint
    raw_motion_change = abs(float(previous_motion) - float(current_motion))

    def scalar(value: Any) -> float:
        return float(torch.as_tensor(value).clamp(0.0, 1.0))

    return {
        "histogram_change_score": scalar(hist_change),
        "edge_change_score": scalar(edge_change),
        "pixel_change_score": scalar(pixel_change),
        "scene_change_score": scalar(scene_change),
        "motion_magnitude_prev": scalar(previous_motion),
        "motion_magnitude_cur": scalar(current_motion),
        "raw_motion_change": scalar(raw_motion_change),
        "motion_change_score": scalar(raw_motion_change / motion_scale),
        "raw_cross_boundary_motion": scalar(raw_cross_motion),
        "cross_boundary_motion_score": scalar(raw_cross_motion / cross_motion_scale),
    }


def encode_subtitle_chunks(
    texts: Sequence[str],
    embedding_model: Any,
    tokenizer: Any,
    batch_size: int = 32,
) -> List[Optional[torch.Tensor]]:
    """Batch-encode non-empty chunk subtitles with the existing BGE encoder."""
    embeddings: List[Optional[torch.Tensor]] = [None] * len(texts)
    nonempty = [(idx, text.strip()) for idx, text in enumerate(texts) if str(text).strip()]
    if not nonempty:
        return embeddings
    device = next(embedding_model.parameters()).device
    for start in range(0, len(nonempty), max(1, int(batch_size))):
        batch = nonempty[start : start + max(1, int(batch_size))]
        encoded = tokenizer(
            [text for _, text in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
        with torch.inference_mode():
            output = embedding_model(**encoded).last_hidden_state[:, 0].float()
            output = F.normalize(output, p=2, dim=1).cpu()
        for (idx, _), embedding in zip(batch, output):
            embeddings[idx] = embedding
    return embeddings


def subtitle_semantic_scores(
    previous_text: str,
    current_text: str,
    previous_embedding: Optional[torch.Tensor],
    current_embedding: Optional[torch.Tensor],
) -> Dict[str, float]:
    """Return stable subtitle similarity/change scores, including empty cases."""
    prev_empty = not str(previous_text).strip()
    cur_empty = not str(current_text).strip()
    if prev_empty and cur_empty:
        similarity = 1.0
    elif prev_empty or cur_empty:
        # High change, but its 0.8 value cannot by itself exceed the default
        # weighted event threshold (subtitle weight 0.25).
        similarity = 0.2
    elif previous_embedding is None or current_embedding is None:
        similarity = 0.0
    else:
        similarity = float(torch.dot(previous_embedding, current_embedding).clamp(0.0, 1.0))
    similarity = min(1.0, max(0.0, similarity))
    return {"subtitle_similarity": similarity, "subtitle_change_score": 1.0 - similarity}


def _frame_to_uint8(frame: torch.Tensor, width: int = 160, height: int = 96) -> np.ndarray:
    x = _as_unit_rgb(frame.unsqueeze(0), max(width, height))[0]
    x = F.interpolate(x.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
    return (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_boundary_contact_sheets(
    chunks: Sequence[torch.Tensor],
    boundaries: Sequence[Dict[str, Any]],
    output_dir: str,
    video_name: str,
    frame_window: int,
    scene_threshold: float,
    event_threshold: float,
) -> List[str]:
    """Save the requested top-scene, top-event, and near-threshold merge sheets."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return []
    if not boundaries:
        return []

    chosen: Dict[int, str] = {}
    for label, key, count in (
        ("scene_top", "scene_change_score", 3),
        ("event_top", "event_boundary_score", 3),
    ):
        ranked = sorted(boundaries, key=lambda item: float(item.get(key, -1.0)), reverse=True)
        for item in ranked[:count]:
            chosen.setdefault(int(item["right_chunk"]), label)
    merged = [item for item in boundaries if item.get("decision") == "merge"]
    merged.sort(
        key=lambda item: min(
            abs(scene_threshold - float(item.get("scene_change_score", 0.0))),
            abs(event_threshold - float(item.get("event_boundary_score", 0.0))),
        )
    )
    for item in merged[:2]:
        chosen.setdefault(int(item["right_chunk"]), "merge_near_threshold")

    safe_video = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(video_name)[0])
    video_dir = os.path.join(output_dir, safe_video)
    os.makedirs(video_dir, exist_ok=True)
    saved: List[str] = []
    thumb_w, thumb_h = 160, 96
    window = max(2, min(4, int(frame_window)))
    by_right = {int(item["right_chunk"]): item for item in boundaries}
    for right_idx, label in sorted(chosen.items()):
        if right_idx <= 0 or right_idx >= len(chunks):
            continue
        item = by_right[right_idx]
        prev_frames = chunks[right_idx - 1][-window:]
        cur_frames = chunks[right_idx][:window]
        cols = max(len(prev_frames), len(cur_frames), 1)
        canvas = Image.new("RGB", (cols * thumb_w, 130 + 2 * thumb_h), "white")
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{video_name}  chunks {right_idx - 1}->{right_idx}  "
            f"t={float(item.get('time_seconds', 0.0)):.2f}s  {item.get('decision')}\n"
            f"scene={float(item.get('scene_change_score', 0.0)):.3f}  "
            f"event={float(item.get('event_boundary_score', 0.0)):.3f}  "
            f"subtitle={float(item.get('subtitle_change_score', 0.0)):.3f}  "
            f"motion={float(item.get('motion_change_score', 0.0)):.3f}  "
            f"cross={float(item.get('cross_boundary_motion_score', 0.0)):.3f}\n"
            f"reason={item.get('reason')}  sample={label}"
        )
        draw.multiline_text((8, 8), title, fill="black", spacing=4)
        for col, frame in enumerate(prev_frames):
            canvas.paste(Image.fromarray(_frame_to_uint8(frame, thumb_w, thumb_h)), (col * thumb_w, 130))
        for col, frame in enumerate(cur_frames):
            canvas.paste(Image.fromarray(_frame_to_uint8(frame, thumb_w, thumb_h)), (col * thumb_w, 130 + thumb_h))
        path = os.path.join(video_dir, f"boundary_{right_idx:04d}_{label}_{item.get('decision')}.jpg")
        canvas.save(path, quality=90)
        saved.append(path)
    return saved
