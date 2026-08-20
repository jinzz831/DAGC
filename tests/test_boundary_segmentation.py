from types import SimpleNamespace

import torch

from utils.boundary_segmentation import (
    appearance_and_motion_scores,
    chunk_motion_magnitude,
    subtitle_semantic_scores,
)
from utils.vgent import Vgent


def make_color(rgb, frames=64, height=24, width=24):
    chunk = torch.zeros(frames, 3, height, width)
    for channel, value in enumerate(rgb):
        chunk[:, channel] = value
    return chunk


def test_identical_chunks_have_low_boundary_scores():
    chunk = make_color((0.25, 0.50, 0.75))
    scores = appearance_and_motion_scores(chunk, chunk, frame_window=4)
    assert scores["scene_change_score"] < 1e-6
    assert scores["cross_boundary_motion_score"] < 1e-6


def test_distinct_colors_trigger_scene_boundary_default_threshold():
    red = make_color((1.0, 0.0, 0.0))
    blue = make_color((0.0, 0.0, 1.0))
    scores = appearance_and_motion_scores(red, blue, frame_window=4)
    assert scores["scene_change_score"] >= 0.45


def test_motion_jump_increases_event_signal():
    static = make_color((0.5, 0.5, 0.5))
    moving = static.clone()
    moving[::2] = 0.0
    moving[1::2] = 1.0
    static_motion = chunk_motion_magnitude(static)
    moving_motion = chunk_motion_magnitude(moving)
    scores = appearance_and_motion_scores(
        static,
        moving,
        previous_motion=static_motion,
        current_motion=moving_motion,
    )
    assert moving_motion > static_motion + 0.5
    assert scores["motion_change_score"] > 0.5


def test_subtitle_semantic_change_and_empty_cases():
    same = torch.tensor([1.0, 0.0])
    other = torch.tensor([0.0, 1.0])
    changed = subtitle_semantic_scores("a person", "a landscape", same, other)
    assert changed["subtitle_change_score"] == 1.0
    empty = subtitle_semantic_scores("", "", None, None)
    assert empty == {"subtitle_similarity": 1.0, "subtitle_change_score": 0.0}
    one_sided = subtitle_semantic_scores("text", "", same, None)
    assert 0.0 < one_sided["subtitle_change_score"] < 1.0


def test_short_final_chunk_does_not_crash():
    full = torch.rand(64, 3, 16, 16)
    short = torch.rand(3, 3, 16, 16)
    scores = appearance_and_motion_scores(full, short, frame_window=4)
    assert all(torch.isfinite(torch.tensor(value)) for value in scores.values())


def test_boundary_switch_off_preserves_original_merge_guard():
    vgent = Vgent.__new__(Vgent)
    vgent.args = SimpleNamespace(chunk_size=64, fps=1.0)
    vgent.max_supernode_span = 4
    vgent.adjacent_sim_threshold = 0.95
    vgent.boundary_aware_merge = False
    vgent.boundary_visual_weight = 0.30
    vgent.boundary_motion_weight = 0.25
    vgent.boundary_cross_motion_weight = 0.20
    vgent.boundary_subtitle_weight = 0.25
    vgent.scene_boundary_threshold = 0.45
    vgent.event_boundary_threshold = 0.45
    vgent.boundary_frame_window = 4
    vgent.boundary_spatial_size = 64
    vgent.boundary_motion_scale = 0.05
    vgent.boundary_cross_motion_scale = 0.40
    identical = make_color((0.2, 0.4, 0.6))
    groups, _, boundaries = vgent._build_supernode_groups([identical, identical], None)
    assert groups == [[0, 1]]
    assert boundaries[0]["decision"] == "merge"
    feat = vgent._chunk_to_feature(identical)
    assert vgent._can_hard_merge(
        feat,
        feat,
        boundary={"hard_scene_boundary": True, "hard_event_boundary": True},
    )
