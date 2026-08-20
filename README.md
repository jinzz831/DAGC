# Boundary-Aware DAGC for Vgent

This repository contains a training-free boundary-aware extension of Dynamic Adjacent Graph Compression (DAGC) for long-video question answering. It preserves Vgent's original graph retrieval, node refinement, and answer aggregation pipeline while adding optional scene, motion, and subtitle boundary guards to adjacent fixed-length chunk merging.

The implementation used in the accompanying experiment keeps 64-frame chunks as the atomic graph units. It does not train or download a separate scene segmentation model.

## Method overview

Original DAGC merges adjacent chunks when:

```python
can_merge = (
    current_group_len < max_supernode_span
    and visual_similarity >= adjacent_sim_threshold
)
```

With `--boundary_aware_merge`, the merge rule becomes:

```python
can_merge = (
    current_group_len < max_supernode_span
    and visual_similarity >= adjacent_sim_threshold
    and not hard_scene_boundary
    and not hard_event_boundary
)
```

The original visual feature is the normalized flattened mean frame of each chunk. Boundary-Aware DAGC adds the following signals at every adjacent chunk join:

- RGB histogram change;
- grayscale edge-structure change;
- mean pixel change;
- change in within-chunk motion magnitude;
- cross-boundary frame difference;
- subtitle embedding change using the existing BGE encoder.

The scene score is:

```text
scene_change_score =
    0.40 * histogram_change
  + 0.30 * edge_change
  + 0.30 * pixel_change
```

The event score is:

```text
event_boundary_score =
    w_visual       * visual_change_score
  + w_motion       * motion_change_score
  + w_cross_motion * cross_boundary_motion_score
  + w_subtitle     * subtitle_change_score
```

The default weights are `0.30`, `0.25`, `0.20`, and `0.25`. The default scene and event thresholds are both `0.45`.

These lightweight scores are diagnostic boundary cues rather than a trained generic event-boundary detector.

## Repository layout

```text
.
├── models/
│   ├── qwenvl.py                    # Qwen2.5-VL adapter
│   └── utils.py                     # Video loading and resizing
├── scripts/
│   ├── probe_boundary_groups.py     # Segmentation-only probe
│   ├── run_mlvu_needle.sh           # DAGC/Boundary-DAGC launcher
│   └── summarize_experiments.py     # Result summarizer
├── tests/
│   └── test_boundary_segmentation.py
├── utils/
│   ├── boundary_segmentation.py     # Boundary signals and contact sheets
│   ├── config.py                    # CLI arguments
│   ├── data.py                      # MLVU/VideoMME/LVB readers
│   ├── prompts.py
│   ├── retrieval.py
│   └── vgent.py                     # DAGC, retrieval, refine, aggregate
├── vgent_rag.py                     # Distributed evaluation entry point
└── requirements.txt
```

No model weights, datasets, graph caches, logs, predictions, diagnostics, or experiment reports are included.

## Installation

Create a Python environment compatible with your CUDA installation, install PyTorch following the official PyTorch instructions, and then run:

```bash
pip install -r requirements.txt
```

The Qwen adapter uses FlashAttention 2. Install a version compatible with your CUDA, PyTorch, and compiler environment:

```bash
pip install flash-attn --no-build-isolation
```

Default model identifiers:

```text
Qwen/Qwen2.5-VL-7B-Instruct
BAAI/bge-large-en-v1.5
```

To avoid downloading models at runtime, pass local directories through `--model_path` and `--embedding_path` or set the environment variables used by the launcher.

## Dataset layout

The MLVU root is expected to follow this structure:

```text
MLVU_ROOT/
├── json/
│   ├── 1_plotQA.json
│   ├── 2_needle.json
│   ├── 3_ego.json
│   ├── 4_count.json
│   ├── 5_order.json
│   ├── 6_anomaly_reco.json
│   └── 7_topic_reasoning.json
└── video/
    ├── 1_plotQA/
    ├── 2_needle/
    ├── 3_ego/
    ├── 4_count/
    ├── 5_order/
    ├── 6_anomaly_reco/
    └── 7_topic_reasoning/
```

Datasets must be obtained separately under their respective licenses.

## Static checks and tests

```bash
python -m py_compile \
  utils/boundary_segmentation.py \
  utils/vgent.py \
  utils/config.py \
  utils/data.py \
  models/qwenvl.py \
  models/utils.py \
  vgent_rag.py \
  scripts/probe_boundary_groups.py \
  scripts/summarize_experiments.py \
  tests/test_boundary_segmentation.py

pytest -q tests/test_boundary_segmentation.py
```

The tests cover identical chunks, color changes, motion changes, subtitle changes, empty subtitles, short final chunks, and disabling the boundary-aware switch.

## Segmentation-only probe

The probe decodes selected videos and runs the exact grouping implementation without loading the vision-language model:

```bash
python scripts/probe_boundary_groups.py \
  --data_path /path/to/MLVU_ROOT \
  --output_dir ./outputs/boundary_probe \
  --video_names needle_1,needle_2 \
  --chunk_size 64 \
  --fps 1.0 \
  --adjacent_sim_threshold 0.95 \
  --max_supernode_span 4 \
  --scene_boundary_threshold 0.45 \
  --event_boundary_threshold 0.45
```

## MLVU Needle evaluation

The launcher requires four positional arguments:

```text
run_mlvu_needle.sh METHOD EXPERIMENT_DIR GRAPH_DIR NPROC [extra arguments]
```

Set dataset and model locations with environment variables:

```bash
export DATA_PATH=/path/to/MLVU_ROOT
export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export EMBEDDING_PATH=/path/to/bge-large-en-v1.5
```

Run the original DAGC baseline:

```bash
bash scripts/run_mlvu_needle.sh \
  dagc \
  ./outputs/mlvu_needle/dagc \
  ./graphs/mlvu_needle/dagc \
  4
```

Run Boundary-Aware DAGC:

```bash
bash scripts/run_mlvu_needle.sh \
  boundary_dagc \
  ./outputs/mlvu_needle/boundary_dagc \
  ./graphs/mlvu_needle/boundary_dagc \
  4 \
  --boundary_debug
```

The launcher keeps the model, dataset, random seed, chunk size, retrieval parameters, and refinement parameters identical between the two methods.

## Important arguments

| Argument | Default | Description |
|---|---:|---|
| `--boundary_aware_merge` | off | Enable hard scene/event boundary guards |
| `--scene_boundary_threshold` | 0.45 | Scene boundary threshold |
| `--event_boundary_threshold` | 0.45 | Event boundary threshold |
| `--boundary_visual_weight` | 0.30 | Visual-change weight |
| `--boundary_motion_weight` | 0.25 | Motion-change weight |
| `--boundary_cross_motion_weight` | 0.20 | Cross-boundary motion weight |
| `--boundary_subtitle_weight` | 0.25 | Subtitle-change weight |
| `--boundary_frame_window` | 4 | Frames used on each side of a join |
| `--boundary_spatial_size` | 64 | Spatial size used for boundary signals |
| `--adjacent_sim_threshold` | 0.95 | Original DAGC redundancy threshold |
| `--max_supernode_span` | 4 | Maximum chunks per super-node |
| `--mlvu_task` | unset | Case-insensitive MLVU subset filter |

## Graph metadata and diagnostics

Each super-node stores:

```text
original_indices
span_start
span_end
num_merged
internal_boundary_scores
max_internal_scene_score
max_internal_event_score
```

Graph summaries include node counts, retained-node ratio, merge spans, boundary counts, split reasons, score averages, and graph-construction runtime.

When `--boundary_debug` is enabled, the evaluator writes per-video JSON diagnostics and boundary contact sheets into the selected experiment directory. These generated files are ignored by Git.

Graph cache names include a hash of the method and segmentation configuration, preventing DAGC and Boundary-DAGC runs from reusing incompatible graphs.

## Scope and limitations

- Boundary estimation is performed only at joins between adjacent 64-frame chunks.
- The boundary module is training-free and is not a substitute for a dedicated shot or generic event-boundary detector.
- Generic boundary quality and question-answering evidence quality are different objectives.
- Super-nodes preserve original chunk indices and chronological order.
- Merged visual inputs are uniformly sampled to the configured target frame count; short inputs are not repeat-padded.
- The original retrieval, node refinement, and answer aggregation logic is unchanged when Boundary-Aware DAGC is enabled.

## License

Add the license required by the upstream Vgent implementation before publishing or redistributing this repository. Model and dataset licenses apply separately.
