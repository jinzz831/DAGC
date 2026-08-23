# Coarse Indexing, Fine Evidence

## Density-Aware Graph Construction for Long-Video RAG

Density-Aware Graph Construction (DAGC) is a training-free method for reducing redundant graph construction in long-video retrieval-augmented generation. Its central idea is simple:

> The temporal granularity used to search a video does not need to be the same as the granularity used to reason about visual evidence.

DAGC builds a compact, density-adaptive coarse index by merging visually redundant neighboring chunks. Every coarse node retains a mapping to its constituent original chunks. Retrieval operates on the compact graph, while downstream reranking, verification, and answer generation recover the original fine-grained evidence.

DAGC therefore compresses the representation used to **search** the video without permanently compressing the evidence used to **reason** about it.

The method is not tied to a particular vision-language model or graph-RAG framework. It changes the temporal units on which graph semantics are instantiated and can be integrated with different graph definitions, retrieval algorithms, refinement procedures, and LVLM backbones.

## Motivation

Long videos have highly non-uniform temporal redundancy:

- rapidly changing regions benefit from fine indexing resolution;
- visually stable regions often contain several neighboring chunks that can share one indexing unit;
- final question answering may still require a specific original chunk, frame, subtitle, or object-state transition.

Conventional graph-based video RAG commonly uses each fixed-length chunk both as an indexing node and as downstream visual evidence. This couples two stages with different requirements:

- **indexing** needs an efficient representation for locating relevant temporal regions;
- **reasoning** needs precise access to the original visual evidence.

DAGC explicitly decouples them.

## Method

DAGC consists of three stages:

1. **Density-adaptive index coarsening** merges locally redundant neighboring chunks.
2. **Bounded-cost coarse graph construction** extracts graph semantics only for the resulting super-nodes.
3. **Coarse-to-fine evidence recovery** maps retrieved super-nodes back to original chunks before question-specific reasoning.

### 1. Density-adaptive index coarsening

Let a video be divided into fixed-length chunks:

```text
X = {c1, c2, ..., cN}
```

Each chunk contains `K` sampled frames. A lightweight visual feature is obtained by temporal pooling and L2 normalization:

```text
vi = Norm(Pool(ci))
```

Adjacent visual similarity is:

```text
sv(i-1, i) = vi-1^T vi
```

Neighboring chunks are greedily merged when they are sufficiently similar and the current super-node remains within the bounded merging span:

```text
sv(i-1, i) >= tau_v
and
|Im| < W
```

where:

- `tau_v` is the adjacent visual-similarity threshold;
- `W` bounds the merging span;
- `Im` stores the original chunk indices assigned to super-node `nm`.

The span constraint prevents a long, visually stable region from collapsing into an excessively coarse indexing unit.

After coarsening, `N` original chunks are represented by `M <= N` indexing units. The retained indexing-unit ratio is:

```text
rho = M / N
```

### 2. Fixed-budget coarse graph construction

Frames from the chunks assigned to one super-node are concatenated in temporal order and uniformly resampled to the same `K`-frame budget used by an original chunk:

```text
{ci | i in Im} -> concatenate -> uniform K-frame sample -> c_tilde_m
```

Increasing a super-node's temporal coverage therefore does not increase its per-node visual input budget.

The host LVLM extracts the structured semantics required by the chosen graph-RAG framework, such as entities, actions, scenes, or textual descriptions. DAGC is orthogonal to the semantic graph definition: temporal, semantic, entity, or hybrid edges can be constructed over the coarse units.

The expensive query-independent extraction stage is executed for `M` super-nodes instead of all `N` original chunks.

### 3. Coarse-to-fine evidence recovery

Given a question `Q`, the host retrieval algorithm ranks nodes in the coarse graph. Let the expanded retrieved super-node set be `R+`. DAGC recovers original-granularity candidates through the stored mappings:

```text
Ccand = union({ci | i in Im}) for nm in R+
```

The resulting candidate set contains original chunks rather than compressed super-nodes. A host pipeline can then apply its normal:

- temporal expansion;
- reranking;
- question-specific visual verification;
- subtitle or speech evidence selection;
- final LVLM answer generation.

This recovery stage is essential. Coarse nodes are designed for efficient localization, not as a replacement for fine-grained evidence.

## Framework integration contract

DAGC requires only a small interface from a host long-video graph-RAG system:

1. Split a sampled video into ordered fixed-length chunks.
2. Provide a visual tensor for each chunk.
3. Construct graph semantics for a supplied temporal unit.
4. Return coarse node identifiers from the host retrieval procedure.
5. Run the host's normal refinement and answer generation over recovered original chunks.

Each DAGC node records:

```text
original_indices
span_start
span_end
num_merged
```

The reference implementation additionally records boundary diagnostics when the optional analysis extension is enabled.

To integrate DAGC into another graph-RAG framework, the framework-specific graph construction and retrieval functions can remain unchanged. The required changes are limited to:

- replacing fixed chunk nodes with DAGC super-nodes during index construction;
- storing the original-index mapping on every super-node;
- expanding retrieved super-nodes through that mapping before fine-grained refinement.

## Reference implementation

```text
.
├── models/
│   ├── qwenvl.py                    # Reference Qwen-family LVLM adapter
│   └── utils.py                     # Video loading and resizing
├── scripts/
│   ├── probe_boundary_groups.py     # Coarsening and boundary probe
│   ├── run_mlvu_needle.sh           # Paired evaluation launcher
│   └── summarize_experiments.py     # Result summarizer
├── tests/
│   └── test_boundary_segmentation.py
├── utils/
│   ├── dagc.py                      # DAGC and reference graph-RAG pipeline
│   ├── boundary_segmentation.py     # Optional boundary-analysis signals
│   ├── config.py                    # CLI arguments
│   ├── data.py                      # Benchmark readers
│   ├── prompts.py
│   └── retrieval.py
├── dagc_rag.py                      # Distributed evaluation entry point
└── requirements.txt
```

The repository contains source code only. It does not include model weights, datasets, graph caches, logs, predictions, diagnostics, or experiment reports.

## Default configuration

The paper uses the following main configuration:

| Setting | Value |
|---|---:|
| Video sampling | 1 FPS |
| Original chunk budget `K` | 64 frames |
| Adjacent similarity threshold `tau_v` | 0.95 |
| Maximum adjacent merge operations | 3 |
| Maximum original chunks per released super-node | 4 |
| Coarse retrieval seeds | 12 |
| Super-node visual budget | 64 frames |

The released `max_supernode_span=4` permits at most four original chunks in one super-node, corresponding to at most three adjacent merge operations.

## Installation

Create a Python environment compatible with the target CUDA installation. Install PyTorch using the official instructions for that environment, then run:

```bash
pip install -r requirements.txt
```

The reference Qwen adapter uses FlashAttention 2:

```bash
pip install flash-attn --no-build-isolation
```

Default public model identifiers are:

```text
Qwen/Qwen2.5-VL-7B-Instruct
BAAI/bge-large-en-v1.5
```

Local checkpoints can be supplied through `--model_path` and `--embedding_path`.

## Dataset layout

For MLVU experiments, the dataset root is expected to follow:

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
  utils/dagc.py \
  utils/boundary_segmentation.py \
  utils/config.py \
  utils/data.py \
  models/qwenvl.py \
  models/utils.py \
  dagc_rag.py \
  scripts/probe_boundary_groups.py \
  scripts/summarize_experiments.py \
  tests/test_boundary_segmentation.py

pytest -q tests/test_boundary_segmentation.py
```

## Running the reference evaluation

Configure dataset and model locations without editing source code:

```bash
export DATA_PATH=/path/to/MLVU_ROOT
export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export EMBEDDING_PATH=/path/to/bge-large-en-v1.5
```

Run DAGC:

```bash
bash scripts/run_mlvu_needle.sh \
  dagc \
  ./outputs/mlvu_needle/dagc \
  ./graphs/mlvu_needle/dagc \
  4
```

The first positional label above selects the released adjacent-similarity DAGC path. To compare against a fully uncompressed graph, run `dagc_rag.py` with `--direct_baseline` and a separate graph directory.

Run the optional boundary-aware analysis variant:

```bash
bash scripts/run_mlvu_needle.sh \
  boundary_dagc \
  ./outputs/mlvu_needle/boundary_dagc \
  ./graphs/mlvu_needle/boundary_dagc \
  4 \
  --boundary_debug
```

The launcher keeps model, dataset, seed, sampling, retrieval, and refinement settings identical between paired runs.

## Optional boundary-aware analysis

The main DAGC method uses adjacent visual similarity as a lightweight proxy for local redundancy. The repository also includes an optional analysis variant that can prevent merging across high boundary scores.

The lightweight scene score combines:

```text
scene_change_score =
    0.40 * histogram_change
  + 0.30 * edge_change
  + 0.30 * pixel_change
```

The optional event score combines:

```text
event_boundary_score =
    w_visual       * visual_change_score
  + w_motion       * motion_change_score
  + w_cross_motion * cross_boundary_motion_score
  + w_subtitle     * subtitle_change_score
```

Default weights are `0.30`, `0.25`, `0.20`, and `0.25`. Scene and event thresholds default to `0.45`.

This extension is provided for controlled analysis. It is not the definition of DAGC and is not a trained generic event-boundary detector.

## Why DAGC is not event segmentation

Event segmentation and DAGC optimize different objectives:

- event segmentation seeks perceptually or semantically coherent temporal partitions;
- DAGC seeks a compact indexing representation while preserving access to question-relevant fine evidence.

A perceptual transition may be irrelevant to a question, while a brief object-state change, subtitle, or visual detail inside a long event may be decisive. Generic event partitions can also introduce a full-video detection pass, more indexing nodes, more retrieval candidates, and variable-duration segments that still require sparse or fixed-budget sampling before LVLM processing.

DAGC does not claim that a super-node is a complete semantic event. It asks a narrower systems question: can neighboring chunks share one coarse indexing representation without deleting access to their original evidence?

## Main findings reported in the paper

Across MLVU, VideoMME, and LongVideoBench, multiple LVLM backbones, and more than one video graph-RAG pipeline, DAGC:

- retains approximately **40-50%** of the original indexing units;
- achieves approximately **1.3-1.7x** end-to-end wall-clock acceleration;
- preserves approximately **99%** of the original QA performance;
- reduces query-independent graph-construction time substantially while adding only a small online recovery cost.

The benefit is primarily computational rather than a guarantee of accuracy improvement on every benchmark.

### Coarsening strategy at a matched graph budget

On LongVideoBench with a 47% retained-node budget:

| Strategy | Accuracy | Retained |
|---|---:|---:|
| Random merge | 61.53 | 47% |
| Uniform merge | 62.28 | 47% |
| DAGC | **63.07** | 47% |

This comparison shows that DAGC's behavior is not explained solely by generic node reduction. Local visual redundancy provides a better indexing-allocation signal than content-agnostic merging under the same graph budget.

### Importance of fine-evidence recovery

Using compressed super-nodes directly as final evidence reduces performance. Recovering and reranking their original chunks restores most of the accuracy, confirming the core design principle:

> Coarse indexing is effective only when fine-grained evidence remains recoverable.

### Event-boundary analysis

Adding learned generic event boundaries as hard merge constraints produced task-dependent changes on the complete MLVU Order, Needle, and Count subsets:

| Task | DAGC | + Event boundary | Delta |
|---|---:|---:|---:|
| Order | 70.66 | 72.97 | +2.32 pp |
| Needle | 82.25 | 81.69 | -0.56 pp |
| Count | 60.68 | 57.77 | -2.91 pp |
| Weighted overall | 73.17 | 72.93 | -0.24 pp |

Explicit event boundaries did not provide a consistent overall QA improvement. This supports treating temporal granularity as an indexing-system variable rather than requiring a single semantic partition throughout the pipeline.

## Important arguments

| Argument | Default | Description |
|---|---:|---|
| `--adjacent_sim_threshold` | 0.95 | Adjacent redundancy threshold |
| `--max_supernode_span` | 4 | Maximum original chunks per super-node |
| `--supernode_target_frames` | 64 | Fixed visual budget for a coarse node |
| `--top_seed_k` | 12 | Initial coarse retrieval seeds |
| `--original_expand_hop` | 1 | Temporal expansion after recovery |
| `--boundary_aware_merge` | off | Enable optional hard boundary guards |
| `--scene_boundary_threshold` | 0.45 | Optional scene threshold |
| `--event_boundary_threshold` | 0.45 | Optional event threshold |
| `--mlvu_task` | unset | Case-insensitive MLVU subset filter |

## Scope and limitations

- Adjacent visual similarity is a lightweight proxy for local redundancy and may miss semantic changes in visually stable regions, including evolving dialogue and subtle object-state transitions.
- Information-dense videos may require more conservative thresholds or merging spans.
- DAGC is designed for efficient long-video QA, not precise event segmentation.
- Super-nodes are not guaranteed to correspond to complete semantic events.
- Broader validation on additional graph structures and retrieval frameworks remains useful future work.
- The reference repository includes one complete integration; porting to another host graph-RAG system requires adapting the framework-specific semantic extraction and retrieval interfaces described above.

## Reproducibility and generated files

Graph cache identities include the coarsening method, chunk size, similarity threshold, maximum span, and optional boundary configuration. Incompatible graph configurations therefore do not reuse the same cache.

Generated artifacts are excluded by `.gitignore`, including:

```text
data/
checkpoints/
graphs/
outputs/
logs/
diagnostics/
boundary_visualizations/
```

## License

Add the license and copyright notice required by the original implementation before public redistribution. Model and dataset licenses apply separately.
