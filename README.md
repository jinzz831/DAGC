# Coarse Indexing, Fine Evidence: Density-Aware Graph Construction for Long-Video RAG

🎞️ Density-Adaptive Coarsening | 🕸️ Compact Graph Indexing | 🔍 Fine-Evidence Recovery

> A training-free method that reduces redundant graph construction while preserving access to the original visual evidence.

---

## 🧭 Overview

<p align="center">
  <img src="assets/motivation.png" width="100%" alt="Comparison of fine indexing, coarse indexing, and DAGC adaptive coarse indexing with fine evidence recovery">
</p>


Long-video graph RAG systems commonly use the same fixed-length chunks for both indexing and reasoning. This is inefficient in visually repetitive regions and may lose important details when coarse units are used directly as evidence.

**Density-Aware Graph Construction (DAGC)** decouples these two granularities:

- **Coarse indexing:** merge visually redundant neighboring chunks into compact super-nodes.
- **Fine evidence:** retain the original chunk mapping and recover fine-grained clips after retrieval.

DAGC is training-free and can be integrated into different long-video graph RAG pipelines and vision-language models.



---

## 🧩 Framework

<p align="center">
  <img src="assets/pipeline.png" width="100%" alt="Density-Aware Graph Construction pipeline">
</p>

The framework contains three stages:

| Stage | Description |
|---|---|
| 1. Density-Adaptive Coarsening | Merge adjacent chunks with high visual redundancy under a bounded temporal span. |
| 2. Coarse Graph Construction | Build graph semantics only for the resulting super-nodes using a fixed visual budget. |
| 3. Fine-Evidence Recovery | Expand retrieved super-nodes back to their original chunks before verification and answering. |

Each super-node stores:

```text
original_indices
span_start
span_end
num_merged
```

This allows the graph to remain compact without permanently replacing the evidence used for reasoning.

---

## ✨ Highlights

- **Training-free:** no additional segmentation or compression model is required.
- **Framework-agnostic:** DAGC changes indexing units while preserving the host system's graph semantics and retrieval logic.
- **Fixed per-node cost:** merged super-nodes are uniformly sampled to the same frame budget as an original chunk.
- **Recoverable evidence:** retrieved nodes are mapped back to original chunks for fine-grained verification.
- **Cache-safe:** graph caches are separated by chunk size, similarity threshold, merge span, and boundary configuration.

---

## 📊 Key Results

Main results on three long-video benchmarks. **Accuracy** is multiple-choice accuracy, **Retained** is the graph-node ratio relative to the Vgent graph, and **Wall Speedup** is end-to-end wall-clock acceleration relative to the corresponding Vgent baseline. **Performance Retention** is computed from the average accuracy across the three benchmarks.

| Model / Method | MLVU Acc. | MLVU Retained | MLVU Speedup | VideoMME Acc. | VideoMME Retained | VideoMME Speedup | LVB Acc. | LVB Retained | LVB Speedup | Avg. Acc. | Performance Retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 69.0 | - | - | 70.1 | - | - | 59.4 | - | - | 66.2 | - |
| + Vgent | 73.3 | 100% | 1.0x | 73.3 | 100% | 1.0x | 63.3 | 100% | 1.0x | 70.0 | 100% |
| **+ DAGC** | **73.4** | **45%** | **1.3x** | **71.1** | **45%** | **1.4x** | **63.1** | **47%** | **1.6x** | **69.2** | **99%** |
| Qwen2.5-VL-3B | 65.0 | - | - | 67.0 | - | - | 56.3 | - | - | 62.8 | - |
| + Vgent | 70.0 | 100% | 1.0x | 69.0 | 100% | 1.0x | 60.0 | 100% | 1.0x | 66.3 | 100% |
| **+ DAGC** | **69.9** | **45%** | **1.3x** | **66.2** | **45%** | **1.3x** | **60.6** | **47%** | **1.3x** | **65.6** | **99%** |
| Qwen2-VL-7B | 65.7 | - | - | 68.6 | - | - | 56.1 | - | - | 63.5 | - |
| + Vgent | 71.7 | 100% | 1.0x | 69.7 | 100% | 1.0x | 58.9 | 100% | 1.0x | 66.8 | 100% |
| **+ DAGC** | **72.1** | **45%** | **1.5x** | **67.3** | **45%** | **1.4x** | **58.6** | **47%** | **1.7x** | **66.0** | **99%** |

---

## 📦 Repository Structure

```text
.
├── assets/                         # README figures
├── models/                         # Vision-language model adapters
├── scripts/                        # Launchers and result utilities
├── tests/                          # Boundary and merge tests
├── utils/
│   ├── dagc.py                     # DAGC and graph-RAG integration
│   ├── boundary_segmentation.py    # Optional boundary signals
│   ├── config.py                   # Command-line arguments
│   ├── data.py                     # Dataset readers
│   ├── prompts.py
│   └── retrieval.py
├── dagc_rag.py                     # Distributed evaluation entry point
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation

Install PyTorch for your CUDA environment, then run:

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Paths

```bash
export DATA_PATH=/path/to/MLVU
export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export EMBEDDING_PATH=/path/to/bge-large-en-v1.5
```

### Run

```bash
bash scripts/run_mlvu_needle.sh \
  dagc \
  ./outputs/mlvu_needle/dagc \
  ./graphs/mlvu_needle/dagc \
  4
```

The final argument specifies the number of distributed processes. Use a separate graph directory for every configuration.

### Test

```bash
python -m py_compile dagc_rag.py utils/*.py models/*.py scripts/*.py
pytest -q tests/test_boundary_segmentation.py
```

---

## ⚙️ Main Configuration

| Argument | Default | Description |
|---|---:|---|
| `--chunk_size` | 64 | Frames in each original chunk |
| `--adjacent_sim_threshold` | 0.95 | Adjacent redundancy threshold |
| `--max_supernode_span` | 4 | Maximum original chunks per super-node |
| `--supernode_target_frames` | 64 | Visual frame budget for each super-node |
| `--top_seed_k` | 12 | Initial coarse retrieval seeds |
| `--original_expand_hop` | 1 | Temporal expansion after recovery |
| `--mlvu_task` | unset | Case-insensitive MLVU subset filter |

The default setup samples video at 1 FPS. Model checkpoints and datasets are not included in this repository.

---

## 📄 License

Please add the license and copyright notice required by the original implementation before redistribution. Model and dataset licenses apply separately.
