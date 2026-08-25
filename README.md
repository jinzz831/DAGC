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

<table>
  <thead>
    <tr>
      <th rowspan="2">Model / Method</th>
      <th colspan="3">MLVU</th>
      <th colspan="3">VideoMME</th>
      <th colspan="3">LVB</th>
      <th rowspan="2">Avg.<br>Accuracy</th>
      <th rowspan="2">Performance<br>Retention</th>
    </tr>
    <tr>
      <th>Accuracy</th>
      <th>Retained</th>
      <th>Wall Speedup</th>
      <th>Accuracy</th>
      <th>Retained</th>
      <th>Wall Speedup</th>
      <th>Accuracy</th>
      <th>Retained</th>
      <th>Wall Speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-VL-7B</td>
      <td>69.0</td><td>&mdash;</td><td>&mdash;</td>
      <td>70.1</td><td>&mdash;</td><td>&mdash;</td>
      <td>59.4</td><td>&mdash;</td><td>&mdash;</td>
      <td>66.2</td><td>&mdash;</td>
    </tr>
    <tr>
      <td>+ Vgent</td>
      <td>73.3</td><td>100%</td><td>1.0&times;</td>
      <td>73.3</td><td>100%</td><td>1.0&times;</td>
      <td>63.3</td><td>100%</td><td>1.0&times;</td>
      <td>70.0</td><td>100%</td>
    </tr>
    <tr>
      <td><strong>+ DAGC</strong></td>
      <td><strong>73.4</strong></td><td><strong>45%</strong></td><td><strong>1.3&times;</strong></td>
      <td><strong>71.1</strong></td><td><strong>45%</strong></td><td><strong>1.4&times;</strong></td>
      <td><strong>63.1</strong></td><td><strong>47%</strong></td><td><strong>1.6&times;</strong></td>
      <td><strong>69.2</strong></td><td><strong>99%</strong></td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-3B</td>
      <td>65.0</td><td>&mdash;</td><td>&mdash;</td>
      <td>67.0</td><td>&mdash;</td><td>&mdash;</td>
      <td>56.3</td><td>&mdash;</td><td>&mdash;</td>
      <td>62.8</td><td>&mdash;</td>
    </tr>
    <tr>
      <td>+ Vgent</td>
      <td>70.0</td><td>100%</td><td>1.0&times;</td>
      <td>69.0</td><td>100%</td><td>1.0&times;</td>
      <td>60.0</td><td>100%</td><td>1.0&times;</td>
      <td>66.3</td><td>100%</td>
    </tr>
    <tr>
      <td><strong>+ DAGC</strong></td>
      <td><strong>69.9</strong></td><td><strong>45%</strong></td><td><strong>1.3&times;</strong></td>
      <td><strong>66.2</strong></td><td><strong>45%</strong></td><td><strong>1.3&times;</strong></td>
      <td><strong>60.6</strong></td><td><strong>47%</strong></td><td><strong>1.3&times;</strong></td>
      <td><strong>65.6</strong></td><td><strong>99%</strong></td>
    </tr>
    <tr>
      <td>Qwen2-VL-7B</td>
      <td>65.7</td><td>&mdash;</td><td>&mdash;</td>
      <td>68.6</td><td>&mdash;</td><td>&mdash;</td>
      <td>56.1</td><td>&mdash;</td><td>&mdash;</td>
      <td>63.5</td><td>&mdash;</td>
    </tr>
    <tr>
      <td>+ Vgent</td>
      <td>71.7</td><td>100%</td><td>1.0&times;</td>
      <td>69.7</td><td>100%</td><td>1.0&times;</td>
      <td>58.9</td><td>100%</td><td>1.0&times;</td>
      <td>66.8</td><td>100%</td>
    </tr>
    <tr>
      <td><strong>+ DAGC</strong></td>
      <td><strong>72.1</strong></td><td><strong>45%</strong></td><td><strong>1.5&times;</strong></td>
      <td><strong>67.3</strong></td><td><strong>45%</strong></td><td><strong>1.4&times;</strong></td>
      <td><strong>58.6</strong></td><td><strong>47%</strong></td><td><strong>1.7&times;</strong></td>
      <td><strong>66.0</strong></td><td><strong>99%</strong></td>
    </tr>
  </tbody>
</table>

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
