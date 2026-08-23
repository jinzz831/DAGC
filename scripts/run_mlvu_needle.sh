#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 {dagc|boundary_dagc} EXPERIMENT_DIR GRAPH_DIR NPROC [extra args...]" >&2
  exit 2
fi

# Override these defaults with environment variables when using local models
# or a non-default dataset location. No machine-specific path is required.
MODEL_NAME="${MODEL_NAME:-qwenvl25_7b}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
EMBEDDING_PATH="${EMBEDDING_PATH:-BAAI/bge-large-en-v1.5}"
DATA_PATH="${DATA_PATH:-./data/MLVU}"

method="$1"
experiment_dir="$2"
graph_dir="$3"
nproc="$4"
shift 4

mkdir -p "$experiment_dir" "$graph_dir"
if [[ "$method" == "boundary_dagc" ]]; then
  enable_boundary=1
elif [[ "$method" != "dagc" ]]; then
  echo "unknown method: $method" >&2
  exit 2
fi

command=(
  torchrun --standalone --nproc_per_node="$nproc" dagc_rag.py
  --model_name "$MODEL_NAME"
  --model_path "$MODEL_PATH"
  --embedding_path "$EMBEDDING_PATH"
  --task mlvu
  --data_path "$DATA_PATH"
  --mlvu_task needle
  --chunk_size 64
  --fps 1.0
  --uniform_frame 450
  --n_retrieval 20
  --n_refine 5
  --adjacent_sim_threshold 0.95
  --max_supernode_span 4
  --seed 42
  --experiment_dir "$experiment_dir"
  --graph_path "$graph_dir"
)
if [[ "${enable_boundary:-0}" == "1" ]]; then
  command+=(--boundary_aware_merge)
fi
command+=("$@")

printf '%q ' "${command[@]}" > "$experiment_dir/command.txt"
printf '\n' >> "$experiment_dir/command.txt"
git diff --binary > "$experiment_dir/git_diff.patch"
"${command[@]}"
