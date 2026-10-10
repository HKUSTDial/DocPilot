#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REASONER="$REPO_ROOT/pipelines/training/reasoner.py"
OUTPUT_ROOT="$REPO_ROOT/pipelines/training/single-ablity-data"
MODEL="qwen3.5-397b-a17b"
LIMIT_ARGS=()
DATASETS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT_ARGS=(--limit "$2")
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --dataset)
      DATASETS+=("$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  echo "Usage: $0 --dataset /path/to/test.jsonl [--dataset /path/to/other.jsonl] [--model MODEL] [--limit N] [--output-root DIR]" >&2
  exit 1
fi

dataset_name() {
  local input="$1"
  local parent
  parent="$(basename "$(dirname "$input")")"
  if [[ "$parent" == "data" || "$parent" == "qa" || "$parent" == "bbox" || "$parent" == "annotations" ]]; then
    basename "$(dirname "$(dirname "$input")")"
  else
    basename "$(dirname "$input")"
  fi
}

for input in "${DATASETS[@]}"; do
  if [[ ! -f "$input" ]]; then
    echo "Missing input file: $input" >&2
    exit 1
  fi

  name="$(dataset_name "$input")"
  split="$(basename "$input" .jsonl)"
  out_dir="$OUTPUT_ROOT/$name"
  out_file="$out_dir/$split.predictions.jsonl"
  mkdir -p "$out_dir"

  python "$REASONER" \
    --input "$input" \
    --output "$out_file" \
    --model "$MODEL" \
    "${LIMIT_ARGS[@]}"
done
