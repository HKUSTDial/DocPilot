# DocPilot Training-Data Pipeline

This directory contains the data conversion and curation utilities used to build DocPilot supervision. It covers the two stages described in the paper:

1. **Single-ability fine-tuning**, which teaches individual document-solving actions from capability-specific traces.
2. **Multi-ability agentic training**, which teaches adaptive action composition from complete DocPilot interactions.

The utilities prepare chat-style training data. Raw datasets, generated traces, model checkpoints, and the downstream training framework are not included.

## Pipeline Overview

```text
Capability datasets                         DocPilot runs
        │                                        │
        ▼                                        ▼
Navigate / Extract / Solve / Answer      *.history.jsonl + state + log
        │                                        │
        ▼                                        ▼
single-ability converters                interaction converter
        │                                        │
        ▼                                        ▼
single-ability chat JSONL                multi-ability chat JSON/JSONL
        └──────────────────┬─────────────────────┘
                           ▼
                  downstream SFT / agentic training
```


## Setup

Run all commands from the artifact root:

```bash
cd guides/iclr/docpilot
python -m pip install -e .
```

Dataset-specific converters require their corresponding source datasets. LLM-backed distillation and rewriting also require a configured model endpoint; keep credentials in the local environment and do not write them into generated data or scripts.

## Single-Ability Data

### Navigate

```bash
python pipelines/training/single-ability-data/convert_navigate.py \
  --input-dir /path/to/docseeker/alr_data \
  --image-root /path/to/source/images \
  --output runs/training/navigate.jsonl \
  --strict-images
```

### Extract

```bash
python pipelines/training/single-ability-data/convert_extract.py \
  --input /path/to/hotpotqa_distractor.json \
  --output runs/training/extract.jsonl
```

The input must be a JSON array. The output is one chat sample per line.

### Solve

```bash
python pipelines/training/single-ability-data/convert_solve.py \
  --input /path/to/file_dataset.jsonl \
  --workspace-root /path/to/referenced/files \
  --output runs/training/solve.jsonl \
  --strict
```

Use `--limit` and `--seed` for reproducible subsampling. `--max-rows` controls the number of rows included when source files are summarized into the prompt.

### Answer Distillation

```bash
python pipelines/training/single-ability-data/distill/answer.py \
  --input /path/to/answer_dataset.jsonl \
  --output runs/training/answer.jsonl \
  --model YOUR_MODEL \
  --workers 1 \
  --resume
```

The script supports generic JSONL plus dataset adapters for InfographicsVQA and InfoChartQA. Use `--limit` for a small endpoint check before a full run.

### Normalize and Combine Capabilities

```bash
python pipelines/training/single-ability-data/convert_single_ability.py \
  --input \
    runs/training/navigate.jsonl \
    runs/training/extract.jsonl \
    runs/training/solve.jsonl \
    runs/training/answer.jsonl \
  --task auto \
  --output runs/training/single_ability_docpilot_chat.jsonl
```

Task inference uses the input path. Pass `--task navigate`, `extract`, `solve`, or `answer` when the filename does not identify the capability. Add `--filter-correct` only when the source rows contain compatible gold and predicted-answer fields.

For a two-stage setup in which routing and action execution are trained separately:

```bash
python pipelines/training/single-ability-data/convert_to_action_only.py \
  --input runs/training/single_ability_docpilot_chat.jsonl \
  --output runs/training/single_ability_action_only.jsonl
```

## Multi-Ability Interaction Data

DocPilot runs normally contain a `*.history.jsonl` trace. When available, the converter also reads the sibling `.state.json` and `.log` files to recover state, reasoning, metadata, and visual evidence.

Convert a run directory:

```bash
python pipelines/training/convert_docpilot_logs_to_agentic_training.py \
  --logs-dir runs/logs \
  --output runs/training/multi_ability_docpilot_chat.jsonl \
  --data-dir runs/training/multi_ability_images
```

Convert one trace:

```bash
python pipelines/training/convert_docpilot_logs_to_agentic_training.py \
  --history-file /path/to/run.history.jsonl \
  --output runs/training/sample.json \
  --data-dir runs/training/sample_images \
  --pretty
```

Alternatively, `--eval-file eval.json` converts only evaluation rows whose `check` field is `true` and whose `trace_path` points to a history file.

For each sample, visual evidence is copied into `<data-dir>/<sample_id>/`. Page images are normalized to `page_<page>.png`; other visual evidence is stored as `block_<index>.png`.

## Output Format

Converters produce JSON or JSONL records with two chat messages and optional image references:

```json
{
  "id": 0,
  "document_id": "document-id",
  "dataset_id": "dataset-id",
  "source": "/path/to/run.history.jsonl",
  "image": ["page_1.png", "block_1.png"],
  "total_tokens": 1024,
  "messages": [
    {"role": "user", "content": "Instruction\n..."},
    {"role": "assistant", "content": "<Analyze>...</Analyze>\n<Navigate>...</Navigate>"}
  ]
}
```

Single-ability converters may use `images` instead of `image`. Preserve image ordering when adapting these records to a training framework.

## Merge Multiple Shards

Validate a merge before writing files:

```bash
python pipelines/training/merge_docpilot_chat.py \
  --input runs/training/part1.jsonl runs/training/part2.jsonl \
  --output runs/training/merged.jsonl \
  --dedupe source \
  --prefer last \
  --dry-run
```

Remove `--dry-run` to write the merged dataset and its `merged.jsonl.progress.json` provenance map. Add `--data-dir` only when every input image reference resolves from the current environment; the merge then copies and renumbers those images.

## Optional Curation Utilities

- `filter/check_true_multipage_extract_multi_reasoning.py` selects correct evaluation cases containing multi-page `Extract` steps and multiple reasoning blocks.
- `filter/rewrite_multipage_extract_reasoning.py` uses a configured LLM to merge those reasoning blocks into candidate-selection decisions. It supports `--resume` and `--workers`.
- `single-ability-data/inspect_json_datasets.py` reports dataset fields, message counts, and image references before conversion.
- `replace_docpilot_paths.py` rewrites dataset roots after relocation. Run it with `--dry-run` first and use `--backup` when modifying the source artifact.

Example path check:

```bash
python pipelines/training/replace_docpilot_paths.py \
  runs/training/single_ability_docpilot_chat.jsonl \
  --old-root /old/dataset/root \
  --new-root /new/dataset/root \
  --dry-run
```

## Validation

Check the action-only conversion contract:

```bash
python pipelines/training/single-ability-data/convert_to_action_only.py --self-test
```

Inspect a small source sample before a full conversion:

```bash
python pipelines/training/single-ability-data/inspect_json_datasets.py \
  --file /path/to/source.jsonl \
  --samples 1 \
  --max-records 10
```

Use `--limit` on conversion and distillation commands for smoke tests. A successful `--help` or syntax check validates the interface only; full generation still depends on source datasets, image paths, and any configured model service.
