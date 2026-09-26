# DocPilot

DocPilot is an agentic document-understanding framework for multimodal document question answering. It couples an LLM policy with document tools for page navigation, evidence extraction, visual focusing, symbolic solving, answer generation, and answer verification.

This directory is a sanitized ICLR artifact snapshot. It keeps the framework, OCR/document preprocessing pipeline, and training-data conversion scripts, while excluding private credentials, local absolute paths, generated datasets, logs, page images, and model checkpoints.

## Contents

```text
app/
  agent/                 ReAct, analyzer, role-specific, and vLLM-tag agents
  tool/                  navigate, extract, focus, solve, execute, answer, analyze
  workflow/              in-memory and MinerU-backed document environments
  prompt/                prompts for action routing and tool use
  cli.py                 main DocPilot inference CLI
  llm.py                 OpenAI-compatible LLM client
  schema.py              state, document node, evidence, and answer schemas
  vllm_protocol.py       parser for trained models that emit action tags

pipelines/
  document/              document indexing, MinerU manifest/run, page-object export
  training/              single-ability and multi-ability training-data converters

configs/
  docpilot.example.yaml          JSON-action agent config
  docpilot-vllm.yaml             tag-protocol trained-model config
  document.example.yaml          OCR/document preprocessing config
```

## Supported Datasets

The document preprocessing and QA runners include dataset adapters for:

- `MMLongBench-Doc`
- `FinRAGBench-V`
- `DocBench`
- `MADQA`
- `MMDocRAG`
- `LongDocURL`
- `DUDE`

Configure local dataset roots in `configs/document.example.yaml` before running OCR or batch QA.

## Installation

Use Python 3.11 or newer.

```bash
cd guides/iclr/docpilot
python -m pip install -e .
```

Main dependencies are listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

MinerU is required only for OCR/document preprocessing stages.

## Configure Models

All endpoint and credential fields are intentionally empty. Fill them in the config you use:

```yaml
llm:
  base_url: ""
  api_key: ""
  judge_base_url: ""
  judge_api_key: ""
```

Use:

- `configs/docpilot.example.yaml` for standard JSON action outputs.
- `configs/docpilot-vllm.yaml` for trained models that emit `<Analyze>`, `<Navigate>`, `<Extract>`, `<Focus>`, `<Solve>`, and `<Answer>` tags.

## Run Pipeline And QA

Optional smoke test with the built-in toy document:

```bash
python -m app \
  --config configs/docpilot.example.yaml \
  --question "What was FY2015 revenue?"
```

For real documents, run the full OCR/document preprocessing pipeline first. Edit `configs/document.example.yaml` to point dataset paths such as `data/DocVQA/...` to your local copies.

```bash
python -m pipelines.document \
  --config configs/document.example.yaml \
  run --stages index,mineru-manifest,mineru-run,page-objects
```

After OCR/preprocessing, run QA on one MinerU-processed document:

```bash
python -m app \
  --config configs/docpilot.example.yaml \
  --mineru-dir runs/environment/mineru/DATASET/DOC_ID \
  --question "Your document question" \
  --show-state
```

Run batch QA from a question file after preprocessing the referenced documents:

```bash
python -m app \
  --config configs/docpilot.example.yaml \
  --question-file data/questions.jsonl \
  --max-samples 10 \
  --num-workers 1
```

Useful outputs are written under the configured `output` roots, usually:

- `runs/logs/`
- `runs/outputs/`
- `runs/.workspace/`

## Training Data Synthesis

Training-data code lives in `pipelines/training`.

Single-ability conversion scripts:

```text
pipelines/training/single-ability-data/convert_navigate.py
pipelines/training/single-ability-data/convert_extract.py
pipelines/training/single-ability-data/convert_solve.py
pipelines/training/single-ability-data/convert_single_ability.py
pipelines/training/single-ability-data/convert_to_action_only.py
```

Multi-ability / agent-trajectory conversion:

```text
pipelines/training/convert_docpilot_logs_to_agentic_training.py
pipelines/training/merge_docpilot_chat.py
pipelines/training/replace_docpilot_paths.py
```

Example: convert DocPilot interaction logs into agentic training records:

```bash
python pipelines/training/convert_docpilot_logs_to_agentic_training.py \
  --logs-dir runs/logs \
  --output pipelines/training/docpilot_agentic_training.json \
  --data-dir pipelines/training/data
```

Example: rewrite dataset image paths after moving data:

```bash
python pipelines/training/replace_docpilot_paths.py \
  pipelines/training/docpilot_agentic_training.json \
  --old-root old/root \
  --new-root new/root \
  --output pipelines/training/docpilot_agentic_training.rebased.json
```

The snapshot intentionally excludes generated `.jsonl` files and page/block images. Provide local dataset roots explicitly when running these converters.

## Validation

Useful artifact checks:

```bash
python -m py_compile $(find app pipelines -name '*.py' | sort)
python -m app --help
python -m pipelines.document --help
```

## Sanitization Notes

- API URLs, API keys, judge credentials, account tokens, and account user IDs are empty.
- User-specific absolute paths were replaced with relative paths such as `data/DocVQA/...`, `external/training/...`, `models/...`, or `./...`.
- Generated runtime artifacts are not included: logs, outputs, OCR results, JSONL datasets, page images, progress files, caches, and checkpoints.
- Remote-sync scripts and local visualization server helpers were omitted from the sanitized snapshot.
