#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_pages(value):
    if value is None:
        return []
    pages = []
    for part in re.split(r"[;,]", str(value)):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            prefix, page_part = part.rsplit(":", 1)
            prefix = prefix.strip()
        else:
            prefix, page_part = "", part
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", page_part.strip())
        if m:
            a, b = map(int, m.groups())
            lo, hi = sorted((a, b))
            pages.extend(f"{prefix}:{i}" if prefix else str(i) for i in range(lo, hi + 1))
        elif page_part.strip():
            pages.append(f"{prefix}:{page_part.strip()}" if prefix else page_part.strip())
    return pages


def extract_pages(args):
    pages = set(parse_pages(args.get("pages")))
    for item in args.get("candidate_pages") or []:
        if isinstance(item, dict) and item.get("page") is not None:
            doc = item.get("doc_id")
            page = str(item["page"])
            pages.add(f"{doc}:{page}" if doc else page)
    return sorted(pages)


def log_path_from_history(path):
    p = Path(path)
    if p.name.endswith(".history.jsonl"):
        return p.with_name(p.name.replace(".history.jsonl", ".log"))
    return p


def result_block(log_path, step):
    start = f"STEP {step} Action execution result:"
    next_step = re.compile(r"^\d{4}-\d{2}-\d{2} .* - INFO - STEP \d+ ")
    lines = []
    collecting = False
    with Path(log_path).open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not collecting:
                if start in line:
                    collecting = True
                continue
            if next_step.match(line):
                break
            lines.append(line)
    return "".join(lines)


def load_multipage_extracts(history_path):
    out = []
    with Path(history_path).open(encoding="utf-8", errors="replace") as f:
        for step, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = row.get("action") or {}
            if action.get("action_type") != "extract":
                continue
            args = action.get("args") or {}
            pages = extract_pages(args)
            if len(pages) > 1:
                out.append(
                    {
                        "step": step,
                        "pages": pages,
                        "page_count": len(pages),
                        "success": (row.get("observation") or {}).get("success"),
                        "intent": action.get("intent"),
                        "instruction": args.get("instruction"),
                    }
                )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="./pipelines/training/filter/2/eval_full.json",
    )
    ap.add_argument(
        "--output",
        default="./pipelines/training/filter/2/eval_full_true_multipage_extract_multi_reasoning.json",
    )
    args = ap.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))["results"]
    results = []
    true_total = 0
    missing_history = 0
    missing_log = 0
    multipage_extract_samples = 0
    step_cache = {}
    block_cache = {}

    for idx, row in enumerate(rows):
        if str(row.get("check")).lower() != "true":
            continue
        true_total += 1
        trace = row.get("trace_path")
        if not trace or not Path(trace).exists():
            missing_history += 1
            continue
        extract_steps = step_cache.get(trace)
        if extract_steps is None:
            extract_steps = load_multipage_extracts(trace)
            step_cache[trace] = extract_steps
        if not extract_steps:
            continue
        multipage_extract_samples += 1

        log_path = log_path_from_history(trace)
        if not log_path.exists():
            missing_log += 1
            continue
        matched = []
        for step in extract_steps:
            cache_key = (str(log_path), step["step"])
            block = block_cache.get(cache_key)
            if block is None:
                block = result_block(log_path, step["step"])
                block_cache[cache_key] = block
            reasoning_tags = len(re.findall(r"<reasoning\b", block))
            reasoning_labels = len(re.findall(r"(?m)^\s*Reasoning:\s*$", block))
            if max(reasoning_tags, reasoning_labels) > 1:
                step.update(
                    {
                        "reasoning_tag_count": reasoning_tags,
                        "reasoning_label_count": reasoning_labels,
                        "result_excerpt": block[:3000],
                    }
                )
                matched.append(step)
        if matched:
            results.append(
                {
                    "index": idx,
                    "dataset": row.get("dataset"),
                    "doc": row.get("doc"),
                    "question": row.get("question"),
                    "pred": row.get("pred"),
                    "gt": row.get("gt"),
                    "trace_path": trace,
                    "log_path": str(log_path),
                    "matched_extract_steps": matched,
                }
            )

    out = {
        "source": args.input,
        "definition": "check=true rows with an extract step whose input covers multiple pages and whose log result contains multiple reasoning blocks",
        "true_total": true_total,
        "missing_history": missing_history,
        "multipage_extract_samples": multipage_extract_samples,
        "missing_log": missing_log,
        "count": len(results),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"true_total: {true_total}")
    print(f"missing_history: {missing_history}")
    print(f"multipage_extract_samples: {multipage_extract_samples}")
    print(f"missing_log: {missing_log}")
    print(f"multi_reasoning_samples: {len(results)}")
    print(f"output: {args.output}")


# python pipelines/training/filter/check_true_multipage_extract_multi_reasoning.py \
#     --input ./pipelines/training/filter/eval_full.json \
#     --output ./pipelines/training/filter/eval_full_true_multipage_extract_multi_reasoning.json
if __name__ == "__main__":
    main()
