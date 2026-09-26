from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "single_ability.openai.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert single-ability data to DocPilot OpenAI chat JSONL.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", choices=["auto", "navigate", "extract", "answer", "solve"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-correct", action="store_true", help="Keep only rows where predicted_answer/pred roughly matches gold.")
    parser.add_argument("--workspace-root", type=Path, default=Path("data/DataScience-Instruct-500K/RL/data/files"))
    parser.add_argument("--max-rows", type=int, default=50, help="Rows per file preview for solve/file data.")
    args = parser.parse_args()

    converted: list[dict[str, Any]] = []
    skipped = 0
    for path in args.input:
        if args.limit is not None and len(converted) >= args.limit:
            break
        task = infer_task(path, args.task)
        for row in iter_rows(path):
            if args.limit is not None and len(converted) >= args.limit:
                break
            sample = convert_row(row, task, workspace_root=args.workspace_root, max_rows=args.max_rows)
            if sample is None or (args.filter_correct and not roughly_correct(row)):
                skipped += 1
                continue
            converted.append(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for sample in converted:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(converted)} samples to {args.output}; skipped={skipped}")


def infer_task(path: Path, task: str) -> str:
    if task != "auto":
        return task
    name = str(path).lower()
    if "navigate" in name:
        return "navigate"
    if "extract" in name or "hotpot" in name:
        return "extract"
    if "file_" in name or "solve" in name:
        return "solve"
    return "answer"


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for child in sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json")):
            yield from iter_rows(child)
        return
    text = path.read_text(encoding="utf-8", errors="replace").lstrip()
    if not text:
        return
    if text.startswith("["):
        yield from json.loads(text)
        return
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict) and isinstance(obj.get("data"), list):
                yield from obj["data"]
            else:
                yield obj
            return
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def convert_row(row: dict[str, Any], task: str, *, workspace_root: Path, max_rows: int) -> dict[str, Any] | None:
    if task == "navigate":
        return convert_navigate(row)
    if task == "extract":
        return convert_extract(row)
    if task == "solve":
        return convert_solve(row, workspace_root=workspace_root, max_rows=max_rows)
    if task == "answer":
        return convert_answer(row)
    raise ValueError(f"Unsupported task: {task}")


def convert_navigate(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = non_system_messages(row)
    if len(messages) < 2:
        return None
    user = str(messages[0].get("content", "")).strip()
    assistant = ensure_analyze_plus_tool(str(messages[-1].get("content", "")), "Navigate", "candidate pages")
    return compact_sample(user, assistant, row.get("images") or row.get("image") or [])


def convert_extract(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = non_system_messages(row)
    if len(messages) >= 2:
        user = strip_task_prompt(str(messages[0].get("content", "")))
        assistant_raw = str(messages[-1].get("content", ""))
    else:
        user = strip_task_prompt(str(row.get("input") or row.get("question") or row.get("prompt") or ""))
        assistant_raw = str(row.get("raw_response") or row.get("response") or "")
    analyze = tag(assistant_raw, "think") or tag(assistant_raw, "Analyze") or "Select the candidate nodes that contain the evidence needed to answer the query."
    answer = boxed(tag(assistant_raw, "answer") or assistant_raw) or boxed_list(row.get("pred") or row.get("ground_truth"))
    if not user or answer is None:
        return None
    assistant = f"<Analyze>\n{clean(analyze)}\n</Analyze>\n<Extract>\ncandidate nodes: \\boxed{{{answer}}}\n</Extract>"
    return compact_sample(user, assistant, row.get("images") or [])


def convert_answer(row: dict[str, Any]) -> dict[str, Any] | None:
    question = first_text(row.get("question") or row.get("query") or row.get("prompt"))
    if not question:
        return None
    evidence = evidence_text(row)
    images = image_list(row)
    user = answer_user(question, evidence, images)
    raw = str(row.get("raw_response") or row.get("response") or "")
    analyze = tag(raw, "think") or tag(raw, "Analyze") or "Use the provided evidence to answer the question."
    answer = boxed(str(row.get("predicted_answer") or "")) or boxed(tag(raw, "answer")) or first_text(row.get("answer") or row.get("answers"))
    if not answer:
        return None
    assistant = f"<Analyze>\n{clean(analyze)}\n</Analyze>\n<Answer>\nanswer: \\boxed{{{clean(answer)}}}\n</Answer>"
    return compact_sample(user, assistant, images)


def convert_solve(row: dict[str, Any], *, workspace_root: Path, max_rows: int) -> dict[str, Any] | None:
    try:
        import convert_solve as solve_converter
    except ImportError:
        return None
    sample = solve_converter.convert_row(row, workspace_root, max_rows=max_rows, strict=False)
    if sample is None:
        return None
    user = sample["messages"][0]["content"]
    solve = str(sample["messages"][1]["content"])
    body = tag(solve, "Solve") or solve
    if "\n\ncode:" in body:
        analyze, code = body.split("\n\ncode:", 1)
    else:
        analyze, code = "Use the extracted evidence to compute the answer.", body
    assistant = f"<Analyze>\n{clean(analyze)}\n</Analyze>\n<Solve>\ncode:{code.strip()}\n</Solve>"
    return compact_sample(user, assistant, [])


def compact_sample(user: str, assistant: str, images: Any) -> dict[str, Any]:
    sample = {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }
    image_values = [str(x) for x in (images if isinstance(images, list) else [images]) if x]
    if image_values:
        sample["images"] = image_values
    return sample


def non_system_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in row.get("messages", []) if m.get("role") != "system"]


def strip_task_prompt(text: str) -> str:
    for marker in ("Search query:", "Question:"):
        idx = text.rfind(marker)
        if idx >= 0:
            return clean(text[idx:])
    return clean(text)


def ensure_analyze_plus_tool(text: str, tool: str, result_label: str) -> str:
    if tag(text, "Analyze") and tag(text, tool):
        return clean(text)
    analyze = tag(text, "think") or tag(text, "Analyze") or f"Use the input to perform the {tool.lower()} task."
    result = boxed(text)
    tool_text = tag(text, tool) or (f"{result_label}: \\boxed{{{result}}}" if result else text)
    return f"<Analyze>\n{clean(analyze)}\n</Analyze>\n<{tool}>\n{clean(tool_text)}\n</{tool}>"


def tag(text: str, name: str) -> str:
    matches = re.findall(rf"<{name}>\s*(.*?)\s*</{name}>", str(text), flags=re.S | re.I)
    return matches[-1].strip() if matches else ""


def boxed(text: str) -> str:
    text = str(text or "")
    start = text.find(r"\boxed{")
    if start < 0:
        return text.strip() if text and not re.search(r"</?(think|answer|Analyze|Extract|Answer|Navigate|Solve)>", text, re.I) else ""
    i = start + len(r"\boxed{")
    depth = 1
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return ""


def boxed_list(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "[" + ", ".join(map(str, value)) + "]"
    return str(value)


def first_text(value: Any) -> str:
    if isinstance(value, list):
        return "" if not value else str(value[0])
    return "" if value is None else str(value)


def evidence_text(row: dict[str, Any]) -> str:
    for key in ("gold_evidence", "paragraph", "context", "evidence", "text"):
        value = row.get(key)
        if value:
            return clean_text_value(value)
    return ""


def clean_text_value(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if x)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def image_list(row: dict[str, Any]) -> list[str]:
    value = row.get("images") or row.get("image") or row.get("image_path") or row.get("img_path")
    if value is None:
        return []
    return [str(x) for x in value] if isinstance(value, list) else [str(value)]


def answer_user(question: str, evidence: str, images: list[str]) -> str:
    parts = ["Question:", question.strip()]
    parts += ["", "Extracted evidence:", clean(evidence) if evidence else "(No text evidence provided.)"]
    if images:
        parts += ["", *["<image>" for _ in images]]
    return "\n".join(parts)


def roughly_correct(row: dict[str, Any]) -> bool:
    pred = row.get("predicted_answer", row.get("pred"))
    gold = row.get("answer", row.get("answers", row.get("ground_truth")))
    if pred is None or gold is None:
        return True
    preds = pred if isinstance(pred, list) else [pred]
    golds = gold if isinstance(gold, list) else [gold]
    pred_norms = {normalize_answer(x) for x in preds}
    gold_norms = {normalize_answer(x) for x in golds}
    return bool(pred_norms & gold_norms) or any(p and g and (p in g or g in p) for p in pred_norms for g in gold_norms)


def normalize_answer(value: Any) -> str:
    text = boxed(str(value)).lower()
    text = text.replace(r"\%", "%")
    text = re.sub(r"[$,]", "", text)
    text = re.sub(r"[^a-z0-9.%/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text).strip())



# python pipelines/training/single-ability-data/convert_single_ability.py \
#     --task answer \
#     --input pipelines/training/single-ability-data/data/InforGraphicsVQA/answer_infographicsvqa_simple_true.jsonl \
#     --output pipelines/training/single-ability-data/data/answer_infographicsvqa_simple_true.chat.jsonl \
#     --filter-correct

# python pipelines/training/single-ability-data/convert_single_ability.py \
#     --task extract \
#     --input pipelines/training/single-ability-data/data/hotpotqa/extract_hotpotqa_distilled_true.jsonl \
#     --output pipelines/training/single-ability-data/data/extract_hotpotqa_distilled.chat.jsonl \
#     --filter-correct

# python pipelines/training/single-ability-data/convert_single_ability.py \
#     --task navigate \
#     --input pipelines/training/single-ability-data/data/Navigate/navigate_dude.jsonl \
#     --output pipelines/training/single-ability-data/data/navigate_dude.chat.jsonl

# python pipelines/training/single-ability-data/convert_single_ability.py \
#     --task solve \
#     --input data/DataScience-Instruct-500K/reasoning/file_csv_3007.json \
#     --output pipelines/training/single-ability-data/data/solve_file_csv.chat.jsonl
if __name__ == "__main__":
    main()
