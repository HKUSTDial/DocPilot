from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2


DEFAULT_LOGS_DIR = Path("./runs/logs")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import token_length
from app.tool.navigate import _resize_page_screenshot_to_thumbnail

PAGE_NODE_CACHE: dict[str, list[dict[str, Any]]] = {}
PAGE_THUMBNAIL_SIZE = 512
WORKSPACE_DIR = REPO_ROOT / "runs/.workspace"

TAG_BY_ACTION = {
    "navigate": "Navigate",
    "extract": "Extract",
    "focus": "Focus",
    "solver": "Solve",
    "execute": "Execute",
    "answer": "Answer",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DocPilot history.jsonl logs to XML-tagged agentic training data."
    )
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--history-file", type=Path, default=None, help="Convert one specific *.history.jsonl file.")
    parser.add_argument("--eval-file", type=Path, default=None, help="Convert only check=true rows from eval.json.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help='Root directory for materialized images. Defaults to <output.parent>/data; each sample uses <data-dir>/<sample_id>.',
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    if args.eval_file is not None:
        history_paths = history_paths_from_eval(args.eval_file)
    else:
        history_paths = [args.history_file] if args.history_file is not None else sorted(args.logs_dir.rglob("*.history.jsonl"))
    data_dir = args.data_dir if args.data_dir is not None else args.output.parent / "data"

    samples = []
    for history_path in history_paths:
        if args.limit is not None and len(samples) >= args.limit:
            break
        if history_path is None or not history_path.exists():
            continue
        sample = convert_history(history_path, len(samples), data_dir)
        if sample is not None:
            samples.append(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        if args.output.suffix.lower() == ".jsonl":
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            json.dump(samples, f, ensure_ascii=False, indent=2 if args.pretty else None)
            f.write("\n")
    print(f"Wrote {len(samples)} samples to {args.output}")


def history_paths_from_eval(eval_path: Path) -> list[Path]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected {eval_path} to be a JSON list or contain a results list")
    return [
        Path(str(row["trace_path"]))
        for row in rows
        if isinstance(row, dict) and str(row.get("check", "")).lower() == "true" and row.get("trace_path")
    ]


def convert_history(history_path: Path, sample_id: int, data_dir: Path) -> dict[str, Any] | None:
    state = load_json(history_path.with_suffix("").with_suffix(".state.json"))
    state_history = state.get("history") if isinstance(state.get("history"), list) else []
    steps = [step for step in state_history if isinstance(step, dict)] or load_jsonl(history_path)
    if not steps:
        return None
    reasoning_trace = parse_log_reasoning(resolve_log_path(history_path))
    metadata = build_metadata(history_path, state)
    question = metadata.get("question") or infer_question(steps) or ""
    assistant_content = render_steps(steps, state, reasoning_trace=reasoning_trace)
    images = collect_sample_images(state, steps)
    image_names, image_labels = materialize_sample_images(images, sample_id, data_dir)
    messages = [
        {
            "role": "user",
            "content": render_user_content(question, metadata),
        },
        {
            "role": "assistant",
            "content": strip_evidence_ids(relabel_content_image_paths(assistant_content, image_labels)),
        },
    ]
    return {
        "id": sample_id,
        "document_id": metadata.get("document_id"),
        "dataset_id": metadata.get("dataset_id"),
        "source": str(history_path),
        "image": image_names,
        "total_tokens": sum(token_length(message["content"]) for message in messages),
        "messages": messages,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def build_metadata(history_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    run_request = state.get("run_request") if isinstance(state.get("run_request"), dict) else {}
    candidate_answer = state.get("candidate_answer") if isinstance(state.get("candidate_answer"), dict) else {}
    verification = state.get("verification") if isinstance(state.get("verification"), dict) else {}
    document_id = (
        run_request.get("doc_name")
        or run_request.get("doc_no")
        or run_request.get("doc_id")
        or history_path.parent.name
    )
    dataset_id = (
        run_request.get("dataset")
        or run_request.get("dataset_id")
        or (history_path.parent.parent.name if len(history_path.parts) >= 2 else None)
    )
    if isinstance(document_id, str) and document_id.endswith(".pdf"):
        document_id = document_id[:-4]
    page_range = run_request.get("page_range") or infer_metadata_page_range(state, document_id)
    return {
        "question": state.get("question") or run_request.get("question"),
        "doc_name": document_id,
        "document_id": str(document_id) if document_id is not None else None,
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "page_range": page_range,
        "q_id": run_request.get("q_id"),
        "dataset": dataset_id,
        "model": history_path.parent.parent.parent.name if len(history_path.parts) >= 3 else None,
        "final_answer": candidate_answer.get("answer"),
        "verification": verification,
    }


def infer_question(steps: list[dict[str, Any]]) -> str | None:
    for step in steps:
        action = step.get("action")
        if not isinstance(action, dict):
            continue
        args = action.get("args")
        if isinstance(args, dict) and isinstance(args.get("question"), str):
            return args["question"]
    return None


def infer_metadata_page_range(state: dict[str, Any], document_id: Any) -> list[int] | None:
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    overview = metadata.get("environment_overview") if isinstance(metadata.get("environment_overview"), dict) else {}
    documents = overview.get("documents") if isinstance(overview.get("documents"), list) else []
    normalized_doc_id = str(document_id or "").removesuffix(".pdf")
    fallback_pages: list[int] | None = None
    for document in documents:
        if not isinstance(document, dict):
            continue
        pages = [int(page) for page in document.get("pages", []) if isinstance(page, int) or str(page).isdigit()]
        if not pages:
            continue
        candidate = [min(pages), max(pages)]
        if fallback_pages is None:
            fallback_pages = candidate
        doc_id = str(document.get("doc_id") or "").removesuffix(".pdf")
        if doc_id == normalized_doc_id:
            return candidate
    return fallback_pages


def format_user_page_range(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            start = int(value[0])
            end = int(value[1])
        except (TypeError, ValueError):
            return ""
        return f"{start}-{end}" if start != end else str(start)
    if isinstance(value, (list, tuple)):
        return format_page_range(list(value))
    return ""


def render_user_content(question: str, metadata: dict[str, Any]) -> str:
    doc_name = metadata.get("doc_name") or ""
    lines = [
        "Instruction",
        question,
        f"doc_name: {doc_name}",
    ]
    page_range = format_user_page_range(metadata.get("page_range"))
    if page_range:
        lines.append(f"page_range: {page_range}")
    return "\n".join(lines)


def render_steps(
    steps: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    reasoning_trace: dict[int, dict[str, str]] | None = None,
) -> str:
    rendered: list[str] = []
    pending_verification = ""
    previous_step: dict[str, Any] | None = None
    previous_action_type = ""

    for index, step in enumerate(steps):
        step_no = index + 1
        action_type = step_action_type(step)
        if action_type == "finish":
            previous_step = step
            previous_action_type = action_type
            continue
        if action_type == "analyze" and is_self_verification_step(step):
            if not has_non_finish_followup(steps, index):
                previous_step = step
                previous_action_type = action_type
                continue
            pending_verification = render_analyze(
                step,
                state,
                previous_step=previous_step,
                react_reasoning=(reasoning_trace or {}).get(step_no, {}).get("react", ""),
            )
            previous_step = step
            previous_action_type = action_type
            continue

        rendered.append(
            render_step(
                step,
                state,
                step_no=step_no,
                previous_action_type=previous_action_type,
                extra_analyze_context=pending_verification,
                reasoning_trace=reasoning_trace or {},
            )
        )
        pending_verification = ""
        previous_step = step
        previous_action_type = action_type

    return "\n".join(part for part in rendered if part)


def step_action_type(step: dict[str, Any]) -> str:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
    return str(action.get("action_type") or observation.get("action_type") or "analyze").lower()


def is_self_verification_step(step: dict[str, Any]) -> bool:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
    return (args.get("task") or payload.get("task")) == "self_verification"


def has_non_finish_followup(steps: list[dict[str, Any]], index: int) -> bool:
    for step in steps[index + 1 :]:
        action_type = step_action_type(step)
        if action_type == "finish":
            continue
        return True
    return False


def render_step(
    step: dict[str, Any],
    state: dict[str, Any],
    *,
    step_no: int,
    previous_action_type: str = "",
    extra_analyze_context: str = "",
    reasoning_trace: dict[int, dict[str, str]] | None = None,
) -> str:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
    action_type = step_action_type(step)
    step_reasoning = (reasoning_trace or {}).get(step_no, {})

    analyze_block = render_analyze(step, state, react_reasoning=step_reasoning.get("react", ""))
    if extra_analyze_context:
        analyze_block = " ".join(part for part in (extra_analyze_context, analyze_block) if part).strip()
    if action_type in {"analyze", "finish"}:
        return tag("Analyze", analyze_block, escape=False)

    tool_tag = TAG_BY_ACTION.get(action_type, "Analyze")
    tool_block = render_tool(action, observation, state, tool_reasoning=step_reasoning.get("tool", ""))
    if action_type == "execute" and previous_action_type == "solver" and not extra_analyze_context:
        return tag(tool_tag, tool_block, escape=False)
    return f"{tag('Analyze', analyze_block, escape=False)}\n{tag(tool_tag, tool_block, escape=False)}"


def render_analyze(
    step: dict[str, Any],
    state: dict[str, Any],
    *,
    previous_step: dict[str, Any] | None = None,
    react_reasoning: str = "",
) -> str:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
    action_type = str(action.get("action_type") or observation.get("action_type") or "unknown")
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    reasoning = clean(react_reasoning)
    lines = [xml_text(reasoning)] if reasoning else []
    if action_type == "analyze":
        payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
        task = args.get("task") or payload.get("task")
        if task == "self_verification":
            verification_input = format_verification_input(state, previous_step)
            if verification_input:
                lines.append(verification_input)
        message = clean(observation.get("message"))
        if task == "self_verification" and message:
            lines.append(message)
        verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
        if verification:
            lines.append(format_verification(verification))
    elif action_type == "finish":
        payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
        lines.append(clean(payload.get("reason") or observation.get("message")))
        candidate = payload.get("candidate_answer") if isinstance(payload.get("candidate_answer"), dict) else {}
        if candidate.get("answer"):
            lines.append(f"The final answer is {clean(candidate.get('answer'))}.")
    elif not lines:
        lines.append(next_step_hint(action_type, state))
    return " ".join(line for line in lines if line).strip()


def render_tool(
    action: dict[str, Any],
    observation: dict[str, Any],
    state: dict[str, Any],
    *,
    tool_reasoning: str = "",
) -> str:
    action_type = str(action.get("action_type") or observation.get("action_type") or "unknown")
    if action_type == "navigate":
        return render_navigate_tool(action, observation, state, tool_reasoning=tool_reasoning)
    if action_type == "extract":
        return render_extract_tool(action, observation, state, tool_reasoning=tool_reasoning)
    if action_type == "focus":
        return render_focus_tool(action, observation, state, tool_reasoning=tool_reasoning)
    if action_type == "answer":
        return render_answer_tool(action, observation, state, tool_reasoning=tool_reasoning)
    return render_generic_tool(action, observation, tool_reasoning=tool_reasoning)


def render_navigate_tool(
    action: dict[str, Any],
    observation: dict[str, Any],
    state: dict[str, Any],
    *,
    tool_reasoning: str = "",
) -> str:
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
    query = clean(args.get("query") or payload.get("query"))
    lines = [xml_text(tool_reasoning), xml_text(f"The navigate query: {query}")]
    page_set = payload.get("page_set")
    if isinstance(page_set, list) and page_set:
        lines.append(xml_text(f"It checked doc_pages {format_page_range(page_set)}."))
    image_paths = page_thumbnail_paths(state, page_set if isinstance(page_set, list) else [])
    image_paths.extend(path for path in extract_image_paths(payload) if path not in image_paths)
    if image_paths:
        lines.append(xml_text(f"The input page thumbnails were {format_page_image_refs(state, page_set, image_paths)}."))
    candidates = payload.get("candidate_pages")
    if isinstance(candidates, list) and candidates:
        lines.append(xml_text("It returned " + format_candidate_pages(candidates) + "."))
    reason = clean(payload.get("reason"))
    if reason:
        lines.append(xml_text(reason))
    return " ".join(lines)


def render_extract_tool(
    action: dict[str, Any],
    observation: dict[str, Any],
    state: dict[str, Any],
    *,
    tool_reasoning: str = "",
) -> str:
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
    instruction = clean(args.get("instruction") or payload.get("query"))
    lines = [xml_text(tool_reasoning), xml_text(f"The extract query: {instruction}")]
    pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
    if not pages:
        pages = [item.get("page") for item in args.get("candidate_pages", []) if isinstance(item, dict) and item.get("page")]
    if pages:
        lines.append(xml_text(f"It read pages {format_page_range(pages)}."))
        page_images = page_thumbnail_paths(state, pages)
        if page_images:
            lines.append(xml_text(f"The input page thumbnails were {format_page_image_refs(state, pages, page_images)}."))
    candidates = args.get("candidate_pages")
    if isinstance(candidates, list) and candidates:
        lines.append(xml_text("The page reasons were " + format_candidate_pages(candidates) + "."))
    extracted = format_extracted_evidence(state, payload.get("evidence_ids"))
    if extracted:
        lines.append(xml_text("The extracted evidence was:\n" + extracted))
    else:
        page_nodes = format_page_nodes(state, pages)
        if page_nodes:
            lines.append(xml_text("The page node contents were:\n" + page_nodes))
    decisions = payload.get("candidate_selection_decisions")
    if isinstance(decisions, list) and decisions:
        lines.append(xml_text("The extraction decision was " + format_selection_decisions(decisions) + "."))
    figure_refs = payload.get("figure_refs")
    if isinstance(figure_refs, list) and figure_refs:
        lines.append(xml_text("It extracted " + format_figure_refs(figure_refs) + "."))
    return "\n".join(line for line in lines if line)


def format_extracted_evidence(state: dict[str, Any], evidence_ids: Any) -> str:
    if not isinstance(evidence_ids, list) or not evidence_ids:
        return ""
    evidence_store = state.get("evidence_store") if isinstance(state.get("evidence_store"), dict) else {}
    parts = []
    for evidence_id in evidence_ids:
        evidence = evidence_store.get(str(evidence_id))
        if not isinstance(evidence, dict):
            continue
        rendered = format_extracted_evidence_record(evidence)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts)


def format_extracted_evidence_record(evidence: dict[str, Any]) -> str:
    page = evidence.get("page")
    node_id = str(evidence.get("node_id") or "")
    block = block_label_from_node_id(node_id)
    location = f"Page {page}" if page is not None else "Unknown page"
    if block:
        location = f"{location}, {block}"
    modality = str(evidence.get("modality") or "").lower()
    content = evidence.get("content")
    if modality == "text":
        text = clean(content)
        return f"{location} text:\n{text}" if text else ""
    if modality == "table":
        table = table_content_to_markdown(content)
        return f"{location} table:\n{table}" if table else ""
    return ""


def block_label_from_node_id(node_id: str) -> str:
    match = re.search(r"::block::(\d+)", node_id)
    return f"Block {match.group(1)}" if match else ""


def render_focus_tool(
    action: dict[str, Any],
    observation: dict[str, Any],
    state: dict[str, Any],
    *,
    tool_reasoning: str = "",
) -> str:
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
    instruction = clean(args.get("instruction"))
    lines = [xml_text(tool_reasoning), xml_text(f"The focus query: {instruction}")]
    evidence_ids = payload.get("evidence_ids") if isinstance(payload.get("evidence_ids"), list) else args.get("evidence_ids", [])
    if evidence_ids:
        pages = evidence_pages(state, evidence_ids)
        if pages:
            lines.append(xml_text(f"It focused on {format_page_labels(pages)}."))
    evidence_context = format_evidence_context(state, evidence_ids)
    if evidence_context:
        lines.append(xml_text(evidence_context))
    operations = payload.get("operations")
    if isinstance(operations, list) and operations:
        lines.append(xml_text("The processed focus operation returned " + format_focus_operations(operations) + "."))
    planner = payload.get("planner") if isinstance(payload.get("planner"), dict) else {}
    if planner.get("reason"):
        lines.append(xml_text(clean(planner.get("reason"))))
    return " ".join(lines)


def render_answer_tool(
    action: dict[str, Any],
    observation: dict[str, Any],
    state: dict[str, Any],
    *,
    tool_reasoning: str = "",
) -> str:
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
    reasoning = clean(tool_reasoning)
    if reasoning:
        return xml_text(reasoning)
    instruction = clean(args.get("instruction"))
    candidate = payload.get("candidate_answer") if isinstance(payload.get("candidate_answer"), dict) else {}
    answer = clean(candidate.get("answer") or (payload.get("answer_summary") or {}).get("answer_preview"))
    rationale = clean(candidate.get("rationale"))
    lines = [xml_text(tool_reasoning), xml_text(f"The answer instruction: {instruction}")]
    if answer:
        lines.append(xml_text(f"It produced the answer {answer}."))
    if rationale:
        lines.append(xml_text(rationale))
    evidence_ids = candidate.get("supporting_evidence_ids") or payload.get("supporting_evidence_ids")
    if isinstance(evidence_ids, list) and evidence_ids:
        pages = evidence_pages(state, evidence_ids)
        if pages:
            lines.append(xml_text(f"The answer used evidence from pages {format_page_range(pages)}."))
        else:
            lines.append(xml_text("The answer used supporting evidence."))
    return " ".join(lines)


def render_generic_tool(
    action: dict[str, Any],
    observation: dict[str, Any],
    *,
    tool_reasoning: str = "",
) -> str:
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
    text = clean(args.get("instruction") or args.get("query") or observation.get("message"))
    details = compact_observation_payload(payload)
    return " ".join(part for part in (xml_text(tool_reasoning), xml_text(text), xml_text(details)) if part)


def resolve_log_path(history_path: Path) -> Path:
    direct = history_path.with_suffix("").with_suffix(".log")
    if direct.exists():
        return direct
    stem = history_path.name.removesuffix(".history.jsonl")
    _, _, suffix = stem.partition("-")
    if not suffix:
        return direct
    candidates = sorted(history_path.parent.glob(f"*-{suffix}.log"))
    return candidates[-1] if candidates else direct


def parse_log_reasoning(log_path: Path) -> dict[int, dict[str, str]]:
    if not log_path.exists():
        return {}
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    result: dict[int, dict[str, str]] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        react_match = re.search(r"STEP (\d+) LLM Response:", line)
        if react_match:
            step_no = int(react_match.group(1))
            block_lines = [line]
            i += 1
            while i < len(lines) and not re.search(r"STEP \d+ (?:Executing action|Action execution result|ReAct input|LLM Response):", lines[i]):
                block_lines.append(lines[i])
                i += 1
            text = "\n".join(block_lines)
            reasoning = extract_reasoning_block(text)
            if reasoning:
                result.setdefault(step_no, {})["react"] = reasoning
            continue
        tool_match = re.search(r"STEP (\d+) Action execution result:", line)
        if tool_match:
            step_no = int(tool_match.group(1))
            i += 1
            block_lines: list[str] = []
            capture = False
            while i < len(lines) and not re.search(r"STEP \d+ (?:ReAct input|LLM Response|Executing action|Action execution result):", lines[i]):
                current = lines[i]
                if current.strip() == "Reasoning:":
                    capture = True
                    i += 1
                    continue
                if capture:
                    if current.strip().startswith("Result:"):
                        break
                    block_lines.append(current)
                i += 1
            reasoning = clean("\n".join(line.strip() for line in block_lines))
            reasoning = extract_reasoning_block(reasoning)
            if reasoning:
                result.setdefault(step_no, {})["tool"] = reasoning
            continue
        i += 1
    return result


def extract_reasoning_block(text: str) -> str:
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return clean(match.group(1))
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return clean(match.group(1))
    return clean(text) if text.strip() else ""


def collect_sample_images(state: dict[str, Any], steps: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for step in steps:
        observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
        action_type = str(observation.get("action_type") or "").lower()
        if action_type == "navigate":
            paths.extend(page_thumbnail_paths(state, payload.get("page_set", [])))
        elif action_type == "extract":
            pages = payload.get("pages", [])
            paths.extend(page_thumbnail_paths(state, pages))
            paths.extend(page_node_image_paths(state, pages))
        elif action_type == "focus":
            paths.extend(evidence_image_paths(state, payload.get("evidence_ids", [])))
        paths.extend(extract_image_paths(payload))
    return unique_existing_paths(paths)


def evidence_image_paths(state: dict[str, Any], evidence_ids: Any) -> list[str]:
    evidence_store = state.get("evidence_store") if isinstance(state.get("evidence_store"), dict) else {}
    paths = []
    for evidence_id in evidence_ids if isinstance(evidence_ids, list) else []:
        evidence = evidence_store.get(str(evidence_id))
        if not isinstance(evidence, dict):
            continue
        content = evidence.get("content") if isinstance(evidence.get("content"), dict) else {}
        paths.extend(extract_image_paths(content))
    return unique_existing_paths(paths)


def page_node_image_paths(state: dict[str, Any], pages: Any) -> list[str]:
    page_numbers = {int(page) for page in pages if isinstance(page, int) or str(page).isdigit()} if isinstance(pages, list) else set()
    if not page_numbers:
        return []
    paths = []
    for item in load_mineru_page_nodes(state):
        if not isinstance(item, dict) or not isinstance(item.get("page_idx"), int):
            continue
        if item["page_idx"] + 1 in page_numbers:
            paths.extend(extract_image_paths(item))
    return unique_existing_paths(paths)


def evidence_pages(state: dict[str, Any], evidence_ids: Any) -> list[int]:
    evidence_store = state.get("evidence_store") if isinstance(state.get("evidence_store"), dict) else {}
    pages = []
    for evidence_id in evidence_ids if isinstance(evidence_ids, list) else []:
        evidence = evidence_store.get(str(evidence_id))
        if not isinstance(evidence, dict):
            continue
        page = evidence.get("page")
        if isinstance(page, int) and page not in pages:
            pages.append(page)
    return pages


def format_page_labels(pages: list[int]) -> str:
    return ", ".join(f"Page {page}" for page in pages)


def relabel_content_image_paths(content: str, image_labels: dict[str, str]) -> str:
    for image_path, label in sorted(image_labels.items(), key=lambda item: len(item[0]), reverse=True):
        content = content.replace(image_path, label)
    content = re.sub(r"/\S+\.(?:png|jpg|jpeg)", "Block image", content, flags=re.IGNORECASE)
    content = re.sub(r"\b(?:image_path|img_path|page_image_path|screenshot_path)\b", "image", content)
    content = re.sub(r"\b(?:crop_path|zoom_path)\b", "block", content)
    return content


def strip_evidence_ids(content: str) -> str:
    return re.sub(r"\S+::page::(\d+)::\S+", r"Page \1 evidence", content)


def materialize_sample_images(images: list[str], sample_id: int, data_dir: Path) -> tuple[list[str], dict[str, str]]:
    data_dir = data_dir / str(sample_id)
    data_dir.mkdir(parents=True, exist_ok=True)
    for old_image in data_dir.iterdir():
        if old_image.is_file():
            old_image.unlink()
    names = []
    labels = {}
    block_index = 0
    for index, image_path in enumerate(images):
        src = Path(image_path)
        page_no = page_no_from_image_path(image_path)
        if page_no is not None:
            name = f"page_{page_no}.png"
            labels[image_path] = f"Page {page_no}"
        else:
            block_index += 1
            name = f"block_{block_index}.png"
            labels[image_path] = f"block {block_index}"
        write_png_image(src, data_dir / name)
        names.append(name)
    return names, labels


def write_png_image(src: Path, dst: Path) -> None:
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
        return
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"Failed to read image: {src}")
    if not cv2.imwrite(str(dst), image):
        raise ValueError(f"Failed to write image: {dst}")


def format_page_image_refs(state: dict[str, Any], pages: Any, image_paths: list[str]) -> str:
    labels = []
    page_values = pages if isinstance(pages, list) else []
    for page, image_path in zip(page_values, image_paths):
        labels.append(f"Page {page}")
    if labels:
        return ", ".join(labels)
    return ", ".join(image_paths)


def page_no_from_image_path(image_path: str) -> int | None:
    path = Path(image_path)
    if "screenshots" in path.parts:
        match = re.search(r"page(\d+)", path.name)
        return int(match.group(1)) if match else None
    if "navigate_thumbnails" in path.parts:
        match = re.match(r".+_(\d+)\.[^.]+\.thumb\.(?:png|jpg|jpeg)$", path.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)) + 1
    return None


def page_thumbnail_paths(state: dict[str, Any], pages: Any) -> list[str]:
    paths = []
    for page in pages if isinstance(pages, list) else []:
        screenshot = page_screenshot_path(state, page)
        if not screenshot:
            continue
        thumbnail = _resize_page_screenshot_to_thumbnail(
            screenshot,
            size=PAGE_THUMBNAIL_SIZE,
            page=page,
        )
        paths.append(thumbnail or screenshot)
    return unique_existing_paths(paths)


def page_screenshot_path(state: dict[str, Any], page: Any) -> str | None:
    try:
        page_no = int(page)
    except (TypeError, ValueError):
        return None
    root = resolve_mineru_root(state)
    if root is None:
        return None
    candidates = [
        root / "screenshots" / f"page{page_no}.png",
        root / root.name / "screenshots" / f"page{page_no}.png",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    matches = sorted(root.rglob(f"screenshots/page{page_no}.png"))
    return str(matches[0]) if matches else None


def resolve_mineru_root(state: dict[str, Any]) -> Path | None:
    run_request = state.get("run_request") if isinstance(state.get("run_request"), dict) else {}
    mineru_dir = run_request.get("mineru_dir")
    if not isinstance(mineru_dir, str) or not mineru_dir:
        return None
    candidates = [normalized_mineru_dir(mineru_dir)]
    doc_name = run_request.get("doc_name")
    if isinstance(doc_name, str) and doc_name:
        base = REPO_ROOT / "runs/environment/mineru"
        candidates.extend(sorted(base.rglob(doc_name)) if base.exists() else [])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalized_mineru_dir(mineru_dir: str) -> Path:
    path = Path(mineru_dir)
    text = str(path)
    for old in ("runs/outputs/document/mineru", "/runs/outputs/document/mineru/"):
        if old in text:
            return Path(text.replace(old, old.replace("outputs/document", "environment")))
    return path


def unique_existing_paths(paths: list[Any]) -> list[str]:
    result = []
    seen = set()
    for path in paths:
        resolved = resolve_existing_image_path(path)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def resolve_existing_image_path(path: Any) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    for candidate in workspace_image_candidates(path):
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return str(candidate)
    return None


def workspace_image_candidates(path: str) -> list[Path]:
    candidates = [Path(path)]
    marker = "docpilot/"
    if marker in path:
        candidates.insert(0, WORKSPACE_DIR / path.split(marker, 1)[1])
    if not Path(path).is_absolute():
        candidates.insert(0, WORKSPACE_DIR / path)
    return candidates


def next_step_hint(action_type: str, state: dict[str, Any]) -> str:
    question = clean(state.get("question"))
    if action_type == "navigate":
        return f"To answer the question, first locate the relevant pages in the document. The question is: {question}"
    if action_type == "extract":
        return "Navigation has identified candidate pages, so the next step is to extract the relevant page evidence."
    if action_type == "focus":
        return "The extracted evidence contains a visual target, so the next step is to inspect it more precisely."
    if action_type == "answer":
        return "The needed evidence has been collected, so the next step is to produce a grounded answer."
    return "Continue the task using the available evidence."


def format_verification(verification: dict[str, Any]) -> str:
    if verification.get("accepted"):
        return "Self-verification accepted the candidate answer."
    issues = verification.get("issues")
    if isinstance(issues, list) and issues:
        return "Self-verification found issues: " + "; ".join(clean(item) for item in issues) + "."
    return "Self-verification did not accept the candidate answer."


def format_verification_input(state: dict[str, Any], previous_step: dict[str, Any] | None) -> str:
    candidate = candidate_answer_from_step(previous_step)
    if not candidate:
        candidate = state.get("candidate_answer") if isinstance(state.get("candidate_answer"), dict) else {}
    if not isinstance(candidate, dict) or not candidate:
        return ""

    answer = clean(candidate.get("answer"))
    rationale = clean(candidate.get("rationale"))
    evidence_ids = candidate.get("supporting_evidence_ids")
    lines = []
    if answer:
        lines.append(f"The verifier checked candidate answer {answer}.")
    if rationale:
        lines.append(f"The candidate rationale was: {rationale}")
    evidence_context = format_text_evidence_context(state, evidence_ids)
    if evidence_context:
        lines.append(evidence_context)
    return " ".join(lines)


def candidate_answer_from_step(step: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
    payload = observation.get("payload") if isinstance(observation.get("payload"), dict) else {}
    candidate = payload.get("candidate_answer")
    return candidate if isinstance(candidate, dict) else {}


def format_text_evidence_context(state: dict[str, Any], evidence_ids: Any) -> str:
    if not isinstance(evidence_ids, list) or not evidence_ids:
        return ""
    evidence_store = state.get("evidence_store") if isinstance(state.get("evidence_store"), dict) else {}
    parts = []
    for evidence_id in evidence_ids:
        evidence = evidence_store.get(str(evidence_id))
        if not isinstance(evidence, dict):
            continue
        page = evidence.get("page")
        modality = evidence.get("modality")
        content = format_supporting_evidence_content(evidence)
        label = f"page {page}" if page is not None else "unknown page"
        if modality:
            label = f"{label} {modality}"
        if content:
            parts.append(f"{label}: {content}")
    if not parts:
        return "The verifier used evidence " + ", ".join(map(str, evidence_ids)) + "."
    return "The verifier used supporting evidence: " + " | ".join(parts) + "."


def format_supporting_evidence_content(evidence: dict[str, Any]) -> str:
    content = label_image_fields(evidence.get("content"), page=evidence.get("page"))
    modality = str(evidence.get("modality") or "").lower()
    if modality in {"image", "figure", "chart"}:
        return format_image_evidence_content(content)
    return clean(content)


def format_image_evidence_content(content: Any) -> str:
    if not isinstance(content, dict):
        return clean(content)
    parts = []
    title = clean(content.get("title") or content.get("caption"))
    summary = clean(content.get("summary"))
    if title:
        parts.append(f"title {title}")
    if summary:
        parts.append(f"summary {summary}")
    if content.get("image"):
        parts.append(f"image {content['image']}")
    if content.get("block"):
        parts.append(f"block {content['block']}")
    if content.get("panel_query"):
        parts.append(f"panel query {clean(content['panel_query'])}")
    matched_panel = content.get("matched_panel") if isinstance(content.get("matched_panel"), dict) else {}
    if matched_panel:
        panel_bits = []
        for key in ("label", "position", "block"):
            if matched_panel.get(key):
                panel_bits.append(f"{key} {matched_panel[key]}")
        if panel_bits:
            parts.append("matched panel " + ", ".join(panel_bits))
    panels = content.get("panels") or content.get("available_panels")
    if isinstance(panels, list) and panels:
        panel_labels = []
        for panel in panels:
            if not isinstance(panel, dict):
                continue
            label = panel.get("label") or panel.get("position") or panel.get("block")
            block = panel.get("block")
            if label and block and label != block:
                panel_labels.append(f"{label} ({block})")
            elif label:
                panel_labels.append(str(label))
        if panel_labels:
            parts.append("available panels " + ", ".join(panel_labels))
    if content.get("text"):
        parts.append(f"text {clean(content['text'])}")
    return "; ".join(parts) or clean(content)


def table_content_to_markdown(content: Any) -> str:
    if isinstance(content, dict):
        caption = clean(content.get("caption"))
        footnote = clean(content.get("footnote"))
        table_text = (
            html_table_to_markdown(content.get("html"))
            or markdown_table_from_rows(content.get("rows"))
            or clean(content.get("markdown"))
            or clean(content.get("raw"))
            or clean(content.get("table_body"))
        )
        parts = []
        if caption:
            parts.append(caption)
        if table_text:
            parts.append(table_text)
        if footnote:
            parts.append(f"Footnote: {footnote}")
        return "\n".join(parts)
    return clean(content)


def html_table_to_markdown(value: Any) -> str:
    if not isinstance(value, str) or "<tr" not in value.lower():
        return ""
    rows = []
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", value, flags=re.IGNORECASE | re.DOTALL):
        cells = []
        for cell_html in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL):
            text = re.sub(r"<[^>]+>", " ", cell_html)
            text = re.sub(r"\s+", " ", html.unescape(text)).strip()
            cells.append(text)
        if cells:
            rows.append(cells)
    return markdown_table_from_rows(rows)


def markdown_table_from_rows(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return ""
    rows = []
    for row in value:
        if isinstance(row, dict):
            if not rows:
                rows.append([clean(key) for key in row.keys()])
            rows.append([clean(item) for item in row.values()])
        elif isinstance(row, list):
            rows.append([clean(cell) for cell in row])
    rows = [row for row in rows if row]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:]
    return "\n".join(
        [
            "| " + " | ".join(escape_markdown_cell(cell) for cell in header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
            *["| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |" for row in body],
        ]
    )


def escape_markdown_cell(value: Any) -> str:
    return clean(value).replace("|", "\\|")


def label_image_fields(value: Any, *, page: Any = None) -> Any:
    if isinstance(value, dict):
        labeled = {}
        for key, item in value.items():
            if key in {"image_path", "img_path", "page_image_path", "screenshot_path"}:
                labeled["image"] = f"Page {page}" if page is not None else "Page image"
            elif key in {"crop_path", "zoom_path"}:
                labeled["block"] = block_label_from_path(item)
            else:
                labeled[key] = label_image_fields(item, page=page)
        return labeled
    if isinstance(value, list):
        return [label_image_fields(item, page=page) for item in value]
    return value


def block_label_from_path(value: Any) -> str:
    if not isinstance(value, str):
        return "Block image"
    match = re.search(r"block[_-]?(\d+)|(?:crop|zoom|panel|split)[_-]?(\d+)", Path(value).stem, flags=re.IGNORECASE)
    number = next((group for group in (match.groups() if match else []) if group), None)
    return f"Block {number}" if number else "Block image"


def format_page_range(pages: list[Any]) -> str:
    values = sorted({int(page) for page in pages if isinstance(page, int) or str(page).isdigit()})
    if not values:
        return ""
    ranges = []
    start = prev = values[0]
    for page in values[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = page
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def extract_image_paths(value: Any) -> list[str]:
    paths: list[str] = []
    seen = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"image_path", "img_path", "page_image_path", "screenshot_path", "thumbnail_path", "crop_path", "zoom_path"}:
                    if isinstance(child, str) and child and child not in seen:
                        seen.add(child)
                        paths.append(child)
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return paths[:12]


def format_candidate_pages(candidates: list[Any]) -> str:
    parts = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        page = item.get("page")
        reason = clean(item.get("reason"))
        confidence = item.get("confidence")
        confidence_text = f", confidence {confidence}" if confidence is not None else ""
        if page is not None and reason:
            parts.append(f"page {page}{confidence_text}, because {reason}")
        elif page is not None:
            parts.append(f"page {page}{confidence_text}")
    return "; ".join(parts)


def format_selection_decisions(decisions: list[Any]) -> str:
    parts = []
    for item in decisions:
        if not isinstance(item, dict):
            continue
        page = item.get("page")
        selected = item.get("selected_candidate_ids")
        reason = clean(item.get("reason"))
        selected_text = ", ".join(format_candidate_node_label(candidate) for candidate in selected) if isinstance(selected, list) and selected else "no node"
        parts.append(f"page {page} selected {selected_text} because {reason}")
    return "; ".join(parts)


def format_candidate_node_label(candidate_id: Any) -> str:
    text = str(candidate_id)
    match = re.fullmatch(r"page(\d+)::([^:]+)::(\d+)", text)
    if not match:
        return text
    page, kind, block = match.groups()
    return f"Page {page} Block {block} {kind}"


def format_figure_refs(refs: list[Any]) -> str:
    parts = []
    for item in refs:
        if not isinstance(item, dict):
            continue
        pieces = []
        if item.get("figure_ref"):
            pieces.append(str(item["figure_ref"]))
        if item.get("page"):
            pieces.append(f"page {item['page']}")
        if item.get("image_path"):
            pieces.append(f"using Page {item['page']}" if item.get("page") else "using Page image")
        caption = clean(item.get("caption") or item.get("caption_preview"))
        if caption:
            pieces.append(f"caption {caption}")
        if pieces:
            parts.append(", ".join(pieces))
    return "; ".join(parts)


def format_focus_operations(operations: list[Any]) -> str:
    parts = []
    for item in operations:
        if not isinstance(item, dict):
            continue
        pieces = []
        if item.get("tool"):
            pieces.append(str(item["tool"]))
        if item.get("page"):
            pieces.append(f"on page {item['page']}")
        if item.get("target_description"):
            pieces.append(clean(item["target_description"]))
        region = item.get("region") if isinstance(item.get("region"), dict) else {}
        if region.get("bbox"):
            pieces.append(f"bbox {region['bbox']}")
        if item.get("visual_caption"):
            pieces.append(f"caption {clean(item['visual_caption'])}")
        if item.get("success") is not None:
            pieces.append("successfully" if item.get("success") else "unsuccessfully")
        if pieces:
            parts.append(", ".join(pieces))
    return "; ".join(parts)


def format_evidence_context(state: dict[str, Any], evidence_ids: list[Any]) -> str:
    evidence_store = state.get("evidence_store") if isinstance(state.get("evidence_store"), dict) else {}
    parts = []
    for evidence_id in evidence_ids:
        evidence = evidence_store.get(str(evidence_id))
        if not isinstance(evidence, dict):
            continue
        content = evidence.get("content") if isinstance(evidence.get("content"), dict) else {}
        page = evidence.get("page")
        context = [f"Page {page} visual evidence" if page is not None else "visual evidence"]
        if content.get("image_path"):
            context.append(f"source Page {page}" if page is not None else "source Page image")
        split_paths = [
            panel.get("crop_path")
            for panel in content.get("panels", [])
            if isinstance(panel, dict) and panel.get("crop_path")
        ]
        if split_paths:
            context.append("split into " + ", ".join(block_label_from_path(path) for path in split_paths))
        matched_panel = content.get("matched_panel") if isinstance(content.get("matched_panel"), dict) else {}
        if matched_panel.get("crop_path"):
            context.append(f"processed {block_label_from_path(matched_panel['crop_path'])}")
        if content.get("zoom_path"):
            context.append(f"processed {block_label_from_path(content['zoom_path'])}")
        caption = clean(content.get("title") or content.get("summary") or content.get("text"))
        if caption:
            context.append(f"caption {caption}")
        parts.append(", ".join(context))
    return "The visual evidence was " + "; ".join(parts) + "." if parts else ""


def format_page_nodes(state: dict[str, Any], pages: list[Any]) -> str:
    page_numbers = sorted({int(page) for page in pages if isinstance(page, int) or str(page).isdigit()})
    if not page_numbers:
        return ""
    nodes = load_mineru_page_nodes(state)
    if not nodes:
        return ""
    by_page: dict[int, list[str]] = {page: [] for page in page_numbers}
    for item in nodes:
        if not isinstance(item, dict):
            continue
        page_idx = item.get("page_idx")
        if not isinstance(page_idx, int):
            continue
        page = page_idx + 1
        if page not in by_page:
            continue
        text = node_text(item)
        if text:
            by_page[page].append(text)
    parts = []
    for page in page_numbers:
        page_parts = [f"Node {index + 1}: {text}" for index, text in enumerate(by_page.get(page, []))]
        if page_parts:
            parts.append(f"Page {page}:\n" + "\n\n".join(page_parts))
    return "\n\n".join(parts)


def load_mineru_page_nodes(state: dict[str, Any]) -> list[dict[str, Any]]:
    root = resolve_mineru_root(state)
    if root is None:
        return []
    cache_key = str(root)
    if cache_key in PAGE_NODE_CACHE:
        return PAGE_NODE_CACHE[cache_key]
    paths = sorted(path for path in root.rglob("*content_list.json") if not path.name.endswith("_v2.json"))
    if not paths:
        paths = sorted(root.rglob("*content_list_v2.json"))
    if not paths:
        PAGE_NODE_CACHE[cache_key] = []
        return []
    try:
        data = json.loads(paths[0].read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        data = []
    nodes = data if isinstance(data, list) else []
    auto_dir = paths[0].parent
    for item in nodes:
        if isinstance(item, dict) and isinstance(item.get("img_path"), str) and not Path(item["img_path"]).is_absolute():
            item["img_path"] = str(auto_dir / item["img_path"])
    PAGE_NODE_CACHE[cache_key] = nodes
    return nodes


def node_text(item: dict[str, Any]) -> str:
    kind = item.get("type") or "node"
    if item.get("text"):
        return f"{kind} {clean(item['text'])}"
    if item.get("table_body"):
        table = table_content_to_markdown({"html": item["table_body"], "raw": item["table_body"]})
        return f"{kind}\n{table}"
    captions = item.get("image_caption")
    if isinstance(captions, list) and captions:
        text = " ".join(clean(caption) for caption in captions)
        img_path = item.get("img_path")
        return f"{kind} {text}; visual {img_path}" if img_path else f"{kind} {text}"
    if item.get("img_path"):
        return f"{kind} visual {item['img_path']}"
    if item.get("page_number"):
        return f"{kind} {item['page_number']}"
    return ""


def compact_observation_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    payload = summarize_payload(payload)
    keys = [
        "accepted",
        "answer_summary",
        "candidate_answer",
        "candidate_pages",
        "evidence",
        "evidence_ids",
        "figure_refs",
        "issues",
        "missing_evidence",
        "operations",
        "pages",
        "result",
        "supporting_evidence_count",
        "tool_name",
        "route",
    ]
    selected = {key: payload[key] for key in keys if key in payload}
    if not selected and payload:
        selected = {key: payload[key] for key in list(payload)[:5]}
    return json.dumps(compact(selected), ensure_ascii=False)


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summarized = dict(payload)
    evidence = summarized.get("evidence")
    if isinstance(evidence, dict):
        summarized["evidence"] = summarize_evidence(evidence)
    candidate_answer = summarized.get("candidate_answer")
    if isinstance(candidate_answer, dict):
        summarized["candidate_answer"] = compact(candidate_answer, max_string=500, max_items=6)
    operations = summarized.get("operations")
    if isinstance(operations, list):
        summarized["operations"] = [summarize_operation(op) for op in operations if isinstance(op, dict)]
    figure_refs = summarized.get("figure_refs")
    if isinstance(figure_refs, list):
        summarized["figure_refs"] = [
            {key: ref.get(key) for key in ("figure_ref", "page", "node_id", "evidence_id", "caption_preview") if key in ref}
            for ref in figure_refs
            if isinstance(ref, dict)
        ]
    return summarized


def summarize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    content = evidence.get("content")
    return {
        "evidence_id": evidence.get("evidence_id"),
        "node_id": evidence.get("node_id"),
        "page": evidence.get("page"),
        "modality": evidence.get("modality"),
        "content_preview": clean(content),
    }


def summarize_operation(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: operation.get(key)
        for key in ("tool", "page", "node_id", "success", "evidence_id", "target_description", "visual_caption")
        if key in operation
    }


def compact(value: Any, *, depth: int = 0, max_string: int = 600, max_items: int = 8) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [compact(item, depth=depth + 1, max_string=max_string, max_items=max_items) for item in value]
    if isinstance(value, dict):
        return {
            str(key): compact(item, depth=depth + 1, max_string=max_string, max_items=max_items)
            for key, item in value.items()
        }
    return repr(value)


def tag(name: str, content: str, *, escape: bool = True) -> str:
    body = html.escape(content, quote=False) if escape else content
    return f"<{name}>\n{body}\n</{name}>"


def xml_text(value: Any) -> str:
    text = clean(value)
    return html.escape(text, quote=False) if text else ""




def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()



# python pipelines/training/convert_docpilot_logs_to_agentic_training.py \
#     --history-file ./runs/logs/qwen3.5-397b-a17b/longdocurl/4017086/20260809T165240Z-summary2title_4017086_61_67_4.history.jsonl \
#     --eval-file ./pipelines/training/eval.json \
#     --output ./pipelines/training/docpilot_single_history_training_longdocurl.json \
#     --data-dir ./pipelines/training/data \
#     --pretty
if __name__ == "__main__":
    main()
