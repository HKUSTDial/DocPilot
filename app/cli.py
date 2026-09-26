from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any

from tqdm import tqdm

from .account_quota import (
    account_endpoint_from_base_url,
    build_pipeline_cost,
    query_account_quota,
)
from .agent import AnalyzeAgent, ReActAgent, VLLMTagsAgent
from .config import AppConfig, load_config
from .environment import InMemoryDocumentEnvironment
from .llm import LLMClient
from .question_loader import (
    _row_doc_names,
    _row_page_range,
    load_all_questions,
    load_question_entry_by_doc_and_text,
    load_question_by_doc_and_text,
    load_question_by_id,
    load_questions_for_doc,
)
from .run_logger import RunLogger
from .schema import BudgetState, CandidateAnswer, DocumentNode
from .tool import build_default_tool_collection
from .workflow import MineruDocumentEnvironment
from .workflow.mineru_loader import load_nodes_from_mineru_output


def _demo_nodes() -> list[DocumentNode]:
    return [
        DocumentNode(
            node_id="doc1-sec1",
            doc_id="doc1",
            page=1,
            node_type="section",
            modality="text",
            title="FY2015 Revenue Overview",
            text="The company reported revenue of 120 million dollars in FY2015.",
        ),
        DocumentNode(
            node_id="doc1-table1",
            doc_id="doc1",
            page=2,
            node_type="table",
            modality="table",
            title="Consolidated Statements",
            content={"Revenue 2015": 120, "Revenue 2014": 100},
            text="Revenue 2015 120; Revenue 2014 100.",
        ),
    ]


def _load_nodes(path: str | None) -> list[DocumentNode]:
    if path is None:
        return _demo_nodes()
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DocumentNode(**item) for item in items]


def _build_agent(config: AppConfig):
    if not config.llm.enabled:
        return AnalyzeAgent(llm_client=None)
    llm_client = LLMClient(config.llm)
    tools = build_default_tool_collection(
        llm_client=llm_client,
        navigate_config=config.navigate,
        extract_config=config.extract,
        workspace_dir=config.output.workspace_dir,
    )
    if config.llm.protocol.strip().lower() == "vllm_tags":
        return VLLMTagsAgent(
            llm_client=llm_client,
            tools=tools,
        )
    return ReActAgent(
        llm_client=llm_client,
        tools=tools,
    )


def _resolved_candidate_answer(state):
    candidate = state.candidate_answer
    if candidate is not None and str(candidate.answer or "").strip():
        return candidate
    payload = state.metadata.get("latest_candidate_answer")
    if not isinstance(payload, dict):
        return None
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        return None

    supporting = payload.get("supporting_evidence_ids") or []
    return CandidateAnswer(
        answer=str(payload.get("answer") or ""),
        rationale=payload.get("rationale"),
        supporting_evidence_ids=list(supporting) if isinstance(supporting, list) else [],
        confidence=float(payload.get("confidence") or 0.0),
    )


def _format_result(state, *, show_state: bool, ground_truth_answer: str | None = None) -> dict[str, Any]:
    candidate = _resolved_candidate_answer(state)
    if show_state:
        payload = state.to_dict()
        payload["predicted_answer"] = candidate.answer if candidate else None
        payload["ground_truth_answer"] = ground_truth_answer
        payload["steps_used"] = state.budget.step_used
        return payload
    return {
        "question": state.question,
        "predicted_answer": candidate.answer if candidate else None,
        "ground_truth_answer": ground_truth_answer,
        "rationale": candidate.rationale if candidate else None,
        "confidence": candidate.confidence if candidate else None,
        "verification": state.verification.to_dict() if state.verification else None,
        "steps_used": state.budget.step_used,
        "evidence_count": len(state.evidence_store),
        "history_len": len(state.history),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="backslashreplace").decode("utf-8")
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            _json_safe(key) if isinstance(key, str) else key: _json_safe(item)
            for key, item in value.items()
        }
    return value


def _json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, **kwargs)


def _account_endpoint_from_base_url(base_url: str) -> str:
    return account_endpoint_from_base_url(base_url) or ""


def _query_account_quota(config: AppConfig) -> dict[str, Any]:
    return query_account_quota(config)


def _build_pipeline_cost(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return build_pipeline_cost(before, after)


def _slugify(text: str, limit: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()
    return text[:limit] or "run"


def _normalize_doc_key(doc_name: str | None) -> str:
    if not doc_name:
        return ""
    raw = Path(str(doc_name).strip())
    suffix = raw.suffix.lower()
    if suffix in {".pdf", ".json", ".jsonl", ".md", ".txt", ".html", ".htm"}:
        return raw.stem.lower()
    return raw.name.lower()


def _run_namespace(config: AppConfig, request: dict[str, Any]) -> tuple[str, str, str]:
    model_name = _slugify(config.llm.model or "no-llm", limit=80)
    dataset = "adhoc"
    if request.get("question_file"):
        dataset = _infer_dataset_name_from_question_file(request["question_file"])
    elif request.get("mineru_dir"):
        mineru_parts = Path(request["mineru_dir"]).parts
        if len(mineru_parts) >= 2:
            dataset = mineru_parts[-2]

    doc_id = None
    if request.get("mineru_dir"):
        doc_id = Path(request["mineru_dir"]).name
    if not doc_id:
        doc_id = _normalize_doc_key(request.get("doc_name"))
    if not doc_id and request.get("nodes_json"):
        doc_id = Path(request["nodes_json"]).stem
    if not doc_id:
        doc_id = request.get("question") or "docpilot"
    return model_name, _slugify(str(dataset), limit=80), _slugify(str(doc_id), limit=120)


def _artifact_dataset_dir(root: str | Path, config: AppConfig, model_name: str, dataset_name: str) -> Path:
    path = Path(root) / model_name
    canonical = {"solve": "solver"}
    disabled = {canonical.get(name, name) for name in config.budget.disabled_tools}
    if not disabled:
        return path / dataset_name
    if disabled == {"extract", "solver", "execute", "focus", "analyze"}:
        return path / f"{dataset_name}_w_navigate+answer"
    aliases = {"solver": "solve"}
    order = ("navigate", "extract", "solver", "execute", "focus", "analyze", "answer")
    names = [aliases.get(name, name) for name in order if name in disabled]
    names.extend(sorted(aliases.get(name, name) for name in disabled.difference(order)))
    return path / f"{dataset_name}_wo_{'_'.join(names)}"


def _infer_dataset_name_from_question_file(question_file: str | Path) -> str:
    question_file_path = Path(question_file)
    question_file_text = str(question_file_path).lower()
    if "mmdocrag" in question_file_text:
        return "MMDocRAG"
    if "longdocurl" in question_file_text:
        return "LongDocURL"
    if "mmlongbench" in question_file_text:
        return "MMLongBench-Doc"
    if "madqa" in question_file_text:
        return "MADQA"
    if "dude" in question_file_text:
        return "DUDE"
    if "finragbench" in question_file_text:
        return "FinRAGBench-V"
    stem = question_file_path.stem
    return question_file_path.parent.name if stem in {"samples", "sample", "questions"} else stem


def _row_doc_name(row: dict[str, Any]) -> str | None:
    names = _row_doc_names(row)
    return names[0] if names else None


def _row_document_titles(row: dict[str, Any], doc_names: list[str]) -> dict[str, str] | None:
    raw_titles = row.get("doc_titles")
    if not isinstance(raw_titles, dict):
        return None
    normalized = {_normalize_doc_key(str(key)): str(value) for key, value in raw_titles.items() if value}
    titles = {
        doc_name: normalized.get(_normalize_doc_key(doc_name), doc_name)
        for doc_name in doc_names
    }
    return titles or None


def _row_question_id(row: dict[str, Any]) -> str | None:
    value = row.get("question_id")
    if value is None:
        value = row.get("questionId")
    if value is None:
        value = row.get("id")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _row_ground_truth(row: dict[str, Any]) -> str | None:
    answers = row.get("answers")
    if answers:
        return answers[0] if isinstance(answers, list) else answers
    if row.get("answer_type") == "not-answerable":
        return "Not answerable"
    return row.get("answer_short") or row.get("answer_interleaved") or row.get("answer") or row.get("answer_variants")


def _row_oracle_candidate_pages(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    if row.get("oracle_page_only") is not True:
        return None
    raw = row.get("oracle_candidate_pages")
    if not isinstance(raw, list):
        raise ValueError("oracle_page_only=true requires oracle_candidate_pages to be a non-empty list")
    candidates: list[dict[str, Any]] = []
    for value in raw:
        item = dict(value) if isinstance(value, dict) else {"page": value}
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid oracle candidate page: {value!r}") from exc
        if page <= 0:
            raise ValueError(f"Invalid oracle candidate page: {value!r}")
        item["page"] = page
        if not any(existing["page"] == page for existing in candidates):
            candidates.append(item)
    if not candidates:
        raise ValueError("oracle_page_only=true requires at least one oracle candidate page")
    return candidates


def _infer_doc_name_from_mineru_dir(mineru_dir: str | Path | None) -> str | None:
    if not mineru_dir:
        return None
    return Path(str(mineru_dir)).name or None


def _resolve_auto_mineru_dirs(config: AppConfig, question_file: str | Path, row: dict[str, Any]) -> list[Path]:
    dataset_name = _infer_dataset_name_from_question_file(question_file)
    doc_names = _row_doc_names(row)
    if not doc_names:
        raise FileNotFoundError("Question row is missing doc_name/doc_id/doc_no; cannot resolve MinerU directory.")
    dataset_dir = Path(config.output.document_dir) / "mineru" / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"MinerU dataset directory does not exist: {dataset_dir}")
    children = {
        _normalize_doc_key(child.name): child
        for child in dataset_dir.iterdir()
        if child.is_dir()
    }
    resolved: list[Path] = []
    for doc_name in doc_names:
        direct = dataset_dir / Path(str(doc_name)).stem
        match = direct if direct.exists() else children.get(_normalize_doc_key(doc_name))
        if match is None:
            raise FileNotFoundError(f"Could not resolve MinerU directory for doc '{doc_name}' under {dataset_dir}")
        resolved.append(match)
    return resolved


def _resolve_auto_mineru_dir(config: AppConfig, question_file: str | Path, row: dict[str, Any]) -> Path:
    return _resolve_auto_mineru_dirs(config, question_file, row)[0]


def _resolve_mineru_content_dir(path: Path) -> Path:
    if list(path.glob("*_content_list_v2.json")):
        return path
    auto = path / path.name / "auto"
    if auto.exists() and list(auto.glob("*_content_list_v2.json")):
        return auto
    candidates = sorted(path.glob("*/*_content_list_v2.json"))
    if candidates:
        return candidates[0].parent
    candidates = sorted(path.glob("*/auto/*_content_list_v2.json"))
    if candidates:
        return candidates[0].parent
    return path


def _request_run_key(request: dict[str, Any]) -> str:
    question_id = request.get("question_id")
    if question_id is not None:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(question_id).strip()).strip("-") or "question"
    return str(request.get("q_id")) if request.get("q_id") is not None else _slugify(request.get("question") or "docpilot")


def _summarize_history_item(item: dict[str, Any]) -> dict[str, Any]:
    action = item.get("action", {}) if isinstance(item, dict) else {}
    observation = item.get("observation", {}) if isinstance(item, dict) else {}
    payload = observation.get("payload", {}) if isinstance(observation, dict) else {}
    payload_summary: dict[str, Any] = {}
    if isinstance(payload, dict):
        if "supporting_evidence_count" in payload:
            payload_summary["supporting_evidence_count"] = payload.get("supporting_evidence_count")
        verification = payload.get("verification")
        if isinstance(verification, dict) and "accepted" in verification:
            payload_summary["accepted"] = verification.get("accepted")
        elif "accepted" in payload:
            payload_summary["accepted"] = payload.get("accepted")
    return {
        "action": action,
        "observation": {
            "action_type": observation.get("action_type"),
            "success": observation.get("success"),
            "message": observation.get("message"),
            "payload": payload_summary,
        },
    }


def _persist_run_artifacts(
    *,
    state,
    brief_result: dict[str, Any],
    config: AppConfig,
    request: dict[str, Any],
    run_id: str,
    pipeline_cost: dict[str, Any] | None = None,
    runtime_seconds: float | None = None,
) -> tuple[Path, Path, Path]:
    model_name, dataset_name, doc_id = _run_namespace(config, request)

    logs_dir = _artifact_dataset_dir(config.output.logs_dir, config, model_name, dataset_name) / doc_id
    outputs_dir = _artifact_dataset_dir(config.output.outputs_dir, config, model_name, dataset_name) / doc_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    state_path = logs_dir / f"{run_id}.state.json"
    text_log_path = logs_dir / f"{run_id}.log"
    trace_path = logs_dir / f"{run_id}.history.jsonl"
    output_path = outputs_dir / f"{run_id}.result.json"

    state_payload = state.to_dict()
    state_payload["run_request"] = request
    if runtime_seconds is not None:
        state_payload["runtime_seconds"] = runtime_seconds
    if pipeline_cost is not None:
        state_payload["pipeline_cost"] = pipeline_cost
    state_payload["artifacts"] = {
        "log_path": str(text_log_path.resolve()),
        "text_log_path": str(text_log_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "result_path": str(output_path.resolve()),
    }
    state_path.write_text(_json_dumps(state_payload, indent=2), encoding="utf-8")

    with trace_path.open("w", encoding="utf-8") as fh:
        for item in state.history:
            fh.write(_json_dumps(_summarize_history_item(item)) + "\n")

    output_payload = {
        **brief_result,
        "run_request": request,
        "log_path": str(text_log_path.resolve()),
        "state_path": str(state_path.resolve()),
        "trace_path": str(trace_path.resolve()),
    }
    if runtime_seconds is not None:
        output_payload["runtime_seconds"] = runtime_seconds
    if pipeline_cost is not None:
        output_payload["pipeline_cost"] = pipeline_cost
    output_path.write_text(_json_dumps(output_payload, indent=2), encoding="utf-8")
    return state_path.resolve(), output_path.resolve(), text_log_path.resolve()


def _find_existing_output(config: AppConfig, request: dict[str, Any]) -> Path | None:
    model_name, dataset_name, doc_id = _run_namespace(config, request)
    outputs_dir = _artifact_dataset_dir(config.output.outputs_dir, config, model_name, dataset_name) / doc_id
    if not outputs_dir.exists():
        return None
    run_key = _request_run_key(request)
    matches = sorted(outputs_dir.glob(f"*-{run_key}.result.json"))
    return matches[-1].resolve() if matches else None


def _load_existing_output_payload(
    existing_output: Path,
    *,
    question: str,
    doc_name: str,
    question_id: str | None = None,
    q_id: int | None,
    ground_truth_answer: str | None,
) -> dict[str, Any]:
    payload = json.loads(existing_output.read_text(encoding="utf-8"))
    merged_payload = dict(payload)
    merged_payload["question"] = merged_payload.get("question") or question
    merged_payload["doc_name"] = merged_payload.get("doc_name") or doc_name
    if question_id is not None and merged_payload.get("question_id") is None:
        merged_payload["question_id"] = question_id
    merged_payload["q_id"] = merged_payload.get("q_id", q_id)
    if ground_truth_answer is not None and merged_payload.get("ground_truth_answer") is None:
        merged_payload["ground_truth_answer"] = ground_truth_answer
    merged_payload["skipped"] = True
    merged_payload["saved_output"] = str(existing_output)
    return merged_payload


def _pipeline_cost_value(value: Any) -> float | int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, dict):
        cost = value.get("actual_cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            return cost
    return None


def _compact_batch_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    compact.pop("run_request", None)
    compact.pop("confidence", None)
    compact.pop("verification", None)
    compact["pipeline_cost"] = _pipeline_cost_value(result.get("pipeline_cost"))
    return compact


def _persist_batch_output(
    *,
    config: AppConfig,
    question_file: str | Path,
    merged_results: list[dict[str, Any]],
    runtime_seconds: float | None = None,
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_name = _slugify(config.llm.model or "no-llm", limit=80)
    dataset_name = _slugify(_infer_dataset_name_from_question_file(question_file), limit=80)
    output_dir = _artifact_dataset_dir(config.output.outputs_dir, config, model_name, dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}-{Path(question_file).stem}.json"
    compact_results = [_compact_batch_result(result) for result in merged_results]
    payload = {
        "question_file": str(Path(question_file).resolve()),
        "dataset": _infer_dataset_name_from_question_file(question_file),
        "run_count": len(compact_results),
        "results": compact_results,
    }
    if runtime_seconds is not None:
        payload["runtime_seconds"] = runtime_seconds
    output_path.write_text(_json_dumps(payload, indent=2), encoding="utf-8")
    return output_path.resolve()


async def _run(question: str, nodes_path: str | None, config: AppConfig, show_state: bool, logger: RunLogger | None, ground_truth_answer: str | None = None):
    nodes = _load_nodes(nodes_path)
    env = InMemoryDocumentEnvironment(nodes, workspace_dir=config.output.workspace_dir)
    agent = _build_agent(config)
    state = await agent.solve(
        question=question,
        env=env,
        document_ids=sorted({node.doc_id for node in nodes}),
        budget=BudgetState(**config.budget.__dict__),
        logger=logger,
    )
    return state, _format_result(state, show_state=show_state, ground_truth_answer=ground_truth_answer)


async def _run_mineru_case(
    mineru_dir: str,
    question: str,
    *,
    doc_name: str | None,
    page_range: tuple[int, int] | None,
    page_filter: set[int] | None,
    initial_candidate_pages: list[dict[str, Any]] | None,
    config: AppConfig,
    show_state: bool,
    logger: RunLogger | None,
    ground_truth_answer: str | None = None,
):
    env = MineruDocumentEnvironment.from_mineru_output_dir(
        _resolve_mineru_content_dir(Path(mineru_dir)),
        doc_id=doc_name,
        workspace_dir=config.output.workspace_dir,
        page_range=page_range,
        page_filter=page_filter,
    )
    agent = _build_agent(config)
    doc_ids = sorted({node.doc_id for node in env.nodes.values()})
    solve_kwargs = {
        "question": question,
        "env": env,
        "document_ids": doc_ids,
        "budget": BudgetState(**config.budget.__dict__),
        "logger": logger,
    }
    if initial_candidate_pages:
        solve_kwargs["initial_candidate_pages"] = initial_candidate_pages
    state = await agent.solve(
        **solve_kwargs,
    )
    return state, _format_result(state, show_state=show_state, ground_truth_answer=ground_truth_answer)


async def _run_mineru_multi_case(
    mineru_dirs: list[str],
    question: str,
    *,
    doc_names: list[str],
    document_titles: dict[str, str] | None,
    page_range: tuple[int, int] | None,
    config: AppConfig,
    show_state: bool,
    logger: RunLogger | None,
    ground_truth_answer: str | None = None,
):
    nodes: list[DocumentNode] = []
    for mineru_dir, doc_name in zip(mineru_dirs, doc_names, strict=True):
        nodes.extend(
            load_nodes_from_mineru_output(
                _resolve_mineru_content_dir(Path(mineru_dir)),
                doc_id=doc_name,
                page_range=page_range,
            )
        )
    env = InMemoryDocumentEnvironment(nodes, workspace_dir=config.output.workspace_dir)
    agent = _build_agent(config)
    state = await agent.solve(
        question=question,
        env=env,
        document_ids=doc_names,
        document_titles=document_titles,
        budget=BudgetState(**config.budget.__dict__),
        logger=logger,
    )
    return state, _format_result(state, show_state=show_state, ground_truth_answer=ground_truth_answer)


def _execute_single_request(
    *,
    config: AppConfig,
    question: str,
    show_state: bool,
    request: dict[str, Any],
    ground_truth_answer: str | None,
    page_range: tuple[int, int] | None = None,
    page_filter: set[int] | None = None,
    initial_candidate_pages: list[dict[str, Any]] | None = None,
    nodes_path: str | None = None,
    mineru_dir: str | None = None,
    doc_name: str | None = None,
    mineru_dirs: list[str] | None = None,
    doc_names: list[str] | None = None,
    document_titles: dict[str, str] | None = None,
) -> tuple[object, dict[str, Any]]:
    model_name, dataset_name, doc_id = _run_namespace(config, request)
    live_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    live_run_key = _request_run_key(request)
    run_id = f"{live_timestamp}-{live_run_key}"
    live_log_dir = _artifact_dataset_dir(config.output.logs_dir, config, model_name, dataset_name) / doc_id
    live_log_path = live_log_dir / f"{run_id}.log"
    logger = RunLogger(live_log_path)
    logger.info(f"Run started with request: {request}")
    started = time.perf_counter()
    cost_before = _query_account_quota(config)
    logger.info(f"Pipeline cost before snapshot: {cost_before}")
    try:
        if mineru_dirs and len(mineru_dirs) > 1:
            state, result = asyncio.run(
                _run_mineru_multi_case(
                    mineru_dirs,
                    question,
                    doc_names=doc_names or [Path(path).name for path in mineru_dirs],
                    document_titles=document_titles,
                    page_range=page_range,
                    config=config,
                    show_state=show_state,
                    logger=logger,
                    ground_truth_answer=ground_truth_answer,
                )
            )
        elif mineru_dir:
            state, result = asyncio.run(
                _run_mineru_case(
                    mineru_dir,
                    question,
                    doc_name=doc_name,
                    page_range=page_range,
                    page_filter=page_filter,
                    initial_candidate_pages=initial_candidate_pages,
                    config=config,
                    show_state=show_state,
                    logger=logger,
                    ground_truth_answer=ground_truth_answer,
                )
            )
        else:
            state, result = asyncio.run(
                _run(
                    question,
                    nodes_path,
                    config,
                    show_state,
                    logger,
                    ground_truth_answer=ground_truth_answer,
                )
            )
        cost_after = _query_account_quota(config)
        pipeline_cost = _build_pipeline_cost(cost_before, cost_after)
        runtime_seconds = round(time.perf_counter() - started, 3)
        logger.info(f"Pipeline cost: {pipeline_cost}")
        logger.info(f"Runtime seconds: {runtime_seconds}")
        state_path, output_path, text_log_path = _persist_run_artifacts(
            state=state,
            brief_result=_format_result(state, show_state=False, ground_truth_answer=ground_truth_answer),
            config=config,
            request=request,
            run_id=run_id,
            pipeline_cost=pipeline_cost,
            runtime_seconds=runtime_seconds,
        )
        if isinstance(result, dict):
            result["runtime_seconds"] = runtime_seconds
            result["pipeline_cost"] = pipeline_cost
            result["saved_log"] = str(text_log_path)
            result["saved_state"] = str(state_path)
            result["saved_output"] = str(output_path)
        logger.info(f"Run finished. Result saved to: {output_path}")
        return state, result
    except Exception as exc:
        logger.info(f"Run failed with fatal exception: {type(exc).__name__}: {exc}")
        raise
    finally:
        logger.close()


def _run_batch(config: AppConfig, args) -> int:
    started = time.perf_counter()
    rows = load_all_questions(args.question_file)
    run_limit = None if args.max_samples is None else max(0, int(args.max_samples))
    run_rows = rows if run_limit is None else rows[:run_limit]
    merged_results: list[dict[str, Any] | None] = [None] * len(rows)
    num_workers = max(1, int(args.num_workers or 1))

    def _batch_label(index: int, row: dict[str, Any]) -> str:
        qid = _row_question_id(row) or row.get("q_id", index)
        doc_names = _row_doc_names(row)
        return f"qid={qid} doc={','.join(doc_names)}"

    def _worker(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        question = str(row.get("question") or "").strip()
        doc_names = _row_doc_names(row)
        doc_name = doc_names[0] if doc_names else ""
        question_id = _row_question_id(row)
        q_id = row.get("q_id", index)
        ground_truth_answer = _row_ground_truth(row)
        page_range = _row_page_range(row)
        try:
            oracle_candidate_pages = _row_oracle_candidate_pages(row)
            if oracle_candidate_pages and len(doc_names) != 1:
                raise ValueError("oracle_page_only currently supports exactly one document")
            if len(doc_names) > 1:
                mineru_dirs = _resolve_auto_mineru_dirs(config, args.question_file, row)
            else:
                mineru_dirs = [_resolve_auto_mineru_dir(config, args.question_file, row)]
        except FileNotFoundError as exc:
            return index, {
                "q_id": q_id,
                "question_id": question_id,
                "doc_name": doc_name,
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "predicted_answer": None,
                "error": f"{type(exc).__name__}: {exc}",
                "failed": True,
                "skipped": True,
            }
        mineru_dir = mineru_dirs[0]
        request = {
            "question": question,
            "question_file": args.question_file,
            "mineru_dir": str(mineru_dir),
            "doc_name": doc_name,
            "question_id": question_id,
            "q_id": q_id,
            "page_range": list(page_range) if page_range is not None else None,
        }
        if oracle_candidate_pages:
            request["oracle_page_only"] = True
            request["oracle_candidate_pages"] = oracle_candidate_pages
        if len(mineru_dirs) > 1:
            request["mineru_dirs"] = [str(path) for path in mineru_dirs]
            request["doc_names"] = doc_names
        existing_output = _find_existing_output(config, request)
        if existing_output is not None:
            return index, _load_existing_output_payload(
                existing_output,
                question=question,
                doc_name=doc_name,
                question_id=question_id,
                q_id=q_id,
                ground_truth_answer=ground_truth_answer,
            )
        try:
            _, result = _execute_single_request(
                config=config,
                question=question,
                show_state=False,
                request=request,
                ground_truth_answer=ground_truth_answer,
                page_range=page_range,
                page_filter={item["page"] for item in oracle_candidate_pages} if oracle_candidate_pages else None,
                initial_candidate_pages=oracle_candidate_pages,
                mineru_dir=str(mineru_dir),
                doc_name=doc_name,
                mineru_dirs=[str(path) for path in mineru_dirs],
                doc_names=doc_names,
                document_titles=_row_document_titles(row, doc_names),
            )
        except Exception as exc:
            result = {
                "q_id": q_id,
                "question_id": question_id,
                "doc_name": doc_name,
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "predicted_answer": None,
                "error": f"{type(exc).__name__}: {exc}",
                "failed": True,
            }
        if isinstance(result, dict):
            result["q_id"] = q_id
            if question_id is not None:
                result["question_id"] = question_id
            result["doc_name"] = doc_name
            result["question"] = question
            result["ground_truth_answer"] = ground_truth_answer
        return index, result

    for i, row in enumerate(rows):
        question = str(row.get("question") or "").strip()
        doc_names = _row_doc_names(row)
        doc_name = doc_names[0] if doc_names else ""
        question_id = _row_question_id(row)
        q_id = row.get("q_id", i)
        ground_truth_answer = _row_ground_truth(row)
        request = {
            "question": question,
            "question_file": args.question_file,
            "doc_name": doc_name,
            "question_id": question_id,
            "q_id": q_id,
        }
        try:
            mineru_dirs = _resolve_auto_mineru_dirs(config, args.question_file, row)
        except FileNotFoundError:
            mineru_dirs = []
        if mineru_dirs:
            request["mineru_dir"] = str(mineru_dirs[0])
        if len(mineru_dirs) > 1:
            request["mineru_dirs"] = [str(path) for path in mineru_dirs]
            request["doc_names"] = doc_names
        existing_output = _find_existing_output(config, request)
        if existing_output is not None:
            merged_results[i] = _load_existing_output_payload(
                existing_output,
                question=question,
                doc_name=doc_name,
                question_id=question_id,
                q_id=q_id,
                ground_truth_answer=ground_truth_answer,
            )

    with tqdm(total=len(run_rows), desc="DocPilot batch", unit="sample") as pbar:
        if num_workers == 1:
            for i, row in enumerate(run_rows):
                label = _batch_label(i, row)
                if merged_results[i] is not None:
                    tqdm.write(f"cached {label}")
                    pbar.update(1)
                    continue
                tqdm.write(f"running {label} ... ", end="")
                idx, payload = _worker(i, row)
                merged_results[idx] = payload
                tqdm.write("done")
                pbar.update(1)
        else:
            pending = [(i, row) for i, row in enumerate(run_rows) if merged_results[i] is None]
            cached_count = len(run_rows) - len(pending)
            if cached_count:
                pbar.update(cached_count)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_labels = {
                    executor.submit(_worker, i, row): _batch_label(i, row)
                    for i, row in pending
                }
                for future in as_completed(future_labels):
                    label = future_labels[future]
                    idx, payload = future.result()
                    merged_results[idx] = payload
                    tqdm.write(f"running {label} ... done")
                    pbar.update(1)

    final_results = [item for item in merged_results if item is not None]
    runtime_seconds = round(time.perf_counter() - started, 3)
    merged_output = _persist_batch_output(
        config=config,
        question_file=args.question_file,
        merged_results=final_results,
        runtime_seconds=runtime_seconds,
    )
    print(_json_dumps(
        {
            "question_file": str(Path(args.question_file).resolve()),
            "run_count": len(final_results),
            "skipped_count": sum(1 for item in final_results if item.get("skipped")),
            "runtime_seconds": runtime_seconds,
            "merged_output": str(merged_output),
        },
        indent=2,
    ))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocPilot.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config.")
    parser.add_argument("--question", default=None, help="DocVQA question to solve.")
    parser.add_argument("--nodes-json", default=None, help="Optional JSON file containing serialized DocumentNode items.")
    parser.add_argument("--mineru-dir", default=None, help="Optional MinerU output directory for one document.")
    parser.add_argument("--question-file", default=None, help="Optional question file for lookup or batch mode.")
    parser.add_argument("--doc-name", default=None, help="Document name/id used to match question_file rows.")
    parser.add_argument("--q-id", default=None, help="Question id inside question_file.")
    parser.add_argument("--list-questions", action="store_true", help="List questions for the specified document.")
    parser.add_argument("--show-state", action="store_true", help="Print full state payload.")
    parser.add_argument("--max-samples", type=int, default=None, help="Batch mode: only run the first N samples.")
    parser.add_argument("--num-workers", type=int, default=1, help="Batch mode concurrency.")
    args = parser.parse_args()

    config = load_config(args.config)
    show_state = bool(args.show_state or config.output.show_state)

    inferred_doc_name = args.doc_name or _infer_doc_name_from_mineru_dir(args.mineru_dir)

    if args.list_questions:
        if not args.question_file:
            parser.error("--list-questions requires --question-file")
        if not inferred_doc_name:
            parser.error("--list-questions requires --doc-name or --mineru-dir")
        rows = load_questions_for_doc(args.question_file, inferred_doc_name)
        print(_json_dumps(rows, indent=2))
        return 0

    if args.question_file and not args.question and args.q_id is None:
        return _run_batch(config, args)

    ground_truth_answer = None
    q_id = args.q_id
    question_id = None
    doc_name = inferred_doc_name
    doc_names: list[str] = [doc_name] if doc_name else []
    document_titles: dict[str, str] | None = None
    question_row: dict[str, Any] | None = None
    page_range = None

    if args.question_file and q_id is not None:
        row = load_question_by_id(args.question_file, q_id)
        question_row = row
        if args.question is None:
            args.question = row.get("question")
        if not doc_names:
            doc_names = _row_doc_names(row)
            doc_name = doc_names[0] if doc_names else None
        document_titles = _row_document_titles(row, doc_names)
        question_id = _row_question_id(row)
        ground_truth_answer = _row_ground_truth(row)
        page_range = _row_page_range(row)
    elif args.question_file and args.question:
        if not doc_name:
            parser.error("--doc-name or --mineru-dir is required when matching question_file by question text")
        entry_index, row = load_question_entry_by_doc_and_text(args.question_file, doc_name, args.question)
        question_row = row
        doc_names = _row_doc_names(row) or ([doc_name] if doc_name else [])
        document_titles = _row_document_titles(row, doc_names)
        q_id = row.get("q_id", entry_index)
        question_id = _row_question_id(row)
        ground_truth_answer = _row_ground_truth(row)
        page_range = _row_page_range(row)

    try:
        oracle_candidate_pages = _row_oracle_candidate_pages(question_row or {})
    except ValueError as exc:
        parser.error(str(exc))
    if oracle_candidate_pages and len(doc_names) != 1:
        parser.error("oracle_page_only currently supports exactly one document")

    if not args.question:
        parser.error("--question is required unless --question-file with --q-id is provided")

    auto_mineru_dirs: list[Path] = []
    if question_row is not None and not args.mineru_dir and not args.nodes_json:
        auto_mineru_dirs = _resolve_auto_mineru_dirs(config, args.question_file, question_row)

    request = {
        "question": args.question,
        "question_file": args.question_file,
        "doc_name": doc_name,
        "question_id": question_id,
        "q_id": q_id,
        "mineru_dir": args.mineru_dir,
        "nodes_json": args.nodes_json,
        "page_range": list(page_range) if page_range is not None else None,
    }
    if oracle_candidate_pages:
        request["oracle_page_only"] = True
        request["oracle_candidate_pages"] = oracle_candidate_pages
    if auto_mineru_dirs:
        request["mineru_dir"] = str(auto_mineru_dirs[0])
        if len(auto_mineru_dirs) > 1:
            request["mineru_dirs"] = [str(path) for path in auto_mineru_dirs]
            request["doc_names"] = doc_names

    if args.mineru_dir or auto_mineru_dirs:
        selected_mineru_dir = args.mineru_dir or str(auto_mineru_dirs[0])
        _, result = _execute_single_request(
            config=config,
            question=args.question,
            show_state=show_state,
            request=request,
            ground_truth_answer=ground_truth_answer,
            page_range=page_range,
            page_filter={item["page"] for item in oracle_candidate_pages} if oracle_candidate_pages else None,
            initial_candidate_pages=oracle_candidate_pages,
            mineru_dir=selected_mineru_dir,
            doc_name=doc_name,
            mineru_dirs=[str(path) for path in auto_mineru_dirs] or None,
            doc_names=doc_names or None,
            document_titles=document_titles,
        )
    else:
        _, result = _execute_single_request(
            config=config,
            question=args.question,
            show_state=show_state,
            request=request,
            ground_truth_answer=ground_truth_answer,
            page_range=page_range,
            nodes_path=args.nodes_json,
        )
    print(_json_dumps(result, indent=2))
    return 0
