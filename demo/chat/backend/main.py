from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import fitz
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import ReActAgent, VLLMTagsAgent
from app.config import AppConfig, ExtractConfig, NavigateConfig, OutputConfig
from app.llm import LLMConfig, LLMClient
from app.multimodal import collect_image_inputs
from app.run_logger import RunLogger
from app.schema import Action, BudgetState, CandidateAnswer, DocVQAState, DocumentNode, Observation
from app.tool import build_default_tool_collection
from app.workflow.in_memory import InMemoryDocumentEnvironment
from app.workflow.mineru_loader import load_nodes_from_mineru_output
from pipelines.document.mineru import _discover_mineru_artifacts, _ensure_page_screenshots


APP_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = APP_ROOT / "workspace"
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
BACKEND_RUNS_ROOT = Path(__file__).resolve().parent / "runs"
UPLOAD_ROOT = BACKEND_RUNS_ROOT / "upload"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
CONVERSATION_HISTORY_PATH = BACKEND_RUNS_ROOT / "chat-history.json"

STATE_FILE_NAME = ".docpilot_state.json"
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".html",
    ".xml",
    ".log",
    ".ini",
    ".toml",
}
PROCESSABLE_PDF_EXTENSIONS = {".pdf"}
CHAT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
ACTION_IMAGE_EXTENSIONS = CHAT_IMAGE_EXTENSIONS | {".bmp", ".tiff"}
SUPPORTED_TEXT_FOR_AGENT = TEXT_EXTENSIONS
PROCESSING_STATUSES = {
    "queued",
    "processing",
    "ready",
    "ready_image",
    "ready_text",
    "skipped",
    "failed",
}
CHAT_PDF_MAX_PAGES = 120
CHAT_PDF_DPI = 144

STATE_LOCK = threading.Lock()
CHAT_RUN_LOCK = threading.Lock()
CONVERSATION_HISTORY_LOCK = threading.RLock()


def utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def normalize_session_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-")
    return cleaned or "default"


def session_dir(session_id: str) -> Path:
    target = WORKSPACE_ROOT / normalize_session_id(session_id)
    target.mkdir(parents=True, exist_ok=True)
    return target


def state_path(session_id: str) -> Path:
    return session_dir(session_id) / STATE_FILE_NAME


def default_state(session_id: str) -> dict[str, Any]:
    return {
        "session_id": normalize_session_id(session_id),
        "updated_at": utcnow_iso(),
        "files": {},
    }


def load_state(session_id: str) -> dict[str, Any]:
    path = state_path(session_id)
    if not path.exists():
        return default_state(session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_state(session_id)
    if not isinstance(data, dict):
        return default_state(session_id)
    data.setdefault("session_id", normalize_session_id(session_id))
    data.setdefault("files", {})
    data.setdefault("updated_at", utcnow_iso())
    return data


def save_state(session_id: str, state: dict[str, Any]) -> None:
    path = state_path(session_id)
    state["updated_at"] = utcnow_iso()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def mutate_state(session_id: str, fn) -> dict[str, Any]:
    with STATE_LOCK:
        current = load_state(session_id)
        updated = fn(current) or current
        save_state(session_id, updated)
        return updated


def get_file_record(session_id: str, file_name: str) -> dict[str, Any] | None:
    state = load_state(session_id)
    item = state.get("files", {}).get(file_name)
    return item if isinstance(item, dict) else None


def source_file_path(session_id: str, file_name: str) -> Path:
    record = get_file_record(session_id, file_name) or {}
    return Path(str(record.get("source_path") or UPLOAD_ROOT / file_name))


def is_internal_workspace_file(path: Path) -> bool:
    return path.name == STATE_FILE_NAME or path.name.startswith(".")


def ensure_not_internal_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if is_internal_workspace_file(path):
        raise HTTPException(status_code=400, detail=f"internal workspace file is not allowed: {path.name}")
    return path


def normalize_base_url(base_url: str, provider: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if provider == "vllm" and not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"
    return cleaned


def build_llm_config(config: "ModelConfig") -> LLMConfig:
    return LLMConfig(
        enabled=True,
        provider="qwen" if config.provider == "vllm" else "",
        model=config.model.strip(),
        base_url=normalize_base_url(config.base_url, config.provider),
        api_key=config.api_key.strip() or "docpilot-local",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        reasoning="enabled" if config.reasoning else "disabled",
    )


def build_agent_config(config: "ModelConfig") -> AppConfig:
    llm_config = build_llm_config(config)
    llm_config.protocol = "vllm_tags" if config.provider == "vllm" else "json"
    return AppConfig(
        llm=llm_config,
        budget=BudgetState(max_steps=28, max_search_calls=6, max_extract_calls=12, max_compute_calls=4),
        navigate=NavigateConfig(top_k=5, modal="image", page_thumbnail_size=256, image_page_set_size=8),
        extract=ExtractConfig(if_enhanced=True, max_pages=4),
        output=OutputConfig(show_state=True),
    )


def build_agent(config: "ModelConfig") -> ReActAgent | VLLMTagsAgent:
    app_config = build_agent_config(config)
    llm_client = LLMClient(app_config.llm)
    tools = build_default_tool_collection(
        llm_client=llm_client,
        navigate_config=app_config.navigate,
        extract_config=app_config.extract,
    )
    if app_config.llm.protocol == "vllm_tags":
        return VLLMTagsAgent(llm_client=llm_client, tools=tools)
    return ReActAgent(llm_client=llm_client, tools=tools)


def advance_react_state(
    agent: ReActAgent,
    action: Action,
    observation: Observation,
    state: DocVQAState,
) -> None:
    agent._remember_latest_candidate(state)
    agent._invalidate_stale_verification_after_recovery(action, state, observation)
    termination_reason = agent._should_terminate(state)
    if termination_reason is not None:
        agent._terminate_as_unanswerable(state, termination_reason)
    elif agent._can_finish(state):
        state.done = True
        state.sync_evidence_state()


def _display_text(value: Any, limit: int = 4000, *, show_truncated_count: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, dict):
        preferred = next(
            (
                value[key]
                for key in ("markdown", "text", "table_body", "raw", "result", "value", "caption")
                if isinstance(value.get(key), str) and value[key].strip()
            ),
            None,
        )
        text = preferred.strip() if isinstance(preferred, str) else json.dumps(value, ensure_ascii=False)
    else:
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple)) else str(value)
    if len(text) <= limit:
        return text
    truncated = len(text) - limit
    suffix = f"…\n\n[Truncated **{truncated}** characters]" if show_truncated_count else "…"
    return f"{text[:limit].rstrip()}{suffix}"


def _display_pages(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    labels: list[str] = []
    for value in values:
        if isinstance(value, dict):
            page = value.get("page")
            doc_id = str(value.get("doc_id") or "").strip()
            label = f"{doc_id}:{page}" if doc_id and page is not None else str(page or "").strip()
        else:
            label = str(value).strip()
        if label and label not in labels:
            labels.append(label)
    return ", ".join(labels)


def _display_evidence(state: DocVQAState, evidence_ids: Any) -> list[str]:
    if not isinstance(evidence_ids, list):
        return []
    rendered: list[str] = []
    for evidence_id in evidence_ids:
        record = state.evidence_store.get(str(evidence_id))
        if record is None:
            continue
        title = f"Page {record.page} · {str(record.modality).title()}"
        if record.modality in {"image", "chart"}:
            provenance = record.provenance if isinstance(record.provenance, dict) else {}
            content = record.content if isinstance(record.content, dict) else {}
            caption = _display_text(
                provenance.get("summary")
                or provenance.get("title")
                or content.get("caption")
                or content.get("visual_caption")
            )
            rendered.append(f"- **{title}:** {caption or 'Visual evidence extracted.'}")
            continue
        content = _display_text(record.content)
        if content:
            rendered.append(f"#### {title}\n\n{content}")
    return rendered


def _display_extracted_evidence(state: DocVQAState, evidence_ids: Any) -> list[str]:
    if not isinstance(evidence_ids, list):
        return []
    rendered: list[str] = []
    for evidence_id in evidence_ids:
        record = state.evidence_store.get(str(evidence_id))
        if record is None or record.modality in {"image", "chart"}:
            continue
        content = _display_text(record.content, limit=800, show_truncated_count=True)
        if content:
            rendered.append(content.replace("\n", "\n   "))
    return rendered


def _action_display_result(action: Action, observation: Observation, state: DocVQAState) -> str:
    payload = observation.payload
    action_type = action.action_type
    lines: list[str] = []

    if action_type == "navigate":
        candidates = payload.get("candidate_pages")
        candidate_pages = _display_pages(candidates)
        lines.append(f"**Candidate pages:** {candidate_pages or 'None'}")
        reason = _display_text(payload.get("reason"), limit=2000)
        if reason:
            lines.append(f"**Reason:** {reason}")
        return "\n\n".join(lines)

    if action_type == "extract":
        evidence = _display_extracted_evidence(state, payload.get("evidence_ids"))
        lines.append("**Extracted evidence:**")
        lines.extend(f"{index}. {content}" for index, content in enumerate(evidence, start=1))
        if not evidence:
            lines.append("No textual evidence extracted; visual evidence is shown below.")
        return "\n\n".join(lines) or observation.message

    if action_type == "focus":
        planner = payload.get("planner") if isinstance(payload.get("planner"), dict) else {}
        reason = _display_text(planner.get("reason"), limit=1500)
        if reason:
            lines.append(f"**Focus target:** {reason}")
        operations = payload.get("operations")
        if isinstance(operations, list) and operations:
            lines.append("**Operations:**")
            for operation in operations:
                if not isinstance(operation, dict):
                    continue
                tool = str(operation.get("tool") or "focus").replace("_", " ").title()
                page = operation.get("page")
                region = operation.get("region") if isinstance(operation.get("region"), dict) else {}
                description = _display_text(region.get("description"), limit=500)
                status = "completed" if operation.get("success") else "failed"
                details = [item for item in (f"Page {page}" if page is not None else "", description, status) if item]
                lines.append(f"- **{tool}:** {' · '.join(details)}")
        lines.extend(_display_evidence(state, payload.get("evidence_ids")))
        return "\n\n".join(lines) or observation.message

    if action_type == "solver":
        summary = _display_text(payload.get("summary"), limit=2000)
        code = _display_text(payload.get("code"), limit=8000)
        execute = payload.get("execute") if isinstance(payload.get("execute"), dict) else {}
        result = _display_text(execute.get("result") or payload.get("result"), limit=4000)
        error = _display_text(execute.get("error") or payload.get("error"), limit=2000)
        if summary:
            lines.append(summary)
        if code:
            lines.append(f"```python\n{code}\n```")
        if result:
            lines.append(f"**Execution result:**\n\n{result}")
        if error:
            lines.append(f"**Error:** {error}")
        return "\n\n".join(lines) or observation.message

    if action_type == "answer":
        candidate = payload.get("candidate_answer") if isinstance(payload.get("candidate_answer"), dict) else {}
        answer = _display_text(candidate.get("answer") or payload.get("answer"), limit=6000)
        rationale = _display_text(candidate.get("rationale"), limit=4000)
        if answer:
            lines.append(f"**Answer:** {answer}")
        if rationale:
            lines.append(f"**Rationale:** {rationale}")
        if candidate.get("confidence") is not None:
            lines.append(f"**Confidence:** {candidate['confidence']}")
        return "\n\n".join(lines) or observation.message

    return ""


def _workspace_image_payload(image_ref: str, session_id: str) -> dict[str, str] | None:
    if image_ref.startswith(("data:image/", "http://", "https://")):
        return {"url": image_ref, "label": "Action image"}
    path = Path(image_ref).expanduser().resolve()
    session_root = (WORKSPACE_ROOT / normalize_session_id(session_id)).resolve()
    if session_root.parent != WORKSPACE_ROOT.resolve():
        return None
    try:
        relative = path.relative_to(session_root)
    except ValueError:
        mineru_root = (BACKEND_RUNS_ROOT / "mineru").resolve()
        try:
            relative = path.relative_to(mineru_root)
        except ValueError:
            return None
        url = f"/api/action-images/{quote(relative.as_posix(), safe='/')}"
    else:
        workspace_relative = Path(normalize_session_id(session_id)) / relative
        url = f"/api/workspace-images/{quote(workspace_relative.as_posix(), safe='/')}"
    if not path.is_file() or path.suffix.lower() not in ACTION_IMAGE_EXTENSIONS:
        return None
    return {
        "url": url,
        "label": path.name,
    }


async def _action_display_images(
    action: Action,
    observation: Observation,
    state: DocVQAState,
    env: InMemoryDocumentEnvironment,
    session_id: str,
) -> list[dict[str, str]]:
    sources: list[Any] = []
    labels_by_ref: dict[str, str] = {}
    if action.action_type == "navigate":
        candidate_pages = observation.payload.get("candidate_pages", [])
        candidate_refs: set[tuple[str, int]] = set()
        unscoped_pages: set[int] = set()
        for candidate in candidate_pages if isinstance(candidate_pages, list) else []:
            if isinstance(candidate, dict):
                try:
                    page = int(candidate.get("page"))
                except (TypeError, ValueError):
                    continue
                doc_id = str(candidate.get("doc_id") or "").strip()
                (candidate_refs if doc_id else unscoped_pages).add((doc_id, page) if doc_id else page)
            elif isinstance(candidate, str) and ":" in candidate:
                doc_id, page_text = candidate.rsplit(":", 1)
                try:
                    candidate_refs.add((doc_id.strip(), int(page_text)))
                except ValueError:
                    continue
        pages = sorted({page for _, page in candidate_refs} | unscoped_pages)
        if pages:
            try:
                page_context = await env.fetch_pages(
                    pages,
                    document_ids=sorted({doc_id for doc_id, _ in candidate_refs}) or state.document_ids,
                    with_image=True,
                )
                candidate_images = []
                for page_item in page_context.get("pages", []):
                    doc_id = str(page_item.get("doc_id") or "").strip()
                    page = int(page_item.get("page"))
                    image_ref = page_item.get("screenshot_path")
                    if not image_ref or (
                        (doc_id, page) not in candidate_refs and page not in unscoped_pages
                    ):
                        continue
                    candidate_images.append(image_ref)
                    labels_by_ref[str(image_ref)] = f"{doc_id} · Page {page}" if doc_id else f"Page {page}"
                sources.append(candidate_images)
            except Exception:
                pass
    elif action.action_type == "extract":
        sources.append(observation.payload)
    elif action.action_type == "focus":
        evidence_ids = observation.payload.get("evidence_ids")
        records = [
            state.evidence_store[item]
            for item in evidence_ids if item in state.evidence_store
        ] if isinstance(evidence_ids, list) else []
        sources.append([record.to_dict() for record in records])

    images: list[dict[str, str]] = []
    for image_ref in collect_image_inputs(sources, max_images=12):
        payload = _workspace_image_payload(image_ref, session_id)
        if payload is None or payload["url"] in {item["url"] for item in images}:
            continue
        payload["label"] = labels_by_ref.get(image_ref, payload["label"])
        images.append(payload)
    return images


def _history_with_display_images(
    history: list[dict[str, Any]],
    images_by_step: dict[int, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    return [
        {**item, **({"display_images": images_by_step[index]} if images_by_step.get(index) else {})}
        for index, item in enumerate(history)
    ]


def file_payload(file_path: Path, record: dict[str, Any] | None = None) -> dict[str, Any]:
    stat = file_path.stat()
    suffix = file_path.suffix.lower()
    record = record or {}
    return {
        "name": file_path.name,
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "content_type": mimetypes.guess_type(file_path.name)[0] or "application/octet-stream",
        "download_url": f"/uploads/{quote(file_path.name)}",
        "extension": suffix,
        "processing_status": record.get("processing_status", "skipped"),
        "processing_error": record.get("processing_error"),
        "doc_id": record.get("doc_id") or file_path.stem,
        "mineru_output_dir": record.get("mineru_output_dir"),
        "artifacts": record.get("artifacts") or {},
        "viewer_url": record.get("viewer_url"),
        "viewer_type": record.get("viewer_type"),
    }


def list_files_payload(session_id: str) -> list[dict[str, Any]]:
    bootstrap_session_files(session_id)
    state = load_state(session_id)
    files_state = state.get("files", {})
    payload: list[dict[str, Any]] = []
    if not isinstance(files_state, dict):
        return payload
    for file_name, record in sorted(files_state.items(), key=lambda item: item[0].lower()):
        if not isinstance(record, dict):
            continue
        file_path = Path(str(record.get("source_path") or UPLOAD_ROOT / file_name))
        if file_path.is_file():
            payload.append(file_payload(file_path, record))
    return payload


def read_document_snippet(file_path: Path, limit: int = 12000) -> str:
    suffix = file_path.suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        return (
            f"[binary-file]\n"
            f"name: {file_path.name}\n"
            f"path: {file_path}\n"
            f"note: 该文件不是纯文本，当前预览请走 Workspace 查看器。"
        )
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    if len(content) <= limit:
        return content
    return f"{content[:limit]}\n\n[truncated {len(content) - limit} chars]"


def mineru_output_dir_for(doc_id: str) -> Path:
    return BACKEND_RUNS_ROOT / "mineru" / doc_id


def session_uses_mineru(session_id: str) -> bool:
    return normalize_session_id(session_id) != "chat"


def initial_processing_status(session_id: str, suffix: str) -> str:
    if suffix in PROCESSABLE_PDF_EXTENSIONS:
        return "queued" if session_uses_mineru(session_id) else "ready"
    if suffix in CHAT_IMAGE_EXTENSIONS:
        return "ready_image"
    if suffix in SUPPORTED_TEXT_FOR_AGENT:
        return "ready_text"
    return "skipped"


def build_mineru_command(file_path: Path, output_dir: Path) -> list[str]:
    configured_executable = os.environ.get("DOCPILOT_MINERU_EXECUTABLE")
    local_kit = PROJECT_ROOT / ".venv" / "bin" / "mineru-kit"
    mineru_kit = str(local_kit) if local_kit.is_file() else shutil.which("mineru-kit")
    executable = (
        configured_executable
        or mineru_kit
        or shutil.which("mineru")
        or str(PROJECT_ROOT / ".venv" / "bin" / "mineru")
    )
    if Path(executable).name == "mineru-kit":
        return [
            executable,
            "parse",
            str(file_path.resolve()),
            "-o",
            str(output_dir.resolve()),
            "-f",
            "zip",
        ]
    cmd = [
        executable,
        "-p",
        str(file_path.resolve()),
        "-o",
        str(output_dir.resolve()),
    ]
    backend = os.environ.get("DOCPILOT_MINERU_BACKEND", "pipeline")
    if backend:
        cmd.extend(["-b", backend])
    return cmd


def extract_mineru_zip(output_dir: Path, doc_id: str) -> None:
    archive_path = output_dir / f"{doc_id}.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(f"MinerU zip output not found: {archive_path}")
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"Unsafe path in MinerU zip: {member.filename}")
        archive.extractall(root)


def build_mineru_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = os.environ.get("DOCPILOT_MINERU_MODEL_SOURCE", "modelscope")
    path_entries = [str(PROJECT_ROOT / ".venv" / "bin")]

    local_nvccs = sorted(
        (PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages/nvidia/cu*/bin/nvcc"),
        reverse=True,
    )
    if local_nvccs:
        cuda_root = local_nvccs[0].parent.parent
        env["CUDA_HOME"] = str(cuda_root)
        path_entries.append(str(cuda_root / "bin"))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(cuda_root / "lib"), env.get("LD_LIBRARY_PATH", ""))
        )
    env["PATH"] = os.pathsep.join((*path_entries, env.get("PATH", "")))
    return env


def update_file_record(
    session_id: str,
    file_name: str,
    **patch: Any,
) -> None:
    def _mutate(state: dict[str, Any]) -> dict[str, Any]:
        files = state.setdefault("files", {})
        record = files.setdefault(file_name, {})
        record.update(patch)
        return state

    mutate_state(session_id, _mutate)


def set_initial_file_record(session_id: str, file_path: Path) -> None:
    suffix = file_path.suffix.lower()
    update_file_record(
        session_id,
        file_path.name,
        doc_id=file_path.stem,
        source_path=str(file_path),
        extension=suffix,
        processing_status=initial_processing_status(session_id, suffix),
        processing_error=None,
        mineru_output_dir=None,
        artifacts={},
    )


def bootstrap_session_files(session_id: str) -> None:
    def _mutate(state: dict[str, Any]) -> dict[str, Any]:
        files_state = state.setdefault("files", {})
        for file_name, record in list(files_state.items()):
            if not isinstance(record, dict):
                files_state.pop(file_name, None)
                continue
            file_path = Path(str(record.get("source_path") or UPLOAD_ROOT / file_name))
            if not file_path.is_file():
                files_state.pop(file_name, None)
                continue
            suffix = file_path.suffix.lower()
            record.update(source_path=str(file_path), extension=suffix)
            record.setdefault("doc_id", file_path.stem)
            record.setdefault("artifacts", {})
            record.setdefault("mineru_output_dir", None)
            record.setdefault(
                "processing_status",
                initial_processing_status(session_id, suffix),
            )
            if not session_uses_mineru(session_id) and suffix in PROCESSABLE_PDF_EXTENSIONS:
                record.update(
                    processing_status="ready",
                    processing_error=None,
                    mineru_output_dir=None,
                    artifacts={},
                )
        return state

    mutate_state(session_id, _mutate)
    state = load_state(session_id)
    for file_name, record in state.get("files", {}).items():
        if not isinstance(record, dict):
            continue
        status = record.get("processing_status")
        extension = str(record.get("extension") or "").lower()
        if (
            session_uses_mineru(session_id)
            and extension in PROCESSABLE_PDF_EXTENSIONS
            and status in {None, "queued", "ready"}
        ):
            mineru_dir = record.get("mineru_output_dir")
            if mineru_dir and Path(str(mineru_dir)).exists():
                try:
                    load_nodes_from_mineru_output(mineru_dir, doc_id=str(record.get("doc_id") or Path(file_name).stem))
                except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    update_file_record(
                        session_id,
                        file_name,
                        processing_status="failed",
                        processing_error=f"MinerU output cannot be loaded: {exc}",
                    )
                else:
                    update_file_record(session_id, file_name, processing_status="ready", processing_error=None)
            elif status != "processing":
                spawn_preprocess(session_id, file_name)


def preprocess_upload(session_id: str, file_name: str) -> None:
    file_path = source_file_path(session_id, file_name)
    if not file_path.exists():
        update_file_record(session_id, file_name, processing_status="failed", processing_error="source file missing")
        return

    suffix = file_path.suffix.lower()
    if not session_uses_mineru(session_id) and suffix in PROCESSABLE_PDF_EXTENSIONS:
        update_file_record(
            session_id,
            file_name,
            processing_status="ready",
            processing_error=None,
            mineru_output_dir=None,
            artifacts={},
        )
        return
    if suffix not in PROCESSABLE_PDF_EXTENSIONS:
        if suffix in SUPPORTED_TEXT_FOR_AGENT:
            update_file_record(session_id, file_name, processing_status="ready_text")
        else:
            update_file_record(
                session_id,
                file_name,
                processing_status="skipped",
                processing_error="auto preprocessing currently supports PDF only",
            )
        return

    doc_id = file_path.stem
    output_dir = mineru_output_dir_for(doc_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    update_file_record(session_id, file_name, processing_status="processing", mineru_output_dir=str(output_dir))
    cmd = build_mineru_command(file_path, output_dir)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT.resolve()),
            env=build_mineru_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        update_file_record(session_id, file_name, processing_status="failed", processing_error=str(exc))
        return

    artifacts = _discover_mineru_artifacts(output_dir)
    load_error: Exception | None = None
    if completed.returncode == 0:
        try:
            if Path(cmd[0]).name == "mineru-kit":
                extract_mineru_zip(output_dir, doc_id)
                artifacts = _discover_mineru_artifacts(output_dir)
            load_nodes_from_mineru_output(output_dir, doc_id=doc_id)
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
            load_error = exc
    if completed.returncode == 0 and load_error is None:
        screenshot_dir = _ensure_page_screenshots(output_dir, input_path=file_path.resolve(), doc_id=doc_id)
        if screenshot_dir:
            artifacts["screenshot_dir"] = str(screenshot_dir)
        update_file_record(
            session_id,
            file_name,
            processing_status="ready",
            processing_error=None,
            mineru_output_dir=str(output_dir),
            artifacts=artifacts,
        )
    else:
        stderr_tail = (completed.stderr or "")[-2000:]
        stdout_tail = (completed.stdout or "")[-2000:]
        reason = str(load_error) if load_error else stderr_tail or stdout_tail or "MinerU produced no loadable output"
        update_file_record(
            session_id,
            file_name,
            processing_status="failed",
            processing_error=reason,
            mineru_output_dir=str(output_dir),
            artifacts=artifacts,
        )


def spawn_preprocess(session_id: str, file_name: str) -> None:
    worker = threading.Thread(target=preprocess_upload, args=(session_id, file_name), daemon=True)
    worker.start()


def build_text_nodes(file_path: Path, doc_id: str) -> list[DocumentNode]:
    snippet = read_document_snippet(file_path, limit=50000)
    node = DocumentNode(
        node_id=f"{doc_id}::page::1",
        doc_id=doc_id,
        page=1,
        node_type="document",
        modality="text",
        title=file_path.name,
        text=snippet,
        summary=snippet[:300],
        metadata={"source": "text_file", "path": str(file_path)},
    )
    para = DocumentNode(
        node_id=f"{doc_id}::page::1::block::0",
        doc_id=doc_id,
        page=1,
        node_type="paragraph",
        modality="text",
        title=file_path.name,
        text=snippet,
        summary=snippet[:240],
        parent_id=node.node_id,
        metadata={"source": "text_file", "path": str(file_path)},
    )
    node.children = [para.node_id]
    return [para, node]


def build_environment(session_id: str, selected_files: list[str]) -> tuple[InMemoryDocumentEnvironment, list[str], list[str]]:
    state = load_state(session_id)
    file_records = state.get("files", {})
    all_nodes: list[DocumentNode] = []
    document_ids: list[str] = []
    skipped: list[str] = []

    for file_name in selected_files:
        if is_internal_workspace_file(Path(file_name)):
            skipped.append(f"{file_name}: internal workspace file is not allowed")
            continue
        record = file_records.get(file_name) if isinstance(file_records, dict) else None
        if not isinstance(record, dict):
            skipped.append(f"{file_name}: missing state record")
            continue
        file_path = Path(str(record.get("source_path") or UPLOAD_ROOT / file_name))
        if not file_path.exists():
            skipped.append(f"{file_name}: source file missing")
            continue
        doc_id = str(record.get("doc_id") or file_path.stem)
        status = record.get("processing_status")

        if status == "ready" and record.get("mineru_output_dir"):
            try:
                nodes = load_nodes_from_mineru_output(record["mineru_output_dir"], doc_id=doc_id)
            except Exception as exc:
                skipped.append(f"{file_name}: failed to load MinerU output ({exc})")
                continue
            all_nodes.extend(nodes)
            document_ids.append(doc_id)
            continue

        if file_path.suffix.lower() in SUPPORTED_TEXT_FOR_AGENT:
            all_nodes.extend(build_text_nodes(file_path, doc_id))
            document_ids.append(doc_id)
            continue

        skipped.append(f"{file_name}: unsupported for DocPilot agent")

    env = InMemoryDocumentEnvironment(all_nodes)
    return env, document_ids, skipped


def build_multimodal_chat_messages(
    session_id: str,
    messages: list["ChatMessage"],
    selected_files: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    formatted: list[dict[str, Any]] = [
        {"role": item.role, "content": item.content}
        for item in messages
        if item.content.strip()
    ]
    if not selected_files:
        return formatted, []

    latest_user_index = next(
        (index for index in range(len(formatted) - 1, -1, -1) if formatted[index]["role"] == "user"),
        None,
    )
    if latest_user_index is None:
        raise HTTPException(status_code=400, detail="消息列表中缺少用户问题。")

    content: list[dict[str, Any]] = [
        {"type": "text", "text": str(formatted[latest_user_index]["content"])}
    ]
    text_context: list[str] = []
    skipped: list[str] = []
    for raw_name in selected_files:
        file_name = Path(raw_name).name
        if file_name != raw_name or is_internal_workspace_file(Path(file_name)):
            skipped.append(f"{raw_name}: invalid workspace file")
            continue
        file_path = source_file_path(session_id, file_name)
        if not file_path.exists() or not file_path.is_file():
            skipped.append(f"{file_name}: source file missing")
            continue

        if file_path.suffix.lower() == ".pdf":
            try:
                with fitz.open(file_path) as pdf:
                    page_count = min(pdf.page_count, CHAT_PDF_MAX_PAGES)
                    for page_index in range(page_count):
                        pixmap = pdf.load_page(page_index).get_pixmap(dpi=CHAT_PDF_DPI, alpha=False)
                        encoded = base64.b64encode(pixmap.tobytes("jpeg")).decode("ascii")
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                            }
                        )
                    if pdf.page_count > page_count:
                        skipped.append(
                            f"{file_name}: only the first {CHAT_PDF_MAX_PAGES} pages were included"
                        )
            except Exception as exc:
                skipped.append(f"{file_name}: failed to render PDF ({exc})")
            continue

        if file_path.suffix.lower() in CHAT_IMAGE_EXTENSIONS:
            mime_type = mimetypes.guess_type(file_path.name)[0] or "image/png"
            encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
            continue

        if file_path.suffix.lower() in TEXT_EXTENSIONS:
            text_context.append(
                f"[Document: {file_name}]\n{read_document_snippet(file_path, limit=50000)}"
            )
            continue

        skipped.append(f"{file_name}: Chat agent supports PDF, image, and text documents")

    if text_context:
        content[0]["text"] += "\n\nAttached document text:\n\n" + "\n\n".join(text_context)
    formatted[latest_user_index]["content"] = content
    return formatted, skipped


def chat_direct(
    config: "ModelConfig",
    messages: list["ChatMessage"],
    session_id: str,
    selected_files: list[str],
    logger: RunLogger | None = None,
) -> dict[str, Any]:
    request_messages, skipped = build_multimodal_chat_messages(session_id, messages, selected_files)
    llm = LLMClient(build_llm_config(config))
    llm.set_logger(logger)
    llm.set_step(1)
    llm.set_call_source("chat_direct")
    try:
        content = llm.generate_text_from_messages(request_messages)
        if logger is not None:
            logger.log_result(
                step=1,
                result=content,
                observation={"action_type": "chat", "success": True, "message": content, "payload": {}},
            )
    finally:
        llm.client.close()
    return {
        "answer": content or "",
        "history": [],
        "state": {"mode": "chat"},
        "skipped_documents": skipped,
        "mode": "chat",
    }


def chat_run_dir_for(session_id: str) -> Path:
    target = session_dir(session_id) / ".chat_runs"
    target.mkdir(parents=True, exist_ok=True)
    return target


def start_chat_run(session_id: str, conversation_id: str) -> tuple[str, RunLogger]:
    run_dir = chat_run_dir_for(session_id)
    normalized_conversation_id = normalize_session_id(conversation_id or "unlinked")
    pattern = re.compile(rf"^{re.escape(normalized_conversation_id)}-(\d+)(?:\.result\.json|\.log)$")
    with CHAT_RUN_LOCK:
        indexes = [
            int(match.group(1))
            for path in [
                *run_dir.glob(f"{normalized_conversation_id}-*.result.json"),
                *run_dir.glob(f"{normalized_conversation_id}-*.log"),
            ]
            if (match := pattern.match(path.name))
        ]
        run_id = f"{normalized_conversation_id}-{max(indexes, default=0) + 1}"
        logger = RunLogger(run_dir / f"{run_id}.log")
    return run_id, logger


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _safe_config_payload(config: "ModelConfig") -> dict[str, Any]:
    payload = _model_dump(config)
    payload.pop("api_key", None)
    return payload


def persist_chat_run(
    *,
    run_id: str,
    log_path: Path,
    session_id: str,
    conversation_id: str,
    mode: str,
    question: str,
    messages: list["ChatMessage"],
    selected_files: list[str],
    config: "ModelConfig",
    answer: str,
    history: list[dict[str, Any]],
    skipped: list[str] | None = None,
    error: str | None = None,
    runtime_seconds: float | None = None,
) -> dict[str, str]:
    run_dir = chat_run_dir_for(session_id)
    normalized_conversation_id = normalize_session_id(conversation_id or "unlinked")
    result_path = run_dir / f"{run_id}.result.json"

    request_payload = {
        "session_id": normalize_session_id(session_id),
        "conversation_id": normalized_conversation_id,
        "mode": mode,
        "question": question,
        "messages": [_model_dump(item) for item in messages],
        "selected_files": selected_files,
        "config": _safe_config_payload(config),
        "skipped_documents": skipped or [],
    }
    result_payload = {
        "run_id": run_id,
        "conversation_id": normalized_conversation_id,
        "created_at": utcnow_iso(),
        "mode": mode,
        "question": question,
        "answer": answer,
        "history_len": len(history),
        "skipped_documents": skipped or [],
        "request": request_payload,
        "log_path": str(log_path.resolve()),
        "error": error,
    }
    if runtime_seconds is not None:
        result_payload["runtime_seconds"] = runtime_seconds

    result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_id": run_id,
        "conversation_id": normalized_conversation_id,
        "log_path": str(log_path.resolve()),
        "result_path": str(result_path.resolve()),
    }


def conversation_to_question(messages: list["ChatMessage"]) -> str:
    user_messages = [item.content.strip() for item in messages if item.role == "user" and item.content.strip()]
    if not user_messages:
        raise HTTPException(status_code=400, detail="消息列表为空。")
    latest = user_messages[-1]
    if len(user_messages) == 1:
        return latest
    prior = "\n".join(f"- {item}" for item in user_messages[:-1][-4:])
    return f"Conversation context:\n{prior}\n\nCurrent question:\n{latest}"


class ModelConfig(BaseModel):
    provider: Literal["vllm", "custom"] = "vllm"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    model: str = ""
    reasoning: bool = False
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=128, le=32768)
    timeout: int = Field(default=180, ge=10, le=1800)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str = "default"
    workspace_session_id: str | None = None
    conversation_id: str = ""
    config: ModelConfig
    agent: Literal["default", "chat"] = "default"
    messages: list[ChatMessage]
    selected_files: list[str] = Field(default_factory=list)


class ServerPathImportRequest(BaseModel):
    session_id: str = "default"
    paths: list[str] = Field(default_factory=list)


class ConversationHistoryMessage(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    role: Literal["user", "assistant"]
    content: str
    agentTrace: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    agentState: dict[str, Any] | None = None
    responseMode: Literal["chat", "docpilot"] = "chat"


class ConversationHistoryItem(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    title: str = Field(max_length=500)
    messages: list[ConversationHistoryMessage] = Field(default_factory=list, max_length=1000)
    selectedFiles: list[str] = Field(default_factory=list, max_length=100)
    agentTrace: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    agentState: dict[str, Any] | None = None
    responseMode: Literal["chat", "docpilot"] = "chat"
    agent: Literal["default", "chat"] = "default"
    updatedAt: str = Field(max_length=64)


class ConversationHistoryStore(BaseModel):
    active_id: str | None = Field(default=None, max_length=200)
    conversations: list[ConversationHistoryItem] = Field(default_factory=list, max_length=50)


def load_conversation_history() -> ConversationHistoryStore:
    with CONVERSATION_HISTORY_LOCK:
        if not CONVERSATION_HISTORY_PATH.is_file():
            return ConversationHistoryStore()
        try:
            history = ConversationHistoryStore.model_validate_json(
                CONVERSATION_HISTORY_PATH.read_text(encoding="utf-8")
            )
            for conversation in history.conversations:
                if not conversation.agentTrace:
                    continue
                for message in reversed(conversation.messages):
                    if message.role != "assistant":
                        continue
                    if not message.agentTrace:
                        message.agentTrace = list(conversation.agentTrace)
                        message.agentState = conversation.agentState
                        message.responseMode = conversation.responseMode
                    break
            return history
        except (OSError, ValueError):
            return ConversationHistoryStore()


def save_conversation_history(history: ConversationHistoryStore) -> ConversationHistoryStore:
    with CONVERSATION_HISTORY_LOCK:
        CONVERSATION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = CONVERSATION_HISTORY_PATH.with_name(
            f".{CONVERSATION_HISTORY_PATH.name}.tmp"
        )
        temporary_path.write_text(
            json.dumps(history.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(CONVERSATION_HISTORY_PATH)
    return history


def delete_conversation_history(conversation_id: str) -> ConversationHistoryStore:
    with CONVERSATION_HISTORY_LOCK:
        history = load_conversation_history()
        removed = next((item for item in history.conversations if item.id == conversation_id), None)
        updated = ConversationHistoryStore(
            active_id=None if history.active_id == conversation_id else history.active_id,
            conversations=[item for item in history.conversations if item.id != conversation_id],
        )
        save_conversation_history(updated)

        if removed is not None:
            session_id = "chat" if removed.agent == "chat" else "default"
            run_dir = WORKSPACE_ROOT / session_id / ".chat_runs"
            normalized_id = normalize_session_id(conversation_id)
            for pattern in (f"{normalized_id}-*.log", f"{normalized_id}-*.result.json"):
                for path in run_dir.glob(pattern):
                    path.unlink(missing_ok=True)
    return updated


app = FastAPI(title="DocPilot Chat Demo", version="0.2.0")
cors_origins = [
    origin.strip()
    for origin in os.environ.get("DOCPILOT_CORS_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")


def bootstrap_all_sessions() -> None:
    for child in WORKSPACE_ROOT.iterdir():
        if child.is_dir():
            bootstrap_session_files(child.name)
    bootstrap_session_files("default")


@app.on_event("startup")
async def on_startup() -> None:
    bootstrap_all_sessions()


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "workspace_root": str(WORKSPACE_ROOT),
        "project_root": str(PROJECT_ROOT),
        "timestamp": utcnow_iso(),
    }


@app.get("/api/presets")
async def presets() -> dict[str, Any]:
    return {
        "vllm": {
            "provider": "vllm",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "",
            "model": "",
            "reasoning": False,
            "temperature": 0.0,
            "timeout": 180,
        },
        "custom": {
            "provider": "custom",
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "model": "",
            "reasoning": False,
            "temperature": 0.0,
            "timeout": 180,
        },
    }


@app.get("/api/history")
async def get_conversation_history() -> ConversationHistoryStore:
    return load_conversation_history()


@app.put("/api/history")
async def put_conversation_history(history: ConversationHistoryStore) -> ConversationHistoryStore:
    return save_conversation_history(history)


@app.delete("/api/history/{conversation_id}")
async def delete_history_item(conversation_id: str) -> ConversationHistoryStore:
    return delete_conversation_history(conversation_id)


@app.get("/api/action-images/{image_path:path}")
async def get_action_image(image_path: str) -> FileResponse:
    mineru_root = (BACKEND_RUNS_ROOT / "mineru").resolve()
    target = (mineru_root / image_path).resolve()
    try:
        target.relative_to(mineru_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Action image not found") from exc
    if not target.is_file() or target.suffix.lower() not in ACTION_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Action image not found")
    return FileResponse(target)


@app.get("/api/workspace-images/{session_id}/{image_path:path}")
async def get_workspace_image(session_id: str, image_path: str) -> FileResponse:
    workspace_root = WORKSPACE_ROOT.resolve()
    session_root = (workspace_root / session_id).resolve()
    if session_root.parent != workspace_root:
        raise HTTPException(status_code=404, detail="Workspace image not found")
    target = (session_root / image_path).resolve()
    try:
        target.relative_to(session_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Workspace image not found") from exc
    if not target.is_file() or target.suffix.lower() not in ACTION_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Workspace image not found")
    return FileResponse(target)


@app.get("/api/models/discover")
async def discover_models(
    provider: Literal["vllm", "custom"] = Query("vllm"),
    base_url: str = Query(...),
    api_key: str = Query(""),
) -> dict[str, Any]:
    client = OpenAI(
        api_key=api_key.strip() or "docpilot-local",
        base_url=normalize_base_url(base_url, provider),
        timeout=20.0,
    )
    try:
        result = client.models.list()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"模型发现失败: {exc}") from exc
    return {"models": sorted([item.id for item in result.data])}


@app.get("/api/files")
async def list_files(session_id: str = Query("default")) -> dict[str, Any]:
    return {"files": list_files_payload(session_id), "workspace_dir": str(session_dir(session_id))}


@app.post("/api/files")
async def upload_files(
    session_id: str = Query("default"),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    bootstrap_session_files(session_id)
    stored: list[str] = []
    for uploaded in files:
        name = Path(uploaded.filename or "upload.bin").name
        if name == STATE_FILE_NAME or name.startswith("."):
            continue
        destination = UPLOAD_ROOT / name
        existing_record = get_file_record(session_id, name)
        if destination.exists() and isinstance(existing_record, dict):
            if (
                session_uses_mineru(session_id)
                and existing_record.get("processing_status") == "failed"
            ):
                update_file_record(
                    session_id,
                    name,
                    processing_status="processing",
                    processing_error=None,
                )
                spawn_preprocess(session_id, name)
            stored.append(name)
            continue
        content = await uploaded.read()
        if destination.exists() and destination.read_bytes() != content:
            raise HTTPException(status_code=409, detail=f"同名文件内容不同: {name}")
        if not destination.exists():
            destination.write_bytes(content)
        set_initial_file_record(session_id, destination)
        if (
            session_uses_mineru(session_id)
            and destination.suffix.lower() in PROCESSABLE_PDF_EXTENSIONS
        ):
            spawn_preprocess(session_id, name)
        stored.append(name)
    return {"uploaded": stored, "files": list_files_payload(session_id)}


@app.post("/api/files/from-paths")
async def import_files_from_paths(request: ServerPathImportRequest) -> dict[str, Any]:
    bootstrap_session_files(request.session_id)
    if not request.paths:
        raise HTTPException(status_code=400, detail="paths 不能为空。")

    imported: list[str] = []
    errors: list[str] = []

    for raw_path in request.paths:
        candidate = ensure_not_internal_path(Path(raw_path).expanduser())
        if not candidate.exists():
            errors.append(f"{raw_path}: path does not exist")
            continue
        if not candidate.is_file():
            errors.append(f"{raw_path}: only file paths are supported")
            continue

        destination = UPLOAD_ROOT / candidate.name
        existing_record = get_file_record(request.session_id, candidate.name)
        if destination.exists() and isinstance(existing_record, dict):
            if (
                session_uses_mineru(request.session_id)
                and existing_record.get("processing_status") == "failed"
            ):
                update_file_record(
                    request.session_id,
                    destination.name,
                    processing_status="processing",
                    processing_error=None,
                )
                spawn_preprocess(request.session_id, destination.name)
            imported.append(destination.name)
            continue
        if destination.exists() and destination.resolve() != candidate.resolve():
            if destination.read_bytes() != candidate.read_bytes():
                errors.append(f"{raw_path}: 同名文件内容不同")
                continue
        elif destination.resolve() != candidate.resolve():
            shutil.copy2(candidate, destination)
        set_initial_file_record(request.session_id, destination)
        if (
            session_uses_mineru(request.session_id)
            and destination.suffix.lower() in PROCESSABLE_PDF_EXTENSIONS
        ):
            spawn_preprocess(request.session_id, destination.name)
        imported.append(destination.name)

    if not imported and errors:
        raise HTTPException(status_code=400, detail="\n".join(errors))

    return {
        "imported": imported,
        "errors": errors,
        "files": list_files_payload(request.session_id),
    }


@app.get("/api/files/content/{file_name}")
async def preview_file(file_name: str, session_id: str = Query("default")) -> PlainTextResponse:
    file_name = ensure_not_internal_path(Path(file_name).name).name
    file_path = source_file_path(session_id, file_name)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在。")
    return PlainTextResponse(read_document_snippet(file_path, limit=20000))


@app.post("/api/files/{file_name}/retry")
async def retry_file(file_name: str, session_id: str = Query("default")) -> dict[str, Any]:
    file_name = ensure_not_internal_path(Path(file_name).name).name
    record = get_file_record(session_id, file_name)
    if not isinstance(record, dict):
        raise HTTPException(status_code=404, detail="文件不存在。")
    file_path = Path(str(record.get("source_path") or UPLOAD_ROOT / file_name))
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="源文件不存在。")
    if not session_uses_mineru(session_id) or file_path.suffix.lower() not in PROCESSABLE_PDF_EXTENSIONS:
        raise HTTPException(status_code=400, detail="该文件不需要 MinerU。")
    if record.get("processing_status") != "failed":
        raise HTTPException(status_code=409, detail="只有 failed 文件可以重试。")

    if record.get("mineru_output_dir"):
        shutil.rmtree(record["mineru_output_dir"], ignore_errors=True)
    update_file_record(
        session_id,
        file_name,
        processing_status="processing",
        processing_error=None,
        mineru_output_dir=None,
        artifacts={},
    )
    spawn_preprocess(session_id, file_name)
    return {"retried": file_name, "files": list_files_payload(session_id)}


@app.delete("/api/files/{file_name}")
async def delete_file(file_name: str, session_id: str = Query("default")) -> dict[str, Any]:
    file_name = ensure_not_internal_path(Path(file_name).name).name
    record = get_file_record(session_id, file_name) or {}
    file_path = Path(str(record.get("source_path") or UPLOAD_ROOT / file_name))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在。")
    if record.get("mineru_output_dir"):
        shutil.rmtree(record["mineru_output_dir"], ignore_errors=True)

    def _mutate(state: dict[str, Any]) -> dict[str, Any]:
        files = state.setdefault("files", {})
        files.pop(file_name, None)
        return state

    mutate_state(session_id, _mutate)
    still_referenced = any(
        Path(str(item.get("source_path") or "")) == file_path
        for session_root in WORKSPACE_ROOT.iterdir()
        if session_root.is_dir()
        for item in (load_state(session_root.name).get("files") or {}).values()
        if isinstance(item, dict)
    )
    if not still_referenced:
        file_path.unlink()
    return {"deleted": file_name, "files": list_files_payload(session_id)}


@app.post("/api/chat/stream")
async def chat_stream(request: Request, chat_request: ChatRequest) -> StreamingResponse:
    use_docpilot = chat_request.agent == "default"
    workspace_session_id = chat_request.workspace_session_id or chat_request.session_id
    question = conversation_to_question(chat_request.messages)
    env, document_ids, skipped = (
        build_environment(workspace_session_id, chat_request.selected_files)
        if use_docpilot
        else (InMemoryDocumentEnvironment([]), [], [])
    )

    def abort_agent(agent: ReActAgent | VLLMTagsAgent | None) -> None:
        if agent is None:
            return
        try:
            agent.llm.client.close()
        except Exception:
            pass

    async def run_cancellable(factory, agent: ReActAgent | VLLMTagsAgent | None):
        task = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(factory())))
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.1)
            if task in done:
                return task.result()
            if await request.is_disconnected():
                abort_agent(agent)
                task.cancel()
                raise asyncio.CancelledError()

    async def event_stream():
        agent: ReActAgent | VLLMTagsAgent | None = None
        latest_state: dict[str, Any] = {}
        latest_history: list[dict[str, Any]] = []
        latest_answer = ""
        display_images_by_step: dict[int, list[dict[str, str]]] = {}
        started = time.perf_counter()
        mode = "docpilot" if use_docpilot else "chat"
        run_id, logger = start_chat_run(chat_request.session_id, chat_request.conversation_id)
        logger.info(f"Chat run started: conversation_id={chat_request.conversation_id or 'unlinked'}, mode={mode}")
        try:
            yield json.dumps(
                {"type": "meta", "model": chat_request.config.model, "mode": mode},
                ensure_ascii=False,
            ) + "\n"

            if skipped:
                notice = "Skipped documents:\n" + "\n".join(skipped) + "\n\n"
                yield json.dumps({"type": "delta", "content": notice}, ensure_ascii=False) + "\n"

            if not use_docpilot:
                result = await asyncio.to_thread(
                    chat_direct,
                    chat_request.config,
                    chat_request.messages,
                    workspace_session_id,
                    chat_request.selected_files,
                    logger,
                )
                chat_skipped = result.get("skipped_documents") or []
                if chat_skipped:
                    notice = "Skipped documents:\n" + "\n".join(chat_skipped) + "\n\n"
                    yield json.dumps({"type": "delta", "content": notice}, ensure_ascii=False) + "\n"
                answer = str(result.get("answer") or "")
                latest_answer = answer
                latest_state = dict(result.get("state") or {})
                runtime_seconds = round(time.perf_counter() - started, 3)
                saved_paths = persist_chat_run(
                    run_id=run_id,
                    log_path=logger.path,
                    session_id=chat_request.session_id,
                    conversation_id=chat_request.conversation_id,
                    mode="chat",
                    question=question,
                    messages=chat_request.messages,
                    selected_files=chat_request.selected_files,
                    config=chat_request.config,
                    answer=answer,
                    history=[],
                    skipped=chat_skipped,
                    runtime_seconds=runtime_seconds,
                )
                logger.info(f"Chat run finished. Result saved to: {saved_paths['result_path']}")
                yield json.dumps(
                    {"type": "delta", "content": answer, "artifacts": saved_paths},
                    ensure_ascii=False,
                ) + "\n"
                yield json.dumps({"type": "done", "artifacts": saved_paths}, ensure_ascii=False) + "\n"
                return

            agent = build_agent(chat_request.config)
            agent.llm.set_logger(logger)
            state = DocVQAState(
                question=question,
                document_ids=document_ids,
                budget=BudgetState(max_steps=28, max_search_calls=6, max_extract_calls=12, max_compute_calls=4),
            )
            state.metadata["environment_overview"] = await env.overview(document_ids)
            if isinstance(agent, VLLMTagsAgent):
                agent.bridge.set_logger(logger)
                agent.messages.clear()
                if len(document_ids) > 1:
                    state.metadata["document_titles"] = {}
                    state.metadata["overview"] = False
                navigate_tool = agent.tools.tools.get("navigate")
                if bool(getattr(navigate_tool, "single_document_overview", False)):
                    state.metadata["overview"] = False
                agent._initialize_navigation(state)
                agent._append_state(state, include_question=True)

            while not state.done and state.budget.step_used < state.budget.max_steps:
                step = state.budget.step_used + 1
                agent.llm.set_step(step)
                thinking_history = [
                    *state.history,
                    {
                        "action": {
                            "action_type": "thinking",
                            "agent_role": "system",
                            "intent": "",
                            "args": {},
                        },
                        "observation": {
                            "action_type": "thinking",
                            "success": False,
                            "message": "",
                            "payload": {"phase": "thinking"},
                        },
                    },
                ]
                yield json.dumps(
                    {
                        "type": "step",
                        "step_index": len(thinking_history) - 1,
                        "history": _history_with_display_images(
                            thinking_history,
                            display_images_by_step,
                        ),
                        "state": state.to_dict(),
                    },
                    ensure_ascii=False,
                ) + "\n"

                if isinstance(agent, VLLMTagsAgent):
                    agent.bridge.set_step(step)
                    action = await run_cancellable(lambda: agent.next_action(state, env), agent)
                    expected = "solve" if action.action_type == "solver" else action.action_type
                    agent.bridge.expect(expected, "")
                else:
                    action = await run_cancellable(lambda: agent.think(state, env), agent)
                logger.log_tool_start(step, action.to_dict())
                if await request.is_disconnected():
                    raise asyncio.CancelledError()

                pending_history = [
                    *state.history,
                    {
                        "action": action.to_dict(),
                        "observation": {
                            "action_type": action.action_type,
                            "success": False,
                            "message": "",
                            "payload": {"phase": "action"},
                        },
                    },
                ]
                yield json.dumps(
                    {
                        "type": "step",
                        "step_index": len(pending_history) - 1,
                        "action": action.to_dict(),
                        "observation": pending_history[-1]["observation"],
                        "history": _history_with_display_images(pending_history, display_images_by_step),
                        "state": state.to_dict(),
                    },
                    ensure_ascii=False,
                ) + "\n"

                if isinstance(agent, VLLMTagsAgent):

                    async def run_vllm_tags_action() -> Observation:
                        result = await agent.dispatch(action.action_type, action, state, env, agent.tools)
                        if action.action_type == "solver" and result.success:
                            return await agent._execute_solver(action, result, state, env, logger, step)
                        if not agent.bridge.generated_action:
                            agent.bridge.append_environment_observation(result.message)
                        return result

                    observation = await run_cancellable(run_vllm_tags_action, agent)
                else:
                    observation = await run_cancellable(lambda: agent.act(action, state, env), agent)
                if await request.is_disconnected():
                    raise asyncio.CancelledError()
                action_reasoning = logger.pending_tool_reasoning(step)
                if action_reasoning:
                    observation.payload["reasoning_content"] = action_reasoning
                display_result = _action_display_result(action, observation, state)
                if display_result:
                    observation.payload["display_result"] = display_result
                state.add_history(action, observation)
                if isinstance(agent, VLLMTagsAgent):
                    if action.action_type == "answer" and observation.success:
                        state.done = True
                        state.sync_evidence_state()
                    if not state.done:
                        agent._append_state(state)
                else:
                    advance_react_state(agent, action, observation, state)
                display_images_by_step[len(state.history) - 1] = await _action_display_images(
                    action,
                    observation,
                    state,
                    env,
                    workspace_session_id,
                )
                latest_state = state.to_dict()
                latest_history = list(state.history)

                logger.log_result(
                    step=step,
                    result=observation.message,
                    observation=observation.to_dict(),
                )

                yield json.dumps(
                    {
                        "type": "step",
                        "step_index": len(state.history) - 1,
                        "action": action.to_dict(),
                        "observation": observation.to_dict(),
                        "history": _history_with_display_images(state.history, display_images_by_step),
                        "state": state.to_dict(),
                    },
                    ensure_ascii=False,
                ) + "\n"

                if state.done:
                    break

            if isinstance(agent, ReActAgent):
                agent._restore_latest_candidate_if_empty(state)
                await run_cancellable(lambda: agent._run_finish_if_needed(state, env, logger), agent)

            if isinstance(agent, VLLMTagsAgent) and not state.done:
                state.candidate_answer = state.candidate_answer or CandidateAnswer(
                    answer="INSUFFICIENT_EVIDENCE",
                    rationale="The vLLM tags rollout reached max_steps before a complete answer.",
                    supporting_evidence_ids=list(state.evidence_store),
                    confidence=0.0,
                )
                state.done = True
                state.sync_evidence_state()

            answer = state.candidate_answer.answer if state.candidate_answer else "INSUFFICIENT_EVIDENCE"
            response_body = answer
            latest_answer = response_body
            latest_state = state.to_dict()
            latest_history = list(state.history)
            runtime_seconds = round(time.perf_counter() - started, 3)
            saved_paths = persist_chat_run(
                run_id=run_id,
                log_path=logger.path,
                session_id=chat_request.session_id,
                conversation_id=chat_request.conversation_id,
                mode="docpilot",
                question=question,
                messages=chat_request.messages,
                selected_files=chat_request.selected_files,
                config=chat_request.config,
                answer=response_body,
                history=latest_history,
                skipped=skipped,
                runtime_seconds=runtime_seconds,
            )
            logger.info(f"Chat run finished. Result saved to: {saved_paths['result_path']}")
            yield json.dumps(
                {
                    "type": "delta",
                    "content": response_body,
                    "state": state.to_dict(),
                    "history": _history_with_display_images(state.history, display_images_by_step),
                    "artifacts": saved_paths,
                    "runtime_seconds": runtime_seconds,
                },
                ensure_ascii=False,
            ) + "\n"
            yield json.dumps(
                {"type": "done", "artifacts": saved_paths, "runtime_seconds": runtime_seconds},
                ensure_ascii=False,
            ) + "\n"
        except asyncio.CancelledError:
            abort_agent(agent)
            logger.info("Chat run stopped by user.")
            persist_chat_run(
                run_id=run_id,
                log_path=logger.path,
                session_id=chat_request.session_id,
                conversation_id=chat_request.conversation_id,
                mode=mode,
                question=question,
                messages=chat_request.messages,
                selected_files=chat_request.selected_files,
                config=chat_request.config,
                answer=latest_answer,
                history=latest_history,
                skipped=skipped,
                error="Stopped by user.",
                runtime_seconds=round(time.perf_counter() - started, 3),
            )
            return
        except Exception as exc:
            abort_agent(agent)
            runtime_seconds = round(time.perf_counter() - started, 3)
            logger.info(f"Chat run failed: {type(exc).__name__}: {exc}")
            saved_paths = persist_chat_run(
                run_id=run_id,
                log_path=logger.path,
                session_id=chat_request.session_id,
                conversation_id=chat_request.conversation_id,
                mode=mode,
                question=question,
                messages=chat_request.messages,
                selected_files=chat_request.selected_files,
                config=chat_request.config,
                answer=latest_answer,
                history=latest_history,
                skipped=skipped,
                error=str(exc),
                runtime_seconds=runtime_seconds,
            )
            yield json.dumps(
                {
                    "type": "error",
                    "message": str(exc),
                    "artifacts": saved_paths,
                    "runtime_seconds": runtime_seconds,
                },
                ensure_ascii=False,
            ) + "\n"
            return
        finally:
            logger.close()

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
