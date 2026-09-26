from __future__ import annotations

import json
import re
from typing import Any

from ..llm import LLMClient
from ..multimodal import collect_image_inputs
from ..prompt.answer import ANSWER_SYSTEM_PROMPT
from ..schema import Action, CandidateAnswer, DocVQAState, Observation
from ..value_extraction import format_number, lookup_metric_value
from utils import compress_images_for_retry, is_request_too_large_error, prefer_original_image_refs
from .base import BaseTool


class AnswerTool(BaseTool):
    name = "answer"
    description = (
        "Generate a grounded candidate answer from the current question, an answer instruction, and the full evidence store."
    )
    parameters = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": (
                    "Ta answering instruction: which evidence to use and what "
                    "comparison, calculation, or constraint to apply. Do not ask for a bare "
                    "JSON list or raw value; this tool wraps the result in its own schema."
                ),
            },
        },
        "required": ["instruction"],
        "additionalProperties": False,
    }

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        instruction = str(action.args.get("instruction") or state.question).strip()
        evidence_ids = _resolve_answer_evidence_ids(action, state)
        candidate_pages = []
        if not evidence_ids and "extract" in state.budget.disabled_tools:
            candidate_pages = await _fetch_candidate_pages(state, env)
        default_confidence = 0.5 if evidence_ids or candidate_pages else 0.1
        execute_result = state.metadata.get("last_execute", {}).get("result")
        generation_source = "fallback"

        if self.llm is not None:
            try:
                answer_payload = _llm_answer(
                    self.llm,
                    state,
                    evidence_ids,
                    instruction,
                    candidate_pages=candidate_pages,
                )
                answer_text = _format_answer_value(answer_payload.get("answer", ""))
                rationale = str(answer_payload.get("rationale", "")).strip()
                confidence = _parse_confidence(answer_payload.get("confidence"), default_confidence)
                generation_source = "answer_generation"
            except Exception:
                if getattr(self.llm, "is_vllm_tags", False):
                    raise
                if execute_result is not None:
                    answer_text = str(execute_result)
                    rationale = instruction
                    confidence = default_confidence
                else:
                    answer_text = _heuristic_answer(state, evidence_ids)
                    rationale = instruction
                    confidence = default_confidence
        elif execute_result is not None:
            answer_text = str(execute_result)
            rationale = instruction
            confidence = default_confidence
        else:
            answer_text = _heuristic_answer(state, evidence_ids)
            rationale = instruction
            confidence = default_confidence

        candidate = CandidateAnswer(
            answer=answer_text,
            rationale=rationale
            or "Candidate answer synthesized from the currently available evidence.",
            supporting_evidence_ids=evidence_ids,
            confidence=confidence,
        )
        state.candidate_answer = candidate
        state.verification = None
        state.sync_evidence_state()
        answer_preview = candidate.answer.strip().replace("\n", " ")
        if len(answer_preview) > 240:
            answer_preview = f"{answer_preview[:237]}..."
        verification_enabled = "analyze" not in state.budget.disabled_tools
        answer_summary = {
            "answer_preview": answer_preview,
            "confidence": candidate.confidence,
            "has_execute_result": state.metadata.get("last_execute") is not None,
            "generation_source": generation_source,
        }
        if candidate_pages:
            answer_summary["candidate_page_count"] = len(candidate_pages)
        payload = {
            "candidate_answer": candidate.to_dict(),
            "question": state.question,
            "supporting_evidence_count": len(evidence_ids),
            "supporting_evidence_ids": evidence_ids,
            "answer_summary": answer_summary,
        }
        if verification_enabled:
            payload["next_task"] = {
                "action_type": "analyze",
                "reason": "Check whether the drafted answer is fully supported by the collected evidence.",
            }
        return Observation(
            action_type="answer",
            success=True,
            message=f"Candidate answer drafted{(' and queued for self-verification. ' if verification_enabled else '. ')}"
            f"Answer preview: {answer_preview or 'EMPTY_ANSWER'}",
            payload=payload,
        )


def _llm_answer(
    llm: LLMClient,
    state: DocVQAState,
    evidence_ids: list[str],
    instruction: str,
    *,
    candidate_pages: list[dict[str, Any]] | None = None,
) -> dict:
    llm.set_call_source("answer_generation")
    try:
        _log_with_run_logger(llm, f"answer_generation SYSTEM_PROMPT:\n{ANSWER_SYSTEM_PROMPT}")
        request = _build_answer_request(
            state,
            evidence_ids,
            instruction,
            prefer_original=False,
            candidate_pages=candidate_pages,
        )
        try:
            return _run_answer_generation(llm, request)
        except Exception as exc:
            if not is_request_too_large_error(exc):
                raise
            _log_with_run_logger(
                llm,
                "answer_generation request too large; retrying with original images instead of zoom/crop.",
            )
        original_request = _build_answer_request(
            state,
            evidence_ids,
            instruction,
            prefer_original=True,
            candidate_pages=candidate_pages,
        )
        try:
            return _run_answer_generation(llm, original_request)
        except Exception as exc:
            if not is_request_too_large_error(exc):
                raise
            _log_with_run_logger(
                llm,
                "answer_generation original images still too large; retrying with compressed images.",
            )
        return _run_answer_generation(
            llm,
            _AnswerRequest(
                user_prompt=original_request.user_prompt,
                images=compress_images_for_retry(original_request.images),
                summary={**original_request.summary, "compress_images": True},
            ),
        )
    finally:
        llm.set_call_source(None)


def _run_answer_generation(llm: LLMClient, request: "_AnswerRequest") -> dict:
    _log_with_run_logger(
        llm,
        "answer_generation request:\n"
        f"{json.dumps(request.summary, ensure_ascii=False)}",
    )
    _log_with_run_logger(llm, f"answer_generation user_prompt:\n{request.user_prompt}")
    response_payload = _coerce_answer_payload(
        llm.generate_json(
            ANSWER_SYSTEM_PROMPT,
            request.user_prompt,
            images=request.images,
            require_object=False,
        )
    )
    _log_with_run_logger(
        llm,
        f"answer_generation llm_response:\n{json.dumps(response_payload, ensure_ascii=False)}",
    )
    return response_payload


class _AnswerRequest:
    def __init__(self, user_prompt: str, images: list[str], summary: dict[str, Any]) -> None:
        self.user_prompt = user_prompt
        self.images = images
        self.summary = summary


def _build_answer_request(
    state: DocVQAState,
    evidence_ids: list[str],
    instruction: str,
    *,
    prefer_original: bool,
    candidate_pages: list[dict[str, Any]] | None = None,
) -> _AnswerRequest:
    evidence = []
    images: list[str] = []
    for evidence_id in evidence_ids:
        record = state.evidence_store.get(evidence_id)
        if record is None:
            continue
        image_payload = prefer_original_image_refs(record.content) if prefer_original else record.content
        for image in collect_image_inputs(image_payload, max_images=8):
            if image not in images:
                images.append(image)
        item = {
            "evidence_id": evidence_id,
            "page": record.page,
            "modality": record.modality,
            "content": record.content,
        }
        if len(state.document_ids) > 1:
            doc_id = str(record.provenance.get("doc_id") or "").strip()
            item["doc_id"] = doc_id
            item["source"] = f"{doc_id}:{record.page}"
        evidence.append(item)
    direct_pages = []
    for page in candidate_pages or []:
        image_payload = prefer_original_image_refs(page) if prefer_original else page
        for image in collect_image_inputs(image_payload, max_images=1):
            if image not in images:
                images.append(image)
        direct_page = {"page": page.get("page")}
        if page.get("doc_id"):
            direct_page["doc_id"] = page["doc_id"]
        # Navigate-only ablation answers from the page screenshot, not MinerU OCR.
        # direct_page["text"] = str(page.get("text") or "")[:2_200]
        direct_pages.append(direct_page)
    prompt_payload = {
        "question": state.question,
        "instruction": instruction,
        "execute_result": state.metadata.get("last_execute"),
        "evidence": evidence,
    }
    summary = {
        "evidence_count": len(evidence),
        "prefer_original": prefer_original,
        "candidate_images": len(images),
    }
    if direct_pages:
        prompt_payload["candidate_pages"] = direct_pages
        summary["candidate_pages"] = len(direct_pages)
    return _AnswerRequest(
        user_prompt=json.dumps(prompt_payload, ensure_ascii=False),
        images=images,
        summary=summary,
    )


async def _fetch_candidate_pages(state: DocVQAState, env) -> list[dict[str, Any]]:
    if env is None or not hasattr(env, "fetch_pages"):
        return []
    pages = []
    for candidate in state.metadata.get("candidate_pages", [])[:8]:
        if not isinstance(candidate, dict) or candidate.get("page") is None:
            continue
        doc_id = str(candidate.get("doc_id") or "").strip()
        try:
            result = await env.fetch_pages(
                [int(candidate["page"])],
                document_ids=[doc_id] if doc_id else state.document_ids,
                with_image=True,
            )
        except Exception:
            continue
        # Do not pass the complete fetch_pages payload: it contains OCR text and nodes.
        # pages.extend(item for item in result.get("pages", []) if isinstance(item, dict))
        for item in result.get("pages", []):
            if not isinstance(item, dict) or not item.get("screenshot_path"):
                continue
            page = {
                "page": item.get("page"),
                "screenshot_path": item["screenshot_path"],
            }
            if len(state.document_ids) > 1 and item.get("doc_id"):
                page["doc_id"] = item["doc_id"]
            pages.append(page)
    return pages


def _prefer_original_image_refs(content: Any) -> Any:
    return prefer_original_image_refs(content)


def _is_request_too_large_error(error: Exception) -> bool:
    return is_request_too_large_error(error)


def _resolve_answer_evidence_ids(action: Action, state: DocVQAState) -> list[str]:
    return list(state.evidence_store.keys())


def _log_with_run_logger(llm: LLMClient, message: str) -> None:
    run_logger = getattr(llm, "logger", None)
    if run_logger is not None:
        run_logger.info(message)


def _coerce_answer_payload(payload) -> dict:
    if isinstance(payload, dict):
        return payload
    return {"answer": payload}


def _format_answer_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def _parse_confidence(value, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    text = str(value).strip()
    if not text:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return default
    parsed = float(match.group(0))
    if parsed > 1.0:
        parsed = parsed / 100.0 if parsed <= 100 else default
    return max(0.0, min(float(parsed), 1.0))


def _heuristic_answer(state: DocVQAState, evidence_ids: list[str]) -> str:
    if not evidence_ids:
        return "INSUFFICIENT_EVIDENCE"

    question = state.question.lower()
    evidence_contents = []
    evidence_text = []
    for evidence_id in evidence_ids:
        evidence = state.evidence_store[evidence_id]
        evidence_contents.append(evidence.content)
        evidence_text.append(str(evidence.content))
    joined = "\n".join(evidence_text)

    if "long-term debt to total liabilities" in question:
        long_term_debt = lookup_metric_value(
            evidence_contents,
            ["long-term debt, excluding current portion"],
        )
        total_liabilities = lookup_metric_value(evidence_contents, ["total liabilities"])
        if long_term_debt is not None and total_liabilities:
            ratio = float(long_term_debt) / float(total_liabilities)
            return f"{ratio:.2f}"

    if "current liabilities to total liabilities" in question:
        current_liabilities = lookup_metric_value(
            evidence_contents,
            ["total current liabilities"],
        )
        total_liabilities = lookup_metric_value(evidence_contents, ["total liabilities"])
        if current_liabilities is not None and total_liabilities:
            ratio = float(current_liabilities) / float(total_liabilities)
            return f"{ratio:.3f}"

    if "operating cash flow to current liabilities" in question:
        operating_cash_flow = lookup_metric_value(
            evidence_contents,
            ["net cash provided by operating activities"],
        )
        current_liabilities = lookup_metric_value(
            evidence_contents,
            ["total current liabilities"],
        )
        if operating_cash_flow is not None and current_liabilities:
            ratio = float(operating_cash_flow) / float(current_liabilities)
            return f"{ratio:.3f}"

    if "membership warehouses" in question:
        match = re.search(r"operated\s+(\d[\d,]*)\s+membership warehouses", joined, flags=re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")

    if "common equity" in question:
        equity = lookup_metric_value(
            evidence_contents,
            ["total equity", "total costco stockholders equity"],
        )
        if equity is not None:
            return format_number(equity)

    if "total liabilities" in question:
        liabilities = lookup_metric_value(evidence_contents, ["total liabilities"])
        if liabilities is not None:
            return format_number(liabilities)

    target = _extract_target_phrase(question)
    if target:
        value = lookup_metric_value(evidence_contents, [target])
        if value is not None:
            return format_number(value)
        match = _find_value_after_label(joined, target)
        if match is not None:
            return match

    first = state.evidence_store[evidence_ids[0]]
    if isinstance(first.content, dict):
        return first.content.get("caption") or first.content.get("footnote") or "INSUFFICIENT_EVIDENCE"
    return str(first.content)


def _extract_target_phrase(question: str) -> str | None:
    question = question.replace("?", " ").strip()
    patterns = [
        r"what is (.+?) for costco in fy ?2021",
        r"what is (.+?) for costco in 2021",
        r"how many (.+?) did costco operate at 2021",
        r"what is the filing date for the '(.+?)' exhibit",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()
    return None


def _find_value_after_label(text: str, label: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text)
    label_pattern = re.escape(label).replace(r"\ ", r"\s+")
    match = re.search(label_pattern + r".{0,120}?(-?\d[\d,]*(?:\.\d+)?)", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    return None


def _collect_number_candidates(text: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        value = float(match.replace(",", ""))
        if 1900 <= value <= 2100 and value.is_integer():
            continue
        values.append(value)
    return values
