from __future__ import annotations

import json
import re

from ..llm import LLMClient
from ..multimodal import collect_image_inputs
from ..prompt.analyze import ANALYZE_SYSTEM_PROMPT
from ..schema import Action, DocVQAState, Observation, VerificationResult
from .base import BaseTool


class AnalyzeTool(BaseTool):
    name = "analyze"
    description = "Return a self-verification result for the current candidate answer."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        if state.candidate_answer is None:
            return Observation(
                action_type="analyze",
                success=False,
                message="No candidate answer available for self-verification.",
                payload={"task": "self_verification"},
            )
        result = await _verify_candidate(self.llm, env, state)
        state.verification = result
        if _should_reopen_navigation(state, result):
            navigation = state.metadata.setdefault("navigation", {})
            navigation["need_more_pages"] = True
            navigation["reopen_reason"] = "missing_page_evidence"
        state.sync_evidence_state()
        if result.accepted:
            _mark_verified_pages(state)
        elif any(action_name in result.suggestion for action_name in ["solver", "execute"]):
            state.metadata.pop("last_solver", None)
            state.metadata.pop("last_execute", None)
        summary = str(result.summary or "").strip()
        return Observation(
            action_type="analyze",
            success=result.accepted,
            message=summary
            or (
                "Candidate answer accepted by self-verification."
                if result.accepted
                else "Candidate answer rejected; continue evidence gathering."
            ),
            payload={
                "task": "self_verification",
                "verification": result.to_dict(),
            },
        )

async def _verify_candidate(llm: LLMClient | None, env, state: DocVQAState) -> VerificationResult:
    rule_result = await env.verify_answer(state.candidate_answer, state)
    consistency_result = _check_answer_rationale_consistency(state)
    merged_rule_result = _merge_verification_results(rule_result, consistency_result)
    if llm is None:
        return merged_rule_result

    evidence = []
    images: list[str] = []
    for evidence_id in state.candidate_answer.supporting_evidence_ids:
        record = state.evidence_store.get(evidence_id)
        if record is None:
            continue
        for image in collect_image_inputs(record.content, max_images=8):
            if image not in images:
                images.append(image)
        evidence.append(
            {
                "evidence_id": evidence_id,
                "page": record.page,
                "modality": record.modality,
                "content": str(record.content),
                "provenance": record.provenance,
            }
        )

    user_prompt = json.dumps(
        {
            "task": "self_verification",
            "question": state.question,
            "candidate_answer": state.candidate_answer.to_dict(),
            "supporting_evidence": evidence,
            "rule_verification": merged_rule_result.to_dict(),
            "observation_state": {
                "evidence_state": state.evidence_state,
                "navigation": state.metadata.get("navigation", {}),
                "candidate_pages": state.metadata.get("candidate_pages", [])[:12],
                "last_extract_observation": state.metadata.get("last_extract_observation", {}),
                "last_focus_observation": state.metadata.get("last_focus_observation", {}),
                "last_solver": state.metadata.get("last_solver"),
                "last_execute": state.metadata.get("last_execute"),
            },
        },
        ensure_ascii=False,
    )
    try:
        llm.set_call_source("analyze_self_verification")
        try:
            payload = llm.judge_json(
                ANALYZE_SYSTEM_PROMPT,
                user_prompt,
                images=images,
                use_reasoning=False,
            )
        finally:
            llm.set_call_source(None)
    except Exception:
        return merged_rule_result
    issues = _normalize_text_items(payload.get("issues", merged_rule_result.issues))
    missing_evidence = _normalize_text_items(payload.get("missing_evidence", merged_rule_result.missing_evidence))
    summary = str(payload.get("summary") or "").strip()
    if not bool(payload.get("result", merged_rule_result.accepted)) and issues and not summary:
        summary = "; ".join(issues)
    llm_result = VerificationResult(
        accepted=bool(payload.get("result", merged_rule_result.accepted)),
        issues=issues,
        missing_evidence=missing_evidence,
        suggestion=_normalize_suggestions(payload.get("suggestion", merged_rule_result.suggestion)),
        summary=summary,
    )
    llm_result = _filter_verification_to_supported_pages(llm_result, state)
    return _merge_verification_results(merged_rule_result, llm_result)

def _mark_verified_pages(state: DocVQAState) -> None:
    if state.candidate_answer is None:
        return
    navigation = state.metadata.setdefault("navigation", {})
    verified = navigation.setdefault("verified_pages", [])
    seen = {
        f"{item.get('doc_id') or '*'}::{item.get('page')}"
        for item in verified
        if isinstance(item, dict)
    }
    for evidence_id in state.candidate_answer.supporting_evidence_ids:
        record = state.evidence_store.get(evidence_id)
        if record is None:
            continue
        doc_id = record.provenance.get("doc_id") or record.provenance.get("metadata", {}).get("doc_id")
        key = f"{doc_id or '*'}::{record.page}"
        if key in seen:
            continue
        verified.append(
            {
                "doc_id": doc_id,
                "page": record.page,
                "evidence_id": evidence_id,
                "reason": "Supporting evidence accepted by self-verification.",
            }
        )
        seen.add(key)


def _check_answer_rationale_consistency(state: DocVQAState) -> VerificationResult:
    candidate = state.candidate_answer
    if candidate is None:
        return VerificationResult(accepted=True)

    answer = str(candidate.answer or "").strip().lower()
    rationale = str(candidate.rationale or "").strip().lower()
    issues: list[str] = []
    missing: list[str] = []
    suggestions: list[str] = []

    if not answer:
        issues.append("The candidate answer is empty.")
        suggestions.append("answer")

    uncertainty_markers = [
        "need more",
        "need to determine",
        "need the detailed",
        "necessary to determine",
        "not enough evidence",
        "insufficient evidence",
        "unclear",
        "cannot determine",
        "missing",
    ]
    if rationale and any(marker in rationale for marker in uncertainty_markers):
        if answer and answer != "insufficient_evidence":
            issues.append("The rationale says the evidence is incomplete, but the answer still makes a definite claim.")
            missing.append("A complete comparison or missing quantity is still required before answering definitively.")
            suggestions.extend(["extract", "answer"])

    comparison_markers = ["greater", "larger", "smaller", "higher", "lower", "more", "less"]
    question = state.question.lower()
    if any(marker in question for marker in comparison_markers):
        if rationale and any(marker in rationale for marker in ["necessary to determine", "need to determine", "missing number", "missing comparison"]):
            issues.append("The question is comparative, but the rationale does not establish the comparison before answering.")
            missing.append("Explicit evidence for both sides of the comparison and their ordering.")
            suggestions.extend(["extract", "answer"])

    supported_pages = _supported_evidence_pages(state)
    if supported_pages:
        unsupported_pages = sorted(_extract_page_numbers(rationale) - supported_pages)
        if unsupported_pages:
            issues.append(
                "The rationale references page numbers that are not present in the supporting evidence: "
                + ", ".join(str(page) for page in unsupported_pages)
                + "."
            )
            missing.append("A rationale that only cites page numbers explicitly present in the supporting evidence.")
            suggestions.extend(["answer"])

    accepted = not issues
    return VerificationResult(
        accepted=accepted,
        issues=issues,
        missing_evidence=missing,
        suggestion=_dedupe_preserve_order(suggestions),
    )


def _merge_verification_results(base: VerificationResult, extra: VerificationResult) -> VerificationResult:
    return VerificationResult(
        accepted=bool(base.accepted and extra.accepted),
        issues=_dedupe_preserve_order([*base.issues, *extra.issues]),
        missing_evidence=_dedupe_preserve_order([*base.missing_evidence, *extra.missing_evidence]),
        suggestion=_dedupe_preserve_order([*base.suggestion, *extra.suggestion]),
        summary=extra.summary or base.summary,
    )


def _normalize_suggestions(raw_suggestions: object) -> list[str]:
    allowed = {"navigate", "extract", "focus", "solver", "answer"}
    if isinstance(raw_suggestions, str):
        items = re.split(r"[,;>\n]+", raw_suggestions)
    elif isinstance(raw_suggestions, list):
        items = [str(item) for item in raw_suggestions]
    else:
        return []
    normalized = [item.strip().lower() for item in items]
    return _dedupe_preserve_order([item for item in normalized if item in allowed])


def _normalize_text_items(raw_items: object) -> list[str]:
    if raw_items is None:
        return []
    if isinstance(raw_items, str):
        text = raw_items.strip()
        return [text] if text else []
    if isinstance(raw_items, list):
        return [str(item).strip() for item in raw_items if str(item).strip()]
    return [str(raw_items).strip()] if str(raw_items).strip() else []


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _supported_evidence_pages(state: DocVQAState) -> set[int]:
    pages: set[int] = set()
    candidate = state.candidate_answer
    if candidate is None:
        return pages
    for evidence_id in candidate.supporting_evidence_ids:
        record = state.evidence_store.get(evidence_id)
        if record is not None:
            pages.add(int(record.page))
    return pages


def _extract_page_numbers(text: str) -> set[int]:
    if not text:
        return set()
    return {
        int(match)
        for match in re.findall(r"\bpages?\s+(\d+)\b", str(text), flags=re.IGNORECASE)
    }


def _filter_verification_to_supported_pages(
    result: VerificationResult,
    state: DocVQAState,
) -> VerificationResult:
    supported_pages = _supported_evidence_pages(state)
    if not supported_pages:
        return result

    def _keep(text: str) -> bool:
        referenced_pages = _extract_page_numbers(text)
        return not referenced_pages or referenced_pages.issubset(supported_pages)

    filtered_issues = [item for item in result.issues if _keep(item)]
    filtered_missing = [item for item in result.missing_evidence if _keep(item)]
    filtered_suggestions = [item for item in result.suggestion if _keep(item)]
    accepted = result.accepted
    if result.issues and not filtered_issues and not filtered_missing:
        accepted = True
    return VerificationResult(
        accepted=accepted,
        issues=filtered_issues,
        missing_evidence=filtered_missing,
        suggestion=filtered_suggestions,
        summary=result.summary,
    )


def _should_reopen_navigation(
    state: DocVQAState,
    result: VerificationResult,
) -> bool:
    supported_pages = _supported_evidence_pages(state)
    rationale = ""
    if state.candidate_answer is not None:
        rationale = str(state.candidate_answer.rationale or "")
    unsupported_rationale_pages = _extract_page_numbers(rationale) - supported_pages
    if unsupported_rationale_pages:
        return True

    issue_text = " ".join(result.issues).lower()
    missing_text = " ".join(result.missing_evidence).lower()
    if "page" in missing_text:
        return True
    page_issue_markers = [
        "not present in the supporting evidence",
        "unprovided page",
        "missing page evidence",
    ]
    return any(marker in issue_text for marker in page_issue_markers)
