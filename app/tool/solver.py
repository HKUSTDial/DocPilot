from __future__ import annotations

import json
from typing import Any

from ..llm import LLMClient
from ..multimodal import collect_image_inputs
from ..prompt.solver import SOLVER_SYSTEM_PROMPT
from ..schema import Action, DocVQAState, Observation
from utils import compress_images_for_retry, is_request_too_large_error, prefer_original_image_refs
from .answer import _log_with_run_logger
from .base import BaseTool


class SolverTool(BaseTool):
    name = "solver"
    description = (
        "Generate Python code when the target information is in local structured data, "
        "especially tables extracted from images or markdown, with explicit rows, columns, "
        "cells, or numeric records. It may also handle a small local text snippet when the "
        "needed value or pattern is explicit. Use it for deterministic filtering, aggregation, "
        "comparison, ranking, or calculation over that local evidence."
    )
    parameters = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "Optional instruction for how the solver should approach code generation.",
            },
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of evidence ids to focus on while generating code.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        evidence_ids = action.args.get("evidence_ids") or list(state.evidence_store.keys())
        if self.llm is None:
            state.budget.compute_used += 1
            payload = {"code": "", "inputs": {}, "error": "missing_llm_client"}
            state.metadata["last_solver"] = payload
            state.metadata.pop("last_execute", None)
            return Observation(
                action_type="solver",
                success=False,
                message="Solver requires an LLM client for code generation.",
                payload=payload,
            )

        try:
            payload = _llm_solver_payload(
                self.llm,
                state,
                evidence_ids,
                action.args.get("instruction"),
            )
        except Exception as exc:
            payload = {"code": "", "inputs": {}, "error": str(exc)}

        state.budget.compute_used += 1
        state.metadata["last_solver"] = payload
        state.metadata.pop("last_execute", None)
        summary = str(payload.get("summary") or "").strip()
        message = (
            f"Solver generated Python code; next action should execute it. Summary: {summary}"
            if payload.get("code")
            else "Solver failed to generate Python code."
        )
        return Observation(
            action_type="solver",
            success=bool(payload.get("code")),
            message=message,
            payload=payload,
        )


def _llm_solver_payload(
    llm: LLMClient,
    state: DocVQAState,
    evidence_ids: list[str],
    instruction: str | None,
) -> dict | None:
    llm.set_call_source("solver_generation")
    try:
        request = _build_solver_request(
            state,
            evidence_ids,
            instruction,
            prefer_original=False,
        )
        try:
            payload = _run_solver_generation(llm, request)
        except Exception as exc:
            if not is_request_too_large_error(exc):
                raise
            _log_with_run_logger(
                llm,
                "solver_generation request too large; retrying with original images instead of zoom/crop.",
            )
            original_request = _build_solver_request(
                state,
                evidence_ids,
                instruction,
                prefer_original=True,
            )
            try:
                payload = _run_solver_generation(llm, original_request)
            except Exception as original_exc:
                if not is_request_too_large_error(original_exc):
                    raise
                _log_with_run_logger(
                    llm,
                    "solver_generation original images still too large; retrying with compressed images.",
                )
                payload = _run_solver_generation(
                    llm,
                    _SolverRequest(
                        user_prompt=original_request.user_prompt,
                        images=compress_images_for_retry(original_request.images),
                        summary={**original_request.summary, "compress_images": True},
                    ),
                )
    finally:
        llm.set_call_source(None)

    code = payload.get("code")
    inputs = payload.get("inputs")
    summary = str(payload.get("summary") or "").strip()
    if isinstance(code, str) and isinstance(inputs, dict):
        return {"summary": summary, "code": code, "inputs": inputs}
    return {"summary": summary, "code": "", "inputs": {}, "error": "invalid_solver_payload"}


class _SolverRequest:
    def __init__(self, user_prompt: str, images: list[str], summary: dict[str, Any]) -> None:
        self.user_prompt = user_prompt
        self.images = images
        self.summary = summary


def _build_solver_request(
    state: DocVQAState,
    evidence_ids: list[str],
    instruction: str | None,
    *,
    prefer_original: bool,
) -> _SolverRequest:
    evidence = {}
    images: list[str] = []
    for evidence_id in evidence_ids:
        record = state.evidence_store.get(evidence_id)
        if record is None:
            continue
        image_payload = prefer_original_image_refs(record.content) if prefer_original else record.content
        for image in collect_image_inputs(image_payload, max_images=8):
            if image not in images:
                images.append(image)
        evidence[evidence_id] = {
            "page": record.page,
            "modality": record.modality,
            "content": str(record.content),
            "provenance": record.provenance,
        }
    user_prompt = json.dumps(
        {
            "question": state.question,
            "instruction": instruction,
            "evidence": evidence,
        },
        ensure_ascii=False,
    )
    return _SolverRequest(
        user_prompt=user_prompt,
        images=images,
        summary={
            "evidence_count": len(evidence),
            "prefer_original": prefer_original,
            "candidate_images": len(images),
        },
    )


def _run_solver_generation(llm: LLMClient, request: _SolverRequest) -> dict:
    _log_with_run_logger(
        llm,
        "solver_generation request:\n"
        f"{json.dumps(request.summary, ensure_ascii=False)}",
    )
    return llm.generate_json(
        SOLVER_SYSTEM_PROMPT,
        request.user_prompt,
        images=request.images,
    )
