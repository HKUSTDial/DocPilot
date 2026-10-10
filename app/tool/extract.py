from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2

from ..config import ExtractConfig
from ..llm import LLMClient
from ..prompt.extract import EXTRACT_SYSTEM_PROMPT
from ..schema import Action, DocVQAState, EvidenceRecord, Observation
from .base import BaseTool


_VISUAL_TERMS = [
    "chart",
    "figure",
    "plot",
    "legend",
    "color",
    "axis",
    "label",
    "visual",
    "map",
    "panel",
    "source line",
    "source note",
    "screenshot",
    "image",
]

_TABLE_TERMS = [
    "table",
    "row",
    "column",
    "cell",
    "tabular",
    "total",
    "rank",
    "ranking",
]


class _BaseExtractTool(BaseTool):
    extract_kind: str = ""

    def _make_evidence(
        self,
        state: DocVQAState,
        payload: dict,
        suffix: str,
    ) -> EvidenceRecord:
        evidence_id = f"{payload['node_id']}::{suffix}::{len(state.evidence_store) + 1}"
        provenance = dict(payload["provenance"])
        provenance.setdefault("doc_id", payload.get("doc_id"))
        return EvidenceRecord(
            evidence_id=evidence_id,
            node_id=payload["node_id"],
            page=payload["page"],
            modality=payload["modality"],
            content=payload["content"],
            provenance=provenance,
            confidence=float(payload.get("confidence", 1.0)),
        )

    def _finalize(self, state: DocVQAState, evidence: EvidenceRecord, *, count_budget: bool = True) -> None:
        state.add_evidence(evidence)
        state.metadata.pop("last_solver", None)
        state.metadata.pop("last_execute", None)
        state.metadata.setdefault("understood_evidence_ids", [])
        if count_budget:
            state.budget.extract_used += 1


class ExtractTextTool(_BaseExtractTool):
    name = "extract_text"
    description = "Extract local textual evidence from a target node, such as paragraphs, captions, definitions, notes, or nearby context needed to ground the answer."
    extract_kind = "text"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "Target node id to extract text from."},
            "span": {"type": "object", "description": "Optional text span or region selector."},
        },
        "required": ["node_id"],
        "additionalProperties": False,
    }

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        payload = await env.extract_text(action.args["node_id"], span=action.args.get("span"))
        evidence = self._make_evidence(state, payload, "text")
        self._finalize(state, evidence, count_budget=bool(action.args.get("_count_budget", True)))
        return Observation(
            action_type=action.action_type,
            success=True,
            message=f"Extracted text evidence from {payload['node_id']}.",
            payload={"tool_name": self.name, "evidence": evidence.to_dict()},
        )


class ExtractTableTool(_BaseExtractTool):
    name = "extract_table"
    description = "Extract table evidence from a target node as markdown-style structured table content."
    extract_kind = "table"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "Target table node id."},
            "cell_query": {"type": "object", "description": "Optional cell targeting query."},
        },
        "required": ["node_id"],
        "additionalProperties": False,
    }

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        payload = await env.extract_table(
            action.args["node_id"],
            cell_query=action.args.get("cell_query"),
        )
        evidence = self._make_evidence(state, payload, "table")
        self._finalize(state, evidence, count_budget=bool(action.args.get("_count_budget", True)))
        return Observation(
            action_type=action.action_type,
            success=True,
            message=f"Extracted table evidence from {payload['node_id']}.",
            payload={"tool_name": self.name, "evidence": evidence.to_dict()},
        )


class ExtractFigureTool(_BaseExtractTool):
    name = "extract_figure"
    description = "Extract figure evidence from a target node."
    extract_kind = "image"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "Target figure node id."},
            "target": {"type": "object", "description": "Optional figure sub-target specification."},
        },
        "required": ["node_id"],
        "additionalProperties": False,
    }

    def __init__(self, if_enhanced: bool = True) -> None:
        self.if_enhanced = bool(if_enhanced)

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        payload = await env.extract_figure(
            action.args["node_id"],
            target=action.args.get("target"),
        )
        payload = _enrich_figure_payload_with_node_context(payload, env)
        if self.if_enhanced:
            payload = _enhance_figure_payload_with_resize(payload)
        evidence = self._make_evidence(state, payload, "figure")
        self._finalize(state, evidence, count_budget=bool(action.args.get("_count_budget", True)))
        return Observation(
            action_type=action.action_type,
            success=True,
            message=f"Extracted figure evidence from {payload['node_id']}.",
            payload={"tool_name": self.name, "evidence": evidence.to_dict()},
        )


class CropTool(_BaseExtractTool):
    name = "crop"
    description = "Crop a page or region using a bounding box."
    extract_kind = "image"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "Target node id for cropping."},
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Optional bounding box [x1, y1, x2, y2].",
            },
        },
        "required": ["node_id"],
        "additionalProperties": False,
    }

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        payload = await env.crop(action.args["node_id"], bbox=action.args.get("bbox"))
        evidence = self._make_evidence(state, payload, "crop")
        self._finalize(state, evidence, count_budget=bool(action.args.get("_count_budget", True)))
        return Observation(
            action_type=action.action_type,
            success=True,
            message=f"Cropped region for {payload['node_id']}.",
            payload={"tool_name": self.name, "evidence": evidence.to_dict()},
        )


class ZoomTool(_BaseExtractTool):
    name = "zoom"
    description = "Zoom into a page or cropped region."
    extract_kind = "image"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "Target node id for zooming."},
            "scale": {"type": "number", "description": "Zoom scale factor."},
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Optional bounding box [x1, y1, x2, y2].",
            },
        },
        "required": ["node_id"],
        "additionalProperties": False,
    }

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        payload = await env.zoom(
            action.args["node_id"],
            scale=float(action.args.get("scale", 2.0)),
            bbox=action.args.get("bbox"),
        )
        evidence = self._make_evidence(state, payload, "zoom")
        self._finalize(state, evidence, count_budget=bool(action.args.get("_count_budget", True)))
        return Observation(
            action_type=action.action_type,
            success=True,
            message=f"Zoomed into region for {payload['node_id']}.",
            payload={"tool_name": self.name, "evidence": evidence.to_dict()},
        )


class SubfigureSplitTool(_BaseExtractTool):
    name = "subfigure_split"
    description = (
        "Split a multi-panel figure into candidate subfigures, or crop one matching subfigure "
        "when panel_query is provided."
    )
    extract_kind = "image"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "Target figure node id."},
            "panel_query": {
                "type": "string",
                "description": (
                    "Optional panel description such as 'panel B', 'top-right panel', "
                    "or 'row 2 column 1'. When provided, returns the matching cropped subfigure."
                ),
            },
            "grid_hint": {
                "type": "object",
                "description": "Optional layout hint such as rows, columns, or expected panel labels.",
            },
            "split_hint": {
                "type": "object",
                "description": "Alias for grid_hint, kept for compatibility with older subfigure crop calls.",
            },
        },
        "required": ["node_id"],
        "additionalProperties": False,
    }

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        panel_query = str(action.args.get("panel_query") or "").strip()
        if panel_query:
            payload = await env.subfigure_crop(
                action.args["node_id"],
                panel_query=panel_query,
                split_hint=action.args.get("split_hint") or action.args.get("grid_hint"),
            )
            suffix = "subfigure_split"
            message = f"Split and cropped subfigure for {payload['node_id']} with query '{panel_query}'."
        else:
            payload = await env.subfigure_split(
                action.args["node_id"],
                grid_hint=action.args.get("grid_hint") or action.args.get("split_hint"),
            )
            suffix = "subfigure_split"
            message = f"Split subfigures for {payload['node_id']}."
        evidence = self._make_evidence(state, payload, suffix)
        self._finalize(state, evidence, count_budget=bool(action.args.get("_count_budget", True)))
        return Observation(
            action_type=action.action_type,
            success=True,
            message=message,
            payload={"tool_name": self.name, "evidence": evidence.to_dict()},
        )


class ExtractTool(BaseTool):
    name = "extract"
    description = (
        "Select question-relevant text, table, and figure candidates from candidate pages, "
        "then extract only those sources into the evidence store."
    )
    parameters = {
        "type": "object",
        "properties": {
            "candidate_pages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                        "doc_id": {"type": "string"},
                    },
                    "required": ["page"],
                },
            },
            "pages": {
                "type": "string",
                "description": "Optional page list or ranges, e.g. 1,3-5.",
            },
            "instruction": {
                "type": "string",
                "description": (
                    "Extraction goal. Ask for all query-relevant nodes/evidence across text, "
                    "tables, figures, charts, captions, source notes, and nearby context needed "
                    "to answer the question; do not restrict to one modality unless the task clearly requires it."
                ),
            },
            "max_pages": {"type": "integer"},
        },
        "additionalProperties": False,
    }

    def __init__(self, llm_client: LLMClient | None = None, extract_config: ExtractConfig | None = None) -> None:
        self.llm = llm_client
        self.extract_config = extract_config or ExtractConfig()
        self._subtools: dict[str, _BaseExtractTool] = {
            "extract_text": ExtractTextTool(),
            "extract_table": ExtractTableTool(),
            "extract_figure": ExtractFigureTool(if_enhanced=bool(self.extract_config.if_enhanced)),
        }

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        requested_candidates = _resolve_candidate_pages(action.args, state)
        candidates = _resolve_candidate_documents(requested_candidates, state, env)
        if not candidates:
            return Observation(
                action_type="extract",
                success=False,
                message=(
                    "Extract candidate pages do not identify a unique document; provide a valid doc_id."
                    if requested_candidates
                    else "Extract requires candidate pages or pages."
                ),
                payload={
                    "error": "ambiguous_extract_target" if requested_candidates else "missing_extract_target"
                },
            )

        default_max_pages = self.extract_config.max_pages or len(candidates)
        max_pages = max(1, int(action.args.get("max_pages") or default_max_pages))
        selected_candidates = candidates[:max_pages]
        state.budget.extract_used += 1

        evidence_ids: list[str] = []
        figure_refs: list[dict[str, Any]] = []
        operations: list[dict[str, Any]] = []
        pages: list[int] = []
        selected_candidate_ids: list[str] = []
        candidate_selection_decisions: list[dict[str, Any]] = []
        extracted_page_keys = state.metadata.setdefault("extracted_page_keys", [])

        prepared_pages: list[dict[str, Any]] = []
        all_page_candidates: list[dict[str, Any]] = []
        for candidate in selected_candidates:
            page_nodes = _nodes_for_candidate_page(env, candidate)
            if not page_nodes:
                continue
            page = int(candidate["page"])
            page_key = _candidate_page_key(candidate)
            pages.append(page)
            page_context = await _fetch_page_context(env, candidate)
            page_candidates = _build_page_candidates(
                candidate=candidate,
                page_nodes=page_nodes,
                page_context=page_context,
            )
            prepared_pages.append(
                {
                    "candidate": candidate,
                    "page": page,
                    "page_key": page_key,
                    "page_nodes": page_nodes,
                    "page_context": page_context,
                    "page_candidates": page_candidates,
                }
            )
            all_page_candidates.extend(page_candidates)

        merged_candidate = {
            "reason": " ".join(
                str(item["candidate"].get("reason") or "") for item in prepared_pages
            ).strip()
        }
        selected = self._select_relevant_candidates(
            action=action,
            state=state,
            candidate=merged_candidate,
            page_candidates=all_page_candidates,
        )
        selected_ids = set(selected["selected_candidate_ids"])

        for prepared_page in prepared_pages:
            candidate = prepared_page["candidate"]
            page = prepared_page["page"]
            page_key = prepared_page["page_key"]
            page_nodes = prepared_page["page_nodes"]
            page_context = prepared_page["page_context"]
            page_candidates = prepared_page["page_candidates"]
            selected_candidates_for_page = [
                item for item in page_candidates if item["candidate_id"] in selected_ids
            ]
            selected_candidate_ids.extend(item["candidate_id"] for item in selected_candidates_for_page)
            candidate_selection_decisions.append(
                {
                    "page": page,
                    "selected_candidate_ids": [item["candidate_id"] for item in selected_candidates_for_page],
                    "reason": selected["reason"],
                }
            )

            for selected_candidate in selected_candidates_for_page:
                kind = selected_candidate["kind"]
                if kind == "text":
                    observation = await self._run_subtool(
                        "extract_text",
                        {"node_id": selected_candidate["node_id"], "instruction": action.args.get("instruction")},
                        state,
                        env,
                    )
                    evidence = observation.payload.get("evidence", {})
                    evidence_id = evidence.get("evidence_id")
                    if evidence_id:
                        evidence_ids.append(evidence_id)
                    operations.append(
                        {
                            "tool": "extract_text",
                            "page": page,
                            "candidate_id": selected_candidate["candidate_id"],
                            "node_id": selected_candidate["node_id"],
                            "success": observation.success,
                            "evidence_id": evidence_id,
                        }
                    )
                    continue
                if kind == "table":
                    observation = await self._run_subtool(
                        "extract_table",
                        {"node_id": selected_candidate["node_id"], "instruction": action.args.get("instruction")},
                        state,
                        env,
                    )
                    evidence = observation.payload.get("evidence", {})
                    evidence_id = evidence.get("evidence_id")
                    if evidence_id:
                        evidence_ids.append(evidence_id)
                    operations.append(
                        {
                            "tool": "extract_table",
                            "page": page,
                            "candidate_id": selected_candidate["candidate_id"],
                            "node_id": selected_candidate["node_id"],
                            "success": observation.success,
                            "evidence_id": evidence_id,
                        }
                    )
                    continue
                if kind == "image":
                    figure_node = _find_figure_node(page_nodes, selected_candidate.get("node_id"))
                    if figure_node is not None:
                        observation = await self._run_subtool(
                            "extract_figure",
                            {"node_id": figure_node.node_id, "instruction": action.args.get("instruction")},
                            state,
                            env,
                        )
                        evidence = observation.payload.get("evidence", {})
                        evidence_id = evidence.get("evidence_id")
                        if evidence_id:
                            evidence_ids.append(evidence_id)
                            figure_refs.append(
                                _build_figure_ref(
                                    candidate=candidate,
                                    node=figure_node,
                                    evidence=evidence,
                                    index=len(figure_refs) + 1,
                                )
                            )
                        operations.append(
                            {
                                "tool": "extract_figure",
                                "page": page,
                                "candidate_id": selected_candidate["candidate_id"],
                                "node_id": figure_node.node_id,
                                "success": observation.success,
                                "evidence_id": evidence_id,
                            }
                        )
                        continue

                    page_visual = _build_page_screenshot_figure_evidence(
                        state=state,
                        candidate=candidate,
                        page_nodes=page_nodes,
                        page_context=page_context,
                    )
                    if page_visual is None:
                        operations.append(
                            {
                                "tool": "page_screenshot",
                                "page": page,
                                "candidate_id": selected_candidate["candidate_id"],
                                "node_id": selected_candidate.get("node_id"),
                                "success": False,
                                "evidence_id": None,
                            }
                        )
                        continue
                    state.add_evidence(page_visual)
                    state.metadata.pop("last_solver", None)
                    state.metadata.pop("last_execute", None)
                    state.metadata.setdefault("understood_evidence_ids", [])
                    evidence_ids.append(page_visual.evidence_id)
                    figure_refs.append(
                        _build_page_screenshot_figure_ref(
                            candidate=candidate,
                            evidence=page_visual,
                        )
                    )
                    operations.append(
                        {
                            "tool": "page_screenshot",
                            "page": page,
                            "candidate_id": selected_candidate["candidate_id"],
                            "node_id": page_visual.node_id,
                            "success": True,
                            "evidence_id": page_visual.evidence_id,
                        }
                    )
                    continue

            if page_key not in extracted_page_keys:
                extracted_page_keys.append(page_key)

        payload = {
            "mode": "page_pipeline",
            "pages": pages,
            "evidence_ids": evidence_ids,
            "figure_refs": figure_refs,
            "selected_candidate_ids": selected_candidate_ids,
            "candidate_selection_decisions": candidate_selection_decisions,
            "operations": operations,
        }
        state.metadata["last_extract_observation"] = payload
        return Observation(
            action_type="extract",
            success=bool(evidence_ids),
            message=f"Extracted targeted evidence from {len(pages)} page(s).",
            payload=payload,
        )

    async def _run_subtool(
        self,
        tool_name: str,
        args: dict[str, Any],
        state: DocVQAState,
        env,
    ) -> Observation:
        action = Action(
            action_type="extract",
            agent_role="extract",
            intent=str(args.get("instruction") or tool_name),
            args={**args, "_count_budget": False},
        )
        return await self._subtools[tool_name].execute(action, state, env)

    def _select_relevant_candidates(
        self,
        *,
        action: Action,
        state: DocVQAState,
        candidate: dict[str, Any],
        page_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not page_candidates:
            return {"selected_candidate_ids": [], "reason": "No page candidates were available."}

        search_query = str(action.args.get("instruction") or action.intent or state.question)
        prompt_text = _render_candidates_for_llm(
            search_query=search_query,
            page_candidates=page_candidates,
        )
        valid_ids = {item["candidate_id"] for item in page_candidates}
        if self.llm is None:
            return _heuristic_select_candidates(
                action=action,
                state=state,
                candidate=candidate,
                page_candidates=page_candidates,
                reason="LLM unavailable; selected candidates with heuristic matching.",
            )

        self.llm.set_call_source("extract_candidate_selector")
        try:
            images = None
            if getattr(self.llm, "is_vllm_tags", False):
                images = [
                    str(item["image_path"])
                    for item in page_candidates
                    if item.get("image_path")
                ]
            payload = self.llm.generate_json(
                EXTRACT_SYSTEM_PROMPT,
                prompt_text,
                images=images,
            )
        finally:
            self.llm.set_call_source(None)

        selected_raw = payload.get("selected_candidate_ids")
        if not isinstance(selected_raw, list):
            return _heuristic_select_candidates(
                action=action,
                state=state,
                candidate=candidate,
                page_candidates=page_candidates,
                reason="Invalid LLM candidate selection payload; selected candidates with heuristic matching.",
            )

        selected_ids: list[str] = []
        for item in selected_raw:
            candidate_id = str(item or "").strip()
            if candidate_id in valid_ids and candidate_id not in selected_ids:
                selected_ids.append(candidate_id)
        return {
            "selected_candidate_ids": selected_ids,
            "reason": str(payload.get("reason") or "Selected relevant page candidates."),
        }


def _resolve_candidate_pages(args: dict[str, Any], state: DocVQAState) -> list[dict[str, Any]]:
    candidate_pages = args.get("candidate_pages")
    if isinstance(candidate_pages, list) and candidate_pages:
        candidates = [item for value in candidate_pages if (item := _normalize_candidate_page(value))]
    else:
        candidates = [
            item
            for value in state.metadata.get("candidate_pages", [])
            if (item := _normalize_candidate_page(value))
        ]

    explicit_targets = _parse_page_targets_arg(args.get("pages"))
    if explicit_targets:
        selected = []
        for target in explicit_targets:
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if int(candidate.get("page")) == target["page"]
                    and (not target.get("doc_id") or candidate.get("doc_id") == target["doc_id"])
                ),
                {},
            )
            selected.append(
                {
                    "page": target["page"],
                    "doc_id": target.get("doc_id") or match.get("doc_id"),
                    "confidence": match.get("confidence"),
                    "reason": match.get("reason") or "Explicit page selection.",
                }
            )
        candidates = selected

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for item in candidates:
        key = _candidate_page_key(item)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def _normalize_candidate_page(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and value.get("page") is not None:
        item = dict(value)
        if isinstance(item.get("page"), str) and ":" in item["page"]:
            parsed = _parse_page_targets_arg(item["page"])
            if not parsed:
                return None
            item.update(parsed[0])
        try:
            item["page"] = int(item["page"])
        except (TypeError, ValueError):
            return None
        return item
    if isinstance(value, str):
        parsed = _parse_page_targets_arg(value)
        return parsed[0] if len(parsed) == 1 else None
    return None


def _parse_page_targets_arg(raw_pages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_pages, str) or not raw_pages.strip():
        return []
    targets: list[dict[str, Any]] = []
    for chunk in raw_pages.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        doc_id = None
        page_text = chunk
        if ":" in chunk:
            doc_id, page_text = chunk.rsplit(":", 1)
            doc_id = doc_id.strip() or None
        for page in _parse_pages_arg(page_text):
            targets.append({"doc_id": doc_id, "page": page})
    return targets


def _parse_pages_arg(raw_pages: Any) -> list[int]:
    if not isinstance(raw_pages, str) or not raw_pages.strip():
        return []
    pages: set[int] = set()
    for chunk in raw_pages.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_raw, end_raw = chunk.split("-", 1)
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError:
                continue
            for page in range(min(start, end), max(start, end) + 1):
                pages.add(page)
            continue
        try:
            pages.add(int(chunk))
        except ValueError:
            continue
    return sorted(pages)


def _candidate_page_key(candidate_page: dict[str, Any]) -> str:
    return f"{candidate_page.get('doc_id') or '*'}::{candidate_page.get('page')}"


def _resolve_candidate_documents(
    candidates: list[dict[str, Any]], state: DocVQAState, env
) -> list[dict[str, Any]]:
    env_nodes = getattr(env, "nodes", None)
    if not isinstance(env_nodes, dict):
        return candidates

    document_ids = set(state.document_ids) or {node.doc_id for node in env_nodes.values()}
    page_to_doc_ids: dict[int, set[str]] = {}
    for node in env_nodes.values():
        if node.doc_id in document_ids:
            page_to_doc_ids.setdefault(node.page, set()).add(node.doc_id)

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        item = dict(candidate)
        try:
            page = int(item["page"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(document_ids) == 1:
            item["doc_id"] = next(iter(document_ids))
        else:
            page_doc_ids = page_to_doc_ids.get(page, set())
            if item.get("doc_id") not in page_doc_ids:
                if len(page_doc_ids) != 1:
                    continue
                item["doc_id"] = next(iter(page_doc_ids))
        key = _candidate_page_key(item)
        if key not in seen:
            seen.add(key)
            resolved.append(item)
    return resolved


def _nodes_for_candidate_page(env, candidate: dict[str, Any]) -> list[Any]:
    env_nodes = getattr(env, "nodes", None)
    if not isinstance(env_nodes, dict):
        return []
    try:
        page = int(candidate.get("page"))
    except (TypeError, ValueError):
        return []
    doc_id = candidate.get("doc_id")
    same_page = [
        node
        for node in env_nodes.values()
        if node.page == page and (not doc_id or node.doc_id == doc_id)
    ]
    return sorted(same_page, key=lambda node: ((node.bbox or [0, 0])[1], (node.bbox or [0, 0])[0], node.node_id))


async def _fetch_page_context(env, candidate: dict[str, Any]) -> dict[str, Any] | None:
    page = candidate.get("page")
    if page is None or not hasattr(env, "fetch_pages"):
        return None
    doc_id = candidate.get("doc_id")
    try:
        result = await env.fetch_pages(
            [int(page)],
            document_ids=[doc_id] if doc_id else None,
            with_image=True,
            zoom=1.0,
        )
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    pages = result.get("pages")
    if not isinstance(pages, list):
        return None
    return next((item for item in pages if isinstance(item, dict) and int(item.get("page", -1)) == int(page)), None)


def _build_page_candidates(
    *,
    candidate: dict[str, Any],
    page_nodes: list[Any],
    page_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    page = int(candidate["page"])
    candidates: list[dict[str, Any]] = []
    for node in _text_nodes(page_nodes):
        preview = _preview_node_text(node)
        if not preview:
            continue
        candidates.append(
            {
                "candidate_id": f"page{page}::text::{_candidate_node_suffix(node.node_id)}",
                "kind": "text",
                "node_id": node.node_id,
                "page": page,
                "title": node.title or "",
                "summary": node.summary or "",
                "text_preview": preview,
                "bbox": node.bbox,
            }
        )
    for node in _table_nodes(page_nodes):
        preview = _preview_node_text(node)
        candidates.append(
            {
                "candidate_id": f"page{page}::table::{_candidate_node_suffix(node.node_id)}",
                "kind": "table",
                "node_id": node.node_id,
                "page": page,
                "title": node.title or "",
                "summary": node.summary or "",
                "text_preview": preview,
                "bbox": node.bbox,
            }
        )
    for node in _figure_nodes(page_nodes):
        candidates.append(
            {
                "candidate_id": f"page{page}::figure::{_candidate_node_suffix(node.node_id)}",
                "kind": "image",
                "node_id": node.node_id,
                "page": page,
                "title": node.title or "",
                "summary": node.summary or "",
                "text_preview": _preview_node_text(node),
                "image_path": _pick_image_path(node.content),
                "bbox": node.bbox,
                "source": "figure_node",
            }
        )
    screenshot_path = page_context.get("screenshot_path") if isinstance(page_context, dict) else None
    if screenshot_path:
        anchor_node = _page_anchor_node(page_nodes)
        doc_id = candidate.get("doc_id") or (anchor_node.doc_id if anchor_node is not None else "")
        candidates.append(
            {
                "candidate_id": f"page{page}::screenshot",
                "kind": "image",
                "node_id": anchor_node.node_id if anchor_node is not None else f"{doc_id}::page::{page}",
                "page": page,
                "title": "Full page screenshot",
                "summary": "Full-page screenshot for visual evidence on this page.",
                "text_preview": "",
                "image_path": str(screenshot_path),
                "bbox": None,
                "source": "page_screenshot",
            }
        )
    return candidates


def _render_candidates_for_llm(
    *,
    search_query: str,
    page_candidates: list[dict[str, Any]],
) -> str:
    lines = [
        "Search query:",
        search_query,
    ]
    lines.extend(["", "Candidate items:"])
    for index, item in enumerate(page_candidates, start=1):
        lines.append(_render_candidate_item(index, item))
    lines.extend(
        [
            "",
            "Select only the candidate ids that are useful evidence for answering the question.",
        ]
    )
    return "\n".join(lines)


def _render_candidate_item(index: int, item: dict[str, Any]) -> str:
    candidate_id = item["candidate_id"]
    kind = item["kind"]
    page = item.get("page")
    title = item.get("title") or ""
    summary = item.get("summary") or ""
    preview = item.get("text_preview") or ""
    if kind == "image":
        if item.get("source") == "figure_node":
            details = " ".join(part for part in [title, summary, preview] if part).strip()
            bbox = item.get("bbox")
            bbox_text = f" bbox={bbox}" if bbox else ""
            return (
                f"{index}. Figure candidate `{candidate_id}` on page {page}{bbox_text}: "
                f"{details or 'figure/image node for visual reading'}."
            )
        return (
            f"{index}. Screenshot candidate `{candidate_id}` on page {page}: "
            f"{summary or 'full-page screenshot for visual reading'}."
        )
    label = "Table" if kind == "table" else "Text"
    details = " ".join(part for part in [title, summary, preview] if part).strip()
    return f"{index}. {label} candidate `{candidate_id}` on page {page}: {details}"


def _heuristic_select_candidates(
    *,
    action: Action,
    state: DocVQAState,
    candidate: dict[str, Any],
    page_candidates: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    query_text = " ".join(
        [
            state.question,
            action.intent,
            str(action.args.get("instruction") or ""),
            str(candidate.get("reason") or ""),
        ]
    )
    query_tokens = _keyword_tokens(query_text)
    visual_query = _contains_any(query_text, _VISUAL_TERMS)
    table_query = _contains_any(query_text, _TABLE_TERMS)

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in page_candidates:
        item_text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("summary") or ""),
                str(item.get("text_preview") or ""),
            ]
        )
        score = sum(1 for token in query_tokens if token in item_text.lower())
        if item["kind"] == "image" and visual_query:
            score += 5
        if item["kind"] == "table" and table_query:
            score += 4
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda pair: (-pair[0], pair[1]["candidate_id"]))
    selected = [item["candidate_id"] for _, item in scored[:3]]
    if not selected:
        image = next((item for item in page_candidates if item["kind"] == "image"), None)
        if image is not None and visual_query:
            selected = [image["candidate_id"]]
        else:
            non_image = next((item for item in page_candidates if item["kind"] != "image"), None)
            selected = [non_image["candidate_id"]] if non_image is not None else []
    return {"selected_candidate_ids": selected, "reason": reason}


def _candidate_node_suffix(node_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", node_id.rsplit("::", 1)[-1])[:80] or "node"


def _preview_node_text(node: Any) -> str:
    return re.sub(r"\s+", " ", node.searchable_text()).strip()


def _keyword_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(token) >= 3}


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _text_nodes(page_nodes: list[Any]) -> list[Any]:
    return [
        node
        for node in page_nodes
        if node.modality in {"text", "mixed"} and node.node_type in {"document", "section", "paragraph", "footnote"}
        and node.searchable_text().strip()
    ]


def _table_nodes(page_nodes: list[Any]) -> list[Any]:
    return [node for node in page_nodes if node.node_type == "table" or node.modality == "table"]


def _figure_nodes(page_nodes: list[Any]) -> list[Any]:
    return [
        node
        for node in page_nodes
        if node.node_type in {"figure", "chart"} or node.modality in {"image", "chart"}
    ]


def _find_figure_node(page_nodes: list[Any], node_id: Any) -> Any | None:
    if not node_id:
        return None
    node_id = str(node_id)
    return next((node for node in _figure_nodes(page_nodes) if node.node_id == node_id), None)


def _build_page_screenshot_figure_evidence(
    *,
    state: DocVQAState,
    candidate: dict[str, Any],
    page_nodes: list[Any],
    page_context: dict[str, Any] | None,
) -> EvidenceRecord | None:
    screenshot_path = None
    if isinstance(page_context, dict):
        screenshot_path = page_context.get("screenshot_path")
    if not screenshot_path:
        return None
    anchor_node = _page_anchor_node(page_nodes)
    doc_id = candidate.get("doc_id") or anchor_node.doc_id
    page = int(candidate["page"])
    node_id = anchor_node.node_id if anchor_node is not None else f"{doc_id}::page::{page}"
    evidence_id = f"{node_id}::page_figure::{len(state.evidence_store) + 1}"
    provenance = dict(getattr(anchor_node, "metadata", {}) or {})
    provenance.setdefault("doc_id", doc_id)
    provenance["source"] = "page_screenshot"
    content = {
        "image_path": str(screenshot_path),
        "kind": "page_screenshot",
        "page": page,
        "doc_id": doc_id,
        "note": "Full page screenshot included for figure-level reading.",
    }
    return EvidenceRecord(
        evidence_id=evidence_id,
        node_id=node_id,
        page=page,
        modality="image",
        content=content,
        provenance=provenance,
        confidence=1.0,
    )


def _page_anchor_node(page_nodes: list[Any]) -> Any | None:
    for node in page_nodes:
        if getattr(node, "node_type", "") == "document":
            return node
    return page_nodes[0] if page_nodes else None


def _build_figure_ref(
    candidate: dict[str, Any],
    node: Any,
    evidence: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    content = evidence.get("content", {})
    figure_ref = f"figure_{index}"
    return {
        "figure_ref": figure_ref,
        "page": int(candidate["page"]),
        "node_id": node.node_id,
        "evidence_id": evidence.get("evidence_id"),
        "image_path": _pick_image_path(content),
        "bbox": node.bbox,
        "caption": node.title or "",
        "caption_preview": (node.title or "")[:240],
        "footnote": node.summary or "",
        "content_preview": _preview_content(content),
        "modality": node.modality,
    }


def _build_page_screenshot_figure_ref(
    *,
    candidate: dict[str, Any],
    evidence: EvidenceRecord,
) -> dict[str, Any]:
    content = evidence.content if isinstance(evidence.content, dict) else {}
    page = int(candidate["page"])
    return {
        "figure_ref": f"page_figure_{page}",
        "page": page,
        "node_id": evidence.node_id,
        "evidence_id": evidence.evidence_id,
        "image_path": _pick_image_path(content),
        "bbox": None,
        "caption": "Full page screenshot fallback",
        "caption_preview": "Full page screenshot fallback",
        "footnote": "",
        "content_preview": _preview_content(content),
        "modality": "image",
        "source": "page_screenshot",
    }


def _pick_image_path(content: Any) -> str | None:
    if isinstance(content, dict):
        for key in ("image_path", "crop_path", "zoom_path"):
            value = content.get(key)
            if value:
                return str(value)
    return None


def _enrich_figure_payload_with_node_context(payload: dict[str, Any], env) -> dict[str, Any]:
    node = getattr(env, "nodes", {}).get(payload.get("node_id"))
    if node is None:
        return payload

    content = payload.get("content")
    if isinstance(content, dict):
        enriched_content = dict(content)
    else:
        enriched_content = {"raw": content}

    node_content = node.content if isinstance(node.content, dict) else {}
    image_path = _pick_image_path(enriched_content) or _pick_image_path(node_content)
    if image_path:
        enriched_content.setdefault("image_path", image_path)
        enriched_content.setdefault("image_url", image_path)

    enriched_content.setdefault("title", node.title or "")
    enriched_content.setdefault("summary", node.summary or "")
    enriched_content.setdefault("text", node.text or "")
    enriched_content.setdefault("bbox", node.bbox)
    enriched_content.setdefault("node_type", node.node_type)
    enriched_content.setdefault("page", node.page)
    enriched_content.setdefault("doc_id", node.doc_id)

    provenance = dict(payload.get("provenance") or {})
    provenance.setdefault("title", node.title or "")
    provenance.setdefault("summary", node.summary or "")
    provenance.setdefault("bbox", node.bbox)
    provenance.setdefault("node_type", node.node_type)
    if image_path:
        provenance.setdefault("image_path", image_path)

    updated = dict(payload)
    updated["content"] = enriched_content
    updated["provenance"] = provenance
    return updated


def _enhance_figure_payload_with_resize(payload: dict[str, Any], scale: float = 2.0) -> dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, dict):
        return payload
    image_path = _pick_image_path(content)
    if not image_path:
        return payload
    resized_path = _resize_image_file(image_path, scale=scale)
    if resized_path is None:
        return payload
    updated_content = dict(content)
    updated_content.setdefault("original_image_path", image_path)
    updated_content["image_path"] = resized_path
    updated_content["image_url"] = resized_path
    updated_content["resize_scale"] = float(scale)
    updated_payload = dict(payload)
    updated_payload["content"] = updated_content
    return updated_payload


def _resize_image_file(image_path: str, scale: float = 2.0) -> str | None:
    try:
        src = Path(image_path)
        if not src.exists() or not src.is_file():
            return None
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        factor = max(float(scale), 1.0)
        resized = cv2.resize(
            image,
            None,
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_CUBIC,
        )
        dst = src.with_name(f"{src.stem}.resized_x{factor:g}{src.suffix}")
        if not cv2.imwrite(str(dst), resized):
            return None
        return str(dst)
    except Exception:
        return None


def _preview_content(content: Any, limit: int | None = 240) -> str:
    text = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
    text = re.sub(r"\s+", " ", text).strip()
    return text if limit is None else text[:limit]
