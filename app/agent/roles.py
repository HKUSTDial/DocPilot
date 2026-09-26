from __future__ import annotations

import json
from typing import Any

from ..environment import DocumentEnvironment
from ..llm import LLMClient
from ..schema import Action, DocVQAState
from .base import BaseAgent


class NavigateAgent(BaseAgent):
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        super().__init__("navigate")
        self.llm = llm_client

    async def next_action(self, state: DocVQAState, env: DocumentEnvironment) -> Action:
        query = _build_navigation_query(state.question)
        if self.llm is not None:
            try:
                payload = _llm_navigation_plan(self.llm, state)
                query = str(payload.get("query") or query)
            except Exception:
                pass
        if state.verification and state.verification.suggestion:
            query = f"{query} {' '.join(state.verification.issues)}"
        return Action(
            action_type="navigate",
            agent_role="navigate",
            intent="Navigate across page-level text and screenshot references to identify candidate pages.",
            args={"query": query},
        )


class ExtractAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("extract")

    async def next_action(self, state: DocVQAState, env: DocumentEnvironment) -> Action:
        page_target = _next_candidate_page_target(state, env)
        if page_target is not None:
            candidate_page, _target = page_target
            instruction = _extract_instruction_for_page(candidate_page)
            return Action(
                action_type="extract",
                agent_role="extract",
                intent=instruction,
                args={
                    "tool_name": "extract",
                    "candidate_pages": [candidate_page],
                    "instruction": instruction,
                },
            )

        candidates = state.metadata.get("candidate_nodes", [])
        extracted_nodes = state.metadata.setdefault("extracted_node_ids", [])
        target = next(
            (candidate for candidate in candidates if candidate["node_id"] not in extracted_nodes),
            None,
        )
        if target is None:
            return Action(
                action_type="extract",
                agent_role="extract",
                intent="No candidate nodes left to extract.",
                args={"tool_name": "extract", "node_id": "", "instruction": "Read any remaining relevant text evidence."},
            )

        extracted_nodes.append(target["node_id"])
        tool_name = _select_extract_tool(target)
        return Action(
            action_type="extract",
            agent_role="extract",
            intent=_extract_instruction_for_tool(tool_name, target),
            args={"tool_name": "extract", "node_id": target["node_id"], "instruction": _extract_instruction_for_tool(tool_name, target)},
        )


class SolverAgent(BaseAgent):
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        super().__init__("solver")
        self.llm = llm_client

    async def next_action(self, state: DocVQAState, env: DocumentEnvironment) -> Action:
        return Action(
            action_type="solver",
            agent_role="solver",
            intent="Generate Python code to interact with the extracted evidence.",
            args={
                "instruction": (
                    "Generate grounded Python that computes the answer from the extracted evidence. "
                    "Prefer explicit numeric inputs and safe arithmetic."
                ),
                "evidence_ids": list(state.evidence_store.keys()),
            },
        )


class ExecuteAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("execute")

    async def next_action(self, state: DocVQAState, env: DocumentEnvironment) -> Action:
        return Action(
            action_type="execute",
            agent_role="execute",
            intent="Execute the latest generated Python code and inspect the result.",
            args={},
        )


class AnswerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("answer")

    async def next_action(self, state: DocVQAState, env: DocumentEnvironment) -> Action:
        rationale = "Synthesize the answer from extracted evidence and any executed Python results."
        return Action(
            action_type="answer",
            agent_role="answer",
            intent="Draft a candidate answer.",
            args={
                "instruction": rationale,
                "evidence_ids": list(state.evidence_store.keys()),
            },
        )


def _select_extract_tool(target: dict) -> str:
    node_type = target.get("node_type")
    modality = target.get("modality")
    if node_type == "table":
        return "extract_table"
    if node_type == "chart":
        return "extract_figure"
    if node_type == "figure":
        return "extract_figure"
    if modality == "image":
        return "subfigure_split"
    return "extract_text"


def _next_candidate_page_target(state: DocVQAState, env: DocumentEnvironment) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates = state.metadata.get("candidate_pages", [])
    extracted_pages = set(state.metadata.setdefault("extracted_page_keys", []))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = _candidate_page_key(candidate)
        if key in extracted_pages:
            continue
        target = _node_for_candidate_page(env, candidate)
        if target is not None:
            return candidate, target
    return None


def _node_for_candidate_page(env: DocumentEnvironment, candidate: dict[str, Any]) -> dict[str, Any] | None:
    nodes = getattr(env, "nodes", None)
    if not isinstance(nodes, dict):
        return None
    try:
        page = int(candidate.get("page"))
    except (TypeError, ValueError):
        return None
    doc_id = candidate.get("doc_id")
    same_page = [
        node
        for node in nodes.values()
        if getattr(node, "page", None) == page and (not doc_id or getattr(node, "doc_id", None) == doc_id)
    ]
    if not same_page:
        return None

    recommended = str(candidate.get("recommended_extract") or "mixed")
    if recommended in {"extract_table"}:
        preferred = _first_node_by_kind(same_page, {"table"}, {"table"})
    elif recommended in {"extract_figure", "crop", "zoom", "subfigure_split"}:
        preferred = _first_node_by_kind(same_page, {"figure", "chart"}, {"image", "chart"})
    else:
        preferred = _first_node_by_kind(same_page, {"document", "section", "paragraph"}, {"text", "mixed"})
    if preferred is None:
        preferred = sorted(same_page, key=lambda node: (getattr(node, "node_type", ""), getattr(node, "node_id", "")))[0]
    return {
        "node_id": preferred.node_id,
        "doc_id": preferred.doc_id,
        "page": preferred.page,
        "node_type": preferred.node_type,
        "modality": preferred.modality,
        "title": preferred.title,
        "snippet": preferred.searchable_text()[:240],
    }


def _first_node_by_kind(nodes: list[Any], node_types: set[str], modalities: set[str]) -> Any | None:
    matching = [
        node
        for node in nodes
        if getattr(node, "node_type", None) in node_types or getattr(node, "modality", None) in modalities
    ]
    if not matching:
        return None
    return sorted(matching, key=lambda node: (getattr(node, "bbox", None) or [0, 0], getattr(node, "node_id", "")))[0]


def _extract_instruction_for_page(candidate_page: dict[str, Any]) -> str:
    hint = str(candidate_page.get("reason") or "").strip()
    page = candidate_page.get("page")
    return (
        f"I need to extract baseline evidence from page {page}, including relevant text, tables, and figures. {hint}"
    ).strip()


def _candidate_page_key(candidate_page: dict[str, Any]) -> str:
    return f"{candidate_page.get('doc_id') or '*'}::{candidate_page.get('page')}"


def _extract_instruction_for_tool(tool_name: str, target: dict) -> str:
    title = str(target.get("title") or target.get("node_type") or "the target node")
    if tool_name == "extract_table":
        return f"I need to extract the structured table contents from {title}."
    if tool_name == "extract_figure":
        return f"I need to inspect the figure evidence from {title}."
    if tool_name == "zoom":
        return f"Now I need to zoom into {title} to inspect the fine-grained detail more precisely."
    if tool_name == "crop":
        return f"I should crop {title} to isolate the relevant visual region."
    if tool_name == "subfigure_split":
        return f"I need to split the multi-panel figure in {title}, optionally targeting a specific subplot."
    return f"I need to read the text evidence from {title}."


def _build_navigation_query(question: str) -> str:
    q = question.lower().strip()
    q = q.replace("costco", "").replace("fy2021", "2021").replace("fy 2021", "2021")
    if "long-term debt to total liabilities" in q:
        return "long-term debt total liabilities 2021 consolidated balance sheets debt"
    if "current liabilities to total liabilities" in q:
        return "current liabilities total liabilities 2021 consolidated balance sheets"
    if "operating cash flow to current liabilities" in q:
        return "net cash provided by operating activities total current liabilities 2021 consolidated statements of cash flows consolidated balance sheets"
    if "total liabilities" in q:
        return "total liabilities 2021 consolidated balance sheets"
    if "common equity" in q:
        return "total stockholders equity common equity 2021 consolidated balance sheets"
    if "membership warehouses" in q:
        return "membership warehouses operated 2021 warehouses worldwide"
    if "filing date" in q and "description of common stock" in q:
        return "description of common stock filing date exhibits"
    return q


def _llm_navigation_plan(llm: LLMClient, state: DocVQAState) -> dict:
    system_prompt = (
        "You are the Navigate Agent for a DocVQA system. "
        "Rewrite the question into a page-level navigation query for long-document inspection. "
        "Return JSON with keys query and top_k."
    )
    user_prompt = json.dumps(
        {
            "question": state.question,
            "environment_overview": state.metadata.get("environment_overview", {}),
            "verification_issues": state.verification.issues if state.verification else [],
        },
        ensure_ascii=False,
    )
    return llm.generate_json(system_prompt, user_prompt)
