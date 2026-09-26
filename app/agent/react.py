from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

from ..environment import DocumentEnvironment
from ..llm import LLMClient
from ..prompt.react import next_step_prompt, system_prompt
from ..run_logger import RunLogger
from ..schema import (
    Action,
    BudgetState,
    CandidateAnswer,
    DocVQAState,
    Memory,
    Message,
    Observation,
    VerificationResult,
)
from ..tool import ToolCollection, build_default_tool_collection
from .base import BaseAgent


class ReActAgent(BaseAgent):
    def __init__(
        self,
        tools: ToolCollection | None = None,
        llm_client: LLMClient | None = None,
    ):
        super().__init__("react")
        if llm_client is None:
            raise ValueError("ReActAgent requires an LLM client.")
        self.tools = tools or build_default_tool_collection(
            llm_client=llm_client,
        )
        self.single_document_overview = bool(
            getattr(self.tools.tools.get("navigate"), "single_document_overview", False)
        )
        self.llm = llm_client
        self.memory = Memory()

    async def next_action(self, state: DocVQAState, env: DocumentEnvironment) -> Action:
        return await self.think(state, env)

    async def think(self, state: DocVQAState, env: DocumentEnvironment) -> Action:
        available_tools = self._available_tools(state)
        actions_text = _format_actions(available_tools)
        available_tool_names = {tool["function"]["name"] for tool in available_tools}
        decision_system_prompt = system_prompt(actions_text)
        self.memory = self._build_memory(state)
        self.memory.add_message(
            Message.user_message(
                next_step_prompt(
                    actions_text,
                    require_verification="analyze" not in state.budget.disabled_tools,
                    direct_answer_from_navigation="extract" in state.budget.disabled_tools,
                    available_tool_names=available_tool_names,
                    multidoc=len(state.document_ids) > 1,
                    single_document_overview=(
                        self.single_document_overview and len(state.document_ids) == 1
                    ),
                )
            )
        )
        self.llm.set_call_source("react_decision")
        try:
            _log_react_input(self.llm, decision_system_prompt, self.memory.messages)
            payload = self.llm.generate_json_from_messages(
                self.memory.messages,
                system_prompt=decision_system_prompt,
            )
        finally:
            self.llm.set_call_source(None)
        thought = str(payload.get("content", "")).strip()
        if not thought:
            thought = str(payload.get("thought", "")).strip()
        action = self._action_from_payload(payload, thought, available_tools)
        self.memory.add_message(
            Message.assistant_message(
                json.dumps(
                    {
                        "thought": thought,
                        "action": payload.get("action"),
                    },
                    ensure_ascii=False,
                )
            )
        )
        return action

    async def act(
        self,
        action: Action,
        state: DocVQAState,
        env: DocumentEnvironment,
    ) -> Observation | None:
        observation = await self.dispatch(self._tool_name_for_action(action), action, state, env, self.tools)
        self.memory.add_message(
            Message.tool_message(
                json.dumps(observation.to_dict(), ensure_ascii=False),
                name=self._tool_name_for_action(action),
            )
        )
        return observation

    async def solve(
        self,
        question: str,
        env: DocumentEnvironment,
        document_ids: list[str],
        document_titles: dict[str, str] | None = None,
        initial_candidate_pages: list[dict[str, Any]] | None = None,
        budget: BudgetState | None = None,
        logger: RunLogger | None = None,
    ) -> DocVQAState:
        self.llm.set_logger(logger)
        state = DocVQAState(
            question=question,
            document_ids=document_ids,
            budget=budget or BudgetState(),
        )
        state.metadata["environment_overview"] = await env.overview(document_ids)
        if len(document_ids) > 1 or self.single_document_overview:
            state.metadata["overview"] = False
        if len(document_ids) > 1:
            state.metadata["document_titles"] = document_titles or {}
        if initial_candidate_pages:
            state.metadata["candidate_pages"] = list(initial_candidate_pages)
            state.metadata["navigation"] = {
                "candidate_pages": list(initial_candidate_pages),
                "pending_page_sets": [],
                "need_more_pages": False,
            }

        while not state.done and state.budget.step_used < state.budget.max_steps:
            step = state.budget.step_used + 1
            self.llm.set_step(step)
            action = await self.think(state, env)
            if logger is not None:
                logger.log_tool_start(step, action.to_dict())
            observation = await self.act(action, state, env)
            state.add_history(action, observation)
            if logger is not None:
                logger.log_result(
                    result=observation.message,
                    step=step,
                    observation=observation.to_dict(),
                )
            self._remember_latest_candidate(state)
            self._invalidate_stale_verification_after_recovery(action, state, observation)
            termination_reason = self._should_terminate(state)
            if termination_reason is not None:
                self._terminate_as_unanswerable(state, termination_reason)
            elif self._can_finish(state):
                state.done = True
                state.sync_evidence_state()
            if state.done:
                break

        self._restore_latest_candidate_if_empty(state)
        await self._run_finish_if_needed(state, env, logger)
        return state

    async def _run_finish_if_needed(
        self,
        state: DocVQAState,
        env: DocumentEnvironment,
        logger: RunLogger | None,
    ) -> None:
        if not state.done:
            return
        if state.history and state.history[-1]["action"]["action_type"] == "finish":
            return
        step = state.budget.step_used + 1
        action = Action(
            action_type="finish",
            agent_role="finish",
            intent="Automatically finishing after the run reached a terminal state.",
            args={},
        )
        if logger is not None:
            logger.log_tool_start(step, action.to_dict())
        observation = await self.dispatch("finish", action, state, env, self.tools)
        state.add_history(action, observation)
        if logger is not None:
            logger.log_result(
                result=observation.message,
                step=step,
                observation=observation.to_dict(),
            )

    def _available_tools(self, state: DocVQAState) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        definitions = self.tools.to_openai_tools()
        if len(state.document_ids) > 1:
            definitions = _with_multidoc_tool_schemas(deepcopy(definitions))
        elif self.single_document_overview:
            definitions = _with_single_document_overview_schema(deepcopy(definitions))
        for tool in definitions:
            name = tool["function"]["name"]
            if name in state.budget.disabled_tools:
                continue
            if name == "finish":
                continue
            if name == "navigate" and not state.budget.can_search():
                continue
            if name in {
                "extract",
                "focus",
            }:
                if not state.budget.can_extract():
                    continue
                if name == "focus":
                    has_focusable_evidence = any(
                        evidence.modality in {"image", "chart"}
                        for evidence in state.evidence_store.values()
                    ) or bool(
                        isinstance(state.metadata.get("last_extract_observation"), dict)
                        and state.metadata["last_extract_observation"].get("figure_refs")
                    )
                    if not has_focusable_evidence:
                        continue
                elif not state.metadata.get("candidate_nodes") and not state.metadata.get("candidate_pages"):
                    continue
            if name == "solver" and (not state.budget.can_compute() or not state.evidence_store):
                continue
            if name == "execute" and "last_solver" not in state.metadata:
                continue
            if name == "answer":
                can_answer_from_navigation = (
                    "extract" in state.budget.disabled_tools
                    and bool(state.metadata.get("candidate_pages"))
                )
                if not state.evidence_store and not can_answer_from_navigation:
                    continue
                if state.candidate_answer is not None and state.verification is None:
                    continue
            if name == "analyze":
                if state.candidate_answer is None:
                    continue
                if state.verification is not None and not state.verification.accepted:
                    continue
            tools.append(tool)
        return tools

    def _can_finish(self, state: DocVQAState) -> bool:
        if "analyze" in state.budget.disabled_tools:
            return state.candidate_answer is not None
        return (
            state.candidate_answer is not None
            and state.verification is not None
            and state.verification.accepted
        )

    def _remember_latest_candidate(self, state: DocVQAState) -> None:
        if not _candidate_answer_text(state.candidate_answer):
            return
        state.metadata["latest_candidate_answer"] = state.candidate_answer.to_dict()

    def _restore_latest_candidate_if_empty(self, state: DocVQAState) -> None:
        if _candidate_answer_text(state.candidate_answer):
            return
        restored = _candidate_from_payload(state.metadata.get("latest_candidate_answer"))
        if restored is None:
            return
        state.candidate_answer = restored

    def _invalidate_stale_verification_after_recovery(
        self,
        action: Action,
        state: DocVQAState,
        observation: Observation,
    ) -> None:
        if not observation.success:
            return
        if state.candidate_answer is None:
            return
        if state.verification is not None and state.verification.accepted:
            return

        if action.action_type in {"navigate", "extract", "focus", "solver", "execute"}:
            self._remember_latest_candidate(state)
            state.verification = None

    def _should_terminate(self, state: DocVQAState) -> str | None:
        # navigation = state.metadata.get("navigation", {})
        # need_more_pages = bool(navigation.get("need_more_pages"))
        # pending_page_sets = navigation.get("pending_page_sets") or []
        # if (
        #     state.candidate_answer is not None
        #     and state.verification is not None
        #     and not state.verification.accepted
        #     and not need_more_pages
        #     and not pending_page_sets
        # ):
        #     return (
        #         "Self-verification rejected the candidate answer and no additional pages remain to inspect. "
        #         "Returning INSUFFICIENT_EVIDENCE."
        #     )
        return None

    def _terminate_as_unanswerable(self, state: DocVQAState, reason: str) -> None:
        supporting_evidence_ids: list[str] = []
        confidence = 0.0
        if state.candidate_answer is not None:
            supporting_evidence_ids = list(state.candidate_answer.supporting_evidence_ids)
            confidence = state.candidate_answer.confidence
        state.candidate_answer = CandidateAnswer(
            answer="INSUFFICIENT_EVIDENCE",
            rationale=reason,
            supporting_evidence_ids=supporting_evidence_ids,
            confidence=confidence,
        )
        if state.verification is None:
            state.verification = VerificationResult(accepted=True)
        state.done = True
        state.sync_evidence_state()

    def _build_action(
        self,
        tool_name: str,
        args: dict[str, Any],
        thought: str,
        available_tools: list[dict[str, Any]],
    ) -> Action:
        available_names = {tool["function"]["name"] for tool in available_tools}
        if tool_name not in available_names:
            tool_name = _fallback_tool_name(available_names)
            args = {}

        if tool_name == "finish" and "finish" not in available_names:
            tool_name = _fallback_tool_name(available_names)
            args = {}

        if tool_name == "finish":
            return Action(
                action_type="finish",
                agent_role="finish",
                intent=thought,
                args={},
            )

        action_type, agent_role = _map_tool_name(tool_name)
        action_args = dict(args)
        if action_type in {"extract", "focus"}:
            action_args["tool_name"] = tool_name
        return Action(
            action_type=action_type,
            agent_role=agent_role,
            intent=thought,
            args=action_args,
        )

    def _tool_name_for_action(self, action: Action) -> str:
        if action.action_type in {"extract", "focus"}:
            return action.args["tool_name"]
        return action.action_type

    def _state_summary(self, state: DocVQAState) -> dict[str, Any]:
        navigation = state.metadata.get("navigation", {})
        multidoc = len(state.document_ids) > 1
        return {
            "evidence_state": state.evidence_state,
            "budget": asdict(state.budget),
            "candidate_pages": (
                _candidate_page_refs(state.metadata.get("candidate_pages", []))[:8]
                if multidoc
                else state.metadata.get("candidate_pages", [])[:8]
            ),
            "top_evidence": [
                {
                    "evidence_id": evidence_id,
                    "page": record.page,
                    "modality": record.modality,
                    "content_preview": str(record.content)[:600],
                    "provenance": record.provenance,
                }
                for evidence_id, record in list(state.evidence_store.items())[:8]
            ],
            "navigation": {
                "need_more_pages": navigation.get("need_more_pages"),
                **(
                    {
                        "overview": bool(state.metadata.get("overview", False)),
                        "document_range": _document_range_from_overview(
                            state.metadata.get("environment_overview", {})
                        ),
                        "active_document": navigation.get("active_document"),
                        "pending_page_sets": navigation.get("pending_page_sets", []),
                    }
                    if multidoc
                    else {
                        "pending_page_sets": navigation.get("pending_page_sets", []),
                        "verified_pages": navigation.get("verified_pages", []),
                        "page_range": _page_range_from_overview(
                            state.metadata.get("environment_overview", {})
                        ),
                    }
                ),
            },
            "last_solver": state.metadata.get("last_solver"),
            "last_execute": state.metadata.get("last_execute"),
            "candidate_answer": (
                state.candidate_answer.to_dict() if state.candidate_answer is not None else None
            ),
            "verification": (
                state.verification.to_dict() if state.verification is not None else None
            ),
        }

    def _build_memory(self, state: DocVQAState) -> Memory:
        memory = Memory()
        multidoc = len(state.document_ids) > 1
        pending_page_sets = state.metadata.get("navigation", {}).get("pending_page_sets", [])
        candidate_pages = _candidate_page_numbers(state.metadata.get("candidate_pages", []))
        page_range = _page_range_from_overview(state.metadata.get("environment_overview", {}))
        memory.add_message(Message.user_message(f"Question: {state.question}"))
        if multidoc:
            navigation = state.metadata.get("navigation", {})
            state_payload = {
                "evidence_state": state.evidence_state,
                "done": state.done,
                "overview": bool(state.metadata.get("overview", False)),
                "document_range": _document_range_from_overview(
                    state.metadata.get("environment_overview", {})
                ),
                "active_document": navigation.get("active_document"),
                "pending_page_sets": navigation.get("pending_page_sets", []),
                "candidate_pages": _candidate_page_refs(state.metadata.get("candidate_pages", [])),
                "extraction": {
                    "extracted_pages": _extracted_page_refs(state),
                },
                "max_navigate_pages_per_step": _max_navigate_pages_per_step(self.tools),
                "max_extract_pages_per_step": _max_extract_pages_per_step(self.tools),
            }
        else:
            state_payload = {
                "evidence_state": state.evidence_state,
                "done": state.done,
                "page_range": page_range,
                "candidate_pages": candidate_pages,
                "pending_page_sets": pending_page_sets,
                "extraction": {
                    "extracted_pages": _extracted_page_numbers(state),
                },
                "max_navigate_pages_per_step": _max_navigate_pages_per_step(self.tools),
                "max_extract_pages_per_step": _max_extract_pages_per_step(self.tools),
            }
            if self.single_document_overview:
                state_payload["overview"] = bool(state.metadata.get("overview", False))
        memory.add_message(
            Message.user_message(
                json.dumps(state_payload, ensure_ascii=False)
            )
        )
        # memory.add_message(
        #     Message.user_message(
        #         json.dumps(
        #             {
        #                 "document_ids": state.document_ids,
        #                 "state_summary": self._state_summary(state),
        #             },
        #             ensure_ascii=False,
        #         )
        #     )
        # )
        observation_history =  _observation_history(state)
        if observation_history:
            memory.add_message(
                Message.user_message("Observation history:\n" + "\n".join(observation_history))
            )
        return memory

    def _action_from_payload(
        self,
        payload: dict[str, Any],
        thought: str,
        available_tools: list[dict[str, Any]],
    ) -> Action:
        action_payload = payload.get("action", {})
        if not isinstance(action_payload, dict):
            action_payload = {}
        tool_name = str(action_payload.get("name") or "").strip()
        args = _normalize_action_properties(action_payload.get("properties"))
        return self._build_action(tool_name, args, thought, available_tools)


def _format_actions(available_tools: list[dict[str, Any]]) -> str:
    actions = []
    for tool in available_tools:
        function = tool.get("function", {})
        actions.append(
            {
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters", {}),
            }
        )
    return json.dumps(actions, ensure_ascii=False, indent=2)


def _with_multidoc_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    descriptions = {
        "navigate": (
            "Optional page targets. Use doc_id:page or doc_id:start-end, or use overview "
            "to inspect all document overview images when the target page is unknown."
        ),
        "extract": (
            "Optional page targets using doc_id:page or doc_id:start-end for multiple documents."
        ),
    }
    for tool in tools:
        function = tool.get("function", {})
        description = descriptions.get(function.get("name"))
        properties = function.get("parameters", {}).get("properties", {})
        if description and isinstance(properties.get("pages"), dict):
            properties["pages"]["description"] = description
    return tools


def _with_single_document_overview_schema(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for tool in tools:
        function = tool.get("function", {})
        if function.get("name") != "navigate":
            continue
        pages = function.get("parameters", {}).get("properties", {}).get("pages")
        if isinstance(pages, dict):
            pages["description"] = (
                "Optional integer page list/ranges, e.g. '1', '3-5', or '1,4'; "
                "use 'overview' to inspect one overview image when the target page is unknown."
            )
    return tools


def _log_react_input(llm: LLMClient, decision_system_prompt: str, messages: list[Message]) -> None:
    run_logger = getattr(llm, "logger", None)
    if run_logger is None:
        return
    step = getattr(llm, "current_step", None)
    prefix = f"STEP {step} " if step is not None else ""
    run_logger.info(
        f"{prefix}ReAct input:\n"
        + json.dumps(
            {
                "system_prompt": decision_system_prompt,
                "messages": [message.to_dict() for message in messages],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _page_range_from_overview(overview: dict[str, Any]) -> str | None:
    if not isinstance(overview, dict):
        return None
    documents = overview.get("documents")
    if not isinstance(documents, list):
        return None
    pages: set[int] = set()
    for document in documents:
        if not isinstance(document, dict):
            continue
        for page in document.get("pages", []) or []:
            try:
                pages.add(int(page))
            except (TypeError, ValueError):
                continue
    return _compact_page_range(sorted(pages)) if pages else None


def _document_range_from_overview(overview: dict[str, Any]) -> str | None:
    documents = overview.get("documents") if isinstance(overview, dict) else None
    if not isinstance(documents, list):
        return None
    ranges: list[str] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        doc_id = str(document.get("doc_id") or "").strip()
        pages = []
        for raw_page in document.get("pages", []) or []:
            try:
                pages.append(int(raw_page))
            except (TypeError, ValueError):
                continue
        if doc_id and pages:
            ranges.append(f"{doc_id}:{_compact_page_range(pages)}")
    return "; ".join(ranges) if ranges else None


def _candidate_page_refs(candidate_pages: Any) -> list[str]:
    if not isinstance(candidate_pages, list):
        return []
    refs: list[str] = []
    for item in candidate_pages:
        if isinstance(item, str) and ":" in item:
            ref = item.strip()
        elif isinstance(item, dict):
            doc_id = str(item.get("doc_id") or "").strip()
            try:
                page = int(item.get("page"))
            except (TypeError, ValueError):
                continue
            ref = f"{doc_id}:{page}" if doc_id else str(page)
        else:
            continue
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _extracted_page_refs(state: DocVQAState) -> list[str]:
    refs: list[str] = []
    for item in state.metadata.get("extracted_page_keys", []):
        text = str(item).strip()
        if "::" not in text:
            continue
        doc_id, page = text.rsplit("::", 1)
        ref = f"{doc_id}:{page}"
        if ref not in refs:
            refs.append(ref)
    return refs


def _candidate_page_numbers(candidate_pages: Any) -> list[int]:
    if not isinstance(candidate_pages, list):
        return []
    pages: list[int] = []
    seen: set[int] = set()
    for item in candidate_pages:
        raw_page = item.get("page") if isinstance(item, dict) else item
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            continue
        if page in seen:
            continue
        seen.add(page)
        pages.append(page)
    return pages


def _extracted_page_numbers(state: DocVQAState) -> list[int]:
    pages: list[int] = []
    seen: set[int] = set()
    for item in state.metadata.get("extracted_page_keys", []):
        raw_page = str(item).rsplit("::", 1)[-1]
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            continue
        if page in seen:
            continue
        seen.add(page)
        pages.append(page)
    return pages


def  _observation_history(state: DocVQAState, *, limit: int | None = None) -> list[str]:
    entries: list[str] = []
    history = state.history if limit is None else state.history[-limit:]
    for index, item in enumerate(history, start=max(1, len(state.history) - len(history) + 1)):
        action_payload = item.get("action") if isinstance(item, dict) else None
        observation_payload = item.get("observation") if isinstance(item, dict) else None
        if not isinstance(observation_payload, dict):
            continue
        action_type = ""
        if isinstance(action_payload, dict):
            action_type = str(action_payload.get("action_type") or "")
        action_type = action_type or str(observation_payload.get("action_type") or "")
        message = _format_observation_history_message(action_type, observation_payload)
        if not message:
            continue
        entries.append(f"Step {index}: {message}")
    return entries


def _format_observation_history_message(action_type: str, observation: dict[str, Any]) -> str:
    payload = observation.get("payload")
    payload = payload if isinstance(payload, dict) else {}

    if action_type == "navigate":
        page_set = _format_pages(payload.get("page_set") or payload.get("target_pages"))
        candidate_pages = _format_candidate_pages(payload.get("candidate_pages"))
        reason = _short_text(payload.get("reason"))
        parts = ["Navigation inspected " + (f"pages {page_set}" if page_set else "the requested pages")]
        if candidate_pages:
            parts.append(f"selected candidate pages {candidate_pages}")
        if reason:
            parts.append(f"reason: {reason}")
        return "; ".join(parts) + "."

    if action_type == "extract":
        selected_ids = [str(item) for item in payload.get("selected_candidate_ids", []) if str(item).strip()]
        decisions = payload.get("candidate_selection_decisions") or []
        reasons = [
            _short_text(item.get("reason"))
            for item in decisions
            if isinstance(item, dict) and item.get("reason")
        ]
        pages = _format_pages(payload.get("pages"))
        parts = ["Extraction processed " + (f"pages {pages}" if pages else "the selected pages")]
        if selected_ids:
            parts.append("selected candidates " + ", ".join(selected_ids[:8]))
        else:
            parts.append("selected no explicit candidates")
        if reasons:
            parts.append("reason: " + " ".join(reasons[:2]))
        evidence_ids = payload.get("evidence_ids") or []
        if evidence_ids:
            parts.append(f"created {len(evidence_ids)} evidence item(s)")
        return "; ".join(parts) + "."

    if action_type == "focus":
        operations = payload.get("operations") or []
        descriptions: list[str] = []
        for operation in operations[:3]:
            if not isinstance(operation, dict):
                continue
            tool_name = str(operation.get("tool") or "visual tool").strip()
            caption = _short_text(operation.get("visual_caption"))
            evidence_id = str(operation.get("evidence_id") or "").strip()
            item = f"used {tool_name}"
            if evidence_id:
                item += f" and produced {evidence_id}"
            if caption:
                item += f"; visual caption: {caption}"
            descriptions.append(item)
        if descriptions:
            return "Focus " + " ".join(descriptions) + "."
        planner = payload.get("planner")
        if isinstance(planner, dict):
            reason = _short_text(planner.get("reason"))
            if reason:
                return f"Focus planner reason: {reason}."

    if action_type == "analyze":
        analysis = payload.get("analysis")
        if isinstance(analysis, dict):
            summary = _short_text(analysis.get("summary"))
            plan = [str(item).strip() for item in analysis.get("plan", []) if str(item).strip()]
            parts = ["Analysis produced a plan"]
            if summary:
                parts.append(f"summary: {summary}")
            if plan:
                parts.append("plan: " + " ".join(plan[:3]))
            return "; ".join(parts) + "."

        verification = payload.get("verification")
        if isinstance(verification, dict):
            accepted = verification.get("accepted")
            summary = _short_text(verification.get("summary"))
            issues = [str(item).strip() for item in verification.get("issues", []) if str(item).strip()]
            suggestion = [str(item).strip() for item in verification.get("suggestion", []) if str(item).strip()]
            result_text = "accepted" if accepted else "rejected"
            parts = [f"Self-verification {result_text} the candidate answer"]
            if summary:
                parts.append(f"summary: {summary}")
            if issues:
                parts.append("issues: " + "; ".join(issues[:3]))
            if suggestion:
                parts.append("suggested next actions: " + ", ".join(suggestion))
            return "; ".join(parts) + "."

    if action_type == "answer":
        candidate = payload.get("candidate_answer")
        if isinstance(candidate, dict):
            answer = _short_text(candidate.get("answer"))
            rationale = _short_text(candidate.get("rationale"))
            evidence_count = payload.get("supporting_evidence_count")
            parts = [f"Answer drafted: {answer or 'EMPTY_ANSWER'}"]
            if evidence_count is not None:
                parts.append(f"supported by {evidence_count} evidence item(s)")
            if rationale:
                parts.append(f"rationale: {rationale}")
            return "; ".join(parts) + "."

    message = _short_text(observation.get("message"))
    return message


def _format_pages(raw_pages: Any) -> str:
    if isinstance(raw_pages, str):
        return raw_pages.strip()
    if isinstance(raw_pages, list):
        refs = _candidate_page_refs(raw_pages)
        if any(":" in ref for ref in refs):
            return ",".join(refs)
        pages: list[int] = []
        for item in raw_pages:
            raw_page = item.get("page") if isinstance(item, dict) else item
            try:
                pages.append(int(raw_page))
            except (TypeError, ValueError):
                continue
        return _compact_page_range(pages)
    return ""


def _format_candidate_pages(raw_pages: Any) -> str:
    if not isinstance(raw_pages, list):
        return ""
    refs = _candidate_page_refs(raw_pages)
    if any(":" in ref for ref in refs):
        return ",".join(refs)
    pages: list[int] = []
    for item in raw_pages:
        raw_page = item.get("page") if isinstance(item, dict) else item
        try:
            pages.append(int(raw_page))
        except (TypeError, ValueError):
            continue
    return _compact_page_range(pages)


def _short_text(value: Any, limit: int = 360) -> str:
    text = str(value or "").strip().replace("\n", " ")
    text = " ".join(text.split())
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _max_navigate_pages_per_step(tools: ToolCollection) -> int | None:
    for tool in _iter_tool_instances(tools):
        if getattr(tool, "name", None) == "navigate":
            value = getattr(tool, "image_page_set_size", None)
            if isinstance(value, int) and value > 0:
                return value
            try:
                value = int(value)
            except (TypeError, ValueError):
                return None
            return value if value > 0 else None
    return None


def _max_extract_pages_per_step(tools: ToolCollection) -> int | None:
    for tool in _iter_tool_instances(tools):
        if getattr(tool, "name", None) == "extract":
            extract_config = getattr(tool, "extract_config", None)
            value = getattr(extract_config, "max_pages", None)
            if value is None:
                return None
            if isinstance(value, int) and value > 0:
                return value
            try:
                value = int(value)
            except (TypeError, ValueError):
                return None
            return value if value > 0 else None
    return None


def _iter_tool_instances(tools: ToolCollection) -> list[Any]:
    tool_map = getattr(tools, "tools", [])
    if isinstance(tool_map, dict):
        return list(tool_map.values())
    return list(tool_map)


def _compact_page_range(pages: list[int]) -> str:
    sorted_pages = sorted(set(pages))
    if not sorted_pages:
        return ""
    ranges: list[str] = []
    start = prev = sorted_pages[0]
    for page in sorted_pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = page
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _candidate_answer_text(candidate: CandidateAnswer | None) -> str:
    if candidate is None:
        return ""
    return str(candidate.answer or "").strip()


def _candidate_from_payload(payload: Any) -> CandidateAnswer | None:
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


def _fallback_tool_name(available_names: set[str]) -> str:
    for name in ("answer", "extract", "navigate", "focus", "solver", "execute", "analyze"):
        if name in available_names:
            return name
    return next(iter(available_names), "finish")


def _normalize_action_properties(raw_properties: Any) -> dict[str, Any]:
    if isinstance(raw_properties, dict):
        return raw_properties
    if isinstance(raw_properties, list):
        properties: dict[str, Any] = {}
        for item in raw_properties:
            if isinstance(item, dict):
                if "name" in item and "value" in item:
                    properties[str(item["name"])] = item["value"]
                else:
                    properties.update(item)
        return properties
    return {}


def _map_tool_name(tool_name: str) -> tuple[str, str]:
    if tool_name == "analyze":
        return "analyze", "analyze"
    if tool_name == "navigate":
        return "navigate", "navigate"
    if tool_name == "extract":
        return "extract", "extract"
    if tool_name == "focus":
        return "focus", "focus"
    if tool_name == "solver":
        return "solver", "solver"
    if tool_name == "execute":
        return "execute", "execute"
    if tool_name == "answer":
        return "answer", "answer"
    return "analyze", "analyze"
