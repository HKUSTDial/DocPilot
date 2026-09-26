from __future__ import annotations

import json
from typing import Any

from ..environment import DocumentEnvironment
from ..llm import LLMClient
from ..run_logger import RunLogger
from ..schema import Action, BudgetState, CandidateAnswer, DocVQAState, Observation
from ..tool import ToolCollection, build_default_tool_collection
from ..tool.navigate import (
    _multidoc_page_pool,
    _multidoc_pending_page_sets,
    _resolve_page_pool,
)
from ..vllm_protocol import (
    VLLMRoute,
    action_system_prompt,
    analyze_system_prompt,
    compact_pages,
    parse_action,
    parse_analyze,
    render_action_observation,
)
from .base import BaseAgent
from .react import (
    _candidate_page_numbers,
    _candidate_page_refs,
    _document_range_from_overview,
    _extracted_page_numbers,
    _extracted_page_refs,
    _page_range_from_overview,
)


def _finish_stopped_tag(text: str, tag: str) -> str:
    stripped = text.strip()
    opening = f"<{tag}>"
    closing = f"</{tag}>"
    lowered = stripped.lower()
    if lowered.count(opening.lower()) == 1 and closing.lower() not in lowered:
        return f"{stripped}\n{closing}"
    return stripped


class VLLMTagsAgent(BaseAgent):
    def __init__(
        self,
        tools: ToolCollection | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        super().__init__("vllm_tags")
        if llm_client is None:
            raise ValueError("VLLMTagsAgent requires an LLM client.")
        self.llm = llm_client
        self.tools = tools or build_default_tool_collection(llm_client=llm_client)
        self.messages: list[dict[str, Any]] = []
        self.bridge = _SharedPolicyToolClient(llm_client, self.messages)
        for name in ("navigate", "extract", "focus", "solver", "answer"):
            tool = self.tools.tools.get(name)
            if tool is not None and hasattr(tool, "llm"):
                tool.llm = self.bridge

    async def next_action(self, state: DocVQAState, env: DocumentEnvironment) -> Action:
        route = self._generate_route(state)
        return self._route_action(route, state)

    async def solve(
        self,
        question: str,
        env: DocumentEnvironment,
        document_ids: list[str],
        document_titles: dict[str, str] | None = None,
        budget: BudgetState | None = None,
        logger: RunLogger | None = None,
    ) -> DocVQAState:
        self.llm.set_logger(logger)
        self.bridge.set_logger(logger)
        self.messages.clear()
        state = DocVQAState(
            question=question,
            document_ids=document_ids,
            budget=budget or BudgetState(),
        )
        state.metadata["environment_overview"] = await env.overview(document_ids)
        if len(document_ids) > 1:
            state.metadata["document_titles"] = document_titles or {}
            state.metadata["overview"] = False
        navigate_tool = self.tools.tools.get("navigate")
        if bool(getattr(navigate_tool, "single_document_overview", False)):
            state.metadata["overview"] = False
        self._initialize_navigation(state)
        self._append_state(state, include_question=True)

        while not state.done and state.budget.step_used < state.budget.max_steps:
            step = state.budget.step_used + 1
            self.llm.set_step(step)
            self.bridge.set_step(step)
            action = await self.next_action(state, env)
            expected = "solve" if action.action_type == "solver" else action.action_type
            self.bridge.expect(expected, "")
            if logger is not None:
                logger.log_tool_start(step, action.to_dict())

            observation = await self.dispatch(action.action_type, action, state, env, self.tools)
            if action.action_type == "solver" and observation.success:
                observation = await self._execute_solver(action, observation, state, env, logger, step)
            elif not self.bridge.generated_action:
                self.bridge.append_environment_observation(observation.message)

            state.add_history(action, observation)
            if action.action_type == "answer" and observation.success:
                state.done = True
                state.sync_evidence_state()
            if logger is not None:
                logger.log_result(
                    result=observation.message,
                    step=step,
                    observation=observation.to_dict(),
                )
            if not state.done:
                self._append_state(state)

        if not state.done:
            state.candidate_answer = state.candidate_answer or CandidateAnswer(
                answer="INSUFFICIENT_EVIDENCE",
                rationale="The vLLM tags rollout reached max_steps before a complete answer.",
                supporting_evidence_ids=list(state.evidence_store),
                confidence=0.0,
            )
            state.done = True
            state.sync_evidence_state()
        return state

    async def _execute_solver(
        self,
        action: Action,
        solver_observation: Observation,
        state: DocVQAState,
        env: DocumentEnvironment,
        logger: RunLogger | None,
        step: int,
    ) -> Observation:
        if "execute" in state.budget.disabled_tools:
            return solver_observation
        inputs = dict(solver_observation.payload.get("inputs") or {})
        inputs.setdefault(
            "evidence",
            {
                evidence_id: record.to_dict()
                for evidence_id, record in state.evidence_store.items()
            },
        )
        execute_action = Action(
            action_type="execute",
            agent_role="execute",
            intent="Execute code generated by the preceding <Solve> action.",
            args={"inputs": inputs},
        )
        execute_observation = await self.dispatch("execute", execute_action, state, env, self.tools)
        self.bridge.append_execute(execute_observation)
        if logger is not None:
            logger.info(f"STEP {step} automatic Execute: {execute_observation.message}")
        payload = dict(solver_observation.payload)
        payload["execute"] = execute_observation.payload
        return Observation(
            action_type="solver",
            success=execute_observation.success,
            message=execute_observation.message,
            payload=payload,
        )

    def _generate_route(self, state: DocVQAState) -> VLLMRoute:
        actions = self._available_actions(state)
        if not actions:
            raise RuntimeError("No vLLM tags action is available.")
        self.llm.set_call_source("vllm_tags_analyze")
        try:
            tag = "Analyze"
            raw = self.llm.generate_text_from_messages(
                self.messages,
                system_prompt=self._system_prompt(state),
                stop=[f"</{tag}>", f"\n<{tag}>"],
            )
            raw = _finish_stopped_tag(raw, tag)
        finally:
            self.llm.set_call_source(None)
        route = parse_analyze(raw, set(actions))
        self.messages.append(
            {"role": "assistant", "content": f"<Analyze>\n{route.reasoning}\n</Analyze>"}
        )
        return route

    def _route_action(self, route: VLLMRoute, state: DocVQAState) -> Action:
        action_type = "solver" if route.action == "solve" else route.action
        args: dict[str, Any] = {"instruction": route.reasoning}
        if route.action in {"navigate", "extract"}:
            pages = compact_pages(route.target_pages)
            if route.action == "navigate" and not pages and not state.metadata.get("overview", True):
                pages = "overview"
            args["pages"] = pages
        elif route.action == "focus":
            args["evidence_ids"] = list(route.evidence_ids)
        return Action(
            action_type=action_type,
            agent_role=action_type,
            intent=route.reasoning,
            args=args,
        )

    def _available_actions(self, state: DocVQAState) -> list[str]:
        disabled = {"solve" if name == "solver" else name for name in state.budget.disabled_tools}
        if "execute" in disabled:
            disabled.add("solve")
        actions: list[str] = []
        if "navigate" not in disabled and state.budget.can_search():
            actions.append("navigate")
        if (
            "extract" not in disabled
            and state.budget.can_extract()
            and bool(state.metadata.get("candidate_pages"))
        ):
            actions.append("extract")
        if "focus" not in disabled and any(
            evidence.modality in {"image", "chart"} for evidence in state.evidence_store.values()
        ):
            actions.append("focus")
        if "solve" not in disabled and state.budget.can_compute() and state.evidence_store:
            actions.append("solve")
        if "answer" not in disabled:
            actions.append("answer")
        return actions

    def _system_prompt(self, state: DocVQAState) -> str:
        return analyze_system_prompt(self._available_actions(state))

    def _initialize_navigation(self, state: DocVQAState) -> None:
        overview = state.metadata.get("environment_overview", {})
        navigate_tool = self.tools.tools.get("navigate")
        set_size = max(1, int(getattr(navigate_tool, "image_page_set_size", 10)))
        navigation = state.metadata.setdefault("navigation", {})
        if len(state.document_ids) > 1:
            available = _multidoc_page_pool(overview)
            navigation["pending_page_sets"] = _multidoc_pending_page_sets(
                available,
                navigation,
                set_size,
            )
            return
        pages = _resolve_page_pool(overview)
        navigation["pending_page_sets"] = [pages[i : i + set_size] for i in range(0, len(pages), set_size)]

    def _append_state(self, state: DocVQAState, *, include_question: bool = False) -> None:
        navigation = state.metadata.get("navigation", {})
        multidoc = len(state.document_ids) > 1
        if multidoc:
            payload = {
                "evidence_state": state.evidence_state,
                "done": state.done,
                "overview": bool(state.metadata.get("overview", False)),
                "document_range": _document_range_from_overview(
                    state.metadata.get("environment_overview", {})
                ),
                "active_document": navigation.get("active_document"),
                "candidate_pages": _candidate_page_refs(state.metadata.get("candidate_pages", [])),
                "pending_page_sets": navigation.get("pending_page_sets", []),
                "extracted_pages": _extracted_page_refs(state),
            }
        else:
            payload = {
                "evidence_state": state.evidence_state,
                "done": state.done,
                "page_range": _page_range_from_overview(state.metadata.get("environment_overview", {})),
                "candidate_pages": _candidate_page_numbers(state.metadata.get("candidate_pages", [])),
                "pending_page_sets": navigation.get("pending_page_sets", []),
                "extracted_pages": _extracted_page_numbers(state),
            }
            if "overview" in state.metadata:
                payload["overview"] = bool(state.metadata["overview"])
        prefix = f"Instruction:\n{state.question}\n\n" if include_question else ""
        self.messages.append(
            {
                "role": "user",
                "content": prefix + "Current context:\nState:\n" + _render_mapping(payload),
            }
        )


class _SharedPolicyToolClient:
    is_vllm_tags = True

    def __init__(self, policy: LLMClient, messages: list[dict[str, Any]]) -> None:
        self.policy = policy
        self.config = policy.config
        self.messages = messages
        self.logger: RunLogger | None = None
        self.current_step: int | None = None
        self.current_call_source: str | None = None
        self.expected_action = ""
        self.policy_prompt = ""
        self.generated_action = False

    def set_logger(self, logger: RunLogger | None) -> None:
        self.logger = logger

    def set_step(self, step: int | None) -> None:
        self.current_step = step

    def set_call_source(self, call_source: str | None) -> None:
        self.current_call_source = call_source

    def expect(self, action: str, prompt: str) -> None:
        self.expected_action = action
        self.policy_prompt = prompt
        self.generated_action = False

    def generate_json(
        self,
        _ignored_system_prompt: str,
        user_prompt: str,
        *,
        images: list[str] | None = None,
        max_tokens: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if not self.expected_action:
            raise RuntimeError("No semantic action is expected.")
        observation = render_action_observation(self.expected_action, user_prompt)
        user_message: dict[str, Any] = {"role": "user", "content": observation}
        if images:
            user_message["images"] = list(images)
        self.messages.append(user_message)
        self.policy.set_call_source(f"vllm_tags_{self.expected_action}")
        try:
            tag = "Solve" if self.expected_action in {"solve", "solver"} else self.expected_action.title()
            raw = self.policy.generate_text_from_messages(
                self.messages,
                system_prompt=action_system_prompt(self.expected_action),
                max_tokens=max_tokens,
                stop=[f"</{tag}>", f"\n<{tag}>"],
            )
            raw = _finish_stopped_tag(raw, tag)
            payload = parse_action(raw, self.expected_action)
        except Exception:
            self.messages.pop()
            raise
        finally:
            self.policy.set_call_source(None)
        user_message.pop("images", None)
        self.messages.append({"role": "assistant", "content": raw})
        self.generated_action = True
        return payload

    def append_execute(self, observation: Observation) -> None:
        payload = observation.payload
        result = payload.get("result") if isinstance(payload, dict) else None
        error = payload.get("error") if isinstance(payload, dict) else None
        content = (
            f"<Execute>\nresult: \\boxed{{{result}}}\n</Execute>"
            if error is None
            else f"<Execute>\nerror: {error}\n</Execute>"
        )
        self.messages.append({"role": "user", "content": content})

    def append_environment_observation(self, message: str) -> None:
        self.messages.append({"role": "user", "content": f"Environment result:\n{message}"})


def _render_mapping(payload: dict[str, Any]) -> str:
    lines = []
    for key, value in payload.items():
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)
