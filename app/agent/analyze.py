from __future__ import annotations

from ..environment import DocumentEnvironment
from ..llm import LLMClient
from ..run_logger import RunLogger
from ..schema import Action, BudgetState, DocVQAState
from ..tool import ToolCollection, build_default_tool_collection
from .base import BaseAgent
from .roles import AnswerAgent, ExecuteAgent, ExtractAgent, NavigateAgent, SolverAgent


class AnalyzeAgent(BaseAgent):
    def __init__(self, tools: ToolCollection | None = None, llm_client: LLMClient | None = None):
        super().__init__("analyze")
        self.tools = tools or build_default_tool_collection(llm_client=llm_client)
        self.navigator = NavigateAgent(llm_client=llm_client)
        self.extractor = ExtractAgent()
        self.solver = SolverAgent(llm_client=llm_client)
        self.executor = ExecuteAgent()
        self.answerer = AnswerAgent()

    async def next_action(self, state: DocVQAState, env: DocumentEnvironment) -> Action:
        if state.candidate_answer is not None and state.verification is None:
            return Action(
                action_type="analyze",
                agent_role="analyze",
                intent="Self-verify the candidate answer against collected evidence.",
                args={},
            )

        if state.verification is not None and not state.verification.accepted:
            state.candidate_answer = None
            if state.budget.can_search() and not _navigation_unavailable(state):
                return await self.navigator.next_action(state, env)

        if (
            not state.metadata.get("candidate_nodes")
            and not state.metadata.get("candidate_pages")
            and state.budget.can_search()
            and not _navigation_unavailable(state)
        ):
            return await self.navigator.next_action(state, env)

        if _has_unextracted_candidate(state) and state.budget.can_extract():
            return await self.extractor.next_action(state, env)

        if self._needs_solve(state) and state.budget.can_compute():
            return await self.solver.next_action(state, env)

        if self._needs_execute(state):
            return await self.executor.next_action(state, env)

        return await self.answerer.next_action(state, env)

    async def solve(
        self,
        question: str,
        env: DocumentEnvironment,
        document_ids: list[str],
        document_titles: dict[str, str] | None = None,
        budget: BudgetState | None = None,
        logger: RunLogger | None = None,
    ) -> DocVQAState:
        if self.navigator.llm is not None:
            self.navigator.llm.set_logger(logger)
        if self.solver.llm is not None:
            self.solver.llm.set_logger(logger)
        state = DocVQAState(
            question=question,
            document_ids=document_ids,
            budget=budget or BudgetState(),
        )
        state.metadata["environment_overview"] = await env.overview(document_ids)
        if len(document_ids) > 1:
            state.metadata["overview"] = False
            state.metadata["document_titles"] = document_titles or {}

        while not state.done and state.budget.step_used < state.budget.max_steps:
            step = state.budget.step_used + 1
            if self.navigator.llm is not None:
                self.navigator.llm.set_step(step)
            if self.solver.llm is not None:
                self.solver.llm.set_step(step)
            action = await self.next_action(state, env)
            if logger is not None:
                logger.log_tool_start(step, action.to_dict())
            observation = await self.dispatch(self._tool_name_for_action(action), action, state, env, self.tools)
            state.add_history(action, observation)
            if logger is not None:
                logger.log_result(
                    result=observation.message,
                    step=step,
                    observation=observation.to_dict(),
                )
            if state.done:
                break

        return state

    def _needs_solve(self, state: DocVQAState) -> bool:
        question = state.question.lower()
        compute_phrases = [
            "ratio",
            "percent",
            "percentage",
            "difference",
            "increase",
            "decrease",
            "average",
            "mean",
            "change from",
            "change over",
            "how did",
        ]
        if not any(signal in question for signal in compute_phrases):
            return False
        if not state.evidence_store:
            return False
        return "last_execute" not in state.metadata and "last_solver" not in state.metadata

    def _needs_execute(self, state: DocVQAState) -> bool:
        return "last_solver" in state.metadata and "last_execute" not in state.metadata

    def _tool_name_for_action(self, action: Action) -> str:
        if action.action_type == "extract":
            return action.args["tool_name"]
        return action.action_type


def _has_unextracted_candidate(state: DocVQAState) -> bool:
    extracted_ids = set(state.metadata.get("extracted_node_ids", []))
    for candidate in state.metadata.get("candidate_nodes", []):
        if candidate["node_id"] not in extracted_ids:
            return True
    extracted_pages = set(state.metadata.get("extracted_page_keys", []))
    for candidate in state.metadata.get("candidate_pages", []):
        if not isinstance(candidate, dict):
            continue
        key = f"{candidate.get('doc_id') or '*'}::{candidate.get('page')}"
        if key not in extracted_pages:
            return True
    return False


def _has_ununderstood_evidence(state: DocVQAState) -> bool:
    return False


def _navigation_unavailable(state: DocVQAState) -> bool:
    navigation = state.metadata.get("navigation", {})
    return navigation.get("last_error") == "missing_llm_client"
