from __future__ import annotations

from ..schema import Action, DocVQAState, Observation
from .base import BaseTool


class FinishTool(BaseTool):
    name = "finish"
    description = "Finish the run and persist the final answer/verification state into the observation."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        state.done = True
        state.sync_evidence_state()
        candidate_answer = state.candidate_answer.to_dict() if state.candidate_answer else None
        verification = state.verification.to_dict() if state.verification else None
        message = "Finish executed."
        if candidate_answer is not None and verification is not None:
            message = "Finish executed after answer and verification were available."
        elif candidate_answer is not None:
            message = "Finish executed with a candidate answer available."
        return Observation(
            action_type="finish",
            success=True,
            message=message,
            payload={
                "reason": action.intent,
                "candidate_answer": candidate_answer,
                "verification": verification,
                "done": True,
            },
        )
