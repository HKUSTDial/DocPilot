from __future__ import annotations

from ..schema import Action, DocVQAState, Observation
from .base import BaseTool


class ExecuteTool(BaseTool):
    name = "execute"
    description = "Execute Python code in the environment and return the result."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. If omitted, the tool uses code from `last_solver`.",
            },
            "inputs": {
                "type": "object",
                "description": "Serializable inputs for the execution. If omitted, uses inputs from `last_solver`.",
            },
        },
        "additionalProperties": False,
    }

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        solver_payload = state.metadata.get("last_solver", {})
        code = action.args.get("code") or solver_payload.get("code", "")
        inputs = action.args.get("inputs")
        if inputs is None:
            inputs = solver_payload.get("inputs", {})
        result = await env.run_python(code, inputs=inputs)
        state.metadata["last_execute"] = result
        return Observation(
            action_type="execute",
            success=result.get("error") is None,
            message=(
                f"Execution finished. Result: {result.get('result')!r}."
                if result.get("error") is None
                else f"Execution failed. Error: {result.get('error')}"
            ),
            payload=result,
        )
