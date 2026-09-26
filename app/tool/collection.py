from __future__ import annotations

from ..environment import DocumentEnvironment
from ..schema import Action, DocVQAState, Observation
from .base import BaseTool


class ToolCollection:
    def __init__(self, *tools: BaseTool):
        self.tools = {tool.name: tool for tool in tools}

    def add_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def to_params(self) -> list[dict]:
        return [tool.to_param() for tool in self.tools.values()]

    def to_openai_tools(self) -> list[dict]:
        return [tool.to_openai_tool() for tool in self.tools.values()]

    async def execute(
        self, tool_name: str, action: Action, state: DocVQAState, env: DocumentEnvironment
    ) -> Observation:
        if tool_name not in self.tools:
            raise KeyError(f"Unknown tool: {tool_name}")
        return await self.tools[tool_name].execute(action, state, env)
