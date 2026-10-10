from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..environment import DocumentEnvironment
from ..schema import Action, DocVQAState, Observation


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }

    @abstractmethod
    async def execute(
        self,
        action: Action,
        state: DocVQAState,
        env: DocumentEnvironment,
    ) -> Observation:
        pass

    def to_param(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": self.to_param(),
        }
