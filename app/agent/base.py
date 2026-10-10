from __future__ import annotations

from abc import ABC, abstractmethod

from ..environment import DocumentEnvironment
from ..schema import Action, DocVQAState, Observation
from ..tool import ToolCollection


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def next_action(
        self, state: DocVQAState, env: DocumentEnvironment
    ) -> Action:
        pass

    async def dispatch(
        self,
        tool_name: str,
        action: Action,
        state: DocVQAState,
        env: DocumentEnvironment,
        tools: ToolCollection,
    ) -> Observation:
        return await tools.execute(tool_name, action, state, env)

