from pathlib import Path

from .analyze import AnalyzeTool
from .answer import AnswerTool
from .base import BaseTool
from .collection import ToolCollection
from .execute import ExecuteTool
from .finish import FinishTool
from .focus import FocusTool
from .solver import SolverTool
from ..config import ExtractConfig, NavigateConfig
from ..llm import LLMClient
from .extract import (
    CropTool,
    ExtractTool,
    ExtractFigureTool,
    ExtractTableTool,
    ExtractTextTool,
    SubfigureSplitTool,
    ZoomTool,
)
from .navigate import NavigateTool


def build_default_tool_collection(
    llm_client: LLMClient | None = None,
    *,
    navigate_config: NavigateConfig | None = None,
    extract_config: ExtractConfig | None = None,
    workspace_dir: str | Path | None = None,
) -> ToolCollection:
    extract_config = extract_config or ExtractConfig()
    return ToolCollection(
        AnalyzeTool(llm_client=llm_client),
        NavigateTool(
            llm_client=llm_client,
            navigate_config=navigate_config,
            workspace_dir=workspace_dir,
        ),
        ExtractTool(llm_client=llm_client, extract_config=extract_config),
        FocusTool(llm_client=llm_client),
        SolverTool(llm_client=llm_client),
        ExecuteTool(),
        AnswerTool(llm_client=llm_client),
        FinishTool(),
    )


__all__ = [
    "AnalyzeTool",
    "AnswerTool",
    "BaseTool",
    "CropTool",
    "ExecuteTool",
    "FinishTool",
    "ExtractTool",
    "ExtractFigureTool",
    "ExtractTableTool",
    "ExtractTextTool",
    "FocusTool",
    "NavigateTool",
    "SolverTool",
    "SubfigureSplitTool",
    "ToolCollection",
    "ZoomTool",
    "build_default_tool_collection",
]
