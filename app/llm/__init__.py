from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .main import DEFAULT_MODEL, async_get_answer, get_answer


def _load_core_llm_module():
    path = Path(__file__).resolve().parent.parent / "llm.py"
    spec = importlib.util.spec_from_file_location("app._core_llm", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load core LLM module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_core_llm = _load_core_llm_module()
LLMConfig = _core_llm.LLMConfig
LLMClient = _core_llm.LLMClient

__all__ = ["DEFAULT_MODEL", "LLMClient", "LLMConfig", "get_answer", "async_get_answer"]
