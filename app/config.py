from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os

from .llm import LLMConfig
from .schema import BudgetState


@dataclass
class OutputConfig:
    show_state: bool = False
    logs_dir: Path = Path("runs/logs")
    outputs_dir: Path = Path("runs/outputs")
    document_dir: Path = Path("runs/environment")
    workspace_dir: Path = Path("runs/.workspace")


@dataclass
class NavigateConfig:
    top_k: int | None = 5
    modal: str = "image"
    overview: bool = False
    page_thumbnail_size: int = 256
    page_thumbnail_type: str = "png"
    image_page_set_size: int = 8
    max_tokens: int | None = None


@dataclass
class ExtractConfig:
    if_enhanced: bool = True
    max_pages: int | None = 4


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    budget: BudgetState = field(default_factory=lambda: BudgetState(max_steps=16))
    navigate: NavigateConfig = field(default_factory=NavigateConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def _as_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "llm": {
            "enabled": config.llm.enabled,
            "provider": config.llm.provider,
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "api_key": config.llm.api_key,
            "account_access_token": config.llm.account_access_token,
            "account_user_id": config.llm.account_user_id,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
            "timeout": config.llm.timeout,
            "max_attempts": config.llm.max_attempts,
            "seed": config.llm.seed,
            "reasoning": config.llm.reasoning,
            "reasoning_effort": config.llm.reasoning_effort,
            "response_format": config.llm.response_format,
            "judge_model": config.llm.judge_model,
            "judge_base_url": config.llm.judge_base_url,
            "judge_api_key": config.llm.judge_api_key,
            "judge_temperature": config.llm.judge_temperature,
            "judge_max_tokens": config.llm.judge_max_tokens,
            "image_type": config.llm.image_type,
            "protocol": config.llm.protocol,
        },
        "budget": {
            "max_steps": config.budget.max_steps,
            "max_search_calls": config.budget.max_search_calls,
            "max_extract_calls": config.budget.max_extract_calls,
            "max_compute_calls": config.budget.max_compute_calls,
            "image_type": config.budget.image_type,
            "disabled_tools": list(config.budget.disabled_tools),
        },
        "navigate": {
            "top_k": config.navigate.top_k,
            "modal": config.navigate.modal,
            "overview": config.navigate.overview,
            "page_thumbnail_size": config.navigate.page_thumbnail_size,
            "image_page_set_size": config.navigate.image_page_set_size,
            "max_tokens": config.navigate.max_tokens,
        },
        "extract": {
            "if_enhanced": config.extract.if_enhanced,
            "max_pages": config.extract.max_pages,
        },
        "output": {
            "show_state": config.output.show_state,
            "logs_dir": str(config.output.logs_dir),
            "outputs_dir": str(config.output.outputs_dir),
            "document_dir": str(config.output.document_dir),
            "workspace_dir": str(config.output.workspace_dir),
        },
    }


def load_config(path: str | Path | None = None) -> AppConfig:
    base = _as_dict(AppConfig())
    if path:
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required for YAML config files.") from exc
            with path.open("r", encoding="utf-8") as f:
                patch = yaml.safe_load(f) or {}
        else:
            import json
            patch = json.loads(path.read_text(encoding="utf-8"))
        navigate_patch = patch.get("navigate") if isinstance(patch, dict) else None
        if isinstance(navigate_patch, dict) and "page_size" in navigate_patch and "page_thumbnail_size" not in navigate_patch:
            navigate_patch["page_thumbnail_size"] = navigate_patch.pop("page_size")
        if isinstance(navigate_patch, dict) and "page_thumbnail_type" in navigate_patch:
            budget_patch = patch.setdefault("budget", {})
            if isinstance(budget_patch, dict) and "image_type" not in budget_patch:
                budget_patch["image_type"] = navigate_patch["page_thumbnail_type"]
            navigate_patch.pop("page_thumbnail_type", None)
        _deep_update(base, _resolve_env(patch))

    protocol = str(base["llm"].get("protocol") or "json").strip().lower()
    if protocol not in {"json", "vllm_tags"}:
        raise ValueError(f"Unsupported llm.protocol: {protocol}")
    base["llm"]["protocol"] = protocol
    base["llm"]["image_type"] = base["budget"].get("image_type", "png")
    return AppConfig(
        llm=LLMConfig(**base["llm"]),
        budget=BudgetState(**base["budget"]),
        navigate=NavigateConfig(**base["navigate"]),
        extract=ExtractConfig(**base["extract"]),
        output=OutputConfig(
            show_state=base["output"]["show_state"],
            logs_dir=Path(base["output"]["logs_dir"]),
            outputs_dir=Path(base["output"]["outputs_dir"]),
            document_dir=Path(base["output"].get("document_dir", "runs/environment")),
            workspace_dir=Path(base["output"].get("workspace_dir", "runs/.workspace")),
        ),
    )
