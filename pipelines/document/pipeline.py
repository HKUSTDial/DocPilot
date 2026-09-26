from __future__ import annotations

from collections.abc import Callable

from .config import DocumentPipelineConfig
from .indexing import build_document_registries
from .mineru import build_mineru_manifest, build_page_objects_from_mineru, run_mineru


StageFn = Callable[[DocumentPipelineConfig], dict[str, int]]


STAGES: dict[str, StageFn] = {
    "index": build_document_registries,
    "mineru-manifest": build_mineru_manifest,
    "mineru-run": run_mineru,
    "page-objects": build_page_objects_from_mineru,
}

DEFAULT_STAGES = [
    "index",
    "mineru-manifest",
    "mineru-run",
    "page-objects",
]


def run_stages(
    config: DocumentPipelineConfig,
    stages: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    selected = stages or DEFAULT_STAGES
    results: dict[str, dict[str, int]] = {}
    for stage in selected:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {', '.join(STAGES)}")
        results[stage] = STAGES[stage](config)
    return results
