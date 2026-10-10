from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import DocumentNode
from .in_memory import InMemoryDocumentEnvironment
from .mineru_loader import load_nodes_from_mineru_output


class MineruDocumentEnvironment(InMemoryDocumentEnvironment):
    """Bootstrap environment for pre-normalized MinerU-derived nodes."""

    @classmethod
    def from_nodes_file(
        cls,
        path: str | Path,
        *,
        workspace_dir: str | Path | None = None,
    ) -> "MineruDocumentEnvironment":
        items: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = [DocumentNode(**item) for item in items]
        return cls(nodes, workspace_dir=workspace_dir)

    @classmethod
    def from_mineru_output_dir(
        cls,
        path: str | Path,
        *,
        doc_id: str | None = None,
        workspace_dir: str | Path | None = None,
        page_range: tuple[int, int] | None = None,
        page_filter: set[int] | None = None,
    ) -> "MineruDocumentEnvironment":
        nodes = load_nodes_from_mineru_output(
            path,
            doc_id=doc_id,
            page_range=page_range,
            page_filter=page_filter,
        )
        return cls(nodes, workspace_dir=workspace_dir)
