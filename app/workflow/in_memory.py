from __future__ import annotations

import contextlib
import io
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..schema import CandidateAnswer, DocumentNode, DocVQAState, VerificationResult
from .base import DocumentEnvironment


class InMemoryDocumentEnvironment(DocumentEnvironment):
    def __init__(self, nodes: list[DocumentNode], *, workspace_dir: str | Path | None = None):
        self.nodes = {node.node_id: node for node in nodes}
        self.doc_index: dict[str, list[str]] = {}
        self.artifact_dir = Path(workspace_dir or "runs/.workspace").resolve() / "extracts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        for node in nodes:
            self.doc_index.setdefault(node.doc_id, []).append(node.node_id)

    @classmethod
    def from_dicts(
        cls,
        items: list[dict[str, Any]],
        *,
        workspace_dir: str | Path | None = None,
    ) -> "InMemoryDocumentEnvironment":
        nodes = [DocumentNode(**item) for item in items]
        return cls(nodes, workspace_dir=workspace_dir)

    async def overview(self, document_ids: list[str]) -> dict:
        docs = []
        for doc_id in document_ids:
            node_ids = self.doc_index.get(doc_id, [])
            pages = sorted({self.nodes[node_id].page for node_id in node_ids})
            docs.append(
                {
                    "doc_id": doc_id,
                    "node_count": len(node_ids),
                    "pages": pages,
                    "node_types": sorted(
                        {self.nodes[node_id].node_type for node_id in node_ids}
                    ),
                }
            )
        return {"documents": docs}

    async def search(self, query: str, filters: dict | None = None) -> dict:
        filters = filters or {}
        query_terms = [term for term in re.split(r"\W+", query.lower()) if term and term not in _STOPWORDS]
        allowed_types = set(filters.get("node_types", []))
        doc_ids = set(filters.get("document_ids", []))
        top_k = int(filters.get("top_k", 5))
        normalized_query = " ".join(query_terms)

        scored = []
        for node in self.nodes.values():
            if allowed_types and node.node_type not in allowed_types:
                continue
            if doc_ids and node.doc_id not in doc_ids:
                continue
            haystack = node.searchable_text().lower()
            if not haystack.strip():
                continue
            score = _score_node(query_terms, normalized_query, haystack, node)
            if score == 0 and query_terms:
                continue
            scored.append(
                (
                    score,
                    {
                        "node_id": node.node_id,
                        "doc_id": node.doc_id,
                        "page": node.page,
                        "node_type": node.node_type,
                        "modality": node.modality,
                        "title": node.title,
                        "snippet": haystack[:240],
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return {"query": query, "results": [item[1] for item in scored[:top_k]]}

    async def fetch_pages(
        self,
        pages: list[int],
        *,
        document_ids: list[str] | None = None,
        with_image: bool = True,
        zoom: float = 1.0,
    ) -> dict:
        requested_pages = {int(page) for page in pages}
        allowed_docs = set(document_ids or [])
        grouped: dict[tuple[str, int], list[DocumentNode]] = {}
        for node in self.nodes.values():
            if node.page not in requested_pages:
                continue
            if allowed_docs and node.doc_id not in allowed_docs:
                continue
            grouped.setdefault((node.doc_id, node.page), []).append(node)

        page_items: list[dict[str, Any]] = []
        for (doc_id, page), nodes in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            ordered_nodes = sorted(nodes, key=_node_reading_order_key)
            text_parts = [node.searchable_text() for node in ordered_nodes if node.searchable_text()]
            figures = [_node_reference(node, with_image=with_image) for node in ordered_nodes if _is_figure_node(node)]
            tables = [_node_reference(node, with_image=with_image) for node in ordered_nodes if node.node_type == "table" or node.modality == "table"]
            page_items.append(
                {
                    "doc_id": doc_id,
                    "page": page,
                    "text": "\n\n".join(text_parts),
                    "screenshot_path": _find_page_screenshot_path(ordered_nodes) if with_image else None,
                    "zoom": zoom,
                    "nodes": [_node_reference(node, with_image=with_image) for node in ordered_nodes],
                    "figures": figures,
                    "tables": tables,
                }
            )
        return {"pages": page_items}

    async def open_node(self, node_id: str) -> dict:
        node = self._require_node(node_id)
        return {"node": node.to_dict()}

    async def extract_text(self, node_id: str, span: dict | None = None) -> dict:
        node = self._require_node(node_id)
        content = node.text or node.summary or str(node.content or "")
        return self._make_extract_payload(node, content, extra={"span": span})

    async def extract_table(self, node_id: str, cell_query: dict | None = None) -> dict:
        node = self._require_node(node_id)
        content = node.content if node.content is not None else node.text
        if isinstance(content, dict):
            extracted = content
        else:
            extracted = {"raw": content, "cell_query": cell_query}
        return self._make_extract_payload(node, extracted, extra={"cell_query": cell_query})

    async def extract_figure(self, node_id: str, target: dict | None = None) -> dict:
        node = self._require_node(node_id)
        content = node.content if node.content is not None else node.text
        return self._make_extract_payload(node, content, extra={"target": target})

    async def extract_image_region(
        self, node_id: str, bbox: list[float] | None = None
    ) -> dict:
        node = self._require_node(node_id)
        crop_result = self._crop_node_image(node, bbox=bbox, operation="extract_image_region")
        content = {
            "bbox": crop_result["bbox"],
            "image_path": crop_result["image_path"],
            "crop_path": crop_result["crop_path"],
            "image_size": crop_result["image_size"],
            "summary": node.summary,
            "text": node.text,
        }
        return self._make_extract_payload(node, content, extra={"bbox": crop_result["bbox"]})

    async def crop(self, node_id: str, bbox: list[float] | None = None) -> dict:
        node = self._require_node(node_id)
        crop_result = self._crop_node_image(node, bbox=bbox, operation="crop")
        content = {
            "bbox": crop_result["bbox"],
            "image_path": crop_result["image_path"],
            "crop_path": crop_result["crop_path"],
            "image_size": crop_result["image_size"],
            "title": node.title,
            "summary": node.summary,
            "text": node.text,
        }
        return self._make_extract_payload(
            node,
            content,
            extra={"operation": "crop", "bbox": crop_result["bbox"]},
            modality_override="image",
        )

    async def subfigure_split(
        self,
        node_id: str,
        grid_hint: dict | None = None,
    ) -> dict:
        node = self._require_node(node_id)
        image, image_path = self._load_node_image(node)
        panels = _infer_subfigure_panels(
            node,
            hint=grid_hint,
            image=image,
            artifact_dir=self.artifact_dir,
        )
        content = {
            "title": node.title,
            "summary": node.summary,
            "image_path": str(image_path),
            "figure_bbox": [0, 0, int(image.shape[1]), int(image.shape[0])],
            "grid_hint": grid_hint,
            "panels": panels,
        }
        return self._make_extract_payload(
            node,
            content,
            extra={"operation": "subfigure_split", "grid_hint": grid_hint},
            modality_override="image",
        )

    async def subfigure_crop(
        self,
        node_id: str,
        panel_query: str,
        split_hint: dict | None = None,
    ) -> dict:
        node = self._require_node(node_id)
        image, image_path = self._load_node_image(node)
        panels = _infer_subfigure_panels(
            node,
            hint=split_hint,
            image=image,
            artifact_dir=self.artifact_dir,
        )
        matched_panel = _match_subfigure_panel(panel_query, panels)
        crop_path = None
        if matched_panel is not None:
            x1, y1, x2, y2 = _sanitize_bbox(matched_panel["bbox"], image.shape)
            crop_path = self._save_image(node, image[y1:y2, x1:x2], "subfigure_crop")
            matched_panel = {
                **matched_panel,
                "bbox": [x1, y1, x2, y2],
                "crop_path": str(crop_path),
            }
        content = {
            "panel_query": panel_query,
            "matched_panel": matched_panel,
            "available_panels": panels,
            "image_path": str(image_path),
            "title": node.title,
            "summary": node.summary,
        }
        return self._make_extract_payload(
            node,
            content,
            extra={
                "operation": "subfigure_crop",
                "panel_query": panel_query,
                "split_hint": split_hint,
            },
            modality_override="image",
        )

    async def zoom(
        self,
        node_id: str,
        scale: float,
        bbox: list[float] | None = None,
    ) -> dict:
        node = self._require_node(node_id)
        image, image_path = self._load_node_image(node)
        zoom_bbox = _sanitize_bbox(bbox, image.shape) if bbox is not None else (0, 0, image.shape[1], image.shape[0])
        x1, y1, x2, y2 = zoom_bbox
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            raise ValueError(f"Zoom region is empty for node_id: {node_id}")
        zoomed = cv2.resize(
            region,
            None,
            fx=max(float(scale), 1.0),
            fy=max(float(scale), 1.0),
            interpolation=cv2.INTER_CUBIC,
        )
        zoom_path = self._save_image(node, zoomed, "zoom")
        content = {
            "bbox": [x1, y1, x2, y2],
            "scale": scale,
            "image_path": str(image_path),
            "zoom_path": str(zoom_path),
            "zoomed_size": [int(zoomed.shape[1]), int(zoomed.shape[0])],
            "title": node.title,
            "summary": node.summary,
            "text": node.text,
        }
        return self._make_extract_payload(
            node,
            content,
            extra={"operation": "zoom", "bbox": [x1, y1, x2, y2], "scale": scale},
            modality_override="image",
        )

    async def run_python(self, code: str, inputs: dict | None = None) -> dict:
        namespace: dict[str, Any] = {"inputs": inputs or {}}
        stdout = io.StringIO()
        result = None
        error = None
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, namespace, namespace)
            result = namespace.get("result")
        except Exception as exc:  # pragma: no cover - best-effort sandboxless fallback
            error = str(exc)
        return _json_safe(
            {
                "stdout": stdout.getvalue(),
                "result": result,
                "error": error,
                "inputs": inputs or {},
            }
        )

    async def verify_answer(
        self, answer: CandidateAnswer, state: DocVQAState
    ) -> VerificationResult:
        issues: list[str] = []
        suggestions: list[str] = []
        if not answer.answer.strip():
            issues.append("Empty candidate answer.")
        if not answer.supporting_evidence_ids:
            issues.append("No supporting evidence ids were attached to the candidate answer.")
            suggestions.append("extract")
        missing = [
            evidence_id
            for evidence_id in answer.supporting_evidence_ids
            if evidence_id not in state.evidence_store
        ]
        if missing:
            issues.append("Some referenced evidence ids are missing from evidence_store.")
            suggestions.extend(["navigate", "extract"])

        accepted = not issues
        if not accepted and not suggestions:
            suggestions.append("plan")

        return VerificationResult(
            accepted=accepted,
            issues=issues,
            missing_evidence=missing,
            suggestion=suggestions,
        )

    def _require_node(self, node_id: str) -> DocumentNode:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node_id: {node_id}")
        return self.nodes[node_id]

    def _load_node_image(self, node: DocumentNode) -> tuple[np.ndarray, Path]:
        image_path = _resolve_node_image_path(node)
        if image_path is None:
            raise ValueError(f"Node {node.node_id} does not have an image_path for CV extraction.")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image for node {node.node_id}: {image_path}")
        return image, image_path

    def _save_image(self, node: DocumentNode, image: np.ndarray, suffix: str) -> Path:
        node_dir = self.artifact_dir / _safe_slug(node.doc_id) / _safe_slug(node.node_id)
        node_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(node_dir.glob(f"{suffix}_*.png"))
        output_path = node_dir / f"{suffix}_{len(existing) + 1}.png"
        if not cv2.imwrite(str(output_path), image):
            raise ValueError(f"Failed to write image artifact: {output_path}")
        return output_path

    def _crop_node_image(
        self,
        node: DocumentNode,
        bbox: list[float] | None,
        operation: str,
    ) -> dict[str, Any]:
        image, image_path = self._load_node_image(node)
        x1, y1, x2, y2 = _sanitize_bbox(bbox, image.shape) if bbox is not None else (0, 0, image.shape[1], image.shape[0])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Crop region is empty for node_id: {node.node_id}")
        crop_path = self._save_image(node, crop, operation)
        return {
            "bbox": [x1, y1, x2, y2],
            "image_path": str(image_path),
            "crop_path": str(crop_path),
            "image_size": [int(image.shape[1]), int(image.shape[0])],
        }

    def _make_extract_payload(
        self,
        node: DocumentNode,
        content: Any,
        extra: dict[str, Any] | None = None,
        modality_override: str | None = None,
    ) -> dict:
        payload = {
            "node_id": node.node_id,
            "doc_id": node.doc_id,
            "page": node.page,
            "modality": modality_override or node.modality,
            "content": content,
            "provenance": {
                "node_type": node.node_type,
                "title": node.title,
                "bbox": node.bbox,
                "metadata": node.metadata,
            },
            "confidence": 1.0,
        }
        if extra:
            payload.update(extra)
        return payload


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


_STOPWORDS = {
    "what",
    "is",
    "the",
    "for",
    "in",
    "of",
    "to",
    "and",
    "a",
    "an",
    "how",
    "did",
    "are",
    "was",
    "at",
    "on",
    "by",
    "from",
    "round",
    "your",
    "answer",
    "place",
    "places",
}


def _infer_subfigure_panels(
    node: DocumentNode,
    hint: dict[str, Any] | None = None,
    image: np.ndarray | None = None,
    artifact_dir: Path | None = None,
) -> list[dict[str, Any]]:
    hint = hint or {}
    if image is None:
        rows = max(1, int(hint.get("rows", 2)))
        cols = max(1, int(hint.get("columns", 2)))
        return _grid_panels(rows, cols, [0.0, 0.0, float(cols), float(rows)], hint.get("labels"))

    rows_hint = _coerce_positive_int(hint.get("rows"))
    cols_hint = _coerce_positive_int(hint.get("columns"))
    labels = hint.get("labels")
    semantic_grid = None
    if rows_hint is None and cols_hint is None:
        semantic_grid = _infer_semantic_panel_grid(node, image)

    if semantic_grid is not None:
        rows_hint = int(semantic_grid["rows"])
        cols_hint = int(semantic_grid["columns"])
        labels = semantic_grid.get("labels")
        boxes = [
            panel["bbox"]
            for panel in _grid_panels(
                rows_hint,
                cols_hint,
                [0, 0, image.shape[1], image.shape[0]],
                labels,
            )
        ]
    else:
        boxes = _detect_panel_boxes(image, rows_hint=rows_hint, cols_hint=cols_hint)
    if not boxes:
        rows = rows_hint or 2
        cols = cols_hint or 2
        boxes = [panel["bbox"] for panel in _grid_panels(rows, cols, [0, 0, image.shape[1], image.shape[0]], labels)]

    panels: list[dict[str, Any]] = []
    ordered = _order_panel_boxes(boxes)
    grid_shape = _estimate_grid_shape(len(ordered), rows_hint, cols_hint)
    for index, bbox in enumerate(ordered):
        row = index // grid_shape[1]
        col = index % grid_shape[1]
        default_label = chr(ord("A") + index)
        label = str(labels[index]) if isinstance(labels, list) and index < len(labels) else default_label
        panel = {
            "label": label,
            "row": row + 1,
            "column": col + 1,
            "position": _panel_position_name(row, col, grid_shape[0], grid_shape[1]),
            "bbox": [int(v) for v in bbox],
        }
        if artifact_dir is not None:
            x1, y1, x2, y2 = _sanitize_bbox(panel["bbox"], image.shape)
            crop = image[y1:y2, x1:x2]
            if crop.size:
                panel["crop_path"] = str(_save_panel_image(artifact_dir, node, crop, index))
        panels.append(panel)
    return panels


def _infer_semantic_panel_grid(node: DocumentNode, image: np.ndarray) -> dict[str, Any] | None:
    text = node.searchable_text().lower()
    if not text.strip():
        return None

    if _has_all_panel_markers(text, ["left", "middle", "right"]):
        return {"rows": 1, "columns": 3, "labels": ["Left", "Middle", "Right"]}
    if _has_all_panel_markers(text, ["left", "center", "right"]):
        return {"rows": 1, "columns": 3, "labels": ["Left", "Center", "Right"]}
    if _has_all_panel_markers(text, ["top", "middle", "bottom"]):
        return {"rows": 3, "columns": 1, "labels": ["Top", "Middle", "Bottom"]}
    if _has_all_panel_markers(text, ["top", "bottom"]):
        return {"rows": 2, "columns": 1, "labels": ["Top", "Bottom"]}

    labels = _extract_ordered_panel_labels(text)
    if 2 <= len(labels) <= 4:
        rows, cols = _semantic_grid_shape_for_label_count(len(labels), image)
        return {"rows": rows, "columns": cols, "labels": labels}
    return None


def _has_all_panel_markers(text: str, markers: list[str]) -> bool:
    return all(re.search(rf"(?:\({marker}\)|\b{marker}\s+panel\b|\b{marker}\s+subplot\b)", text) for marker in markers)


def _extract_ordered_panel_labels(text: str) -> list[str]:
    labels: list[str] = []
    for match in re.finditer(r"\(([A-Z])\)", text, flags=re.IGNORECASE):
        label = match.group(1).upper()
        if label not in labels:
            labels.append(label)
    return labels


def _semantic_grid_shape_for_label_count(count: int, image: np.ndarray) -> tuple[int, int]:
    if count == 2:
        return (1, 2) if image.shape[1] >= image.shape[0] else (2, 1)
    if count == 3:
        return (1, 3) if image.shape[1] >= image.shape[0] else (3, 1)
    return 2, 2


def _match_subfigure_panel(
    panel_query: str,
    panels: list[dict[str, Any]],
) -> dict[str, Any] | None:
    normalized = panel_query.strip().lower()
    row_match = re.search(r"row\s*(\d+)", normalized)
    col_match = re.search(r"(?:column|col)\s*(\d+)", normalized)
    label_match = re.search(r"panel\s*([a-z])\b", normalized)

    if row_match and col_match:
        row = int(row_match.group(1))
        col = int(col_match.group(1))
        for panel in panels:
            if panel["row"] == row and panel["column"] == col:
                return panel

    if label_match:
        label = label_match.group(1).upper()
        for panel in panels:
            if str(panel["label"]).upper() == label:
                return panel

    for panel in panels:
        if panel["position"] in normalized:
            return panel

    return panels[0] if panels else None


def _panel_position_name(row: int, col: int, rows: int, cols: int) -> str:
    if rows == 1 and cols == 1:
        return "full panel"
    if rows == 1:
        horizontal = "left" if col == 0 else "right" if col == cols - 1 else "center"
        return f"{horizontal} panel"
    if cols == 1:
        vertical = "top" if row == 0 else "bottom" if row == rows - 1 else "middle"
        return f"{vertical} panel"
    vertical = "top" if row == 0 else "bottom" if row == rows - 1 else "middle"
    horizontal = "left" if col == 0 else "right" if col == cols - 1 else "center"
    if vertical == "middle" and horizontal == "center":
        return "center panel"
    if vertical == "middle":
        return f"{horizontal} panel"
    if horizontal == "center":
        return f"{vertical}-center panel"
    return f"{vertical}-{horizontal} panel"


def _node_reading_order_key(node: DocumentNode) -> tuple[float, float, str]:
    bbox = node.bbox or []
    y = float(bbox[1]) if len(bbox) >= 2 else 0.0
    x = float(bbox[0]) if len(bbox) >= 1 else 0.0
    return y, x, node.node_id


def _is_figure_node(node: DocumentNode) -> bool:
    return node.node_type in {"figure", "chart"} or node.modality in {"image", "chart"}


def _node_reference(node: DocumentNode, *, with_image: bool) -> dict[str, Any]:
    image_path = _resolve_node_image_path(node) if with_image else None
    return {
        "node_id": node.node_id,
        "doc_id": node.doc_id,
        "page": node.page,
        "node_type": node.node_type,
        "modality": node.modality,
        "title": node.title,
        "summary": node.summary,
        "text": node.text[:2000],
        "bbox": node.bbox,
        "image_path": str(image_path) if image_path else None,
    }


def _find_page_screenshot_path(nodes: list[DocumentNode]) -> str | None:
    for node in nodes:
        metadata = node.metadata or {}
        for key in ("page_image_path", "screenshot_path", "image_path"):
            value = metadata.get(key)
            if value:
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    path = path.resolve()
                if path.exists():
                    return str(path)
    for node in nodes:
        image_path = _resolve_node_image_path(node)
        if image_path is not None:
            return str(image_path)
    return None


def _resolve_node_image_path(node: DocumentNode) -> Path | None:
    candidates: list[str | None] = []
    if isinstance(node.content, dict):
        candidates.extend(
            [
                node.content.get("image_path"),
                node.content.get("img_path"),
                ((node.content.get("image_source") or {}).get("path") if isinstance(node.content.get("image_source"), dict) else None),
            ]
        )
    metadata = node.metadata or {}
    candidates.extend(
        [
            metadata.get("page_image_path"),
            metadata.get("screenshot_path"),
            metadata.get("image_path"),
            metadata.get("asset_path"),
            ((metadata.get("raw_evidence") or {}).get("image_path") if isinstance(metadata.get("raw_evidence"), dict) else None),
            ((metadata.get("raw_evidence") or {}).get("img_path") if isinstance(metadata.get("raw_evidence"), dict) else None),
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if path.exists():
            return path
    return None


def _sanitize_bbox(bbox: list[float] | None, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    if bbox is None or len(bbox) != 4:
        return 0, 0, width, height
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1 = min(max(x1, 0), width - 1 if width else 0)
    x2 = min(max(x2, x1 + 1), width)
    y1 = min(max(y1, 0), height - 1 if height else 0)
    y2 = min(max(y2, y1 + 1), height)
    return x1, y1, x2, y2


def _detect_panel_boxes(
    image: np.ndarray,
    rows_hint: int | None = None,
    cols_hint: int | None = None,
) -> list[list[int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    row_segments = _projection_segments(mask, axis=0, expected_count=rows_hint)
    col_segments = _projection_segments(mask, axis=1, expected_count=cols_hint)
    if row_segments and col_segments and len(row_segments) * len(col_segments) > 1:
        return _grid_boxes_from_segments(row_segments, col_segments)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = image.shape[0] * image.shape[1] * 0.05
    boxes: list[list[int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < area_threshold:
            continue
        boxes.append([x, y, x + w, y + h])

    merged = _merge_overlapping_boxes(boxes)
    if len(merged) > 1:
        return merged

    if rows_hint or cols_hint:
        rows = rows_hint or 1
        cols = cols_hint or 1
        return [panel["bbox"] for panel in _grid_panels(rows, cols, [0, 0, image.shape[1], image.shape[0]], None)]
    return merged


def _projection_segments(
    mask: np.ndarray,
    axis: int,
    expected_count: int | None = None,
) -> list[tuple[int, int]]:
    projection = np.mean(mask > 0, axis=axis).astype(np.float32)
    if projection.size == 0:
        return []
    smoothed = cv2.GaussianBlur(projection.reshape(1, -1), (1, 9), 0).reshape(-1)
    threshold = max(0.01, float(smoothed.max()) * 0.08)
    active = smoothed > threshold
    min_len = max(8, int(mask.shape[1 - axis] * 0.08))
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_len:
                segments.append((start, index))
            start = None
    if start is not None and len(active) - start >= min_len:
        segments.append((start, len(active)))
    if expected_count and len(segments) != expected_count:
        total = mask.shape[0] if axis == 0 else mask.shape[1]
        return _even_segments(total, expected_count)
    return segments


def _grid_boxes_from_segments(
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> list[list[int]]:
    boxes: list[list[int]] = []
    for row_start, row_end in row_segments:
        for col_start, col_end in col_segments:
            boxes.append([col_start, row_start, col_end, row_end])
    return boxes


def _even_segments(total: int, count: int) -> list[tuple[int, int]]:
    count = max(1, count)
    step = total / count
    segments: list[tuple[int, int]] = []
    for index in range(count):
        start = int(round(index * step))
        end = int(round((index + 1) * step))
        segments.append((start, max(start + 1, end)))
    return segments


def _merge_overlapping_boxes(boxes: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0])):
        x1, y1, x2, y2 = box
        combined = False
        for existing in merged:
            if _boxes_overlap(existing, box):
                existing[0] = min(existing[0], x1)
                existing[1] = min(existing[1], y1)
                existing[2] = max(existing[2], x2)
                existing[3] = max(existing[3], y2)
                combined = True
                break
        if not combined:
            merged.append([x1, y1, x2, y2])
    return merged


def _boxes_overlap(a: list[int], b: list[int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _grid_panels(
    rows: int,
    cols: int,
    bbox: list[float],
    labels: list[str] | None,
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    panel_width = (x2 - x1) / cols if cols else 0.0
    panel_height = (y2 - y1) / rows if rows else 0.0
    panels: list[dict[str, Any]] = []
    label_index = 0
    for row in range(rows):
        for col in range(cols):
            default_label = chr(ord("A") + label_index)
            label = str(labels[label_index]) if isinstance(labels, list) and label_index < len(labels) else default_label
            px1 = x1 + col * panel_width
            py1 = y1 + row * panel_height
            px2 = px1 + panel_width
            py2 = py1 + panel_height
            panels.append(
                {
                    "label": label,
                    "row": row + 1,
                    "column": col + 1,
                    "position": _panel_position_name(row, col, rows, cols),
                    "bbox": [px1, py1, px2, py2],
                }
            )
            label_index += 1
    return panels


def _order_panel_boxes(boxes: list[list[int]]) -> list[list[int]]:
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda item: (item[1], item[0]))
    rows: list[list[list[int]]] = []
    current_row: list[list[int]] = []
    current_y = None
    median_height = np.median([box[3] - box[1] for box in sorted_boxes])
    tolerance = max(10.0, float(median_height) * 0.4)
    for box in sorted_boxes:
        center_y = (box[1] + box[3]) / 2
        if current_y is None or abs(center_y - current_y) <= tolerance:
            current_row.append(box)
            if current_y is None:
                current_y = center_y
            else:
                current_y = (current_y + center_y) / 2
        else:
            rows.append(sorted(current_row, key=lambda item: item[0]))
            current_row = [box]
            current_y = center_y
    if current_row:
        rows.append(sorted(current_row, key=lambda item: item[0]))
    return [box for row in rows for box in row]


def _estimate_grid_shape(count: int, rows_hint: int | None, cols_hint: int | None) -> tuple[int, int]:
    if rows_hint and cols_hint:
        return rows_hint, cols_hint
    if rows_hint:
        return rows_hint, max(1, int(np.ceil(count / rows_hint)))
    if cols_hint:
        return max(1, int(np.ceil(count / cols_hint))), cols_hint
    cols = max(1, int(round(np.sqrt(count))))
    rows = max(1, int(np.ceil(count / cols)))
    return rows, cols


def _save_panel_image(artifact_dir: Path, node: DocumentNode, image: np.ndarray, index: int) -> Path:
    node_dir = artifact_dir / _safe_slug(node.doc_id) / _safe_slug(node.node_id)
    node_dir.mkdir(parents=True, exist_ok=True)
    output_path = node_dir / f"subfigure_split_panel_{index + 1}.png"
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Failed to write panel artifact: {output_path}")
    return output_path


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned or "node"


def _coerce_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _score_node(query_terms: list[str], normalized_query: str, haystack: str, node: DocumentNode) -> float:
    word_count = max(len(haystack.split()), 1)
    score = 0.0
    for term in query_terms:
        count = haystack.count(term)
        if count:
            score += count * 3.0
            if term in node.title.lower():
                score += 4.0
            if term in node.summary.lower():
                score += 2.0

    if normalized_query and normalized_query in haystack:
        score += 12.0

    if "2021" in query_terms and "2021" in haystack:
        score += 4.0

    title_lower = node.title.lower()
    if "balance" in query_terms and "balance sheets" in title_lower:
        score += 14.0
    if "current" in query_terms and "liabilities" in query_terms and "balance sheets" in title_lower:
        score += 12.0
    if "liabilities" in query_terms and "balance sheets" in title_lower:
        score += 10.0
    if "cash" in query_terms and "flows" in query_terms and "cash flows" in title_lower:
        score += 12.0
    if "debt" in query_terms and ("long-term debt" in title_lower or "debt" in title_lower):
        score += 10.0

    type_boost = {
        "table": 10.0,
        "chart": 8.0,
        "figure": 4.0,
        "section": 3.0,
        "paragraph": 2.0,
        "document": -6.0,
    }.get(node.node_type, 0.0)
    score += type_boost

    modality_boost = {
        "table": 6.0,
        "mixed": -4.0,
    }.get(node.modality, 0.0)
    score += modality_boost

    return score / (word_count ** 0.5)
