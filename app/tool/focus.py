from __future__ import annotations

import json
import re
from typing import Any

import cv2

from ..llm import LLMClient
from ..prompt.focus import FOCUS_PLANNER_SYSTEM_PROMPT
from ..schema import Action, DocVQAState, Observation
from .base import BaseTool
from .extract import (
    CropTool,
    SubfigureSplitTool,
    ZoomTool,
    _parse_pages_arg,
    _pick_image_path,
    _preview_content,
)


class FocusTool(BaseTool):
    name = "focus"
    description = (
        "Perform local high-precision visual reading on already extracted figure or image evidence. "
        "Use it to inspect small labels, legends, axes, chart marks, subfigures, or spatial relationships "
        "by planning zoom, crop, or subfigure operations."
    )
    parameters = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "Natural-language focus goal.",
            },
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional figure evidence ids to inspect.",
            },
            "figure_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional figure refs from the latest extract observation.",
            },
            "pages": {
                "type": "string",
                "description": "Optional page list/ranges used to restrict candidate figures.",
            },
            "max_visual_ops": {"type": "integer"},
        },
        "additionalProperties": False,
    }

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client
        self._subtools = {
            "crop": CropTool(),
            "zoom": ZoomTool(),
            "subfigure_split": SubfigureSplitTool(),
        }

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        figure_records = _resolve_focus_targets(action.args, state)
        page_filter = _parse_pages_arg(action.args.get("pages"))
        if page_filter:
            figure_records = [record for record in figure_records if int(record.get("page", -1)) in page_filter]
        if not figure_records:
            return Observation(
                action_type="focus",
                success=False,
                message="No figure evidence available for focus.",
                payload={"error": "missing_focus_target"},
            )

        plan = self._plan_focus(action, state, figure_records)
        operations = list(plan.get("operations") or [])
        max_visual_ops = max(1, int(action.args.get("max_visual_ops") or 2))
        operations = operations[:max_visual_ops]
        if not operations:
            return Observation(
                action_type="focus",
                success=False,
                message="Focus planner did not find a useful visual operation.",
                payload={"planner": plan},
            )

        executed_operations: list[dict[str, Any]] = []
        evidence_ids: list[str] = []
        for operation in operations:
            figure = _match_focus_record(operation, figure_records)
            if figure is None:
                continue
            tool_name = str(operation["tool"])
            tool_args = _build_focus_tool_args(tool_name, operation, figure)
            observation = await self._subtools[tool_name].execute(
                Action(
                    action_type="focus",
                    agent_role="focus",
                    intent=str(action.args.get("instruction") or state.question),
                    args=tool_args,
                ),
                state,
                env,
            )
            evidence = observation.payload.get("evidence", {})
            evidence = _replace_focus_source_evidence(state, figure, evidence)
            evidence_id = evidence.get("evidence_id")
            if evidence_id:
                evidence_ids.append(evidence_id)
            executed_operations.append(
                {
                    **operation,
                    "node_id": tool_args["node_id"],
                    "success": observation.success,
                    "evidence_id": evidence_id,
                }
            )

        payload = {
            "mode": "focus_pipeline",
            "planner": plan,
            "evidence_ids": evidence_ids,
            "operations": executed_operations,
        }
        state.metadata["last_focus_observation"] = payload
        return Observation(
            action_type="focus",
            success=bool(evidence_ids),
            message=f"Focused visual extraction completed with {len(executed_operations)} operation(s).",
            payload=payload,
        )

    def _plan_focus(
        self,
        action: Action,
        state: DocVQAState,
        figure_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if "_operations" in action.args:
            operations = action.args.get("_operations")
            return _normalize_focus_plan(
                {"operations": operations if isinstance(operations, list) else []},
                figure_records,
            )
        instruction = str(action.args.get("instruction") or "").strip()
        verification = state.verification.to_dict() if state.verification else None
        llm_payload = self._plan_focus_with_llm(instruction, state, figure_records, verification)
        if llm_payload is not None:
            normalized = _normalize_focus_plan(llm_payload, figure_records)
            if normalized.get("operations"):
                return normalized
        return _heuristic_focus_plan(
            question=state.question,
            instruction=instruction,
            figure_records=figure_records,
            verification=verification,
        )

    def _plan_focus_with_llm(
        self,
        instruction: str,
        state: DocVQAState,
        figure_records: list[dict[str, Any]],
        verification: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self.llm is None:
            return None
        prompt = json.dumps(
            {
                "question": state.question,
                "instruction": instruction,
                "figure_evidence": [
                    {
                        "evidence_id": item.get("evidence_id"),
                        "figure_ref": item.get("figure_ref"),
                        "node_id": item.get("node_id"),
                        "page": item.get("page"),
                        "image_path": item.get("image_path"),
                        "caption": item.get("caption"),
                        "footnote": item.get("footnote"),
                        "content_preview": item.get("content_preview"),
                    }
                    for item in figure_records
                ],
                "understanding": state.metadata.get("understanding", {}),
                "verification": verification,
            },
            ensure_ascii=False,
        )
        try:
            self.llm.set_call_source("focus_planner")
            try:
                images = None
                if getattr(self.llm, "is_vllm_tags", False):
                    images = [
                        str(item["image_path"])
                        for item in figure_records
                        if item.get("image_path")
                    ]
                return self.llm.generate_json(
                    FOCUS_PLANNER_SYSTEM_PROMPT,
                    prompt,
                    images=images,
                )
            finally:
                self.llm.set_call_source(None)
        except Exception:
            if getattr(self.llm, "is_vllm_tags", False):
                raise
            return None


def _replace_focus_source_evidence(
    state: DocVQAState,
    figure: dict[str, Any],
    evidence_payload: dict[str, Any],
) -> dict[str, Any]:
    source_evidence_id = str(figure.get("evidence_id") or "").strip()
    new_evidence_id = str(evidence_payload.get("evidence_id") or "").strip()
    if not source_evidence_id or not new_evidence_id or source_evidence_id == new_evidence_id:
        return evidence_payload

    source_record = state.evidence_store.get(source_evidence_id)
    new_record = state.evidence_store.get(new_evidence_id)
    if source_record is None or new_record is None:
        return evidence_payload

    replacement = type(new_record)(
        evidence_id=source_evidence_id,
        node_id=new_record.node_id,
        page=new_record.page,
        modality=new_record.modality,
        content=new_record.content,
        provenance=new_record.provenance,
        confidence=new_record.confidence,
    )
    state.evidence_store.pop(new_evidence_id, None)
    state.evidence_store[source_evidence_id] = replacement

    updated_payload = dict(evidence_payload)
    updated_payload["evidence_id"] = source_evidence_id
    return updated_payload


def _resolve_focus_targets(args: dict[str, Any], state: DocVQAState) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    evidence_ids = args.get("evidence_ids")
    if isinstance(evidence_ids, list):
        for evidence_id in evidence_ids:
            if not isinstance(evidence_id, str):
                continue
            evidence = state.evidence_store.get(evidence_id)
            if evidence is None or evidence.modality not in {"image", "chart"}:
                continue
            resolved.append(
                {
                    "evidence_id": evidence_id,
                    "figure_ref": None,
                    "node_id": evidence.node_id,
                    "page": evidence.page,
                    "image_path": _pick_image_path(evidence.content),
                    "caption": evidence.provenance.get("title", ""),
                    "footnote": evidence.provenance.get("summary", ""),
                    "content_preview": _preview_content(evidence.content, limit=None),
                }
            )
    if resolved:
        return resolved

    figure_ref_names = args.get("figure_refs")
    last_extract = state.metadata.get("last_extract_observation", {})
    last_refs = last_extract.get("figure_refs", []) if isinstance(last_extract, dict) else []
    if isinstance(figure_ref_names, list):
        wanted = {str(item) for item in figure_ref_names}
        resolved = [item for item in last_refs if str(item.get("figure_ref")) in wanted]
    if resolved:
        return resolved

    recent_figure_ids = [
        evidence_id
        for evidence_id, evidence in reversed(list(state.evidence_store.items()))
        if evidence.modality in {"image", "chart"}
    ]
    for evidence_id in recent_figure_ids[:4]:
        evidence = state.evidence_store[evidence_id]
        resolved.append(
            {
                "evidence_id": evidence_id,
                "figure_ref": None,
                "node_id": evidence.node_id,
                "page": evidence.page,
                "image_path": _pick_image_path(evidence.content),
                "caption": evidence.provenance.get("title", ""),
                "footnote": evidence.provenance.get("summary", ""),
                "content_preview": _preview_content(evidence.content, limit=None),
            }
        )
    return resolved


def _heuristic_focus_plan(
    *,
    question: str,
    instruction: str,
    figure_records: list[dict[str, Any]],
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    focus_text = " ".join(
        item
        for item in [
            question,
            instruction,
            " ".join(verification.get("issues", [])) if verification else "",
            " ".join(verification.get("missing_evidence", [])) if verification else "",
            " ".join(verification.get("suggestion", [])) if verification else "",
        ]
        if item
    ).lower()
    primary = figure_records[0]
    tool = "zoom"
    target_description = "the visually relevant region needed to answer the question"
    region = {"type": "full", "description": target_description}
    visual_caption = "Read the fine-grained visual details in the figure region needed to answer the question."
    if _needs_specific_panel(focus_text):
        panel_query = _extract_panel_query(focus_text)
        tool = "subfigure_split"
        target_description = panel_query or "split the multi-panel figure into candidate panels"
        region = {
            "type": "panel",
            "description": target_description,
            "panel_query": panel_query,
        }
        visual_caption = f"Read the visual content of {target_description}."
    elif any(term in focus_text for term in ["source line", "source note", "footnote", "footer"]):
        tool = "crop"
        target_description = "bottom source note area of the figure"
        region = {"type": "source_note", "bbox": [0.0, 0.75, 1.0, 1.0], "description": target_description}
        visual_caption = "Read the source note or footer text in the bottom region of the figure."
    elif any(term in focus_text for term in ["legend", "color key"]):
        tool = "zoom"
        target_description = "legend area and nearby labels"
        region = {"type": "legend", "bbox": [0.66, 0.0, 1.0, 1.0], "description": target_description}
        visual_caption = "Read the legend entries and nearby labels that define the chart encodings."
    elif any(term in focus_text for term in ["axis", "label", "%", "percent", "percentage", "small text"]):
        tool = "zoom"
        target_description = "chart labels and adjacent category text"
        region = {"type": "center", "description": target_description}
        visual_caption = "Read the small chart labels, percentages, and adjacent category text."
    return {
        "needs_focus": True,
        "reason": visual_caption,
        "operations": [
            {
                "tool": tool,
                "page": primary.get("page"),
                "figure_ref": primary.get("figure_ref"),
                "evidence_id": primary.get("evidence_id"),
                "region": region,
                "visual_caption": visual_caption,
            }
        ],
    }


def _normalize_focus_plan(payload: dict[str, Any], figure_records: list[dict[str, Any]]) -> dict[str, Any]:
    operations = []
    known_refs = {str(item.get("figure_ref")): item for item in figure_records if item.get("figure_ref")}
    known_evidence_ids = {str(item.get("evidence_id")): item for item in figure_records if item.get("evidence_id")}
    raw_operations = payload.get("operations")
    operation_items = raw_operations if isinstance(raw_operations, list) else [payload]
    for item in operation_items:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        if tool == "subfigure_crop":
            tool = "subfigure_split"
        if tool not in {"zoom", "crop", "subfigure_split"}:
            continue
        figure_ref = str(item.get("figure_ref") or "").strip() or None
        evidence_id = str(item.get("evidence_id") or "").strip() or None
        if figure_ref and figure_ref not in known_refs:
            figure_ref = None
        if evidence_id and evidence_id not in known_evidence_ids:
            evidence_id = None
        region = _normalize_focus_region(item.get("region"))
        visual_caption = str(item.get("visual_caption") or item.get("caption") or "").strip()
        operations.append(
            {
                "tool": tool,
                "page": item.get("page"),
                "figure_ref": figure_ref,
                "evidence_id": evidence_id,
                "region": region,
                "grid_hint": _normalize_grid_hint(item.get("grid_hint") or item.get("split_hint")),
                "visual_caption": visual_caption,
                "reason": str(item.get("reason") or "").strip(),
                "rationale": str(item.get("rationale") or "").strip(),
            }
        )
    return {
        "needs_focus": bool(payload.get("needs_focus", bool(operations))),
        "reason": str(payload.get("reason") or (operations[0].get("reason") if operations else "") or ""),
        "operations": operations,
    }


def _match_focus_record(operation: dict[str, Any], figure_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    evidence_id = operation.get("evidence_id")
    if evidence_id:
        for item in figure_records:
            if item.get("evidence_id") == evidence_id:
                return item
    figure_ref = operation.get("figure_ref")
    if figure_ref:
        for item in figure_records:
            if item.get("figure_ref") == figure_ref:
                return item
    return figure_records[0] if figure_records else None


def _build_focus_tool_args(
    tool_name: str,
    operation: dict[str, Any],
    figure: dict[str, Any],
) -> dict[str, Any]:
    args: dict[str, Any] = {"node_id": figure["node_id"]}
    region = operation.get("region") if isinstance(operation.get("region"), dict) else {}
    bbox = _normalize_region_bbox(region.get("bbox"))
    if bbox is None:
        bbox = _infer_coarse_focus_bbox(
            figure=figure,
            region=region,
        )
    if bbox is None:
        bbox = _bbox_for_region(region)
    if bbox is None:
        bbox = _bbox_for_region_description(str(region.get("description") or ""))
    if tool_name == "zoom":
        args["scale"] = 2.5
        if bbox is not None:
            args["bbox"] = bbox
    elif tool_name == "crop":
        if bbox is not None:
            args["bbox"] = bbox
    elif tool_name == "subfigure_split":
        region = operation.get("region") if isinstance(operation.get("region"), dict) else {}
        panel_query_text = str(region.get("panel_query") or region.get("description") or "")
        panel_query = _extract_panel_query(panel_query_text) if _needs_specific_panel(panel_query_text) else ""
        if panel_query:
            args["panel_query"] = panel_query
        grid_hint = operation.get("grid_hint")
        if isinstance(grid_hint, dict) and grid_hint:
            args["grid_hint"] = grid_hint
    return args


def _infer_coarse_focus_bbox(
    *,
    figure: dict[str, Any],
    region: dict[str, Any],
) -> list[float] | None:
    region_type = str(region.get("type") or "").strip().lower()
    text = " ".join(
        item
        for item in [
            region_type,
            str(region.get("description") or ""),
        ]
        if item
    ).lower()
    image = _load_focus_image(figure.get("image_path"))
    if image is None:
        return None

    if region_type in {"legend", "color_key"} or any(term in text for term in ["legend", "color key"]):
        return _detect_legend_bbox(image)
    if region_type in {"source_note", "footer", "footnote", "bottom"} or any(
        term in text for term in ["source note", "footer", "footnote"]
    ):
        return _detect_bottom_text_bbox(image)
    if region_type in {"x_axis", "x-axis"} or "x-axis" in text or "x axis" in text:
        return _detect_axis_bbox(image, axis="x")
    if region_type in {"y_axis", "y-axis"} or "y-axis" in text or "y axis" in text:
        return _detect_axis_bbox(image, axis="y")
    return None


def _load_focus_image(image_path: Any):
    if not image_path:
        return None
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    except Exception:
        return None
    if image is None or image.size == 0:
        return None
    return image


def _detect_legend_bbox(image) -> list[int] | None:
    height, width = image.shape[:2]
    boxes = _foreground_component_boxes(image)
    right_candidates = [
        box for box in boxes
        if _box_area(box) <= width * height * 0.18
        and _box_center(box)[0] >= width * 0.55
    ]
    if right_candidates:
        return _merged_bbox(right_candidates, image.shape, pad=8)

    top_candidates = [
        box for box in boxes
        if _box_area(box) <= width * height * 0.12
        and _box_center(box)[1] <= height * 0.28
    ]
    return _merged_bbox(top_candidates, image.shape, pad=8)


def _detect_bottom_text_bbox(image) -> list[int] | None:
    height, width = image.shape[:2]
    boxes = [
        box for box in _foreground_component_boxes(image)
        if _box_center(box)[1] >= height * 0.65 and _box_area(box) <= width * height * 0.12
    ]
    bbox = _merged_bbox(boxes, image.shape, pad=6)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return bbox if (x2 - x1) >= width * 0.2 and y1 >= height * 0.55 else None


def _detect_axis_bbox(image, *, axis: str) -> list[int] | None:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        3.141592653589793 / 180,
        threshold=max(30, min(width, height) // 8),
        minLineLength=max(20, min(width, height) // 4),
        maxLineGap=12,
    )
    if lines is None:
        return None

    if axis == "x":
        horizontal = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = [int(v) for v in line]
            if abs(y2 - y1) <= max(3, height * 0.015) and (y1 + y2) / 2 >= height * 0.45:
                horizontal.append([x1, y1, x2, y2])
        if not horizontal:
            return None
        x1 = min(min(line[0], line[2]) for line in horizontal)
        x2 = max(max(line[0], line[2]) for line in horizontal)
        y = max((line[1] + line[3]) // 2 for line in horizontal)
        return _clamp_bbox([x1, y - int(height * 0.08), x2, y + int(height * 0.18)], image.shape)

    vertical = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]
        if abs(x2 - x1) <= max(3, width * 0.015) and (x1 + x2) / 2 <= width * 0.55:
            vertical.append([x1, y1, x2, y2])
    if not vertical:
        return None
    x = min((line[0] + line[2]) // 2 for line in vertical)
    y1 = min(min(line[1], line[3]) for line in vertical)
    y2 = max(max(line[1], line[3]) for line in vertical)
    return _clamp_bbox([x - int(width * 0.12), y1, x + int(width * 0.08), y2], image.shape)


def _foreground_component_boxes(image) -> list[list[int]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(8, int(width * height * 0.00008))
    max_area = int(width * height * 0.2)
    boxes: list[list[int]] = []
    for index in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[index]]
        if area < min_area or area > max_area or w < 2 or h < 2:
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes


def _merged_bbox(boxes: list[list[int]], image_shape: tuple[int, ...], *, pad: int = 0) -> list[int] | None:
    if not boxes:
        return None
    x1 = min(box[0] for box in boxes) - pad
    y1 = min(box[1] for box in boxes) - pad
    x2 = max(box[2] for box in boxes) + pad
    y2 = max(box[3] for box in boxes) + pad
    return _clamp_bbox([x1, y1, x2, y2], image_shape)


def _clamp_bbox(bbox: list[int], image_shape: tuple[int, ...]) -> list[int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = min(max(int(x1), 0), max(width - 1, 0))
    x2 = min(max(int(x2), x1 + 1), width)
    y1 = min(max(int(y1), 0), max(height - 1, 0))
    y2 = min(max(int(y2), y1 + 1), height)
    return [x1, y1, x2, y2]


def _box_center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _normalize_grid_hint(raw_hint: Any) -> dict[str, Any] | None:
    if not isinstance(raw_hint, dict):
        return None
    hint: dict[str, Any] = {}
    rows = _coerce_positive_int(raw_hint.get("rows"))
    columns = _coerce_positive_int(raw_hint.get("columns") or raw_hint.get("cols"))
    if rows is not None:
        hint["rows"] = rows
    if columns is not None:
        hint["columns"] = columns
    labels = raw_hint.get("labels")
    if isinstance(labels, list):
        cleaned = [str(item).strip() for item in labels if str(item).strip()]
        if cleaned:
            hint["labels"] = cleaned
    return hint or None


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalize_focus_region(raw_region: Any) -> dict[str, Any]:
    if isinstance(raw_region, dict):
        region_type = str(raw_region.get("type") or "").strip().lower() or "custom"
        region: dict[str, Any] = {
            "type": region_type,
            "description": str(raw_region.get("description") or "").strip(),
        }
        bbox = _normalize_region_bbox(raw_region.get("bbox"))
        if bbox is not None:
            region["bbox"] = bbox
        panel_query = str(raw_region.get("panel_query") or "").strip()
        if panel_query:
            region["panel_query"] = panel_query
        return region

    if isinstance(raw_region, str) and raw_region.strip():
        return _region_from_target_description(raw_region)
    return {"type": "full", "description": ""}


def _region_from_target_description(target_description: str) -> dict[str, Any]:
    text = target_description.lower()
    if any(term in text for term in ["bottom", "source note", "footer", "footnote"]):
        return {"type": "source_note", "bbox": [0.0, 0.75, 1.0, 1.0], "description": target_description}
    if any(term in text for term in ["legend", "right side", "color key"]):
        return {"type": "legend", "bbox": [0.66, 0.0, 1.0, 1.0], "description": target_description}
    if "x-axis" in text or "x axis" in text:
        return {"type": "x_axis", "bbox": [0.0, 0.66, 1.0, 1.0], "description": target_description}
    if "y-axis" in text or "y axis" in text:
        return {"type": "y_axis", "bbox": [0.0, 0.0, 0.34, 1.0], "description": target_description}
    if any(term in text for term in ["title", "top"]):
        return {"type": "top", "bbox": [0.0, 0.0, 1.0, 0.3], "description": target_description}
    if _needs_specific_panel(target_description):
        return {
            "type": "panel",
            "description": target_description,
            "panel_query": _extract_panel_query(target_description),
        }
    return {"type": "full", "description": target_description}


def _bbox_for_region(region: Any) -> list[float] | None:
    if not isinstance(region, dict):
        return None
    bbox = _normalize_region_bbox(region.get("bbox"))
    if bbox is not None:
        return bbox
    region_type = str(region.get("type") or "").strip().lower()
    if region_type in {"bottom", "source_note", "footer", "footnote"}:
        return [0.0, 0.75, 1.0, 1.0]
    if region_type in {"right", "legend", "color_key"}:
        return [0.66, 0.0, 1.0, 1.0]
    if region_type == "left":
        return [0.0, 0.0, 0.34, 1.0]
    if region_type in {"top", "title"}:
        return [0.0, 0.0, 1.0, 0.3]
    if region_type in {"x_axis", "x-axis"}:
        return [0.0, 0.66, 1.0, 1.0]
    if region_type in {"y_axis", "y-axis"}:
        return [0.0, 0.0, 0.34, 1.0]
    if region_type == "center":
        return [0.15, 0.15, 0.85, 0.85]
    return None


def _normalize_region_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if max(abs(item) for item in bbox) <= 1.0:
        return [min(max(item, 0.0), 1.0) for item in bbox]
    return bbox


def _bbox_for_region_description(description: str) -> list[float] | None:
    text = description.lower()
    if any(term in text for term in ["bottom", "source note", "footer"]):
        return [0.0, 0.75, 1.0, 1.0]
    if any(term in text for term in ["legend", "right side"]):
        return [0.66, 0.0, 1.0, 1.0]
    if "x-axis" in text or "x axis" in text:
        return [0.0, 0.66, 1.0, 1.0]
    if "y-axis" in text or "y axis" in text:
        return [0.0, 0.0, 0.34, 1.0]
    if any(term in text for term in ["title", "top"]):
        return [0.0, 0.0, 1.0, 0.3]
    return None


def _needs_specific_panel(text: str) -> bool:
    return bool(
        re.search(r"\bpanel\s+[a-z0-9]+\b", text, flags=re.IGNORECASE)
        or "subplot" in text
        or "top-right panel" in text
        or "top right panel" in text
        or re.search(r"\brow\s+\d+\s+(?:column|col)\s+\d+\b", text, flags=re.IGNORECASE)
    )


def _extract_panel_query(instruction: str) -> str:
    normalized = instruction.strip()
    label_match = re.search(r"\bpanel\s+([A-Za-z0-9]+)\b", normalized, flags=re.IGNORECASE)
    if label_match:
        return f"panel {label_match.group(1)}"
    row_col_match = re.search(r"\brow\s+\d+\s+(?:column|col)\s+\d+\b", normalized, flags=re.IGNORECASE)
    if row_col_match:
        return row_col_match.group(0)
    if re.search(r"\btop[- ]right panel\b", normalized, flags=re.IGNORECASE):
        return "top-right panel"
    if re.search(r"\btop[- ]left panel\b", normalized, flags=re.IGNORECASE):
        return "top-left panel"
    if re.search(r"\bbottom[- ]right panel\b", normalized, flags=re.IGNORECASE):
        return "bottom-right panel"
    if re.search(r"\bbottom[- ]left panel\b", normalized, flags=re.IGNORECASE):
        return "bottom-left panel"
    return normalized or "panel A"
