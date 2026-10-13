from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any

from ..schema import DocumentNode


def load_nodes_from_mineru_output(
    output_dir: str | Path,
    *,
    doc_id: str | None = None,
    page_range: tuple[int, int] | None = None,
    page_filter: set[int] | None = None,
) -> list[DocumentNode]:
    output_dir = Path(output_dir)
    auto_dir = output_dir / output_dir.name / "auto" if (output_dir / output_dir.name / "auto").exists() else output_dir
    content_v2 = auto_dir / f"{auto_dir.parent.name}_content_list_v2.json"
    markdown_path = auto_dir / f"{auto_dir.parent.name}.md"
    is_middle_json = False

    if not content_v2.exists():
        candidates = list(auto_dir.glob("*_content_list_v2.json"))
        if candidates:
            content_v2 = candidates[0]
        else:
            middle_json = auto_dir / "middle_json.json"
            if not middle_json.exists():
                candidates = [
                    path
                    for path in auto_dir.glob("*.json")
                    if path.name not in {"structured_content.json", "model_output.json"}
                ]
                middle_json = next(
                    (
                        path
                        for path in candidates
                        if json.loads(path.read_text(encoding="utf-8")).get("schema") == "docvortex.middle"
                    ),
                    middle_json,
                )
            if not middle_json.exists():
                raise FileNotFoundError(f"No MinerU content-list or middle JSON found under {auto_dir}")
            content_v2 = middle_json
            is_middle_json = True
    if not markdown_path.exists():
        md_candidates = list(auto_dir.glob("*.md"))
        markdown_path = md_candidates[0] if md_candidates else markdown_path

    if is_middle_json:
        from mineru.parser.base import ParseResult
        from mineru.render.content_list_v2 import render_content_list_v2

        result = ParseResult.from_json(content_v2.read_text(encoding="utf-8"))
        data = render_content_list_v2(result.middle_json)
    else:
        data = json.loads(content_v2.read_text(encoding="utf-8"))
    doc_id = doc_id or content_v2.stem.removesuffix("_content_list_v2")
    markdown_pages = _load_markdown_pages(markdown_path) if markdown_path.exists() and not is_middle_json else {}
    page_id_by_index = _load_page_id_by_index(auto_dir, len(data))

    nodes: list[DocumentNode] = []
    for page_idx, blocks in enumerate(data):
        internal_page_no = page_idx + 1
        page_no = page_id_by_index.get(page_idx, internal_page_no)
        if page_filter is not None and page_no not in page_filter:
            continue
        if page_range is not None:
            start_page, end_page = page_range
            if page_no < start_page or page_no > end_page:
                continue
        page_node_id = f"{doc_id}::page::{page_no}"
        page_text_parts: list[str] = []
        child_ids: list[str] = []
        page_title = ""
        current_section_title = ""
        page_image_path = _resolve_page_screenshot_path(auto_dir, page_no)

        for block_idx, block in enumerate(blocks):
            node = _block_to_node(
                doc_id,
                page_no,
                block_idx,
                block,
                current_section_title=current_section_title,
                auto_dir=auto_dir,
            )
            if node is None:
                continue
            if node.node_type == "title" and not page_title:
                page_title = node.title
            if node.node_type == "section":
                current_section_title = node.title
            if node.text:
                page_text_parts.append(node.text)
            child_ids.append(node.node_id)
            nodes.append(node)

        page_markdown = markdown_pages.get(page_no) or markdown_pages.get(internal_page_no) or markdown_pages.get(page_idx) or ""
        page_summary = _truncate(page_markdown or " ".join(page_text_parts), 320)
        nodes.append(
            DocumentNode(
                node_id=page_node_id,
                doc_id=doc_id,
                page=page_no,
                node_type="document",
                modality="mixed",
                title=page_title or f"Page {page_no}",
                text=_truncate(" ".join(page_text_parts), 4000),
                summary=page_summary,
                children=child_ids,
                metadata={
                    "source": "mineru",
                    "page_markdown": page_markdown,
                    "mineru_page_index": page_idx,
                    "auto_dir": str(auto_dir),
                    "page_image_path": page_image_path,
                },
            )
        )

    return nodes


def _load_page_id_by_index(auto_dir: Path, page_count: int) -> dict[int, int]:
    mapping = _load_report_page_id_by_index(auto_dir)
    if mapping:
        return mapping

    screenshots = _screenshot_page_ids(auto_dir.parent / "screenshots")
    if len(screenshots) == page_count:
        return {idx: page_id for idx, page_id in enumerate(screenshots)}
    return {}


def _load_report_page_id_by_index(auto_dir: Path) -> dict[int, int]:
    for report_path in (auto_dir.parent.parent / "parse_report.json", auto_dir.parent / "parse_report.json"):
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
        raw_mapping = metadata.get("page_id_by_index")
        if not isinstance(raw_mapping, dict):
            continue
        parsed: dict[int, int] = {}
        for key, value in raw_mapping.items():
            try:
                parsed[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
        if parsed:
            return parsed
    return {}


def _screenshot_page_ids(screenshots_dir: Path) -> list[int]:
    if not screenshots_dir.exists():
        return []
    page_ids: set[int] = set()
    for path in screenshots_dir.glob("page*.png"):
        match = re.fullmatch(r"page(\d+)", path.stem)
        if match:
            page_ids.add(int(match.group(1)))
    return sorted(page_ids)


def _block_to_node(
    doc_id: str,
    page_no: int,
    block_idx: int,
    block: dict[str, Any],
    *,
    current_section_title: str = "",
    auto_dir: Path | None = None,
) -> DocumentNode | None:
    block_type = str(block.get("type") or "paragraph").lower()
    content = block.get("content") or {}
    bbox = block.get("bbox")
    node_id = f"{doc_id}::page::{page_no}::block::{block_idx}"

    if block_type == "table":
        html = str(content.get("html") or "")
        caption = _join_rich_text(content.get("table_caption"))
        footnote = _join_rich_text(content.get("table_footnote"))
        text = _table_to_text(html)
        summary = _truncate(" ".join(part for part in [caption, text, footnote] if part), 320)
        title = caption or current_section_title or f"Table on page {page_no}"
        return DocumentNode(
            node_id=node_id,
            doc_id=doc_id,
            page=page_no,
            node_type="table",
            modality="table",
            title=title,
            text=text,
            summary=summary,
            content={"html": html, "caption": caption, "footnote": footnote},
            bbox=bbox,
            metadata={"source": "mineru", "block_type": block_type, "auto_dir": str(auto_dir) if auto_dir else None},
        )

    if block_type in {"image", "figure", "chart"}:
        if block_type == "chart":
            caption = _join_rich_text(content.get("chart_caption") or content.get("image_caption"))
            footnote = _join_rich_text(content.get("chart_footnote") or content.get("image_footnote"))
        else:
            caption = _join_rich_text(content.get("image_caption"))
            footnote = _join_rich_text(content.get("image_footnote"))
        path = ((content.get("image_path") or content.get("img_path") or content.get("image_source") or {}).get("path")
                if isinstance(content, dict) else None)
        if isinstance(content.get("image_source"), dict) and not path:
            path = content["image_source"].get("path")
        resolved_path = _resolve_asset_path(auto_dir, path)
        node_type = "figure"
        title = caption or current_section_title or f"{node_type.title()} on page {page_no}"
        return DocumentNode(
            node_id=node_id,
            doc_id=doc_id,
            page=page_no,
            node_type=node_type,
            modality="image",
            title=title,
            text="\n".join(part for part in [caption, footnote] if part),
            summary=_truncate("\n".join(part for part in [caption, footnote] if part), 320),
            content={"image_path": resolved_path or path, "caption": caption, "footnote": footnote},
            bbox=bbox,
            metadata={
                "source": "mineru",
                "block_type": block_type,
                "auto_dir": str(auto_dir) if auto_dir else None,
                "original_image_path": path,
            },
        )

    text = _content_to_text(content)
    if not text.strip():
        return None
    title = text if block_type == "title" else _truncate(text, 120)
    return DocumentNode(
        node_id=node_id,
        doc_id=doc_id,
        page=page_no,
        node_type=_normalize_node_type(block_type),
        modality="text",
        title=title,
        text=text,
        summary=_truncate(text, 240),
        bbox=bbox,
        metadata={"source": "mineru", "block_type": block_type, "auto_dir": str(auto_dir) if auto_dir else None},
    )


def _normalize_node_type(block_type: str) -> str:
    if block_type == "title":
        return "section"
    if block_type == "paragraph":
        return "paragraph"
    return "paragraph"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(_content_to_text(item) for item in content if item).strip()
    if isinstance(content, dict):
        texts: list[str] = []
        for key in (
            "title_content",
            "paragraph_content",
            "text",
            "content",
            "table_caption",
            "table_footnote",
            "chart_caption",
            "chart_footnote",
            "image_caption",
            "image_footnote",
        ):
            if key in content and content.get(key):
                texts.append(_content_to_text(content.get(key)))
        if texts:
            return "\n".join(part for part in texts if part).strip()
        return " ".join(_content_to_text(value) for value in content.values() if value).strip()
    return str(content).strip()


def _join_rich_text(value: Any) -> str:
    return _content_to_text(value)


def _table_to_text(html: str) -> str:
    text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", "\t", html, flags=re.IGNORECASE)
    text = re.sub(r"</tr>\s*<tr[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _load_markdown_pages(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "\f" in text:
        chunks = [chunk.strip() for chunk in text.split("\f")]
        return {idx + 1: chunk for idx, chunk in enumerate(chunks) if chunk}
    pattern = re.compile(r"^\s*[-#]*\s*page\s+(\d+)\s*[-#]*\s*$", re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return {1: text.strip()} if text.strip() else {}
    pages: dict[int, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            pages[int(match.group(1))] = chunk
    return pages


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def _resolve_asset_path(auto_dir: Path | None, raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return str(path)
    if auto_dir is not None:
        candidate = (auto_dir / path).resolve()
        if candidate.exists():
            return str(candidate)
    return str(path)


def _resolve_page_screenshot_path(auto_dir: Path | None, page_no: int) -> str | None:
    if auto_dir is None:
        return None
    candidates = [
        auto_dir.parent / "screenshots" / f"page{page_no}.png",
        auto_dir / "screenshots" / f"page{page_no}.png",
        auto_dir / "images" / "screenshots" / f"page{page_no}.png",
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return str(candidate)
    return None
