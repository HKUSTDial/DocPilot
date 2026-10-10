from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentRecord:
    dataset: str
    doc_id: str
    pdf_path: str | None = None
    page_image_dir: str | None = None
    page_images: list[str] = field(default_factory=list)
    total_pages: int | None = None
    domain: str | None = None
    available_annotations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceSample:
    source_sample_id: str
    dataset: str
    doc_id: str
    source_file: str
    question: str | None = None
    answer: str | None = None
    question_type: str | None = None
    evidence_pages: list[int] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageObject:
    doc_id: str
    dataset: str
    page_id: int
    image_path: str | None = None
    source_pdf_path: str | None = None
    mineru_json_path: str | None = None
    mineru_markdown_path: str | None = None
    text_blocks: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    captions: list[dict[str, Any]] = field(default_factory=list)
    formulas: list[dict[str, Any]] = field(default_factory=list)
    layout_blocks: list[dict[str, Any]] = field(default_factory=list)
    ocr_text: str = ""
    reading_order_text: str = ""
    page_summary: str = ""
    parse_quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
