from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import fitz

from .io import ensure_dir, read_json, read_jsonl, safe_text, stable_id, write_json, write_jsonl
from .models import PageObject


TEXT_TYPES = {"text", "paragraph", "title", "list", "text_block"}
TABLE_TYPES = {"table", "table_body"}
FIGURE_TYPES = {"image", "figure", "fig", "picture"}
CHART_TYPES = {"chart"}
CAPTION_TYPES = {"caption", "figure_caption", "table_caption"}
FORMULA_TYPES = {"formula", "equation", "latex"}
LAYOUT_TYPES = {"layout", "block", "section", "header", "footer"}
MINERU_CONTENT_LIST_V2_SUFFIX = "_content_list_v2.json"
MINERU_CONTENT_LIST_SUFFIX = "_content_list.json"
MINERU_MIDDLE_SUFFIX = "_middle.json"
MINERU_MODEL_SUFFIX = "_model.json"


def build_mineru_manifest(config: Any) -> dict[str, int]:
    out = ensure_dir(config.output_dir).resolve()
    docs = list(read_jsonl(out / "document_registry.jsonl"))
    rows: list[dict[str, Any]] = []
    for doc in docs:
        dataset = doc["dataset"]
        doc_id = doc["doc_id"]
        output_dir = (out / "mineru" / dataset / doc_id).resolve()
        input_path = doc.get("pdf_path") or doc.get("page_image_dir")
        source_input_path = input_path
        if not input_path:
            page_images = doc.get("page_images") or []
            if page_images and dataset in {"LongDocURL", "FinRAGBench-V"}:
                input_path = str(_prepare_longdocurl_input(output_dir, doc))
            elif len(page_images) == 1:
                input_path = page_images[0]
            elif page_images:
                input_path = str(_prepare_page_screenshot_input(output_dir, doc))
        elif Path(str(input_path)).suffix.lower() == ".pdf":
            input_path = str(_prepare_mineru_pdf_input(output_dir, doc_id, Path(str(input_path))))
        if not input_path:
            continue
        rows.append(
            {
                "manifest_id": stable_id(dataset, doc_id, input_path, prefix="mineru_"),
                "dataset": dataset,
                "doc_id": doc_id,
                "input_path": input_path,
                "source_input_path": source_input_path,
                "output_dir": str(output_dir),
                "expected_pages": doc.get("total_pages"),
                "backend": config.mineru.backend,
            }
        )
    max_docs = getattr(config.mineru, "max_docs", None)
    if max_docs is not None:
        rows = rows[:max_docs]
    count = write_jsonl(out / "mineru_input_manifest.jsonl", rows)
    return {"mineru_manifest": count}


def _prepare_mineru_pdf_input(output_dir: Path, doc_id: str, source: Path) -> Path:
    source = source.resolve()
    target = ensure_dir(output_dir) / f"{doc_id}.pdf"
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == source:
                return target
        except OSError:
            pass
        if target.is_file() or target.is_symlink():
            target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)
    return target


def _prepare_longdocurl_input(output_dir: Path, doc: dict[str, Any]) -> Path:
    doc_id = str(doc["doc_id"])
    screenshot_dir = ensure_dir(output_dir / doc_id / "screenshots")
    page_ids = _doc_page_ids(doc)
    for index, image_path in enumerate(doc.get("page_images") or []):
        source = Path(image_path)
        if not source.exists():
            continue
        page_number = _display_page_number(index, page_ids, source, one_based=doc.get("dataset") == "LongDocURL")
        target = screenshot_dir / f"page{page_number}{source.suffix.lower()}"
        if target.exists():
            continue
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    pdf_path = output_dir / f"{doc_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    pdf = fitz.open()
    try:
        screenshot_paths = sorted(
            screenshot_dir.glob("page*"),
            key=lambda path: _screenshot_page_number(path) or 0,
        )
        for image_path in screenshot_paths:
            pix = fitz.Pixmap(str(image_path))
            try:
                page = pdf.new_page(width=float(pix.width), height=float(pix.height))
                page.insert_image(page.rect, filename=str(image_path))
            finally:
                pix = None
        pdf.save(str(pdf_path))
    finally:
        pdf.close()
    return pdf_path


def _prepare_page_screenshot_input(output_dir: Path, doc: dict[str, Any]) -> Path:
    screenshot_dir = ensure_dir(output_dir / str(doc["doc_id"]) / "screenshots")
    page_ids = _doc_page_ids(doc)
    for index, image_path in enumerate(doc.get("page_images") or []):
        source = Path(image_path)
        if not source.exists():
            continue
        page_number = _display_page_number(index, page_ids, source, one_based=False)
        target = screenshot_dir / f"page{page_number:06d}{source.suffix.lower()}"
        if target.exists():
            continue
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    return screenshot_dir


def _doc_page_ids(doc: dict[str, Any]) -> list[int]:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    page_ids = metadata.get("page_ids") if isinstance(metadata, dict) else None
    if not isinstance(page_ids, list):
        return []
    parsed = []
    for value in page_ids:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            continue
    return parsed


def _display_page_number(index: int, page_ids: list[int], source: Path, *, one_based: bool) -> int:
    if index < len(page_ids):
        page_number = page_ids[index]
    else:
        page_number = _image_stem_page_id(source)
        if page_number is None:
            page_number = index
    return page_number + 1 if one_based else page_number


def _screenshot_page_number(path: Path) -> int | None:
    match = re.fullmatch(r"page(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def run_mineru(config: Any) -> dict[str, int]:
    out = ensure_dir(config.output_dir).resolve()
    manifest_path = out / "mineru_input_manifest.jsonl"
    rows = list(read_jsonl(manifest_path))
    reports = []
    if not config.mineru.enabled:
        for row in rows:
            reports.append({**row, "parse_status": "skipped", "reason": "mineru.disabled"})
        write_jsonl(out / "mineru_parse_reports.jsonl", reports)
        return {"mineru_skipped": len(reports)}

    executable = shutil.which(config.mineru.executable)
    if not executable:
        for row in rows:
            reports.append({**row, "parse_status": "failed", "reason": f"{config.mineru.executable} not found"})
        write_jsonl(out / "mineru_parse_reports.jsonl", reports)
        return {"mineru_failed": len(reports)}

    for row in rows:
        reports.append(_run_one(executable, row, config))
    write_jsonl(out / "mineru_parse_reports.jsonl", reports)
    return {
        "mineru_success": sum(1 for r in reports if r.get("parse_status") == "success"),
        "mineru_skipped": sum(1 for r in reports if r.get("parse_status") == "skipped"),
        "mineru_failed": sum(1 for r in reports if r.get("parse_status") == "failed"),
    }


def build_page_objects_from_mineru(config: Any) -> dict[str, int]:
    out = Path(config.output_dir)
    docs = {(d["dataset"], d["doc_id"]): d for d in read_jsonl(out / "document_registry.jsonl")}
    reports = list(read_jsonl(out / "mineru_parse_reports.jsonl"))
    pages: list[PageObject] = []
    failures: list[dict[str, Any]] = []

    for report in reports:
        dataset = report.get("dataset")
        doc_id = report.get("doc_id")
        doc = docs.get((dataset, doc_id), {})
        if report.get("parse_status") == "failed":
            failures.append(report)
            continue
        output_dir = Path(report.get("output_dir") or "")
        parsed = _parse_output_dir(output_dir, dataset, doc_id, doc)
        if not parsed:
            failures.append({**report, "reason": "no parseable MinerU markdown/json found"})
        pages.extend(parsed)

    count = write_jsonl(out / "page_objects.jsonl", (p.to_dict() for p in pages))
    write_jsonl(out / "parse_failure_report.jsonl", failures)
    return {"page_objects": count, "parse_failures": len(failures)}


def _run_one(executable: str, row: dict[str, Any], config: Any) -> dict[str, Any]:
    output_dir = ensure_dir(Path(row["output_dir"])).resolve()
    doc_id = str(row.get("doc_id") or output_dir.name)
    input_path = Path(row["input_path"]).absolute()
    existing_report, reused_output_dir = _load_existing_success_report(
        output_dir,
        doc_id=doc_id,
        require_preferred_doc_root=input_path.suffix.lower() == ".pdf",
    )
    if existing_report is not None:
        effective_output_dir = reused_output_dir or output_dir
        screenshot_dir = _ensure_page_screenshots(
            effective_output_dir,
            input_path=input_path,
            doc_id=doc_id,
        )
        report = {
            **row,
            "parse_status": "skipped",
            "reason": "existing successful MinerU output found",
            "output_dir": str(effective_output_dir),
            "artifacts": existing_report.get("artifacts") or _discover_mineru_artifacts(output_dir),
            "reused_parse_report": str(effective_output_dir / "parse_report.json"),
            "screenshot_dir": str(screenshot_dir) if screenshot_dir else None,
        }
        return _normalize_report_paths(report, effective_output_dir)
    cmd = [executable, "-p", str(input_path), "-o", str(output_dir)]
    if config.mineru.backend:
        cmd.extend(["-b", config.mineru.backend])
    env = os.environ.copy()
    if config.mineru.model_source:
        env["MINERU_MODEL_SOURCE"] = config.mineru.model_source
    report = {**row, "cmd": cmd}
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(Path(config.output_dir).resolve()),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        report.update({"parse_status": "failed", "reason": str(exc)})
        write_json(output_dir / "parse_report.json", report)
        return report
    report.update(
        {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    artifacts = _discover_mineru_artifacts(output_dir)
    report["artifacts"] = artifacts
    if completed.returncode == 0 and (artifacts["json_files"] or artifacts["markdown_files"]):
        report["parse_status"] = "success"
        screenshot_dir = _ensure_page_screenshots(
            output_dir,
            input_path=input_path,
            doc_id=doc_id,
        )
        if screenshot_dir:
            report["screenshot_dir"] = str(screenshot_dir)
    else:
        report["parse_status"] = "failed"
        report["reason"] = "mineru produced no markdown/json artifacts" if completed.returncode == 0 else "mineru command failed"
    report = _normalize_report_paths(report, output_dir)
    write_json(output_dir / "parse_report.json", report)
    return report


def _load_existing_success_report(
    output_dir: Path,
    *,
    doc_id: str,
    require_preferred_doc_root: bool,
) -> tuple[dict[str, Any] | None, Path | None]:
    for candidate_dir in _existing_output_dir_candidates(output_dir):
        report_path = candidate_dir / "parse_report.json"
        report = read_json(report_path, default=None)
        if not isinstance(report, dict):
            continue
        if report.get("parse_status") != "success":
            continue
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = _discover_mineru_artifacts(candidate_dir)
        json_files = artifacts.get("json_files") if isinstance(artifacts, dict) else None
        markdown_files = artifacts.get("markdown_files") if isinstance(artifacts, dict) else None
        if json_files or markdown_files:
            if require_preferred_doc_root and not _artifacts_use_preferred_doc_root(artifacts, candidate_dir, doc_id):
                continue
            report["artifacts"] = artifacts
            report = _normalize_report_paths(report, candidate_dir)
            write_json(report_path, report)
            return report, candidate_dir
    return None, None


def _existing_output_dir_candidates(output_dir: Path) -> list[Path]:
    candidates = [output_dir]
    name = output_dir.name
    if not name.lower().endswith(".pdf"):
        candidates.append(output_dir.with_name(f"{name}.pdf"))
    return candidates


def _artifacts_use_preferred_doc_root(artifacts: dict[str, Any], output_dir: Path, doc_id: str) -> bool:
    preferred_root = (output_dir / doc_id).resolve()
    paths: list[str] = []
    for key in ("json_files", "markdown_files", "image_files"):
        values = artifacts.get(key)
        if isinstance(values, list):
            paths.extend(str(value) for value in values)
    for value in paths:
        try:
            path = Path(value).resolve()
        except OSError:
            path = Path(value)
        if path == preferred_root or preferred_root in path.parents:
            return True
    return False


def _discover_mineru_artifacts(output_dir: Path) -> dict[str, list[str]]:
    json_files = []
    markdown_files = []
    image_files = []
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.name == "parse_report.json":
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            json_files.append(str(path))
        elif suffix == ".md":
            markdown_files.append(str(path))
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            image_files.append(str(path))
    return {
        "json_files": sorted(json_files),
        "markdown_files": sorted(markdown_files),
        "image_files": sorted(image_files),
    }


def _normalize_report_paths(payload: Any, output_dir: Path) -> Any:
    current_root = _pipeline_output_root(output_dir)
    if current_root is None:
        return payload
    normalized = _rewrite_legacy_paths(payload, current_root)
    if isinstance(normalized, dict):
        normalized["output_dir"] = str(output_dir)
        if "artifacts" in normalized and not isinstance(normalized.get("artifacts"), dict):
            normalized["artifacts"] = _discover_mineru_artifacts(output_dir)
    return normalized


def _pipeline_output_root(path: Path) -> Path | None:
    try:
        parts = path.resolve().parts
    except FileNotFoundError:
        parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index] == "runs":
            if index + 1 < len(parts) and parts[index + 1] in {"environment", "outputs"}:
                end = index + 2
                if parts[index + 1] == "outputs" and index + 2 < len(parts) and parts[index + 2] == "document":
                    end = index + 3
                return Path(*parts[:end])
    return None


def _rewrite_legacy_paths(value: Any, current_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_legacy_paths(item, current_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_legacy_paths(item, current_root) for item in value]
    if not isinstance(value, str):
        return value
    legacy_marker = "/runs/outputs/document"
    if legacy_marker not in value:
        return value
    prefix, suffix = value.split(legacy_marker, 1)
    return f"{prefix}{current_root.as_posix()}{suffix}"


def _ensure_page_screenshots(output_dir: Path, *, input_path: Path, doc_id: str) -> Path | None:
    if input_path.is_dir() and input_path.name == "screenshots":
        existing = list(input_path.glob("page*"))
        return input_path if existing else None
    if input_path.suffix.lower() != ".pdf" or not input_path.exists():
        return None
    doc_root = _mineru_doc_root(output_dir, doc_id)
    screenshot_dir = ensure_dir(doc_root / "screenshots")
    existing = list(screenshot_dir.glob("page*.png"))
    if existing:
        return screenshot_dir
    previous_errors = None
    previous_warnings = None
    try:
        if hasattr(fitz, "TOOLS"):
            previous_errors = fitz.TOOLS.mupdf_display_errors(False)
            previous_warnings = fitz.TOOLS.mupdf_display_warnings(False)
            fitz.TOOLS.reset_mupdf_warnings()
        with fitz.open(str(input_path)) as pdf:
            matrix = fitz.Matrix(2.0, 2.0)
            for page_index in range(pdf.page_count):
                pix = pdf.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
                pix.save(str(screenshot_dir / f"page{page_index + 1}.png"))
    except Exception:
        return None
    finally:
        if hasattr(fitz, "TOOLS"):
            if previous_errors is not None:
                fitz.TOOLS.mupdf_display_errors(previous_errors)
            if previous_warnings is not None:
                fitz.TOOLS.mupdf_display_warnings(previous_warnings)
    return screenshot_dir


def _mineru_doc_root(output_dir: Path, doc_id: str) -> Path:
    preferred = output_dir / doc_id
    if preferred.exists():
        return preferred
    child_dirs = [path for path in output_dir.iterdir() if path.is_dir()] if output_dir.exists() else []
    if len(child_dirs) == 1:
        return child_dirs[0]
    return output_dir


def _parse_output_dir(output_dir: Path, dataset: str, doc_id: str, doc: dict[str, Any]) -> list[PageObject]:
    if not output_dir.exists():
        return []
    json_files = [p for p in sorted(output_dir.rglob("*.json")) if p.name != "parse_report.json"]
    markdown_files = sorted(output_dir.rglob("*.md"))
    page_map: dict[int, PageObject] = {}

    content_v2_files = [p for p in json_files if p.name.endswith(MINERU_CONTENT_LIST_V2_SUFFIX)]
    content_files = [p for p in json_files if p.name.endswith(MINERU_CONTENT_LIST_SUFFIX)]
    middle_files = [p for p in json_files if p.name.endswith(MINERU_MIDDLE_SUFFIX)]
    model_files = [p for p in json_files if p.name.endswith(MINERU_MODEL_SUFFIX)]

    parsed_any = False
    if content_v2_files:
        for json_path in content_v2_files:
            parsed_any = _parse_content_list_v2(json_path, page_map, dataset, doc_id, doc) or parsed_any
    elif content_files:
        for json_path in content_files:
            parsed_any = _parse_content_list(json_path, page_map, dataset, doc_id, doc) or parsed_any
    elif middle_files:
        for json_path in middle_files:
            parsed_any = _parse_middle_json(json_path, page_map, dataset, doc_id, doc, include_content=True) or parsed_any
    elif model_files:
        for json_path in model_files:
            parsed_any = _parse_model_json(json_path, page_map, dataset, doc_id, doc) or parsed_any

    if page_map and middle_files:
        for json_path in middle_files:
            _parse_middle_json(json_path, page_map, dataset, doc_id, doc, include_content=False)

    if not parsed_any:
        for json_path in json_files:
            _parse_generic_json(json_path, page_map, dataset, doc_id, doc)

    for md_path in markdown_files:
        text = _safe_read(md_path)
        if not text.strip():
            continue
        for page_id, chunk in _split_markdown_pages(text):
            page_id = _map_doc_page_id(doc, page_id)
            page = _get_page(page_map, dataset, doc_id, page_id, doc, markdown_path=md_path)
            page.mineru_markdown_path = page.mineru_markdown_path or str(md_path)
            page.reading_order_text = _join_text(page.reading_order_text, chunk)
            page.page_summary = _summarize_text(page.reading_order_text)
            page.parse_quality["has_markdown"] = True

    for page in page_map.values():
        page.image_path = _find_page_image(output_dir, page.page_id, doc)
        page.ocr_text = page.ocr_text or page.reading_order_text
        page.parse_quality.setdefault("source", "mineru")
        page.parse_quality.setdefault("has_text", bool(page.text_blocks or page.reading_order_text))
        page.parse_quality.setdefault("has_table_html", bool(page.tables))
        page.parse_quality.setdefault("has_layout", bool(page.layout_blocks))
        page.parse_quality.setdefault("needs_manual_or_vlm_check", False)

    return sorted(page_map.values(), key=lambda p: p.page_id)


def _parse_content_list_v2(
    json_path: Path,
    page_map: dict[int, PageObject],
    dataset: str,
    doc_id: str,
    doc: dict[str, Any],
) -> bool:
    data = _read_json(json_path)
    if not isinstance(data, list):
        return False
    parsed = False
    for page_idx, blocks in enumerate(data):
        if not isinstance(blocks, list):
            continue
        page_id = _map_doc_page_id(doc, _page_id({"page_idx": page_idx}))
        page = _get_page(page_map, dataset, doc_id, page_id, doc, json_path=json_path)
        page.parse_quality["mineru_format"] = "content_list_v2"
        for order, block in enumerate(blocks):
            if isinstance(block, dict):
                _append_mineru_block(page, _normalize_content_v2_block(block, json_path, order), json_path)
                parsed = True
    return parsed


def _parse_content_list(
    json_path: Path,
    page_map: dict[int, PageObject],
    dataset: str,
    doc_id: str,
    doc: dict[str, Any],
) -> bool:
    data = _read_json(json_path)
    if not isinstance(data, list):
        return False
    parsed = False
    for order, block in enumerate(data):
        if not isinstance(block, dict):
            continue
        page = _get_page(page_map, dataset, doc_id, _map_doc_page_id(doc, _page_id(block)), doc, json_path=json_path)
        page.parse_quality["mineru_format"] = "content_list"
        _append_mineru_block(page, _normalize_content_block(block, json_path, order), json_path)
        parsed = True
    return parsed


def _parse_middle_json(
    json_path: Path,
    page_map: dict[int, PageObject],
    dataset: str,
    doc_id: str,
    doc: dict[str, Any],
    *,
    include_content: bool,
) -> bool:
    data = _read_json(json_path)
    pdf_info = data.get("pdf_info") if isinstance(data, dict) else None
    if not isinstance(pdf_info, list):
        return False
    parsed = False
    for page_idx, page_info in enumerate(pdf_info):
        if not isinstance(page_info, dict):
            continue
        page_id = _page_id(page_info) if _has_page_key(page_info) else page_idx
        page_id = _map_doc_page_id(doc, page_id)
        page = _get_page(page_map, dataset, doc_id, page_id, doc, json_path=json_path)
        page.parse_quality["mineru_format"] = page.parse_quality.get("mineru_format") or "middle"
        page.parse_quality["has_middle_json"] = True
        if page_info.get("page_size"):
            page.parse_quality["page_size"] = page_info.get("page_size")
        if page_info.get("need_drop"):
            page.parse_quality["needs_manual_or_vlm_check"] = True
        if page_info.get("drop_reason"):
            page.parse_quality["drop_reason"] = page_info.get("drop_reason")
        if not include_content:
            continue
        for field in ("para_blocks", "preproc_blocks", "images", "tables", "discarded_blocks", "interline_equations"):
            for order, block in enumerate(_as_list(page_info.get(field))):
                if not isinstance(block, dict):
                    continue
                _append_mineru_block(page, _normalize_middle_block(block, field, json_path, order), json_path)
                parsed = True
    return parsed


def _parse_model_json(
    json_path: Path,
    page_map: dict[int, PageObject],
    dataset: str,
    doc_id: str,
    doc: dict[str, Any],
) -> bool:
    data = _read_json(json_path)
    pages = data if isinstance(data, list) else data.get("layout_dets") if isinstance(data, dict) else None
    if not isinstance(pages, list):
        return False
    parsed = False
    for page_idx, page_value in enumerate(pages):
        blocks = page_value if isinstance(page_value, list) else _as_list(page_value.get("layout_dets") if isinstance(page_value, dict) else page_value)
        page_id = _page_id(page_value) if isinstance(page_value, dict) and _has_page_key(page_value) else page_idx
        page_id = _map_doc_page_id(doc, page_id)
        page = _get_page(page_map, dataset, doc_id, page_id, doc, json_path=json_path)
        page.parse_quality["mineru_format"] = page.parse_quality.get("mineru_format") or "model"
        for order, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            _append_mineru_block(page, _normalize_model_block(block, json_path, order), json_path)
            parsed = True
    return parsed


def _parse_generic_json(
    json_path: Path,
    page_map: dict[int, PageObject],
    dataset: str,
    doc_id: str,
    doc: dict[str, Any],
) -> bool:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    parsed = False
    for block in _iter_blocks(data):
        page_id = _map_doc_page_id(doc, _page_id(block))
        page = _get_page(page_map, dataset, doc_id, page_id, doc, json_path=json_path)
        page.parse_quality["mineru_format"] = page.parse_quality.get("mineru_format") or "generic"
        _append_mineru_block(page, _normalize_content_block(block, json_path, None), json_path)
        parsed = True
    return parsed


def _get_page(
    page_map: dict[int, PageObject],
    dataset: str,
    doc_id: str,
    page_id: int,
    doc: dict[str, Any],
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
) -> PageObject:
    page = page_map.get(page_id)
    if page is None:
        page = PageObject(
            doc_id=doc_id,
            dataset=dataset,
            page_id=page_id,
            source_pdf_path=doc.get("pdf_path"),
            mineru_json_path=str(json_path) if json_path else None,
            mineru_markdown_path=str(markdown_path) if markdown_path else None,
            parse_quality={"source": "mineru"},
        )
        page_map[page_id] = page
    if json_path and not page.mineru_json_path:
        page.mineru_json_path = str(json_path)
    if markdown_path and not page.mineru_markdown_path:
        page.mineru_markdown_path = str(markdown_path)
    if json_path:
        page.parse_quality["has_json"] = True
    if markdown_path:
        page.parse_quality["has_markdown"] = True
    return page


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _has_page_key(block: dict[str, Any]) -> bool:
    return any(key in block for key in ("page_id", "page_idx", "page_no", "page", "pageNumber"))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_content_v2_block(block: dict[str, Any], json_path: Path, order: int | None) -> dict[str, Any]:
    content = block.get("content") if isinstance(block.get("content"), dict) else {}
    typ = safe_text(content.get("sub_type") or block.get("type") or block.get("block_type") or "text").lower()
    text = _content_text(content) or _content_text(block)
    normalized = _normalize_content_block({**block, **content, "type": typ, "text": text}, json_path, order)
    normalized["raw"] = _compact_block({**block, "content": content})
    return normalized


def _normalize_content_block(block: dict[str, Any], json_path: Path, order: int | None) -> dict[str, Any]:
    typ = _block_type(block)
    text = _block_text(block)
    img_path = _resolve_asset_path(json_path, block.get("img_path") or block.get("image_path") or block.get("table_img_path"))
    caption = _caption_text(block)
    if caption and caption not in text:
        text = _join_text(text, caption)
    table_html = safe_text(block.get("table_body") or block.get("html"))
    if table_html and table_html not in text:
        text = _join_text(text, table_html)
    return {
        "type": typ,
        "text": text,
        "html": table_html or None,
        "bbox": _bbox(block),
        "asset_path": img_path,
        "caption": caption or None,
        "raw": _compact_block(block),
        "reading_order_index": order,
        "mineru_json_path": str(json_path),
    }


def _normalize_middle_block(block: dict[str, Any], field: str, json_path: Path, order: int | None) -> dict[str, Any]:
    field_type = {
        "para_blocks": "text",
        "preproc_blocks": "layout",
        "images": "image",
        "tables": "table",
        "discarded_blocks": "layout",
        "interline_equations": "formula",
    }.get(field, "layout")
    typ = safe_text(block.get("type") or block.get("block_type") or field_type).lower()
    return {
        "type": typ,
        "text": _block_text(block),
        "html": None,
        "bbox": _bbox(block),
        "asset_path": _resolve_asset_path(json_path, block.get("img_path") or block.get("image_path")),
        "caption": _caption_text(block) or None,
        "raw": _compact_block(block),
        "reading_order_index": order,
        "mineru_json_path": str(json_path),
    }


def _normalize_model_block(block: dict[str, Any], json_path: Path, order: int | None) -> dict[str, Any]:
    category = block.get("category") or block.get("type") or block.get("layout_type") or "layout"
    return {
        "type": safe_text(category).lower(),
        "text": _block_text(block),
        "html": safe_text(block.get("html") or block.get("table_body")) or None,
        "bbox": _bbox(block),
        "asset_path": _resolve_asset_path(json_path, block.get("img_path") or block.get("image_path")),
        "caption": _caption_text(block) or None,
        "raw": _compact_block(block),
        "reading_order_index": order,
        "mineru_json_path": str(json_path),
    }


def _iter_blocks(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if _looks_like_block(value):
            yield value
        for child in value.values():
            yield from _iter_blocks(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_blocks(item)


def _looks_like_block(value: dict[str, Any]) -> bool:
    keys = set(value)
    return bool(keys & {"type", "block_type", "category", "text", "content", "html", "bbox", "page_id", "page_idx"})


def _page_id(block: dict[str, Any]) -> int:
    for key in ("page_id", "page_idx", "page_no", "page", "pageNumber"):
        if key in block:
            try:
                return int(block[key])
            except (TypeError, ValueError):
                pass
    return 0


def _map_doc_page_id(doc: dict[str, Any], page_id: int) -> int:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    mapping = metadata.get("page_id_by_index") if isinstance(metadata, dict) else None
    if not isinstance(mapping, dict):
        return page_id
    mapped = mapping.get(str(page_id))
    if mapped is None:
        mapped = mapping.get(page_id)
    try:
        return int(mapped)
    except (TypeError, ValueError):
        return page_id


def _block_type(block: dict[str, Any]) -> str:
    raw = block.get("type") or block.get("block_type") or block.get("category") or block.get("kind") or "text"
    return str(raw).lower()


def _block_text(block: dict[str, Any]) -> str:
    for key in (
        "text",
        "content",
        "markdown",
        "html",
        "latex",
        "caption",
        "table_body",
        "table_caption",
        "image_caption",
        "equation",
    ):
        if block.get(key):
            return _content_text(block.get(key))
    spans = block.get("spans") or block.get("lines") or block.get("blocks")
    if isinstance(spans, list):
        return " ".join(_block_text(span) if isinstance(span, dict) else safe_text(span) for span in spans)
    return ""


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, dict):
        texts = []
        for key in (
            "text",
            "content",
            "title_content",
            "text_content",
            "table_body",
            "table_caption",
            "table_footnote",
            "image_caption",
            "image_footnote",
            "formula",
            "latex",
            "equation",
        ):
            if value.get(key):
                texts.append(_content_text(value.get(key)))
        if texts:
            return "\n".join(t for t in texts if t)
        return " ".join(_content_text(v) for v in value.values() if v)
    if isinstance(value, list):
        return " ".join(_content_text(item) for item in value if item)
    return safe_text(value)


def _caption_text(block: dict[str, Any]) -> str:
    texts = []
    for key in ("caption", "table_caption", "image_caption", "table_footnote", "image_footnote"):
        if block.get(key):
            texts.append(_content_text(block.get(key)))
    return "\n".join(t for t in texts if t)


def _append_mineru_block(page: PageObject, block: dict[str, Any], json_path: Path) -> None:
    typ = _block_type(block)
    text = safe_text(block.get("text"))
    normalized = {
        "type": typ,
        "text": text,
        "html": block.get("html"),
        "bbox": block.get("bbox"),
        "asset_path": block.get("asset_path"),
        "caption": block.get("caption"),
        "raw": block.get("raw") or _compact_block(block),
        "reading_order_index": block.get("reading_order_index"),
        "mineru_json_path": str(json_path),
    }
    if typ in TABLE_TYPES or "table" in typ:
        normalized["html"] = _table_html(block)
        page.tables.append(normalized)
    elif typ in FIGURE_TYPES or "image" in typ or "figure" in typ:
        page.figures.append(normalized)
    elif typ in CHART_TYPES or "chart" in typ:
        page.charts.append(normalized)
    elif typ in CAPTION_TYPES or "caption" in typ:
        page.captions.append(normalized)
    elif typ in FORMULA_TYPES or "formula" in typ:
        page.formulas.append(normalized)
    elif typ in LAYOUT_TYPES:
        page.layout_blocks.append(normalized)
    elif text:
        page.text_blocks.append(normalized)
        page.reading_order_text = _join_text(page.reading_order_text, text)
        page.page_summary = _summarize_text(page.reading_order_text)


def _table_html(block: dict[str, Any]) -> str:
    raw = block.get("raw") if isinstance(block.get("raw"), dict) else {}
    return safe_text(block.get("html") or raw.get("html") or raw.get("table_body"))


def _bbox(block: dict[str, Any]) -> list[float] | None:
    value = block.get("bbox") or block.get("box") or block.get("coordinates") or block.get("poly")
    if isinstance(value, dict):
        value = value.get("bbox") or value.get("points")
    if isinstance(value, list) and len(value) >= 4:
        try:
            return [float(x) for x in value[:4]]
        except (TypeError, ValueError):
            return None
    return None


def _compact_block(block: dict[str, Any]) -> dict[str, Any]:
    keep = {}
    for key in (
        "id",
        "type",
        "block_type",
        "category",
        "page_id",
        "page_idx",
        "bbox",
        "text",
        "content",
        "html",
        "latex",
        "table_body",
        "table_caption",
        "table_footnote",
        "image_caption",
        "image_footnote",
        "img_path",
        "image_path",
        "table_img_path",
    ):
        if key in block:
            keep[key] = block[key]
    return keep


def _resolve_asset_path(json_path: Path, value: Any) -> str | None:
    if not value:
        return None
    text = safe_text(value)
    path = Path(text)
    if path.is_absolute():
        return str(path)
    candidate = json_path.parent / path
    return str(candidate) if candidate.exists() else text


def _split_markdown_pages(text: str) -> list[tuple[int, str]]:
    if "\f" in text:
        return [(i, chunk.strip()) for i, chunk in enumerate(text.split("\f")) if chunk.strip()]
    pattern = re.compile(r"^\s*[-#]*\s*page\s+(\d+)\s*[-#]*\s*$", re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [(0, text.strip())]
    chunks = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunks.append((int(match.group(1)), text[start:end].strip()))
    return [(page_id, chunk) for page_id, chunk in chunks if chunk]


def _find_page_image(output_dir: Path, page_id: int, doc: dict[str, Any]) -> str | None:
    screenshot_names = [f"page{page_id}.png", f"page{page_id:06d}.png"]
    if page_id >= 0:
        screenshot_names.extend([f"page{page_id + 1}.png", f"page{page_id + 1:06d}.png"])
    for screenshot_name in screenshot_names:
        screenshot_candidates = sorted(output_dir.rglob(f"screenshots/{screenshot_name}"))
        if screenshot_candidates:
            return str(screenshot_candidates[0])
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    image_by_page = metadata.get("page_image_by_page_id") if isinstance(metadata, dict) else None
    if isinstance(image_by_page, dict):
        image_path = image_by_page.get(str(page_id)) or image_by_page.get(page_id)
        if image_path:
            return str(image_path)
    for path in doc.get("page_images") or []:
        if _image_stem_page_id(Path(path)) == page_id:
            return path
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    candidates = [path for path in output_dir.rglob("*") if path.suffix.lower() in image_suffixes]
    for path in candidates:
        if str(page_id) in path.stem:
            return str(path)
    return str(candidates[0]) if len(candidates) == 1 else None


def _image_stem_page_id(path: Path) -> int | None:
    _, sep, page_text = path.stem.rpartition("_")
    if not sep:
        return None
    try:
        return int(page_text)
    except ValueError:
        return None


def _join_text(a: str, b: str) -> str:
    if not a:
        return b.strip()
    if not b:
        return a.strip()
    return f"{a.strip()}\n{b.strip()}"


def _summarize_text(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
