from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import ensure_dir, norm_path, read_jsonl, stable_id, write_json, write_jsonl
from .models import DocumentRecord, SourceSample


def build_document_registries(config: Any) -> dict[str, int]:
    out = ensure_dir(config.output_dir)
    documents: dict[tuple[str, str], DocumentRecord] = {}
    samples: list[SourceSample] = []
    path_report: dict[str, Any] = {"missing_pdf": [], "missing_images": [], "datasets": {}}
    include_specs = _build_include_set(getattr(config.datasets, "include", []))

    m_docs, m_samples, m_report = _index_mmdocrag(
        config.datasets.mmdocrag_dir,
        sample_files=getattr(config.datasets, "mmdocrag_sample_files", []),
        max_samples=getattr(config.datasets, "mmdocrag_max_samples", None),
    )
    l_docs, l_samples, l_report = _index_longdocurl(
        config.datasets.longdocurl_dir,
        max_samples=getattr(config.datasets, "longdocurl_max_samples", None),
        max_docs=getattr(config.datasets, "longdocurl_max_docs", None),
        sample_jsonl=getattr(config.datasets, "longdocurl_sample_jsonl", None),
    )
    ml_docs, ml_samples, ml_report = _index_mmlongbench_doc(
        getattr(config.datasets, "mmlongbench_dir"),
        getattr(config.datasets, "mmlongbench_documents_dir"),
        sample_file=getattr(config.datasets, "mmlongbench_sample_file"),
        max_samples=getattr(config.datasets, "mmlongbench_max_samples", None),
    )
    madqa_docs, madqa_samples, madqa_report = _index_madqa(
        getattr(config.datasets, "madqa_dir"),
        getattr(config.datasets, "madqa_pdf_dir"),
        sample_file=getattr(config.datasets, "madqa_sample_file"),
        max_samples=getattr(config.datasets, "madqa_max_samples", None),
    )
    dude_docs, dude_samples, dude_report = _index_dude(
        getattr(config.datasets, "dude_dir"),
        getattr(config.datasets, "dude_pdf_dir"),
        sample_file=getattr(config.datasets, "dude_sample_file"),
        max_samples=getattr(config.datasets, "dude_max_samples", None),
    )
    finrag_docs, finrag_samples, finrag_report = _index_finragbench(
        getattr(config.datasets, "finragbench_dir"),
        getattr(config.datasets, "finragbench_image_dir"),
        sample_file=getattr(config.datasets, "finragbench_sample_file"),
        max_samples=getattr(config.datasets, "finragbench_max_samples", None),
    )
    docbench_docs, docbench_samples, docbench_report = _index_docbench(
        getattr(config.datasets, "docbench_dir"),
        getattr(config.datasets, "docbench_data_dir"),
        max_samples=getattr(config.datasets, "docbench_max_samples", None),
    )

    for doc in m_docs + l_docs + ml_docs + madqa_docs + dude_docs + finrag_docs + docbench_docs:
        documents[(doc.dataset, doc.doc_id)] = doc
    samples.extend(m_samples)
    samples.extend(l_samples)
    samples.extend(ml_samples)
    samples.extend(madqa_samples)
    samples.extend(dude_samples)
    samples.extend(finrag_samples)
    samples.extend(docbench_samples)

    if include_specs:
        documents = {key: doc for key, doc in documents.items() if key in include_specs}
        samples = [sample for sample in samples if (sample.dataset, sample.doc_id) in include_specs]

    path_report["datasets"]["MMDocRAG"] = m_report
    path_report["datasets"]["LongDocURL"] = l_report
    path_report["datasets"]["MMLongBench-Doc"] = ml_report
    path_report["datasets"]["MADQA"] = madqa_report
    path_report["datasets"]["DUDE"] = dude_report
    path_report["datasets"]["FinRAGBench-V"] = finrag_report
    path_report["datasets"]["DocBench"] = docbench_report
    path_report["missing_pdf"].extend(m_report.get("missing_pdf", []))
    path_report["missing_pdf"].extend(l_report.get("missing_pdf", []))
    path_report["missing_pdf"].extend(ml_report.get("missing_pdf", []))
    path_report["missing_pdf"].extend(madqa_report.get("missing_pdf", []))
    path_report["missing_pdf"].extend(dude_report.get("missing_pdf", []))
    path_report["missing_pdf"].extend(docbench_report.get("missing_pdf", []))
    path_report["missing_images"].extend(l_report.get("missing_images", []))
    path_report["missing_images"].extend(finrag_report.get("missing_images", []))
    if include_specs:
        path_report["include"] = [
            {"dataset": dataset, "doc_id": doc_id}
            for dataset, doc_id in sorted(include_specs)
        ]

    doc_count = write_jsonl(out / "document_registry.jsonl", (doc.to_dict() for doc in documents.values()))
    sample_count = write_jsonl(out / "source_sample_registry.jsonl", (sample.to_dict() for sample in samples))
    write_json(out / "path_mapping_report.json", path_report)
    return {"documents": doc_count, "source_samples": sample_count}


def _index_mmdocrag(
    root: Path,
    sample_files: list[Path],
    max_samples: int | None = None,
) -> tuple[list[DocumentRecord], list[SourceSample], dict[str, Any]]:
    docs: dict[str, DocumentRecord] = {}
    samples: list[SourceSample] = []
    report: dict[str, Any] = {
        "root": str(root),
        "sample_files": {},
        "missing_pdf": [],
        "max_samples": max_samples,
        "kept_samples": 0,
    }
    pdf_dir = root / "doc_pdfs"
    pdf_by_stem = {p.stem: p for p in pdf_dir.glob("*.pdf")} if pdf_dir.exists() else {}

    for path in sample_files:
        if max_samples is not None and len(samples) >= max_samples:
            break
        path = _resolve_dataset_path(root, path)
        filename = str(path)
        seen = 0
        kept = 0
        for idx, row in enumerate(read_jsonl(path)):
            seen += 1
            if max_samples is not None and len(samples) >= max_samples:
                break
            doc_name = str(row.get("doc_name") or row.get("doc_id") or row.get("document") or "unknown")
            doc_id = _normalize_doc_id(doc_name)
            pdf_path = _resolve_pdf_path(pdf_by_stem, doc_name)
            if doc_id not in docs:
                if not pdf_path:
                    report["missing_pdf"].append({"doc_id": doc_id, "source_file": filename})
                docs[doc_id] = DocumentRecord(
                    dataset="MMDocRAG",
                    doc_id=doc_id,
                    pdf_path=str(pdf_path) if pdf_path else None,
                    domain=row.get("domain"),
                    available_annotations=["text_quotes", "img_quotes", "gold_quotes"],
                )
            elif row.get("domain") and not docs[doc_id].domain:
                docs[doc_id].domain = row.get("domain")

            source_id = stable_id("MMDocRAG", filename, idx, doc_id, prefix="src_")
            samples.append(
                SourceSample(
                    source_sample_id=source_id,
                    dataset="MMDocRAG",
                    doc_id=doc_id,
                    source_file=str(path),
                    question=row.get("question"),
                    answer=row.get("answer_short") or row.get("answer") or row.get("answer_interleaved"),
                    question_type=row.get("question_type"),
                    evidence_pages=_extract_pages(row),
                    evidence_sources=_as_list(row.get("evidence_modality_type")),
                    raw=row,
                )
            )
            kept += 1
        report["sample_files"][filename] = {"seen": seen, "kept": kept}

    report["kept_samples"] = len(samples)
    return list(docs.values()), samples, report


def _index_longdocurl(
    root: Path,
    max_samples: int | None = None,
    max_docs: int | None = None,
    sample_jsonl: Path | None = None,
) -> tuple[list[DocumentRecord], list[SourceSample], dict[str, Any]]:
    docs: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    use_sample_jsonl = max_samples is not None
    report: dict[str, Any] = {
        "root": str(root),
        "image_root": str(root / "data" / "pdf_pngs"),
        "sample_jsonl": str(sample_jsonl) if sample_jsonl else None,
        "selection_mode": "sample_jsonl" if use_sample_jsonl else "image_root",
        "image_files": 0,
        "documents": 0,
        "selected_doc_ids": [],
        "skipped_files": [],
        "sample_files": {},
        "missing_pdf": [],
        "missing_images": [],
        "max_samples": max_samples,
        "max_docs": max_docs,
        "kept_samples": 0,
    }
    image_root = root / "data" / "pdf_pngs"
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}

    if image_root.exists():
        for path in image_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in image_suffixes:
                continue
            parsed = _parse_longdocurl_image_name(path)
            if parsed is None:
                report["skipped_files"].append(str(path))
                continue
            doc_id, page_id = parsed
            docs[doc_id].append((page_id, path))
            report["image_files"] += 1
    else:
        report["missing_images"].append({"image_root": str(image_root), "reason": "directory not found"})

    records: list[DocumentRecord] = []
    if use_sample_jsonl:
        doc_ids = _select_longdocurl_doc_ids(
            sample_jsonl=sample_jsonl,
            sample_limit=max_samples,
            all_doc_ids=set(docs),
            report=report,
        )
    else:
        doc_ids = sorted(docs, key=_natural_doc_sort_key)
        if max_docs is not None:
            if max_docs <= 0:
                doc_ids = []
            else:
                doc_ids = doc_ids[:max_docs]
        report["selected_doc_ids"] = doc_ids
    for doc_id in doc_ids:
        if doc_id not in docs:
            report["missing_images"].append({"doc_id": doc_id, "reason": "no full document pages found in pdf_pngs"})
            continue
        pages = sorted(docs[doc_id], key=lambda item: item[0])
        page_ids = [page_id for page_id, _ in pages]
        page_images = [str(path) for _, path in pages]
        records.append(
            DocumentRecord(
                dataset="LongDocURL",
                doc_id=doc_id,
                page_image_dir=None,
                page_images=page_images,
                total_pages=len(page_images),
                available_annotations=["page_images"],
                metadata={
                    "source": "pdf_pngs",
                    "page_ids": page_ids,
                    "page_image_by_page_id": {
                        str(page_id): str(path)
                        for page_id, path in pages
                    },
                    "page_id_by_index": {
                        str(index): page_id
                        for index, page_id in enumerate(page_ids)
                    },
                },
            )
        )

    report["documents"] = len(docs)
    report["kept_samples"] = len(records)
    return records, [], report


def _select_longdocurl_doc_ids(
    *,
    sample_jsonl: Path | None,
    sample_limit: int | None,
    all_doc_ids: set[str],
    report: dict[str, Any],
) -> list[str]:
    if sample_limit is not None and sample_limit <= 0:
        report["selected_doc_ids"] = []
        return []
    if sample_jsonl is None or not sample_jsonl.exists():
        report["missing_images"].append(
            {
                "sample_jsonl": str(sample_jsonl) if sample_jsonl else None,
                "reason": "sample jsonl not found",
            }
        )
        report["selected_doc_ids"] = []
        return []

    selected: list[str] = []
    seen: set[str] = set()
    seen_rows = 0
    kept_rows = 0
    missing_doc_ids: list[str] = []
    for row in read_jsonl(sample_jsonl):
        seen_rows += 1
        doc_id = str(row.get("doc_no") or row.get("doc_id") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id not in all_doc_ids:
            missing_doc_ids.append(doc_id)
            continue
        selected.append(doc_id)
        kept_rows += 1
        if sample_limit is not None and len(selected) >= sample_limit:
            break

    report["sample_files"][str(sample_jsonl)] = {
        "seen": seen_rows,
        "kept": kept_rows,
        "missing_doc_ids": missing_doc_ids[:100],
    }
    report["selected_doc_ids"] = selected
    return selected


def _index_mmlongbench_doc(
    root: Path,
    documents_dir: Path,
    sample_file: Path,
    max_samples: int | None = None,
) -> tuple[list[DocumentRecord], list[SourceSample], dict[str, Any]]:
    docs: dict[str, DocumentRecord] = {}
    samples: list[SourceSample] = []
    report: dict[str, Any] = {
        "root": str(root),
        "documents_dir": str(documents_dir),
        "sample_files": {},
        "missing_pdf": [],
        "max_samples": max_samples,
        "kept_samples": 0,
    }
    sample_path = _resolve_dataset_path(root, sample_file)
    pdf_index = _build_basename_index(documents_dir, suffixes={".pdf"})
    seen = 0
    kept = 0

    for idx, row in enumerate(_read_json_list(sample_path)):
        if max_samples is not None and len(samples) >= max_samples:
            break
        seen += 1
        doc_id = _normalize_doc_id(
            str(row.get("doc_id") or row.get("document_id") or row.get("doc_name") or "unknown")
        )
        pdf_path = _resolve_local_path(doc_id, pdf_index)
        if doc_id not in docs:
            if not pdf_path:
                report["missing_pdf"].append({"doc_id": doc_id, "source_file": str(sample_path)})
            docs[doc_id] = DocumentRecord(
                dataset="MMLongBench-Doc",
                doc_id=doc_id,
                pdf_path=pdf_path,
                available_annotations=["evidence_pages", "evidence_sources", "answer_format"],
                metadata={"doc_type": row.get("doc_type")},
            )
        source_id = stable_id("MMLongBench-Doc", sample_path.name, idx, doc_id, prefix="src_")
        samples.append(
            SourceSample(
                source_sample_id=source_id,
                dataset="MMLongBench-Doc",
                doc_id=doc_id,
                source_file=str(sample_path),
                question=row.get("question"),
                answer=row.get("answer"),
                question_type=row.get("doc_type"),
                evidence_pages=_parse_int_list(row.get("evidence_pages")),
                evidence_sources=[str(x) for x in _parse_literal_list(row.get("evidence_sources"))],
                raw=row,
            )
        )
        kept += 1

    report["sample_files"][sample_path.name] = {"seen": seen, "kept": kept}
    report["kept_samples"] = len(samples)
    return list(docs.values()), samples, report


def _index_madqa(
    root: Path,
    pdf_dir: Path,
    sample_file: Path,
    max_samples: int | None = None,
) -> tuple[list[DocumentRecord], list[SourceSample], dict[str, Any]]:
    docs: dict[str, DocumentRecord] = {}
    samples: list[SourceSample] = []
    sample_path = _resolve_dataset_path(root, sample_file)
    pdf_index = _build_basename_index(pdf_dir, suffixes={".pdf"})
    _add_madqa_document_aliases(root, pdf_index)
    report: dict[str, Any] = {
        "root": str(root),
        "pdf_dir": str(pdf_dir),
        "sample_file": str(sample_path),
        "missing_pdf": [],
        "max_samples": max_samples,
        "kept_samples": 0,
    }

    for idx, row in enumerate(_read_json_list(sample_path)):
        if max_samples is not None and len(samples) >= max_samples:
            break
        evidence = _parse_madqa_evidence(row.get("evidence"))
        documents = [str(item["document"]) for item in evidence if item.get("document")]
        doc_name = documents[0] if documents else str(row.get("document") or row.get("doc_id") or "unknown")
        doc_id = _normalize_doc_id(doc_name)
        pdf_path = _resolve_local_path(doc_name, pdf_index)

        if doc_id not in docs:
            if not pdf_path:
                report["missing_pdf"].append({"doc_id": doc_id, "source_file": str(sample_path)})
            docs[doc_id] = DocumentRecord(
                dataset="MADQA",
                doc_id=doc_id,
                pdf_path=pdf_path,
                domain=row.get("domain"),
                available_annotations=["answer_variants", "evidence_pages"],
                metadata={
                    "document_category": row.get("document_category"),
                    "sample_file": str(sample_path),
                },
            )

        source_id = stable_id("MADQA", str(sample_path), row.get("id") or idx, doc_id, prefix="src_")
        samples.append(
            SourceSample(
                source_sample_id=source_id,
                dataset="MADQA",
                doc_id=doc_id,
                source_file=str(sample_path),
                question=row.get("question"),
                answer=row.get("answer_variants"),
                question_type=row.get("document_category"),
                evidence_pages=sorted({
                    page
                    for page in (_to_int(item.get("page")) for item in evidence)
                    if page is not None
                }),
                evidence_sources=documents,
                raw=row,
            )
        )

    report["kept_samples"] = len(samples)
    return list(docs.values()), samples, report


def _index_dude(
    root: Path,
    pdf_dir: Path,
    sample_file: Path,
    max_samples: int | None = None,
) -> tuple[list[DocumentRecord], list[SourceSample], dict[str, Any]]:
    docs: dict[str, DocumentRecord] = {}
    samples: list[SourceSample] = []
    sample_path = _resolve_dataset_path(root, sample_file)
    pdf_index = _build_basename_index(pdf_dir, suffixes={".pdf"})
    report: dict[str, Any] = {
        "root": str(root),
        "pdf_dir": str(pdf_dir),
        "sample_file": str(sample_path),
        "missing_pdf": [],
        "max_samples": max_samples,
        "kept_samples": 0,
    }

    for idx, row in enumerate(_read_json_collection(sample_path, "data")):
        if max_samples is not None and len(samples) >= max_samples:
            break
        doc_id = _normalize_doc_id(row.get("docId") or row.get("doc_id") or row.get("document") or "unknown")
        pdf_path = _resolve_local_path(doc_id, pdf_index)

        if doc_id not in docs:
            if not pdf_path:
                report["missing_pdf"].append({"doc_id": doc_id, "source_file": str(sample_path)})
            docs[doc_id] = DocumentRecord(
                dataset="DUDE",
                doc_id=doc_id,
                pdf_path=pdf_path,
                available_annotations=["answers", "answer_bounding_boxes"],
                metadata={
                    "sample_file": str(sample_path),
                    "data_split": row.get("data_split"),
                },
            )

        source_id = stable_id("DUDE", str(sample_path), row.get("questionId") or idx, doc_id, prefix="src_")
        samples.append(
            SourceSample(
                source_sample_id=source_id,
                dataset="DUDE",
                doc_id=doc_id,
                source_file=str(sample_path),
                question=row.get("question"),
                answer=row.get("answers"),
                question_type=row.get("answer_type"),
                evidence_pages=_extract_dude_pages(row),
                raw=row,
            )
        )

    report["kept_samples"] = len(samples)
    return list(docs.values()), samples, report


def _index_finragbench(
    root: Path,
    image_dir: Path,
    sample_file: Path,
    max_samples: int | None = None,
) -> tuple[list[DocumentRecord], list[SourceSample], dict[str, Any]]:
    sample_path = _resolve_dataset_path(root, sample_file)
    rows = _read_json_list(sample_path)
    if max_samples is not None:
        rows = rows[:max(0, max_samples)]

    samples: list[SourceSample] = []
    wanted_docs: dict[str, None] = {}
    for idx, row in enumerate(rows):
        query_id = str(row.get("query-id") or idx)
        doc_id = _finrag_doc_id(query_id)
        if not doc_id:
            continue
        wanted_docs.setdefault(doc_id, None)
        samples.append(
            SourceSample(
                source_sample_id=stable_id("FinRAGBench-V", query_id, prefix="src_"),
                dataset="FinRAGBench-V",
                doc_id=doc_id,
                source_file=str(sample_path),
                question=row.get("query"),
                answer=row.get("answer"),
                question_type=row.get("category"),
                evidence_pages=_parse_int_list(row.get("from_pages")),
                raw=row,
            )
        )

    pages_by_doc: dict[str, dict[int, Path]] = defaultdict(dict)
    if image_dir.exists():
        for path in sorted(image_dir.rglob("*")):
            parsed = _parse_finrag_image_name(path)
            if parsed and parsed[0] in wanted_docs:
                pages_by_doc[parsed[0]].setdefault(parsed[1], path)

    docs: list[DocumentRecord] = []
    missing_images = []
    for doc_id in wanted_docs:
        pages = sorted(pages_by_doc.get(doc_id, {}).items())
        if not pages:
            missing_images.append({"doc_id": doc_id, "reason": "no document page images found"})
            continue
        page_ids = [page_id for page_id, _ in pages]
        docs.append(
            DocumentRecord(
                dataset="FinRAGBench-V",
                doc_id=doc_id,
                page_images=[str(path) for _, path in pages],
                total_pages=len(pages),
                available_annotations=["page_images", "answers", "evidence_pages"],
                metadata={
                    "source": image_dir.name,
                    "page_ids": page_ids,
                    "page_image_by_page_id": {str(page_id): str(path) for page_id, path in pages},
                    "page_id_by_index": {
                        str(index): page_id
                        for index, page_id in enumerate(page_ids)
                    },
                },
            )
        )

    return docs, samples, {
        "root": str(root),
        "image_dir": str(image_dir),
        "sample_file": str(sample_path),
        "max_samples": max_samples,
        "kept_samples": len(samples),
        "documents": len(docs),
        "image_files": sum(len(pages) for pages in pages_by_doc.values()),
        "missing_images": missing_images,
    }


def _index_docbench(
    root: Path,
    data_dir: Path,
    max_samples: int | None = None,
) -> tuple[list[DocumentRecord], list[SourceSample], dict[str, Any]]:
    docs: dict[str, DocumentRecord] = {}
    samples: list[SourceSample] = []
    report: dict[str, Any] = {
        "root": str(root),
        "data_dir": str(data_dir),
        "missing_pdf": [],
        "missing_qa": [],
        "max_samples": max_samples,
        "kept_samples": 0,
    }
    if max_samples is not None and max_samples <= 0:
        return [], [], report
    if not data_dir.exists():
        report["missing_qa"].append({"data_dir": str(data_dir), "reason": "directory not found"})
        return [], [], report

    for folder in sorted((p for p in data_dir.iterdir() if p.is_dir()), key=lambda p: _natural_doc_sort_key(p.name)):
        if max_samples is not None and len(samples) >= max_samples:
            break
        qa_path = folder / f"{folder.name}_qa.jsonl"
        if not qa_path.exists():
            report["missing_qa"].append({"doc_id": folder.name, "qa_path": str(qa_path)})
            continue
        pdfs = sorted(folder.glob("*.pdf"))
        pdf_path = pdfs[0] if pdfs else None
        doc_id = _normalize_doc_id(folder.name)
        if doc_id not in docs:
            if not pdf_path:
                report["missing_pdf"].append({"doc_id": doc_id, "folder": str(folder)})
            docs[doc_id] = DocumentRecord(
                dataset="DocBench",
                doc_id=doc_id,
                pdf_path=str(pdf_path) if pdf_path else None,
                available_annotations=["answers", "evidence", "type"],
                metadata={
                    "folder": str(folder),
                    "pdf_name": pdf_path.name if pdf_path else None,
                },
            )
        for idx, row in enumerate(read_jsonl(qa_path)):
            if max_samples is not None and len(samples) >= max_samples:
                break
            samples.append(
                SourceSample(
                    source_sample_id=stable_id("DocBench", folder.name, idx, prefix="src_"),
                    dataset="DocBench",
                    doc_id=doc_id,
                    source_file=str(qa_path),
                    question=row.get("question"),
                    answer=row.get("answer"),
                    question_type=row.get("type"),
                    raw=row,
                )
            )

    report["documents"] = len(docs)
    report["kept_samples"] = len(samples)
    return list(docs.values()), samples, report


def _finrag_doc_id(query_id: str) -> str:
    image_stem = Path(str(query_id).rsplit("-", 1)[0]).stem
    if "_multipage_" in image_stem:
        return image_stem.rpartition("_multipage_")[0]
    return image_stem.rpartition("_")[0]


def _parse_finrag_image_name(path: Path) -> tuple[str, int] | None:
    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None
    doc_id, sep, page_text = path.stem.rpartition("_")
    if not sep or not doc_id or "_multipage_" in doc_id:
        return None
    try:
        return doc_id, int(page_text)
    except ValueError:
        return None


def _add_madqa_document_aliases(root: Path, pdf_index: dict[str, str]) -> None:
    pdf_root = root / "data" / "pdfs"
    for row in _read_json_list(root / "data" / "documents" / "document_urls.json"):
        document = row.get("document")
        url = row.get("url")
        if not document:
            continue

        doc_name = str(document)
        doc_pdf = pdf_root / f"{_normalize_doc_id(doc_name)}.pdf"
        if doc_pdf.exists():
            pdf_path = str(doc_pdf)
        else:
            if not url:
                continue
            url_name = Path(str(url).split("?", 1)[0]).name
            pdf_path = pdf_index.get(url_name) or pdf_index.get(url_name.lower())

        if pdf_path:
            pdf_index[doc_name] = pdf_path
            pdf_index[doc_name.lower()] = pdf_path


def _resolve_pdf_path(pdf_by_stem: dict[str, Path], doc_name: str) -> Path | None:
    path = Path(str(doc_name))
    candidates = [path.stem, str(doc_name), str(doc_name).removesuffix(".pdf")]
    for candidate in candidates:
        if candidate in pdf_by_stem:
            return pdf_by_stem[candidate]
    if path.exists() and path.suffix.lower() == ".pdf":
        return path
    return None


def _build_basename_index(root: Path, suffixes: set[str] | None = None) -> dict[str, str]:
    index: dict[str, str] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        index.setdefault(path.name, str(path))
        index.setdefault(path.name.lower(), str(path))
    return index


def _resolve_local_path(raw: Any, basename_index: dict[str, str]) -> str | None:
    raw_path = norm_path(raw)
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return str(path)
    candidates = [path.name]
    if path.suffix.lower() != ".pdf":
        candidates.append(f"{path.name}.pdf")
    else:
        candidates.append(path.stem)
    for candidate in candidates:
        if candidate in basename_index:
            return basename_index[candidate]
        folded = candidate.lower()
        if folded in basename_index:
            return basename_index[folded]
    return None


def _resolve_dataset_path(root: Path, path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def _infer_page_image_dir(paths: list[str]) -> str | None:
    if not paths:
        return None
    parents = {str(Path(p).parent) for p in paths}
    if len(parents) == 1:
        return next(iter(parents))
    parent_counts: dict[str, int] = defaultdict(int)
    for p in paths:
        parent_counts[str(Path(p).parent)] += 1
    return max(parent_counts.items(), key=lambda item: item[1])[0]


def _parse_longdocurl_image_name(path: Path) -> tuple[str, int] | None:
    doc_id, sep, page_text = path.stem.rpartition("_")
    if not sep or not doc_id:
        return None
    try:
        # LongDocURL image filenames are zero-based, e.g. 4137832_0.png.
        # Keep the original zero-based page ids in registry metadata.
        return doc_id, int(page_text)
    except ValueError:
        return None


def _natural_doc_sort_key(doc_id: str) -> tuple[int, str]:
    try:
        return int(doc_id), doc_id
    except ValueError:
        return 10**18, doc_id


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import json

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _read_json_collection(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import json

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get(key, []) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_literal_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        return [parsed]
    return [value]


def _parse_madqa_evidence(value: Any) -> list[dict[str, Any]]:
    items = _parse_literal_list(value)
    if len(items) == 1 and isinstance(items[0], str):
        items = [
            {"document": document, "page": int(page)}
            for document, page in re.findall(r"'document': '([^']+)', 'page': (\d+)", items[0])
        ]
    return [item for item in items if isinstance(item, dict)]


def _parse_int_list(value: Any) -> list[int]:
    result: list[int] = []
    for item in _parse_literal_list(value):
        parsed = _to_int(item)
        if parsed is not None:
            result.append(parsed)
    return result


def _extract_dude_pages(row: dict[str, Any]) -> list[int]:
    pages: set[int] = set()
    for group in _as_list(row.get("answers_page_bounding_boxes")):
        for box in _as_list(group):
            if not isinstance(box, dict):
                continue
            page = _to_int(box.get("page"))
            if page is not None:
                pages.add(page)
    return sorted(pages)


def _extract_pages(row: dict[str, Any]) -> list[int]:
    pages: set[int] = set()
    for key in ("text_quotes", "img_quotes", "detailed_evidences"):
        for item in _as_list(row.get(key)):
            if not isinstance(item, dict):
                continue
            page = _to_int(item.get("page_id") or item.get("page") or item.get("page_no"))
            if page is not None:
                pages.add(page)
    for page in _as_list(row.get("evidence_pages")):
        value = _to_int(page)
        if value is not None:
            pages.add(value)
    return sorted(pages)


def _normalize_doc_id(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        return "unknown"
    path = Path(raw)
    name = path.name or raw
    if name.lower().endswith(".pdf"):
        return name[:-4]
    return name


def _build_include_set(items: list[Any]) -> set[tuple[str, str]]:
    include_specs: set[tuple[str, str]] = set()
    for item in items or []:
        if isinstance(item, dict):
            dataset = item.get("dataset")
            doc_id = item.get("doc_id")
        else:
            dataset = getattr(item, "dataset", None)
            doc_id = getattr(item, "doc_id", None)
        if dataset and doc_id:
            for candidate_doc_id in _doc_id_aliases(str(doc_id)):
                include_specs.add((str(dataset), candidate_doc_id))
    return include_specs


def _doc_id_aliases(doc_id: str) -> set[str]:
    raw = str(doc_id).strip()
    if not raw:
        return set()
    aliases = {raw}
    if raw.lower().endswith(".pdf"):
        aliases.add(raw[:-4])
    else:
        aliases.add(f"{raw}.pdf")
    return aliases


__all__ = ["build_document_registries"]
