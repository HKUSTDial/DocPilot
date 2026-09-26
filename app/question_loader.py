from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from typing import Any

_KNOWN_DOC_SUFFIXES = {
    ".pdf",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".html",
    ".htm",
}


def _iter_question_rows(path: str | Path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    for row in data:
        yield row


def _row_doc_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in names:
            names.append(text)

    explicit = row.get("doc_names") or row.get("doc_ids")
    if isinstance(explicit, list):
        for value in explicit:
            add(value)
    elif explicit:
        add(explicit)

    if not names:
        add(row.get("doc_name") or row.get("doc_id") or row.get("doc_no") or row.get("docId"))

    evidence = row.get("raw_evidence") or row.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                add(item.get("document"))
    elif isinstance(evidence, str):
        try:
            parsed = ast.literal_eval(evidence)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    add(item.get("document"))
        for match in re.finditer(r"['\"]document['\"]\s*:\s*['\"]([^'\"]+)['\"]", evidence):
            add(match.group(1))

    if not names:
        evidence_sources = row.get("evidence_sources")
        if isinstance(evidence_sources, list) and evidence_sources:
            add(evidence_sources[0])
    return names


def _row_doc_name(row: dict[str, Any]) -> str | None:
    names = _row_doc_names(row)
    return names[0] if names else None


def _normalize_doc_name(doc_name: str | None) -> str:
    if not doc_name:
        return ""
    path = Path(str(doc_name).strip())
    suffix = path.suffix.lower()
    if suffix in _KNOWN_DOC_SUFFIXES:
        return path.stem.lower()
    return path.name.lower()


def _row_question_id(row: dict[str, Any]) -> str | None:
    value = row.get("q_id")
    if value is None:
        value = row.get("question_id")
    if value is None:
        value = row.get("questionId")
    if value is None:
        value = row.get("id")
    if value is None:
        return None
    return str(value)


def _row_ground_truth(row: dict[str, Any]) -> str | None:
    answers = row.get("answers")
    if answers:
        return answers[0] if isinstance(answers, list) else answers
    if row.get("answer_type") == "not-answerable":
        return "Not answerable"
    return row.get("answer_short") or row.get("answer_interleaved") or row.get("answer") or row.get("answer_variants")


def _row_page_range(row: dict[str, Any]) -> tuple[int, int] | None:
    value = row.get("start_end_idx")
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError):
        return None
    if start <= 0 or end <= 0:
        return None
    if start > end:
        start, end = end, start
    return start, end


def _normalize_question_text(text: str | None) -> str:
    return " ".join((text or "").split())


def load_questions_for_doc(path: str | Path, doc_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalized_doc_name = _normalize_doc_name(doc_name)
    for row in _iter_question_rows(path):
        if _normalize_doc_name(_row_doc_name(row)) == normalized_doc_name:
            rows.append(row)
    return rows


def load_all_questions(path: str | Path) -> list[dict[str, Any]]:
    return list(_iter_question_rows(path))


def load_question_by_id(path: str | Path, q_id: str | int) -> dict[str, Any]:
    target_qid = str(q_id)
    for row in _iter_question_rows(path):
        row_qid = _row_question_id(row)
        if row_qid is not None and row_qid == target_qid:
            return row
    raise KeyError(f"Question id {q_id} not found in {path}")


def load_question_by_doc_and_text(path: str | Path, doc_name: str, question: str) -> dict[str, Any]:
    normalized_doc_name = _normalize_doc_name(doc_name)
    normalized_question = _normalize_question_text(question)
    for row in _iter_question_rows(path):
        if _normalize_doc_name(_row_doc_name(row)) == normalized_doc_name and _normalize_question_text(row.get("question")) == normalized_question:
            return row
    raise KeyError(f"Question not found for doc {doc_name} in {path}")


def load_question_entry_by_doc_and_text(
    path: str | Path,
    doc_name: str,
    question: str,
) -> tuple[int, dict[str, Any]]:
    normalized_doc_name = _normalize_doc_name(doc_name)
    normalized_question = _normalize_question_text(question)
    for index, row in enumerate(_iter_question_rows(path)):
        if (
            _normalize_doc_name(_row_doc_name(row)) == normalized_doc_name
            and _normalize_question_text(row.get("question")) == normalized_question
        ):
            return index, row
    raise KeyError(f"Question not found for doc {doc_name} in {path}")
