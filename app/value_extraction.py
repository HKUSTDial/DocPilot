from __future__ import annotations

import re
from html import unescape
from typing import Any


_OCR_FIXES = {
    "liaities": "liabilities",
    "liabiities": "liabilities",
    "libities": "liabilities",
    "curent": "current",
    "activiies": "activities",
    "contrlling": "controlling",
    "ssued": "issued",
}


def normalize_label(text: str) -> str:
    normalized = unescape(str(text or "")).lower()
    for bad, good in _OCR_FIXES.items():
        normalized = normalized.replace(bad, good)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def parse_number(text: str) -> float | None:
    value = str(text or "").strip()
    if not value:
        return None
    value = value.replace("$", "").replace("€", "").replace("£", "")
    value = value.replace("—", "").replace("–", "").replace("一", "").strip()
    if not value:
        return None
    match = re.search(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", value)
    if not match:
        return None
    token = match.group(0)
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").replace(",", "")
    try:
        number = float(token)
    except ValueError:
        return None
    return -number if negative else number


def format_number(value: float, decimals: int | None = None) -> str:
    if decimals is not None:
        return f"{value:.{decimals}f}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value}".rstrip("0").rstrip(".")


def lookup_metric_value(contents: list[Any], labels: list[str], *, year: str = "2021") -> float | None:
    for content in contents:
        value = lookup_metric_value_in_content(content, labels, year=year)
        if value is not None:
            return value
    return None


def lookup_metric_value_in_content(content: Any, labels: list[str], *, year: str = "2021") -> float | None:
    rows = _extract_rows(content)
    if rows:
        value = _lookup_in_rows(rows, labels, year=year)
        if value is not None:
            return value

    text = _content_to_text(content)
    if text:
        return _lookup_in_text(text, labels, year=year)
    return None


def _extract_rows(content: Any) -> list[list[str]]:
    if isinstance(content, dict) and content.get("html"):
        return _parse_html_rows(str(content["html"]))
    if isinstance(content, str) and "<table" in content.lower():
        return _parse_html_rows(content)
    return []


def _parse_html_rows(html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
        cleaned = [_clean_cell(cell) for cell in cells]
        if cleaned:
            rows.append(cleaned)
    return rows


def _clean_cell(cell: str) -> str:
    text = re.sub(r"<[^>]+>", " ", cell)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _lookup_in_rows(rows: list[list[str]], labels: list[str], *, year: str) -> float | None:
    label_candidates = [normalize_label(label) for label in labels]
    year_columns = _find_year_columns(rows, year)
    if not year_columns:
        year_columns = [1]

    for label in label_candidates:
        for row in rows:
            if not row:
                continue
            row_label = normalize_label(row[0])
            if not row_label or not _label_matches(row_label, label):
                continue
            for col_idx in year_columns:
                if col_idx < len(row):
                    value = parse_number(row[col_idx])
                    if value is not None:
                        return value
            for cell in row[1:]:
                value = parse_number(cell)
                if value is not None:
                    return value
    return None


def _find_year_columns(rows: list[list[str]], year: str) -> list[int]:
    for row in rows[:3]:
        cols = [idx for idx, cell in enumerate(row) if year in normalize_label(cell)]
        if cols:
            return cols
    return []


def _label_matches(row_label: str, target_label: str) -> bool:
    if row_label == target_label:
        return True
    if row_label.startswith(target_label) or target_label.startswith(row_label):
        return True
    row_tokens = row_label.split()
    target_tokens = target_label.split()
    if not target_tokens or len(target_tokens) > len(row_tokens):
        return False
    for start in range(len(row_tokens) - len(target_tokens) + 1):
        if row_tokens[start : start + len(target_tokens)] == target_tokens:
            return True
    return False


def _lookup_in_text(text: str, labels: list[str], *, year: str) -> float | None:
    compact = re.sub(r"\s+", " ", text)
    for label in labels:
        label_pattern = re.escape(label).replace(r"\ ", r"\s+")
        patterns = [
            rf"{label_pattern}.{{0,60}}?\$?\s*(\(?-?\d[\d,]*(?:\.\d+)?\)?)\s*.{{0,30}}?{year}",
            rf"{label_pattern}.{{0,120}}?(\(?-?\d[\d,]*(?:\.\d+)?\)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, compact, flags=re.IGNORECASE)
            if match:
                value = parse_number(match.group(1))
                if value is not None:
                    return value
    return None


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return " ".join(str(value) for value in content.values() if value)
    return str(content)
