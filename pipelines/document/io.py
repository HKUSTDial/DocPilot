from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path | str, skip_invalid: bool = False) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if skip_invalid:
                    continue
                raise ValueError(f"Invalid JSONL at {path}:{line_no}")
            if isinstance(value, dict):
                yield value
            else:
                yield {"value": value}


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def stable_id(*parts: Any, prefix: str = "") -> str:
    import hashlib

    raw = "||".join(safe_text(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def norm_path(path: Any) -> str | None:
    if not path:
        return None
    return str(Path(str(path)).expanduser())


def qa_output_dir(base_dir: Path | str, model_name: str | None) -> Path:
    base = Path(base_dir)
    model_name = (model_name or "").strip()
    if not model_name:
        return base
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")
    return ensure_dir(base / (safe or "unknown-model"))
