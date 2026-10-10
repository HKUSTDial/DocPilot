from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
from typing import Any


DEFAULT_ROOT = Path("data/DocVQA/single-ablity-data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect json/jsonl datasets under single-ablity-data.")
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--file", type=Path, default=None, help="Inspect one file instead of scanning root.")
    parser.add_argument("--samples", type=int, default=1, help="Samples to print per file.")
    parser.add_argument("--max-files", type=int, default=None, help="Only inspect the first N files.")
    parser.add_argument("--max-records", type=int, default=None, help="Only read the first N records per file.")
    parser.add_argument("--width", type=int, default=220, help="Max chars for printed sample values.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    files = [args.file] if args.file else sorted(args.root.rglob("*.json")) + sorted(args.root.rglob("*.jsonl"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No json/jsonl files found under {args.root}")

    for path in files:
        inspect_file(path, base=args.root, samples=max(0, args.samples), max_records=args.max_records, width=args.width)


def iter_rows(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with path.open(encoding="utf-8", errors="replace") as f:
        first = first_nonspace(f)
    if first == "[":
        yield from iter_json_array(path)
        return

    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        yield from data["data"]
    else:
        yield data


def first_nonspace(f) -> str:
    while True:
        char = f.read(1)
        if not char or not char.isspace():
            return char


def iter_json_array(path: Path):
    decoder = json.JSONDecoder()
    buf = ""
    started = False
    with path.open(encoding="utf-8", errors="replace") as f:
        while True:
            if len(buf) < 65536:
                chunk = f.read(65536)
                if chunk:
                    buf += chunk
            buf = buf.lstrip()
            if not started:
                if not buf:
                    break
                if buf[0] != "[":
                    raise ValueError(f"Expected JSON array in {path}")
                buf = buf[1:]
                started = True
                continue
            buf = buf.lstrip()
            if buf.startswith(","):
                buf = buf[1:]
                continue
            if buf.startswith("]"):
                break
            try:
                item, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = f.read(65536)
                if not chunk:
                    raise
                buf += chunk
                continue
            yield item
            buf = buf[end:]


def inspect_file(path: Path, *, base: Path, samples: int, max_records: int | None, width: int) -> None:
    key_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    image_fields: Counter[str] = Counter()
    first_rows: list[Any] = []
    records = 0
    error = ""

    try:
        for row in iter_rows(path):
            records += 1
            if len(first_rows) < samples:
                first_rows.append(row)
            if isinstance(row, dict):
                key_counts.update(row.keys())
                collect_roles(row, role_counts)
                collect_image_fields(row, image_fields)
            if max_records is not None and records >= max_records:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    rel = str(path.relative_to(base)) if path.is_relative_to(base) else str(path)
    size_mb = path.stat().st_size / 1024 / 1024
    print("\n" + "=" * 100)
    print(f"{rel}")
    print(f"size={size_mb:.2f}MB records={records}{'+' if max_records and records >= max_records else ''} error={error or '-'}")
    print(f"keys={format_counter(key_counts)}")
    if role_counts:
        print(f"roles={format_counter(role_counts)}")
    if image_fields:
        print(f"image_fields={format_counter(image_fields)}")

    for i, row in enumerate(first_rows):
        print(f"\n--- sample {i} ---")
        print(short_json(row, width))


def collect_roles(row: dict[str, Any], counts: Counter[str]) -> None:
    for field in ("messages", "conversations"):
        items = row.get(field)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    counts[str(item.get("role") or item.get("from") or "?")] += 1


def collect_image_fields(row: dict[str, Any], counts: Counter[str]) -> None:
    for key in ("image", "images", "image_urls", "img_path"):
        value = row.get(key)
        if value:
            counts[key] += len(value) if isinstance(value, list) else 1


def format_counter(counter: Counter[str], limit: int = 12) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def short_json(value: Any, width: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    lines = []
    for line in text.splitlines():
        lines.append(line if len(line) <= width else line[: width - 3] + "...")
    return "\n".join(lines[:80])


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.jsonl").write_text('{"messages":[{"role":"user"}],"image":"x.png"}\n', encoding="utf-8")
        (root / "b.json").write_text(json.dumps({"data": [{"question": "q", "answers": ["a"]}]}), encoding="utf-8")
        assert sum(1 for _ in iter_rows(root / "a.jsonl")) == 1
        assert sum(1 for _ in iter_rows(root / "b.json")) == 1
    print("self-test ok")


if __name__ == "__main__":
    main()
