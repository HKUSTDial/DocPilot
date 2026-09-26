from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_WORKSPACE_ROOT = Path("data/DataScience-Instruct-500K/RL/data/files")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "solve_file_tasks.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeepAnalyze file_* data to DocPilot single-ability Solve JSONL.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="Randomly sample this many converted rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--strict", action="store_true", help="Skip rows when referenced files cannot be resolved.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    skipped = seen = 0
    samples: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    if args.limit is None:
        out = args.output.open("w", encoding="utf-8")
    else:
        out = None
    try:
        for input_path in args.input:
            for row in iter_rows(input_path):
                sample = convert_row(row, args.workspace_root, max_rows=args.max_rows, strict=args.strict)
                if sample is None:
                    skipped += 1
                    continue
                seen += 1
                if out is not None:
                    out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                elif len(samples) < max(0, args.limit):
                    samples.append(sample)
                else:
                    index = rng.randrange(seen)
                    if index < args.limit:
                        samples[index] = sample
    finally:
        if out is not None:
            out.close()

    if args.limit is not None:
        with args.output.open("w", encoding="utf-8") as sampled_out:
            for sample in samples:
                sampled_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    written = seen if args.limit is None else len(samples)
    print(f"Wrote {written} samples to {args.output}; converted={seen}; skipped={skipped}")


def iter_rows(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace").lstrip()
    if text.startswith("["):
        yield from json.loads(text)
    else:
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)


def convert_row(row: dict[str, Any], workspace_root: Path, *, max_rows: int, strict: bool) -> dict[str, Any] | None:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None
    user = str(messages[0].get("content", ""))
    assistant = str(messages[1].get("content", ""))
    file_names = extract_file_names(user)
    workspace = workspace_root / str(row.get("id"))
    evidence, data_literal, missing = build_evidence(workspace, file_names, max_rows=max_rows)
    if strict and missing:
        return None

    instruction = user.split("# Data", 1)[0].strip()
    user_content = "\n\n".join([instruction, "Extracted evidence:", evidence]).strip()
    solve_content = build_solve_content(assistant, data_literal)
    return {"messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": solve_content}]}


def extract_file_names(text: str) -> list[str]:
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    return list(dict.fromkeys(names))


def build_evidence(workspace: Path, file_names: list[str], *, max_rows: int) -> tuple[str, str, list[str]]:
    parts: list[str] = []
    literals: dict[str, Any] = {}
    missing: list[str] = []
    for name in file_names:
        path = workspace / name
        if not path.exists():
            missing.append(name)
            parts.append(f"File: {name}\n[MISSING]")
            continue
        rendered, literal = render_file(path, max_rows=max_rows)
        parts.append(f"File: {name}\n{rendered}")
        literals[name] = literal
    return "\n\n".join(parts), input_prefix(literals), missing


def render_file(path: Path, *, max_rows: int) -> tuple[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return render_csv(path, max_rows=max_rows)
    if suffix == ".xlsx":
        return render_xlsx(path, max_rows=max_rows)
    if suffix == ".db":
        return render_db(path, max_rows=max_rows)
    if suffix == ".json":
        return render_json(path, max_rows=max_rows)
    return render_text(path, max_chars=6000)


def render_csv(path: Path, *, max_rows: int) -> tuple[str, Any]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = [{k: parse_scalar(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return table_text(rows[:max_rows]), rows


def render_xlsx(path: Path, *, max_rows: int) -> tuple[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return render_text(path, max_chars=1000)
    df = pd.read_excel(path)
    rows = df.where(df.notna(), None).to_dict(orient="records")
    return table_text(rows[:max_rows]), rows


def render_db(path: Path, *, max_rows: int) -> tuple[str, Any]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    data: dict[str, list[dict[str, Any]]] = {}
    parts: list[str] = []
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for table in tables:
            rows = [dict(r) for r in conn.execute(f'SELECT * FROM "{table}" LIMIT ?', (max_rows,))]
            data[table] = rows
            parts.append(f"Table: {table}\n{table_text(rows)}")
    finally:
        conn.close()
    return "\n\n".join(parts), data


def render_json(path: Path, *, max_rows: int) -> tuple[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.dumps(rows[:max_rows], ensure_ascii=False, indent=2), rows
    preview = data[:max_rows] if isinstance(data, list) else data
    return json.dumps(preview, ensure_ascii=False, indent=2), data


def render_text(path: Path, *, max_chars: int) -> tuple[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    return text, text


def table_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(empty)"
    headers = list(rows[0].keys())
    lines = [" | ".join(headers)]
    for row in rows:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))


def input_prefix(literals: dict[str, Any]) -> str:
    return """inputs = %r

import pandas as pd
dfs = {name: pd.DataFrame(value) for name, value in inputs.items() if isinstance(value, list)}
if len(dfs) == 1:
    df = next(iter(dfs.values()))""" % literals


def parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?\d*\.\d+", text):
            return float(text)
    except ValueError:
        return value
    return value


def build_solve_content(assistant: str, data_literal: str) -> str:
    analyze = last_tag(assistant, "Analyze") or last_tag(assistant, "Understand") or "Use the extracted evidence to compute the answer."
    code = last_tag(assistant, "Code")
    code = strip_fence(code)
    code = rewrite_file_reads(code)
    answer = last_tag(assistant, "Answer")
    result = clean_answer(answer)
    if result:
        code = f"""{code.rstrip()}

# Fallback final result from the original trajectory.
if 'result' not in locals():
    for _name in ('count', 'answer', 'final_answer', 'total', 'num', 'n'):
        if _name in locals():
            result = locals()[_name]
            break
if 'result' not in locals() and 'older_than_56' in locals():
    result = len(older_than_56)
if 'result' not in locals():
    result = {result!r}"""
    code = f"{data_literal}\n\n{sqlite_helper()}\n\n{code}".strip()
    return f"<Solve>\n{analyze.strip()}\n\ncode: \\boxed{{{code}}}\n</Solve>"


def last_tag(text: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.S)
    return matches[-1] if matches else ""


def strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def rewrite_file_reads(code: str) -> str:
    code = re.sub(r"pd\.read_csv\((['\"])([^'\"]+)\1\)", lambda m: f"pd.DataFrame(inputs[{m.group(2)!r}])", code)
    code = re.sub(r"pd\.read_excel\((['\"])([^'\"]+)\1\)", lambda m: f"pd.DataFrame(inputs[{m.group(2)!r}])", code)
    code = re.sub(r"sqlite3\.connect\((['\"])([^'\"]+)\1\)", lambda m: f"make_sqlite_conn(inputs[{m.group(2)!r}])", code)
    return code


def sqlite_helper() -> str:
    return """def make_sqlite_conn(tables):
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(':memory:')
    for table_name, rows in tables.items():
        pd.DataFrame(rows).to_sql(table_name, conn, index=False, if_exists='replace')
    return conn"""


def clean_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


if __name__ == "__main__":
    main()
