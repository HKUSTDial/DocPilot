from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "data" / "single_ability_docpilot_chat.jsonl"
DEFAULT_OUTPUT = HERE / "data" / "single_ability_docpilot_action_only.jsonl"
ACTION_TAGS = ("Navigate", "Extract", "Focus", "Solve", "Answer")
FINAL_FIELDS = {
    "Navigate": ("candidate_pages", "candidate pages"),
    "Extract": ("candidate_nodes", "candidate nodes", "evidence_ids", "evidence ids"),
    "Focus": ("operation",),
    "Solve": ("code",),
    "Answer": ("answer",),
}


def tag_bodies(text: str, tag: str) -> list[str]:
    return [
        value.strip()
        for value in re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.I | re.S)
    ]


def normalize_final_field(action: str, body: str) -> str:
    canonical, *aliases = FINAL_FIELDS[action]
    labels = (canonical, *aliases)
    pattern = "|".join(re.escape(label).replace(r"\_", r"[ _]") for label in labels)
    normalized, count = re.subn(
        rf"(?im)^(\s*)(?:{pattern})(\s*:)",
        rf"\1{canonical}\2",
        body,
    )
    if count == 0:
        raise ValueError(f"<{action}> is missing final field {canonical!r}")
    return normalized.strip()


def convert_sample(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise ValueError("expected exactly one user message followed by one assistant message")

    user = str(messages[0].get("content", "")).strip()
    assistant = str(messages[1].get("content", "")).strip()
    analyses = tag_bodies(assistant, "Analyze")
    if len(analyses) != 1:
        raise ValueError(f"expected one <Analyze> block, found {len(analyses)}")

    actions = [(tag, bodies) for tag in ACTION_TAGS if (bodies := tag_bodies(assistant, tag))]
    if len(actions) != 1 or len(actions[0][1]) != 1:
        found = {tag: len(bodies) for tag, bodies in actions}
        raise ValueError(f"expected one action block, found {found}")

    action, bodies = actions[0]
    action_body = normalize_final_field(action, bodies[0])
    combined = "\n\n".join(part for part in (analyses[0], action_body) if part)

    converted = dict(row)
    converted["messages"] = [
        {
            "role": "user",
            "content": f"Expected action: <{action}>\n\n{user}",
        },
        {
            "role": "assistant",
            "content": f"<{action}>\n{combined}\n</{action}>",
        },
    ]
    return converted, action


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
    skip_invalid: bool = False,
    limit: int | None = None,
) -> Counter[str]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different files")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    counts: Counter[str] = Counter()
    try:
        with input_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, 1):
                if limit is not None and counts["converted"] >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    converted, action = convert_sample(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    if not skip_invalid:
                        raise ValueError(f"line {line_number}: {exc}") from exc
                    counts["skipped"] += 1
                    print(f"skip line {line_number}: {exc}", file=sys.stderr)
                    continue
                target.write(json.dumps(converted, ensure_ascii=False) + "\n")
                counts[action] += 1
                counts["converted"] += 1
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return counts


def self_test() -> None:
    fields = {
        "Navigate": r"candidate pages: \boxed{[2]}",
        "Extract": r"candidate nodes: \boxed{[node2]}",
        "Focus": r'operation: \boxed{{"tool":"zoom","evidence_id":"e1"}}',
        "Solve": r"code: \boxed{print(1)}",
        "Answer": r"answer: \boxed{1}",
    }
    for action, result in fields.items():
        row = {
            "messages": [
                {"role": "user", "content": "Question\n<image>"},
                {
                    "role": "assistant",
                    "content": f"<Analyze>reasoning</Analyze>\n<{action}>{result}</{action}>",
                },
            ],
            "images": ["page.png"],
        }
        converted, actual = convert_sample(row)
        assert actual == action
        assert converted["messages"][0]["content"].startswith(f"Expected action: <{action}>")
        assert converted["messages"][1]["content"].count(f"<{action}>") == 1
        assert "<Analyze>" not in converted["messages"][1]["content"]
        assert "reasoning" in converted["messages"][1]["content"]
        assert converted["images"] == row["images"]
    print("self-test passed: Navigate, Extract, Focus, Solve, Answer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DocPilot single-ability data to action-only two-stage executor data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    counts = convert_file(
        args.input,
        args.output,
        overwrite=args.overwrite,
        skip_invalid=args.skip_invalid,
        limit=args.limit,
    )
    print(json.dumps(dict(counts), ensure_ascii=False, sort_keys=True))
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
