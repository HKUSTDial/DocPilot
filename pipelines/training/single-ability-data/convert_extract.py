from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/hotpot/dataset/hotpot_dev_distractor_v1.json")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "extract_hotpotqa.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HotpotQA distractor dev data to Extract SFT JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {args.input}")
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    samples = [convert_row(row) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} samples to {args.output}")


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    contexts = row.get("context")
    supporting_facts = row.get("supporting_facts")
    if not isinstance(contexts, list) or not isinstance(supporting_facts, list):
        raise ValueError(f"Invalid HotpotQA row: {row.get('_id', '<unknown>')}")

    title_to_node: dict[str, str] = {}
    candidates: list[str] = []
    for index, context in enumerate(contexts, start=1):
        if not isinstance(context, list) or len(context) != 2 or not isinstance(context[1], list):
            raise ValueError(f"Invalid context in row: {row.get('_id', '<unknown>')}")
        title = str(context[0])
        node_id = f"node{index}"
        title_to_node[title] = node_id
        paragraph = re.sub(r"\s+", " ", "".join(map(str, context[1]))).strip()
        candidates.append(f"{node_id}: [{title}] {paragraph}")

    gold_titles = list(dict.fromkeys(str(fact[0]) for fact in supporting_facts))
    missing = [title for title in gold_titles if title not in title_to_node]
    if missing:
        raise ValueError(f"Gold titles missing from context for {row.get('_id', '<unknown>')}: {missing}")
    selected_ids = [title_to_node[title] for title in gold_titles]

    user = "\n".join(
        [
            "Search query:",
            str(row.get("question") or "").strip(),
            "",
            "Candidate paragraphs:",
            *candidates,
            "",
            "Select only the node ids that contain evidence needed to answer the query.",
        ]
    )
    assistant = "<Extract>\n" + f"candidate nodes: \\boxed{{[{', '.join(selected_ids)}]}}" + "\n</Extract>"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


if __name__ == "__main__":
    main()
