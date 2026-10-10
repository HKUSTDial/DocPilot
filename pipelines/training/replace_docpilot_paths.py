#!/usr/bin/env python3
"""Replace DocPilot absolute paths inside text files."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Files to rewrite, e.g. *.jsonl *.json")
    parser.add_argument("--old-root", default=".")
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--backup", action="store_true", help="Write <file>.bak before replacing")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_root = args.old_root.rstrip("/")
    new_root = args.new_root.rstrip("/")

    for name in args.files:
        path = Path(name)
        text = path.read_text(encoding="utf-8")
        count = text.count(old_root)
        print(f"{path}: {count} replacements")
        if args.dry_run or count == 0:
            continue
        if args.backup:
            path.with_suffix(path.suffix + ".bak").write_text(text, encoding="utf-8")
        path.write_text(text.replace(old_root, new_root), encoding="utf-8")



# python ./pipelines/training/replace_docpilot_paths.py \
#     ./pipelines/training/single-ability-data/data/single_ability_docpilot_chat.jsonl \
#     --old-root data/DocVQA \
#     --new-root remote/DocVQA \
#     --dry-run
if __name__ == "__main__":
    main()
