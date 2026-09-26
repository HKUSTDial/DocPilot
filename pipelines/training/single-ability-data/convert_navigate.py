from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("data/DocVQA/single-ablity-data/navigate/DocSeeker/ALR_data")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "navigate_alr.openai.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DocSeeker ALR data to DocPilot Navigate SFT chat jsonl.")
    parser.add_argument("--input-dir", "--alr-dir", dest="input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-root", type=Path, default=None, help="Optional root for relative image filenames.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict-images", action="store_true", help="Fail if any image path cannot be resolved.")
    args = parser.parse_args()

    files = input_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No jsonl files found in {args.input_dir}")

    image_index = build_image_index(args.image_root or args.input_dir.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as out:
        for path in files:
            for row in read_jsonl(path):
                if args.limit is not None and written >= args.limit:
                    print(f"Wrote {written} samples to {args.output}")
                    return
                sample = convert_row(row, image_index, strict_images=args.strict_images)
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
    print(f"Wrote {written} samples to {args.output}")


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_image_index(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    index: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index.setdefault(path.name, str(path.resolve()))
    return index


def convert_row(row: dict[str, Any], image_index: dict[str, str], *, strict_images: bool) -> dict[str, Any]:
    rel_images = [str(x) for x in row.get("image", [])]
    images = [resolve_image(path, image_index, strict_images) for path in rel_images]
    question = human_question(row)
    assistant = assistant_content(row, rel_images)
    return {
        "messages": [
            {"role": "user", "content": user_content(question, len(images))},
            {"role": "assistant", "content": assistant},
        ],
        "images": images,
    }


def resolve_image(path: str, image_index: dict[str, str], strict: bool) -> str:
    p = Path(path)
    resolved = str(p.resolve()) if p.is_absolute() and p.exists() else image_index.get(p.name, path)
    if strict and not Path(resolved).exists():
        raise FileNotFoundError(f"Cannot resolve image: {path}")
    return resolved


def human_question(row: dict[str, Any]) -> str:
    for msg in row.get("conversations", []):
        if msg.get("from") == "human":
            return re.sub(r"\npage\s+\d+:\s*<image>", "", str(msg.get("value", "")), flags=re.I).strip()
    return ""


def user_content(question: str, image_count: int) -> str:
    pages = "\n".join(f"page {i}: <image>" for i in range(1, image_count + 1))
    return f"{question}\n\n{pages}".strip()


def assistant_content(row: dict[str, Any], rel_images: list[str]) -> str:
    gpt = next((str(m.get("value", "")) for m in row.get("conversations", []) if m.get("from") == "gpt"), "")
    analysis = extract_section(gpt, "Question Analysis")
    localization = extract_section(gpt, "Evidence Localization")
    pages = gt_pages(row.get("gt_page", []), rel_images)
    return (
        f"<Analyze>\n{analysis or 'Identify the question intent and the page-level evidence needed to answer it.'}\n</Analyze>\n"
        f"<Navigate>\n{localization or 'Locate the document pages that contain the required evidence.'}\n\n"
        f"candidate pages: \\boxed{{{', '.join(map(str, pages))}}}\n</Navigate>"
    )


def extract_section(text: str, title: str) -> str:
    pattern = rf"\\boxed{{{re.escape(title)}}}\s*(.*?)(?=\\boxed{{|</think>|<answer>)"
    match = re.search(pattern, text, flags=re.S)
    return clean_text(match.group(1)) if match else ""


def clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def gt_pages(gt_page: Any, rel_images: list[str]) -> list[int]:
    values = gt_page if isinstance(gt_page, list) else [gt_page]
    image_names = [Path(x).name for x in rel_images]
    pages: list[int] = []
    for value in values:
        if isinstance(value, int):
            page = value + 1
        else:
            name = Path(str(value)).name
            page = image_names.index(name) + 1 if name in image_names else infer_page_from_name(name)
        if page and page not in pages:
            pages.append(page)
    return pages


def infer_page_from_name(name: str) -> int:
    match = re.search(r"(?:_p|_)(\d+)(?:\.[^.]+)?$", name)
    return int(match.group(1)) + 1 if match else 0



# python ./pipelines/training/single-ablity-data/convert_navigate.py \
#     --input-dir data/DocVQA/single-ablity-data/navigate/DocSeeker/ALR_data/sft_dude_expand_data_correct.jsonl \
#     --image-root data/DocVQA/DUDE/rendered_pages_jpg \
#     --output ./pipelines/training/single-ablity-data/navigate_dude.jsonl \
#     --strict-images

# python pipelines/training/single-ablity-data/convert_navigate.py \
#     --input-dir data/DocVQA/single-ablity-data/navigate/DocSeeker/ALR_data/sft_mpdocvqa_expand_data_correct.jsonl \
#     --image-root data/DocVQA/MP-DocVQA/images \
#     --output ./pipelines/training/single-ablity-data/navigate_mpdocvqa.jsonl \
#     --strict-images
if __name__ == "__main__":
    main()
