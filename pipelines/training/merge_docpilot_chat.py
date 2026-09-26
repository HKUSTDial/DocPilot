from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge DocPilot chat JSON/JSONL files and their progress maps.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional merged image root; samples are written under data-dir/<index>.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the planned merge without writing files or copying images.")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--prefer", choices=("first", "last"), default="last")
    parser.add_argument("--dedupe", choices=("source", "none"), default="none")
    parser.add_argument(
        "--source-progress",
        type=Path,
        nargs="*",
        default=None,
        help="Optional original progress files, one per input, used to keep source ids after separate refinement.",
    )
    args = parser.parse_args()
    source_progress = args.source_progress or []
    if source_progress and len(source_progress) != len(args.input):
        raise SystemExit("--source-progress must be omitted or have the same count as --input")

    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    duplicates = 0
    input_counts = []
    for i, path in enumerate(args.input):
        progress_path = source_progress[i] if source_progress else None
        count = 0
        for entry in load_entries(path, source_progress=progress_path):
            count += 1
            key = entry["source"] if args.dedupe == "source" else f"{entry['source']}#{len(merged)}"
            if key in positions:
                duplicates += 1
                if args.prefer == "last":
                    merged[positions[key]] = entry
                continue
            positions[key] = len(merged)
            merged.append(entry)
        input_counts.append(count)

    data_dir = args.data_dir.resolve() if args.data_dir is not None else None
    if args.dry_run:
        print_dry_run(args.output, merged, input_counts, duplicates=duplicates, data_dir=data_dir)
        return
    write_output(args.output, merged, pretty=args.pretty, data_dir=data_dir)
    print(f"merged={len(merged)} duplicates={duplicates} output={args.output}")


def load_entries(path: Path, *, source_progress: Path | None = None) -> list[dict[str, Any]]:
    if source_progress is not None:
        samples = load_samples(path)
        sources = load_progress_sources(source_progress)
        if len(samples) != len(sources):
            raise ValueError(f"{path} has {len(samples)} samples but {source_progress} has {len(sources)} sources")
        return [{"source": source, "sample": sample} for source, sample in zip(sources, samples)]

    progress_path = path.with_suffix(path.suffix + ".progress.json")
    if progress_path.exists():
        entries = load_progress_entries(progress_path)
        if entries:
            return [
                {"source": str(entry.get("source") or f"{path}:{i}"), "sample": entry["sample"]}
                for i, entry in enumerate(entries)
                if isinstance(entry, dict) and isinstance(entry.get("sample"), dict)
            ]
    return [
        {"source": f"{path}:{i}", "sample": sample}
        for i, sample in enumerate(load_samples(path))
    ]


def load_progress_sources(path: Path) -> list[str]:
    entries = load_progress_entries(path)
    if not entries:
        raise ValueError(f"No progress entries found in {path}")
    return [str(entry.get("source") or f"{path}:{i}") for i, entry in enumerate(entries)]


def load_progress_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    return [
        entry
        for entry in sorted(entries, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        if isinstance(entry, dict)
    ]


def load_samples(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        samples: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                samples.append(value)
            elif isinstance(value, list):
                samples.extend(item for item in value if isinstance(item, dict))
        return samples
    value = json.loads(text)
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        value = value["data"]
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list or JSONL in {path}")
    return [item for item in value if isinstance(item, dict)]


def write_output(output_path: Path, entries: list[dict[str, Any]], *, pretty: bool, data_dir: Path | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    entries = [
        {**entry, "sample": renumber_sample(entry["sample"], index, data_dir=data_dir)}
        for index, entry in enumerate(entries)
    ]
    progress_entries = [dict(entry, index=i) for i, entry in enumerate(entries)]
    samples = [entry["sample"] for entry in entries]

    tmp_progress = progress_path.with_suffix(progress_path.suffix + f".tmp.{os.getpid()}")
    tmp_progress.write_text(json.dumps({"entries": progress_entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_progress.replace(progress_path)

    tmp_output = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with tmp_output.open("w", encoding="utf-8") as f:
        if output_path.suffix.lower() == ".jsonl":
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            json.dump(samples, f, ensure_ascii=False, indent=2 if pretty else None)
            f.write("\n")
    tmp_output.replace(output_path)


def print_dry_run(
    output_path: Path,
    entries: list[dict[str, Any]],
    input_counts: list[int],
    *,
    duplicates: int,
    data_dir: Path | None,
) -> None:
    missing_images = []
    placeholder_mismatches = []
    image_refs = 0
    for index, entry in enumerate(entries):
        sample = entry["sample"]
        images = sample.get("images") if isinstance(sample.get("images"), list) else sample.get("image") if isinstance(sample.get("image"), list) else []
        placeholders = sum(
            str(message.get("content") or "").count("<image>")
            for message in sample.get("messages", [])
            if isinstance(message, dict)
        )
        image_refs += len(images)
        if placeholders != len(images):
            placeholder_mismatches.append((index, placeholders, len(images)))
        for image in images:
            if isinstance(image, str) and not Path(image).exists():
                missing_images.append((index, image))

    cursor = 0
    print("dry_run: true")
    print(f"output: {output_path}")
    print(f"progress: {output_path.with_suffix(output_path.suffix + '.progress.json')}")
    print(f"merged: {len(entries)}")
    print(f"duplicates: {duplicates}")
    print(f"image_refs: {image_refs}")
    for input_index, count in enumerate(input_counts):
        end = cursor + count - 1
        print(f"input[{input_index}]: count={count} output_range={cursor}-{end if count else cursor - 1}")
        cursor += count
    if data_dir is not None:
        append_start = input_counts[0] if input_counts else 0
        existing = [index for index in range(append_start, len(entries)) if (data_dir / str(index)).exists()]
        print(f"data_dir: {data_dir}")
        print(f"append_range: {append_start}-{len(entries) - 1 if len(entries) > append_start else append_start - 1}")
        print(f"append_target_dirs_existing: {len(existing)}")
        if existing:
            print(f"append_target_dirs_existing_first20: {existing[:20]}")
    print(f"missing_images: {len(missing_images)}")
    if missing_images:
        print(f"missing_images_first5: {missing_images[:5]}")
    print(f"placeholder_mismatches: {len(placeholder_mismatches)}")
    if placeholder_mismatches:
        print(f"placeholder_mismatches_first5: {placeholder_mismatches[:5]}")
    print(f"dry_run_ok: {not missing_images and not placeholder_mismatches}")


def renumber_sample(sample: dict[str, Any], index: int, *, data_dir: Path | None) -> dict[str, Any]:
    result = json.loads(json.dumps(sample, ensure_ascii=False))
    for message_index, message in enumerate(result.get("messages") or []):
        if isinstance(message, dict):
            message["id"] = f"sample_{index}-{message_index}"
    if data_dir is not None:
        for key in ("images", "image"):
            if isinstance(result.get(key), list):
                result[key] = remap_images(result[key], data_dir / str(index))
    return result


def remap_images(images: list[Any], dst_dir: Path) -> list[Any]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    used: dict[str, int] = {}
    copied: dict[str, str] = {}
    remapped = []
    for image in images:
        if not isinstance(image, str):
            remapped.append(image)
            continue
        src = Path(image)
        src_key = str(src)
        if src_key in copied:
            remapped.append(copied[src_key])
            continue
        name = unique_name(src.name, used)
        dst = dst_dir / name
        try:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing image referenced by sample: {src}") from None
        copied[src_key] = str(dst)
        remapped.append(copied[src_key])
    return remapped


def unique_name(name: str, used: dict[str, int]) -> str:
    count = used.get(name, 0)
    used[name] = count + 1
    if count == 0:
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    return f"{stem}_{count}{suffix}"


# python pipelines/training/merge_docpilot_chat.py \
#     --input \
#       pipelines/training/multi_ability_docpilot.chat.refined.jsonl \
#       pipelines/training/multi_ability_docpilot.chat2.refined.jsonl \
#     --source-progress \
#       pipelines/training/multi_ability_docpilot.chat.jsonl.progress.json \
#       pipelines/training/multi_ability_docpilot.chat2.jsonl.progress.json \
#     --output pipelines/training/multi_ability_docpilot.chat.refined.merged.jsonl \
#     --data-dir pipelines/training/multi-ability-data \
#     --dry-run
if __name__ == "__main__":
    main()
