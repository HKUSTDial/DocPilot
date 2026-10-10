#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "docpilot-4b"
DEFAULT_SAMPLE = "pipelines/training/multi_ability_docpilot.chat.refined.jsonl"


def main() -> None:
    args = parse_args()
    if args.test:
        test(args)
        return

    from openai import OpenAI

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    output_path = Path(args.output)
    if args.prompt:
        text = run_infer(client, args, load_messages(args))
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        text = "\n\n".join(replay_sample(client, args, output_path))
    print(text)
    print(f"\nSaved to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DocPilot Qwen3.5 vLLM inference.")
    parser.add_argument("--base-url", default=os.getenv("DOCPILOT_VLLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("DOCPILOT_VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("DOCPILOT_VLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", default=str(Path(__file__).with_suffix(".txt")))
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen thinking mode in the chat template.")
    parser.add_argument("--test", action="store_true", help="Run one text chat and one image QA example.")

    parser.add_argument("--sample-jsonl", default=DEFAULT_SAMPLE)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--keep-last-assistant", action="store_true")

    parser.add_argument("--prompt", help="Manual prompt. If set, sample-jsonl is ignored.")
    parser.add_argument("--image", action="append", default=[], help="Manual image path or URL; repeatable.")
    return parser.parse_args()


def test(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    image = args.image[0] if args.image else first_sample_image(args.sample_jsonl, args.sample_index)
    examples = [
        ("text", [{"role": "user", "content": "你是什么模型？请用一句话回答。"}]),
        ("image", [{"role": "user", "content": content_with_images("这张图片里主要有什么内容？", [image])}]),
    ]
    outputs = []
    for name, messages in examples:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=extra_body(args),
        )
        text = response.choices[0].message.content or ""
        outputs.append(f"## {name}\n{text}")
        print(f"\n## {name}\n{text}")
    Path(args.output).write_text("\n\n".join(outputs) + "\n", encoding="utf-8")
    print(f"\nSaved to {args.output}")


def load_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prompt:
        return [{"role": "user", "content": content_with_images(args.prompt, args.image)}]

    sample = read_jsonl_item(Path(args.sample_jsonl), args.sample_index)
    raw_messages = list(sample["messages"])
    if not args.keep_last_assistant:
        raw_messages = [next(message for message in raw_messages if message.get("role") == "user")]
    return render_messages(raw_messages, sample.get("images") or [])


def replay_sample(client: Any, args: argparse.Namespace, output_path: Path) -> list[str]:
    sample = read_jsonl_item(Path(args.sample_jsonl), args.sample_index)
    raw_messages = list(sample["messages"])
    outputs: list[str] = []
    output_path.write_text("", encoding="utf-8")
    for i in range(len(raw_messages) - 1):
        expected = raw_messages[i + 1]
        if expected.get("role") != "assistant":
            continue
        messages = render_messages(raw_messages[: i + 1], sample.get("images") or [])
        prediction = run_infer(client, args, messages)
        block = (
            f"## step {i} -> {i + 1}\n"
            f"expected role: {expected.get('role')}\n\n"
            f"{prediction}"
        )
        outputs.append(block)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(block + "\n\n")
            f.flush()
        print(block, flush=True)
    return outputs


def run_infer(client: Any, args: argparse.Namespace, messages: list[dict[str, Any]]) -> str:
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body=extra_body(args),
    )
    return response.choices[0].message.content or ""



def render_messages(raw_messages: list[dict[str, Any]], images: list[str]) -> list[dict[str, Any]]:
    images = [str(Path(image)) for image in images]
    cursor = 0
    messages: list[dict[str, Any]] = []
    for message in raw_messages:
        role = message["role"]
        content = str(message.get("content") or "")
        if role == "user" and "<image>" in content:
            rendered, cursor = replace_image_placeholders(content, images, cursor)
        else:
            rendered = content
        messages.append({"role": role, "content": rendered})
    return messages


def extra_body(args: argparse.Namespace) -> dict[str, Any]:
    return {"chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)}}


def read_jsonl_item(path: Path, index: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        data = json.loads(text)
        try:
            item = data[index]
        except IndexError:
            raise IndexError(f"{path} has no item at index {index}") from None
        if isinstance(item, dict):
            return item
        raise TypeError(f"{path} item {index} is not an object")
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no == index:
                item = json.loads(line)
                if isinstance(item, dict):
                    return item
                raise TypeError(f"{path} item {index} is not an object")
    raise IndexError(f"{path} has no item at index {index}")


def first_sample_image(sample_jsonl: str, sample_index: int) -> str:
    images = read_jsonl_item(Path(sample_jsonl), sample_index).get("images") or []
    if not images:
        raise ValueError("No sample image found. Pass --image for image QA test.")
    return str(Path(images[0]))


def replace_image_placeholders(text: str, images: list[str], cursor: int) -> tuple[list[dict[str, Any]], int]:
    parts: list[dict[str, Any]] = []
    chunks = text.split("<image>")
    append_text(parts, chunks[0])
    for chunk in chunks[1:]:
        if cursor >= len(images):
            raise ValueError("Not enough images for <image> placeholders.")
        parts.append({"type": "image_url", "image_url": {"url": image_url(images[cursor])}})
        cursor += 1
        append_text(parts, chunk)
    return parts, cursor


def content_with_images(text: str, images: list[str]) -> str | list[dict[str, Any]]:
    if not images:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": image_url(image)}})
    return parts


def append_text(parts: list[dict[str, Any]], text: str) -> None:
    if text:
        parts.append({"type": "text", "text": text})


def image_url(value: str) -> str:
    if value.startswith(("http://", "https://", "data:image/")):
        return value
    path = Path(value).expanduser()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


# python app/llm/docpilot_vllm_infer.py \
#     --model docpilot-4b \
#     --prompt "你是什么模型？"

# python app/llm/docpilot_vllm_infer.py \
#     --base-url http://localhost:8080/v1 \
#     --model docpilot-4b_agenticrl\
#     --prompt "请描述这张文档图片的主要内容，并读出可见的标题或表格信息。" \
#     --image pipelines/training/multi-ability-data/1382/page_34.png

# python app/llm/docpilot_vllm_infer.py --sample-index 0 --model docpilot-4b_onestage --base-url http://localhost:8088/v1
if __name__ == "__main__":
    main()
