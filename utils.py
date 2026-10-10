import re
import ast
import json
import random
import base64
from pathlib import Path
from typing import Any

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


RETRY_MAX_SIDE = 1024
RETRY_JPEG_QUALITY = 45


def token_length(text):
    return len(encoding.encode(text, disallowed_special=()))


def is_request_too_large_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 413:
        return True
    message = str(error).lower()
    return "413" in message or "request entity too large" in message or "payload too large" in message


def prefer_original_image_refs(content: Any) -> Any:
    if isinstance(content, dict):
        original = content.get("image_path") or content.get("img_path") or content.get("page_image_path")
        compact: dict[str, Any] = {}
        for key, value in content.items():
            lowered = str(key).lower()
            if original and lowered in {"zoom_path", "crop_path", "screenshot_path"}:
                continue
            compact[key] = prefer_original_image_refs(value)
        return compact
    if isinstance(content, list):
        return [prefer_original_image_refs(item) for item in content]
    if isinstance(content, tuple):
        return [prefer_original_image_refs(item) for item in content]
    return content


def compress_images_for_retry(images: list[str]) -> list[str]:
    compressed: list[str] = []
    for image in images:
        compressed.append(encode_image_for_retry(image) or image)
    return compressed


def encode_image_for_retry(path: str) -> str | None:
    if path.startswith(("http://", "https://", "data:image/")):
        return None
    src = Path(path).expanduser()
    try:
        if not src.exists() or not src.is_file():
            return None
        import cv2

        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > RETRY_MAX_SIDE:
            scale = RETRY_MAX_SIDE / float(longest)
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, RETRY_JPEG_QUALITY])
        if not ok:
            return None
        payload = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
    except Exception:
        return None

