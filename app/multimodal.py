from __future__ import annotations

import base64
import io
import math
import re
from pathlib import Path
from typing import Any


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_IMAGE_HINT_KEYS = {
    "image",
    "image_path",
    "img_path",
    "screenshot_path",
    "page_image_path",
    "crop_path",
    "zoom_path",
    "image_url",
}


def collect_image_inputs(payload: Any, *, max_images: int = 12) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        if value in seen or len(collected) >= max_images:
            return
        seen.add(value)
        collected.append(value)

    def _walk(node: Any, parent_key: str | None = None) -> None:
        if len(collected) >= max_images:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, str(key).lower())
            return
        if isinstance(node, list):
            for item in node:
                _walk(item, parent_key)
            return
        if not isinstance(node, str):
            return
        candidate = _normalize_image_candidate(node, parent_key)
        if candidate is not None:
            _push(candidate)

    _walk(payload)
    return collected


def _normalize_image_candidate(value: str, parent_key: str | None) -> str | None:
    text = value.strip()
    if not text:
        return None
    key = parent_key or ""
    if text.startswith("data:image/"):
        return text
    if text.startswith(("http://", "https://")):
        if _is_likely_image_url(text) or _is_image_key(key):
            return text
        return None
    if not _is_image_key(key) and not _has_image_extension(text):
        return None
    direct_candidate = text if _looks_like_path_string(text) else None
    if direct_candidate is not None:
        resolved = _resolve_existing_image_path(direct_candidate)
        if resolved is not None:
            return resolved

    for embedded in _extract_embedded_image_candidates(text):
        resolved = _resolve_existing_image_path(embedded)
        if resolved is not None:
            return resolved
    return None


def _has_image_extension(value: str) -> bool:
    return Path(value).suffix.lower() in _IMAGE_EXTENSIONS


def _is_likely_image_url(url: str) -> bool:
    lowered = url.lower()
    return bool(re.search(r"\.(png|jpe?g|webp|gif|bmp|tiff)(?:$|[?#])", lowered))


def _is_image_key(key: str) -> bool:
    if key in _IMAGE_HINT_KEYS:
        return True
    return "image" in key or "screenshot" in key or "crop" in key or "zoom" in key


def _looks_like_path_string(value: str) -> bool:
    if len(value) > 512 or "\n" in value or "\r" in value:
        return False
    if any(ch in value for ch in ("{", "}", "<", ">")):
        return False
    if value.count(" ") > 6:
        return False
    return _has_image_extension(value)


def _extract_embedded_image_candidates(value: str) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r"(?P<path>(?:/[\w.\-]+)+\.(?:png|jpe?g|webp|gif|bmp|tiff))",
        r"(?P<path>(?:[\w.\-]+/)+[\w.\-]+\.(?:png|jpe?g|webp|gif|bmp|tiff))",
        r"(?P<path>[\w.\-]+\.(?:png|jpe?g|webp|gif|bmp|tiff))",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            path = match.group("path").strip().strip("'\"")
            if path not in candidates:
                candidates.append(path)
    return candidates


def _resolve_existing_image_path(value: str) -> str | None:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    try:
        exists = path.exists()
    except OSError:
        return None
    if not exists or not _has_image_extension(path.name):
        return None
    return str(path)


def estimate_image_tokens(image_ref: str, *, model: str = "") -> int:
    """Estimate OpenAI-style vision tokens for one image (high detail by default)."""
    size = _read_image_size(image_ref)
    if size is None:
        return _fallback_image_tokens(model)
    width, height = size
    if _uses_patch_tokenization(model):
        return _estimate_patch_image_tokens(width, height)
    return _estimate_tile_image_tokens(width, height)


def estimate_image_list_tokens(images: list[str], *, model: str = "") -> int:
    return sum(estimate_image_tokens(image, model=model) for image in images)


def _estimate_tile_image_tokens(width: int, height: int) -> int:
    """GPT-4o / GPT-4.1 high-detail tiling: 85 base + 170 per 512x512 tile."""
    if width <= 0 or height <= 0:
        return 85
    if width > 2048 or height > 2048:
        if width >= height:
            height = max(1, round(height * 2048 / width))
            width = 2048
        else:
            width = max(1, round(width * 2048 / height))
            height = 2048
    if width > 768 and height > 768:
        if width <= height:
            height = max(1, round(height * 768 / width))
            width = 768
        else:
            width = max(1, round(width * 768 / height))
            height = 768
    tiles_w = math.ceil(width / 512)
    tiles_h = math.ceil(height / 512)
    return 85 + 170 * tiles_w * tiles_h


def _estimate_patch_image_tokens(width: int, height: int, *, patch_size: int = 32, max_patches: int = 1536) -> int:
    """GPT-4.1-mini / nano style: one token per 32x32 patch, capped at 1536."""
    if width <= 0 or height <= 0:
        return 1
    patches_w = math.ceil(width / patch_size)
    patches_h = math.ceil(height / patch_size)
    patch_count = patches_w * patches_h
    if patch_count <= max_patches:
        return patch_count
    scale = math.sqrt(max_patches / patch_count)
    scaled_w = max(1, round(width * scale))
    scaled_h = max(1, round(height * scale))
    return math.ceil(scaled_w / patch_size) * math.ceil(scaled_h / patch_size)


def _uses_patch_tokenization(model: str) -> bool:
    lowered = model.lower()
    return bool(re.search(r"(^|[-_.])(mini|nano)($|[-_.])", lowered))


def _fallback_image_tokens(model: str) -> int:
    return 765 if not _uses_patch_tokenization(model) else 512


def _read_image_size(image_ref: str) -> tuple[int, int] | None:
    if image_ref.startswith("data:image/"):
        return _read_data_url_size(image_ref)
    path = Path(image_ref).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        return None
    return _read_file_image_size(path)


def _read_data_url_size(data_url: str) -> tuple[int, int] | None:
    match = re.match(r"data:image/[^;]+;base64,(.+)", data_url, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        raw = base64.b64decode(match.group(1), validate=False)
    except Exception:
        return None
    return _read_bytes_image_size(raw)


def _read_file_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        pass
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            return width, height
    except Exception:
        pass
    try:
        return _read_bytes_image_size(path.read_bytes())
    except Exception:
        return None


def _read_bytes_image_size(raw: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            return image.size
    except Exception:
        return None
