from __future__ import annotations

import ast
import base64
import hashlib
import json
import mimetypes
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

from .run_logger import RunLogger
from .schema import Message


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    account_access_token: str = ""
    account_user_id: str = ""
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout: int = 120
    max_attempts: int = 3
    seed: int | None = None
    reasoning: str = "auto"
    reasoning_effort: str = ""
    response_format: str = ""
    judge_model: str = ""
    judge_base_url: str = ""
    judge_api_key: str = ""
    judge_temperature: float = 0.0
    judge_max_tokens: int = 800
    image_type: str = "png"
    protocol: str = "json"


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        if not config.enabled:
            raise ValueError("LLM is disabled. Set llm.enabled=true.")
        if not config.model:
            raise ValueError("Missing llm.model in config.")
        if not config.base_url:
            raise ValueError("Missing llm.base_url in config.")
        if not config.api_key:
            raise ValueError("Missing llm.api_key in config.")
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=float(config.timeout),
        )
        self.judge_client = None
        if config.judge_base_url or config.judge_api_key:
            self.judge_client = OpenAI(
                api_key=config.judge_api_key or config.api_key,
                base_url=config.judge_base_url or config.base_url,
                timeout=float(config.timeout),
            )
        self.logger: RunLogger | None = None
        self.current_step: int | None = None
        self.current_call_source: str | None = None

    def set_logger(self, logger: RunLogger | None) -> None:
        self.logger = logger

    def set_step(self, step: int | None) -> None:
        self.current_step = step

    def set_call_source(self, call_source: str | None) -> None:
        self.current_call_source = call_source

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: list[str] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        require_object: bool = True,
    ) -> Any:
        resolved_model = model or self.config.model
        resolved_temperature = self.config.temperature if temperature is None else temperature
        resolved_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _content_with_images(user_prompt, images, image_type=self.config.image_type)},
            ],
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        kwargs.update(_response_format_kwargs(self.config))
        kwargs.update(_model_extra_kwargs(self.config, resolved_model))
        return self._request_json(
            call_kind="generate_json",
            kwargs=kwargs,
            request={
                "model": resolved_model,
                "temperature": resolved_temperature,
                "max_tokens": resolved_max_tokens,
                "seed": self.config.seed,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "images": images or [],
            },
            require_object=require_object,
        )

    def generate_json_from_messages(
        self,
        messages: list[dict[str, Any]] | list[Message],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        resolved_model = model or self.config.model
        resolved_temperature = self.config.temperature if temperature is None else temperature
        resolved_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        formatted_messages: list[dict[str, Any]] = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            if isinstance(message, Message):
                formatted_messages.append(_normalize_openai_message(message.to_dict(), image_type=self.config.image_type))
            else:
                formatted_messages.append(_normalize_openai_message(dict(message), image_type=self.config.image_type))
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": formatted_messages,
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        kwargs.update(_response_format_kwargs(self.config))
        kwargs.update(_model_extra_kwargs(self.config, resolved_model))
        return self._request_json(
            call_kind="generate_json_from_messages",
            kwargs=kwargs,
            request={
                "model": resolved_model,
                "temperature": resolved_temperature,
                "max_tokens": resolved_max_tokens,
                "seed": self.config.seed,
                "system_prompt": system_prompt,
                "messages": formatted_messages,
            },
        )

    def generate_text_from_messages(
        self,
        messages: list[dict[str, Any]] | list[Message],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        resolved_model = model or self.config.model
        resolved_temperature = self.config.temperature if temperature is None else temperature
        resolved_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        formatted_messages: list[dict[str, Any]] = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            if isinstance(message, Message):
                formatted_messages.append(_normalize_openai_message(message.to_dict(), image_type=self.config.image_type))
            else:
                formatted_messages.append(_normalize_openai_message(dict(message), image_type=self.config.image_type))
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": formatted_messages,
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        if stop:
            kwargs["stop"] = stop
        kwargs.update(_model_extra_kwargs(self.config, resolved_model))
        return self._request_text(
            call_kind="generate_text_from_messages",
            kwargs=kwargs,
            request={
                "model": resolved_model,
                "temperature": resolved_temperature,
                "max_tokens": resolved_max_tokens,
                "stop": stop,
                "seed": self.config.seed,
                "system_prompt": system_prompt,
                "messages": formatted_messages,
            },
        )

    def ask_tool(
        self,
        messages: list[dict[str, Any]] | list[Message],
        *,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        tool_choice: str | dict[str, Any] = "required",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        resolved_model = model or self.config.model
        resolved_temperature = self.config.temperature if temperature is None else temperature
        resolved_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        formatted_messages: list[dict[str, Any]] = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            if isinstance(message, Message):
                formatted_messages.append(_normalize_openai_message(message.to_dict(), image_type=self.config.image_type))
            else:
                formatted_messages.append(_normalize_openai_message(dict(message), image_type=self.config.image_type))
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": formatted_messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        kwargs.update(_response_format_kwargs(self.config))
        kwargs.update(_model_extra_kwargs(self.config, resolved_model))
        return self._request_tool_payload(
            kwargs=kwargs,
            request={
                "model": resolved_model,
                "temperature": resolved_temperature,
                "max_tokens": resolved_max_tokens,
                "seed": self.config.seed,
                "system_prompt": system_prompt,
                "messages": formatted_messages,
                "tools": tools,
                "tool_choice": tool_choice,
            },
        )

    def judge_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: list[str] | None = None,
        use_reasoning: bool = True,
    ) -> dict[str, Any]:
        client = self.judge_client or self.client
        return self._generate_json_with_client(
            client,
            system_prompt,
            user_prompt,
            images=images,
            model=self.config.judge_model or self.config.model,
            temperature=self.config.judge_temperature,
            max_tokens=self.config.judge_max_tokens,
            use_reasoning=use_reasoning,
        )

    def _generate_json_with_client(
        self,
        client: OpenAI,
        system_prompt: str,
        user_prompt: str,
        *,
        images: list[str] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_reasoning: bool = True,
    ) -> dict[str, Any]:
        resolved_model = model or self.config.model
        resolved_temperature = self.config.temperature if temperature is None else temperature
        resolved_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _content_with_images(user_prompt, images, image_type=self.config.image_type)},
            ],
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        kwargs.update(_response_format_kwargs(self.config))
        if use_reasoning:
            kwargs.update(_model_extra_kwargs(self.config, resolved_model))
        return self._request_json(
            client=client,
            call_kind="generate_json",
            kwargs=kwargs,
            request={
                "model": resolved_model,
                "temperature": resolved_temperature,
                "max_tokens": resolved_max_tokens,
                "seed": self.config.seed,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "images": images or [],
            },
        )

    def _request_json(
        self,
        *,
        client: OpenAI | None = None,
        call_kind: str,
        kwargs: dict[str, Any],
        request: dict[str, Any],
        require_object: bool = True,
    ) -> Any:
        last_error: Exception | None = None
        attempts = max(1, int(self.config.max_attempts or 1))
        include_reasoning = _reasoning_enabled(self.config, str(kwargs.get("model") or ""))
        active_client = client or self.client
        for attempt in range(1, attempts + 1):
            response = None
            content = ""
            reasoning_content = ""
            try:
                response = active_client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content, reasoning_content = _message_content_and_reasoning(message, include_reasoning=include_reasoning)
                if not content.strip():
                    raise ValueError("Empty LLM response.")
                parsed = _extract_json_object(content) if require_object else _extract_json_value_or_text(content)
                if self.logger is not None:
                    self.logger.log_llm_call(
                        step=self.current_step,
                        call_kind=call_kind,
                        call_source=self.current_call_source,
                        raw_response=_raw_response_with_reasoning(content, reasoning_content),
                        parsed_response=parsed,
                    )
                return parsed
            except Exception as exc:
                last_error = exc
                self._log_llm_failure(
                    call_kind=call_kind,
                    attempt=attempt,
                    request=request,
                    response=response,
                    raw_response=_raw_response_with_reasoning(content, reasoning_content),
                    error=exc,
                )
                if attempt < attempts and _is_retryable_llm_error(exc):
                    time.sleep(_retry_delay_seconds(attempt))
                    continue
                raise
        raise last_error if last_error is not None else RuntimeError("LLM request failed.")

    def _request_text(
        self,
        *,
        call_kind: str,
        kwargs: dict[str, Any],
        request: dict[str, Any],
    ) -> str:
        last_error: Exception | None = None
        attempts = max(1, int(self.config.max_attempts or 1))
        include_reasoning = _reasoning_enabled(self.config, str(kwargs.get("model") or ""))
        for attempt in range(1, attempts + 1):
            response = None
            content = ""
            reasoning_content = ""
            try:
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content, reasoning_content = _message_content_and_reasoning(
                    message,
                    include_reasoning=include_reasoning,
                )
                if not content.strip():
                    raise ValueError("Empty LLM response.")
                if self.logger is not None:
                    self.logger.log_llm_call(
                        step=self.current_step,
                        call_kind=call_kind,
                        call_source=self.current_call_source,
                        raw_response=_raw_response_with_reasoning(content, reasoning_content),
                        parsed_response=None,
                    )
                return content.strip()
            except Exception as exc:
                last_error = exc
                self._log_llm_failure(
                    call_kind=call_kind,
                    attempt=attempt,
                    request=request,
                    response=response,
                    raw_response=_raw_response_with_reasoning(content, reasoning_content),
                    error=exc,
                )
                if attempt < attempts and _is_retryable_llm_error(exc):
                    time.sleep(_retry_delay_seconds(attempt))
                    continue
                raise
        raise last_error if last_error is not None else RuntimeError("LLM request failed.")

    def _request_tool_payload(self, *, kwargs: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        attempts = max(1, int(self.config.max_attempts or 1))
        include_reasoning = _reasoning_enabled(self.config, str(kwargs.get("model") or ""))
        for attempt in range(1, attempts + 1):
            response = None
            payload: dict[str, Any] | None = None
            raw_response = ""
            try:
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content, reasoning_content = _message_content_and_reasoning(message, include_reasoning=include_reasoning)
                tool_calls = []
                if message.tool_calls:
                    for call in message.tool_calls:
                        tool_calls.append(
                            {
                                "id": call.id,
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            }
                        )
                payload = {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls,
                }
                raw_response = _raw_response_with_reasoning(json.dumps(payload, ensure_ascii=False), reasoning_content)
                if self.logger is not None:
                    self.logger.log_llm_call(
                        step=self.current_step,
                        call_kind="ask_tool",
                        call_source=self.current_call_source,
                        raw_response=raw_response,
                        parsed_response=payload,
                    )
                return payload
            except Exception as exc:
                last_error = exc
                self._log_llm_failure(
                    call_kind="ask_tool",
                    attempt=attempt,
                    request=request,
                    response=response,
                    raw_response=raw_response,
                    error=exc,
                )
                if attempt < attempts and _is_retryable_llm_error(exc):
                    time.sleep(_retry_delay_seconds(attempt))
                    continue
                raise
        raise last_error if last_error is not None else RuntimeError("LLM request failed.")

    def _log_llm_failure(
        self,
        *,
        call_kind: str,
        attempt: int,
        request: dict[str, Any],
        response: Any,
        raw_response: str,
        error: Exception,
    ) -> None:
        if self.logger is None:
            return
        response_meta = _summarize_response(response)
        self.logger.info(
            "LLM failure:\n"
            f"{json.dumps({'call_kind': call_kind, 'call_source': self.current_call_source, 'attempt': attempt, 'request': self._sanitize(request), 'response_meta': response_meta, 'raw_response': raw_response, 'error': repr(error)}, ensure_ascii=False, indent=2)}"
        )

    def _sanitize(self, value: Any, depth: int = 0) -> Any:
        if depth >= 5:
            return "<max-depth-reached>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= 2000 else value[:2000] + "...<truncated>"
        if isinstance(value, list):
            items = [self._sanitize(item, depth + 1) for item in value[:20]]
            if len(value) > 20:
                items.append(f"<truncated {len(value) - 20} more items>")
            return items
        if isinstance(value, tuple):
            return self._sanitize(list(value), depth + 1)
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            items = list(value.items())
            for key, item in items[:50]:
                sanitized[str(key)] = self._sanitize(item, depth + 1)
            if len(items) > 50:
                sanitized["_truncated_keys"] = len(items) - 50
            return sanitized
        if hasattr(value, "to_dict"):
            try:
                return self._sanitize(value.to_dict(), depth + 1)
            except Exception:
                pass
        return repr(value)


def _model_extra_kwargs(config: LLMConfig, model: str) -> dict[str, Any]:
    if (config.protocol or "json").strip().lower() == "vllm_tags":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    family = _model_family(config, model)
    effort = (config.reasoning_effort or "").strip().lower()

    if family in {"deepseek", "qwen"}:
        if not effort:
            effort = "high" if family == "deepseek" else "low"
        return {
            "extra_body": {"thinking": {"type": _resolved_reasoning_mode(config, model)}},
            "reasoning_effort": effort,
        }

    if family == "openai" and model.lower().startswith("gpt-5"):
        return {"reasoning_effort": effort or "medium"}

    return {}


def _response_format_kwargs(config: LLMConfig) -> dict[str, Any]:
    if (config.protocol or "json").strip().lower() == "vllm_tags":
        return {}
    mode = (config.response_format or "").strip().lower()
    if mode in {"json", "json_object"}:
        return {"response_format": {"type": "json_object"}}
    return {}


def _is_retryable_llm_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {400, 408, 409, 425, 429} or status_code >= 500
    error_name = type(error).__name__.lower()
    if any(token in error_name for token in ("timeout", "connection", "connect", "server")):
        return True
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "timed out",
            "timeout",
            "connection error",
            "connection reset",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "upstream_error",
            "error code: 400",
        )
    )


def _retry_delay_seconds(attempt: int) -> float:
    base_delay = min(2 ** max(0, attempt - 1), 8)
    jitter = random.uniform(0.0, 0.5)
    return base_delay + jitter


def _reasoning_enabled(config: LLMConfig, model: str) -> bool:
    return _resolved_reasoning_mode(config, model) == "enabled"


def _resolved_reasoning_mode(config: LLMConfig, model: str) -> str:
    reasoning_mode = (config.reasoning or "auto").strip().lower()
    if reasoning_mode not in {"auto", "enabled", "disabled"}:
        reasoning_mode = "auto"
    if reasoning_mode != "auto":
        return reasoning_mode
    return "enabled" if _model_family(config, model) == "deepseek" else "disabled"


def _model_family(config: LLMConfig, model: str) -> str:
    provider = (config.provider or "").strip().lower()
    if provider:
        return provider

    normalized_model = model.lower()
    normalized_base_url = (config.base_url or "").lower()
    if "deepseek" in normalized_model or "deepseek" in normalized_base_url:
        return "deepseek"
    if "qwen" in normalized_model or "dashscope" in normalized_base_url:
        return "qwen"
    if "claude" in normalized_model or "anthropic" in normalized_base_url:
        return "claude"
    return "openai"


def _message_content_and_reasoning(message: Any, *, include_reasoning: bool) -> tuple[str, str]:
    content = getattr(message, "content", None) or ""
    if not include_reasoning:
        return str(content), ""
    reasoning = getattr(message, "reasoning_content", None) or ""
    if not reasoning:
        reasoning = getattr(message, "reasoning", None) or ""
    return str(content), str(reasoning)


def _raw_response_with_reasoning(content: str, reasoning_content: str) -> str:
    if not reasoning_content:
        return content
    return f"<think>\n{reasoning_content}\n</think>\n{content}"


def _content_with_images(
    text: str,
    images: list[str] | None,
    *,
    image_type: str = "png",
) -> str | list[dict[str, Any]]:
    if not images:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": _image_url(image, image_type=image_type)}})
    return parts


def _normalize_openai_message(message: dict[str, Any], *, image_type: str = "png") -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [_normalize_content_part(part, image_type=image_type) for part in content]
    elif "images" in message:
        message["content"] = _content_with_images(
            str(content or ""),
            [str(item) for item in message.get("images") or []],
            image_type=image_type,
        )
        message.pop("images", None)
    return message


def _normalize_content_part(part: Any, *, image_type: str = "png") -> dict[str, Any]:
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}
    if part.get("type") == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": _image_url(image_url, image_type=image_type)}}
        if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            return {"type": "image_url", "image_url": {"url": _image_url(image_url["url"], image_type=image_type)}}
    if part.get("type") in {"text", "image_url"}:
        return part
    if "image" in part:
        return {"type": "image_url", "image_url": {"url": _image_url(str(part["image"]), image_type=image_type)}}
    if "text" in part:
        return {"type": "text", "text": str(part["text"])}
    return {"type": "text", "text": json.dumps(part, ensure_ascii=False)}


def _image_url(value: str, *, image_type: str = "png") -> str:
    if value.startswith(("http://", "https://", "data:image/")):
        return value
    path = Path(value).expanduser()
    path = _prepare_local_image(path, image_type=image_type)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _prepare_local_image(path: Path, *, image_type: str) -> Path:
    normalized = _normalize_image_type(image_type)
    if normalized != "jpeg":
        return path
    try:
        if not path.exists() or not path.is_file():
            return path
        stat = path.stat()
        cache_key = hashlib.sha1(
            f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{normalized}:75".encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = Path("runs/.workspace/llm_images").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / f"{path.stem}.{cache_key}.jpg"
        if dst.exists():
            return dst
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return path
        if not cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 75]):
            return path
        return dst
    except Exception:
        return path


def _normalize_image_type(value: Any) -> str:
    text = str(value or "png").strip().lower()
    if text in {"jpg", "jpeg", "jepg"}:
        return "jpeg"
    return "png"


def _extract_json_value(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty LLM response.")
    candidates = _candidate_json_strings(text)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError) as inner_exc:
                last_error = inner_exc
    salvaged = _salvage_truncated_json_object(text)
    if salvaged is None:
        salvaged = _salvage_react_action_payload(text)
    if salvaged is None:
        raise ValueError(f"LLM response is not valid JSON: {text[:500]} | last_error={last_error}")
    return salvaged


def _extract_json_value_or_text(text: str) -> Any:
    try:
        return _extract_json_value(text)
    except ValueError:
        stripped = text.strip()
        if not stripped:
            raise
        return stripped


def _extract_json_object(text: str) -> dict[str, Any]:
    value = _extract_json_value(text)
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object.")
    return value


def _summarize_response(response: Any) -> dict[str, Any]:
    if response is None:
        return {"available": False}
    choice = None
    try:
        choice = response.choices[0]
    except Exception:
        choice = None
    message = getattr(choice, "message", None) if choice is not None else None
    usage = getattr(response, "usage", None)
    return {
        "available": True,
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "created": getattr(response, "created", None),
        "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
        "message_content": getattr(message, "content", None) if message is not None else None,
        "tool_calls": [
            {
                "id": getattr(call, "id", None),
                "name": getattr(getattr(call, "function", None), "name", None),
            }
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        if message is not None
        else [],
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
        },
    }


def _candidate_json_strings(text: str) -> list[str]:
    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    balanced = _extract_balanced_braces(stripped)
    if balanced:
        candidates.append(balanced)

    repaired = _repair_truncated_json_object(stripped)
    if repaired:
        candidates.append(repaired)

    unique: list[str] = []
    seen = set()
    for item in candidates:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _extract_balanced_braces(text: str) -> str | None:
    return _extract_balanced(text, "{", "}")


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    quote_char = ""
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in {'"', "'"}:
            in_string = True
            quote_char = ch
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _repair_truncated_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    candidate = text[start:].strip()
    if not candidate:
        return None

    depth = 0
    in_string = False
    escape = False
    quote_char = ""
    repaired_chars: list[str] = []

    for ch in candidate:
        repaired_chars.append(ch)
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in {'"', "'"}:
            in_string = True
            quote_char = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1

    if in_string:
        repaired_chars.append(quote_char)
    if depth > 0:
        repaired_chars.extend("}" for _ in range(depth))

    repaired = "".join(repaired_chars).strip()
    if repaired == candidate:
        return None
    return repaired


def _salvage_truncated_json_object(text: str, *, max_trim: int = 512) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    max_steps = min(max_trim, len(stripped))
    for trim in range(max_steps + 1):
        candidate = stripped[:-trim] if trim else stripped
        candidate = candidate.rstrip()
        if not candidate:
            continue
        repaired = _repair_truncated_json_object(candidate) or candidate
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(repaired)
            except (ValueError, SyntaxError):
                continue
        if isinstance(value, dict):
            return value
    return None


def _salvage_react_action_payload(text: str) -> dict[str, Any] | None:
    thought = _extract_quoted_field(text, "thought")
    action_name = _extract_quoted_field(text, "name")
    if not thought and not action_name:
        return None

    payload: dict[str, Any] = {}
    if thought:
        payload["thought"] = thought

    action: dict[str, Any] = {}
    if action_name:
        action["name"] = action_name

    properties = _extract_properties_object(text)
    action["properties"] = properties if isinstance(properties, dict) else {}
    payload["action"] = action
    return payload if payload.get("action", {}).get("name") or payload.get("thought") else None


def _extract_quoted_field(text: str, field_name: str) -> str | None:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"', text)
    if not match:
        return None
    start = match.end()
    chars: list[str] = []
    escape = False
    for ch in text[start:]:
        if escape:
            chars.append(ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            return "".join(chars)
        chars.append(ch)
    return "".join(chars).strip() or None


def _extract_properties_object(text: str) -> dict[str, Any] | None:
    marker = re.search(r'"properties"\s*:\s*{', text)
    if not marker:
        return None
    start = marker.end() - 1
    fragment = text[start:]
    repaired = _repair_truncated_json_object(fragment)
    if not repaired:
        return None
    try:
        value = json.loads(repaired)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(repaired)
        except (ValueError, SyntaxError):
            return None
    return value if isinstance(value, dict) else None
