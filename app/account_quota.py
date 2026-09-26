from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


QUOTA_UNIT_DENOMINATOR = 500000

_NEWAPI_HOST_MARKERS = (
    "dmxapi",
    "aigcbest",
    "scitrace",
    "yi-zhan",
    "newapi",
    "oneapi",
)
_OFFICIAL_HOST_MARKERS = (
    "dashscope.aliyuncs.com",
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "openai.azure.com",
)
_USER_HEADER_NAMES = (
    "New-Api-User",
    "New-API-User",
    "Dmx-Api-User",
    "X-Api-User",
)


def detect_platform(base_url: str) -> str:
    host = (urlsplit(base_url or "").netloc or "").lower()
    if not host:
        return "unknown"
    if any(marker in host for marker in _OFFICIAL_HOST_MARKERS):
        return "official"
    if "dmxapi" in host:
        return "dmxapi"
    if any(marker in host for marker in _NEWAPI_HOST_MARKERS):
        return "newapi"
    return "newapi"


def origin_from_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url or "")
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def account_endpoint_from_base_url(base_url: str, path: str = "/api/user/self") -> str:
    origin = origin_from_base_url(base_url)
    if not origin:
        return ""
    return f"{origin}{path}"


def _authorization(token: str) -> str:
    return token if token.lower().startswith("bearer ") else f"Bearer {token}"


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
    return None


def _user_headers(user_id: str) -> dict[str, str]:
    if not user_id:
        return {}
    return {name: user_id for name in _USER_HEADER_NAMES}


def _resolve_credentials(config: Any) -> tuple[str, str, str]:
    system_token = (
        os.environ.get("DOCPILOT_ACCOUNT_ACCESS_TOKEN")
        or getattr(config.llm, "account_access_token", "")
        or ""
    ).strip()
    api_key = (config.llm.api_key or "").strip()
    user_id = (
        os.environ.get("DOCPILOT_ACCOUNT_USER_ID")
        or str(getattr(config.llm, "account_user_id", "") or "")
    ).strip()
    return system_token, api_key, user_id


def _http_get_json(url: str, headers: dict[str, str], timeout: float) -> tuple[Any | None, str | None]:
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload, None
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()[:240]
        except Exception:
            detail = ""
        suffix = f" {detail}" if detail else ""
        return None, f"HTTP Error {exc.code}: {exc.reason}{suffix}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _parse_user_self(payload: Any) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    quota = _as_int(data.get("quota"))
    used_quota = _as_int(data.get("used_quota"))
    if quota is None or used_quota is None:
        return None
    parsed = {"quota": quota, "used_quota": used_quota}
    request_count = _as_int(data.get("request_count"))
    if request_count is not None:
        parsed["request_count"] = request_count
    return parsed


def _parse_usage_token(payload: Any) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        data = payload
    quota = _as_int(data.get("quota"))
    used_quota = _as_int(data.get("used_quota") if data.get("used_quota") is not None else data.get("total_used"))
    if quota is not None and used_quota is not None:
        parsed = {"quota": quota, "used_quota": used_quota}
    else:
        granted = _as_int(data.get("total_granted"))
        used = _as_int(data.get("total_used"))
        available = _as_int(data.get("total_available"))
        if used is None:
            return None
        if available is None and granted is not None:
            available = granted - used
        if available is None:
            return None
        parsed = {"quota": available, "used_quota": used}
    request_count = _as_int(data.get("request_count"))
    if request_count is not None:
        parsed["request_count"] = request_count
    return parsed


def _success_payload(endpoint: str, platform: str, parsed: dict[str, int]) -> dict[str, Any]:
    quota = parsed["quota"]
    used_quota = parsed["used_quota"]
    return {
        "available": True,
        "platform": platform,
        "endpoint": endpoint,
        "quota": quota,
        "quota_actual": quota / QUOTA_UNIT_DENOMINATOR,
        "used_quota": used_quota,
        "used_quota_actual": used_quota / QUOTA_UNIT_DENOMINATOR,
        "request_count": parsed.get("request_count"),
    }


def _query_attempts(config: Any) -> list[dict[str, Any]]:
    platform = detect_platform(config.llm.base_url)
    origin = origin_from_base_url(config.llm.base_url)
    system_token, api_key, user_id = _resolve_credentials(config)
    attempts: list[dict[str, Any]] = []
    if not origin:
        return attempts

    user_self = f"{origin}/api/user/self"
    usage_token = f"{origin}/api/usage/token"
    extra_user = _user_headers(user_id)

    if system_token:
        attempts.append(
            {
                "endpoint": user_self,
                "token": system_token,
                "headers": extra_user,
                "parser": _parse_user_self,
            }
        )
    if api_key and api_key != system_token:
        attempts.append(
            {
                "endpoint": user_self,
                "token": api_key,
                "headers": extra_user,
                "parser": _parse_user_self,
            }
        )
    if api_key:
        attempts.append(
            {
                "endpoint": usage_token,
                "token": api_key,
                "headers": {},
                "parser": _parse_usage_token,
            }
        )
    elif system_token:
        attempts.append(
            {
                "endpoint": usage_token,
                "token": system_token,
                "headers": {},
                "parser": _parse_usage_token,
            }
        )
    return attempts


def query_account_quota(config: Any) -> dict[str, Any]:
    platform = detect_platform(config.llm.base_url)
    if not config.llm.enabled:
        return {"available": False, "platform": platform, "error": "llm is disabled"}
    if platform == "official":
        return {
            "available": False,
            "platform": platform,
            "error": "official provider has no NewAPI-compatible quota endpoint",
        }

    system_token, api_key, _user_id = _resolve_credentials(config)
    if not system_token and not api_key:
        return {"available": False, "platform": platform, "error": "account access token is not configured"}

    endpoint = account_endpoint_from_base_url(config.llm.base_url)
    if not endpoint:
        return {"available": False, "platform": platform, "error": "llm.base_url is missing a host"}

    timeout = min(max(float(config.llm.timeout or 10), 1.0), 30.0)
    errors: list[str] = []
    last_endpoint = endpoint
    for attempt in _query_attempts(config):
        last_endpoint = attempt["endpoint"]
        headers = {
            "Accept": "application/json",
            "Authorization": _authorization(attempt["token"]),
            **attempt["headers"],
        }
        payload, error = _http_get_json(attempt["endpoint"], headers, timeout)
        if error:
            errors.append(f"{attempt['endpoint']}: {error}")
            continue
        parsed = attempt["parser"](payload)
        if parsed is None:
            message = payload.get("message") if isinstance(payload, dict) else None
            detail = message if isinstance(message, str) and message else "missing quota fields"
            errors.append(f"{attempt['endpoint']}: {detail}")
            continue
        return _success_payload(attempt["endpoint"], platform, parsed)

    return {
        "available": False,
        "platform": platform,
        "endpoint": last_endpoint,
        "error": "; ".join(errors) or "account quota query failed",
    }


def build_pipeline_cost(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": bool(before.get("available") and after.get("available")),
        "quota_unit_denominator": QUOTA_UNIT_DENOMINATOR,
        "before": before,
        "after": after,
    }
    if payload["available"]:
        raw_cost = int(after["used_quota"]) - int(before["used_quota"])
        payload.update(
            {
                "used_quota_delta": raw_cost,
                "actual_cost": raw_cost / QUOTA_UNIT_DENOMINATOR,
                "request_count_delta": (
                    int(after["request_count"]) - int(before["request_count"])
                    if isinstance(before.get("request_count"), int) and isinstance(after.get("request_count"), int)
                    else None
                ),
            }
        )
    else:
        errors = []
        if before.get("error"):
            errors.append(f"before: {before['error']}")
        if after.get("error"):
            errors.append(f"after: {after['error']}")
        payload["error"] = "; ".join(errors) or "account quota query failed"
    return payload
