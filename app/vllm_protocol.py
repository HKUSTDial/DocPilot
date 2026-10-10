from __future__ import annotations

import ast
from dataclasses import dataclass
import html
import json
import re
from typing import Any


ACTION_TAGS = ("Navigate", "Extract", "Focus", "Solve", "Answer")

ANALYZE_SYSTEM_PROMPT_TEMPLATE = r"""You are the DocPilot multimodal document agent.

Available next actions: {actions}

Output exactly one <Analyze>...</Analyze> block with next_action.
For navigate and extract, include target_pages: \boxed{{[...]}}.
For focus, include evidence_ids: \boxed{{[...]}}.

Choose next_action only from the available next actions.
Do not fabricate environment state, observations, or tool results."""

ACTION_SYSTEM_PROMPT_TEMPLATE = r"""You are the DocPilot multimodal document agent.

The required action is <{tag}>. This is a command, not a choice.
Output exactly one <{tag}>...</{tag}> block and nothing else.
The first tag must be <{tag}> and the final tag must be </{tag}>.
Do not output <Analyze> or any other action tag.
{contract}

Use only pages, candidate IDs, and evidence IDs provided by the environment.
Do not reproduce the observation or fabricate State, page images, extracted
evidence, or execution results."""

ACTION_CONTRACTS = {
    "navigate": r"End with candidate_pages: \boxed{...}.",
    "extract": r"End with candidate nodes: \boxed{...}.",
    "focus": r"End with operation: \boxed{...}.",
    "solve": (
        r"End with code: \boxed{...}. "
        r"The code must assign the final answer to a Python variable named result. "
        'Read evidence only from inputs["evidence"]. '
        'Do not invent inputs["table_data"], inputs["table_html"], '
        'inputs["financial_table"], or other top-level input keys.'
    ),
    "answer": r"End with answer: \boxed{...}.",
}

@dataclass(frozen=True)
class VLLMRoute:
    action: str
    target_pages: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    reasoning: str = ""


def parse_analyze(text: str, available_actions: set[str]) -> VLLMRoute:
    body = _single_tag_body(text, "Analyze")
    match = re.search(r"(?im)^\s*next_action\s*:\s*([a-z_]+)\s*$", body)
    if not match:
        raise ValueError("<Analyze> is missing next_action.")
    action = match.group(1).lower()
    if action == "solver":
        action = "solve"
    if action not in available_actions:
        raise ValueError(f"Unavailable next_action: {action}")

    raw_pages = _boxed_field(body, "target_pages")
    target_pages = tuple(_split_boxed_values(raw_pages))
    evidence_ids = tuple(_split_boxed_values(_boxed_field(body, "evidence_ids")))
    if action == "extract" and raw_pages is None:
        raise ValueError(f"{action} requires target_pages.")
    if action == "focus" and not evidence_ids:
        raise ValueError("focus requires evidence_ids.")
    return VLLMRoute(
        action=action,
        target_pages=target_pages,
        evidence_ids=evidence_ids,
        reasoning=body.strip(),
    )


def parse_action(text: str, expected_action: str) -> dict[str, Any]:
    expected = "Solve" if expected_action in {"solve", "solver"} else expected_action.title()
    if expected not in ACTION_TAGS:
        raise ValueError(f"Unsupported action: {expected_action}")
    body = _single_tag_body(text, expected)

    if expected == "Navigate":
        raw_candidates = _boxed_or_plain_field(body, "candidate_pages", aliases=("candidate pages",))
        if raw_candidates is None:
            raise ValueError("<Navigate> is missing candidate_pages.")
        values = _split_boxed_values(raw_candidates)
        candidate_pages: list[dict[str, Any]] = []
        for value in values:
            if ":" in value:
                candidate_pages.append({"page": value, "confidence": None})
            else:
                try:
                    candidate_pages.append({"page": int(value), "confidence": None})
                except ValueError:
                    continue
        return {
            "candidate_pages": candidate_pages,
            "reason": body.strip(),
            "need_more_pages": False,
        }

    if expected == "Extract":
        raw_candidates = _boxed_or_plain_field(
            body,
            "candidate_nodes",
            aliases=("candidate nodes", "evidence_ids", "evidence ids"),
        )
        if raw_candidates is None:
            raise ValueError("<Extract> is missing candidate_nodes.")
        values = _split_boxed_values(raw_candidates)
        return {"selected_candidate_ids": values, "reason": body.strip()}

    if expected == "Focus":
        raw_operations = _boxed_fields(body, "operation")
        if not raw_operations:
            raise ValueError("<Focus> is missing operation.")
        operations = [_parse_mapping(raw) for raw in raw_operations]
        for operation in operations:
            if "bbox" in operation and "region" not in operation:
                operation["region"] = {"bbox": operation.pop("bbox")}
            operation.setdefault("reason", body.strip())
        return {"operations": operations, "reason": body.strip()}

    if expected == "Solve":
        code = _boxed_field(body, "code")
        if code is None:
            raise ValueError("<Solve> is missing code.")
        raw_inputs = _boxed_field(body, "inputs")
        inputs = _parse_mapping(raw_inputs) if raw_inputs is not None else {}
        return {
            "summary": body.strip(),
            "code": html.unescape(code).strip(),
            "inputs": inputs,
        }

    answer = _boxed_field(body, "answer")
    if answer is None:
        answer = _last_boxed(body)
    if answer is None:
        raise ValueError("<Answer> is missing a boxed answer.")
    return {
        "answer": html.unescape(answer).strip(),
        "rationale": body.strip(),
        "confidence": 0.8,
    }


def compact_pages(values: tuple[str, ...]) -> str:
    return ",".join(values)


def analyze_system_prompt(actions: list[str] | tuple[str, ...]) -> str:
    return ANALYZE_SYSTEM_PROMPT_TEMPLATE.format(actions=", ".join(actions))


def action_system_prompt(action: str) -> str:
    normalized = "solve" if action == "solver" else action
    tag = "Solve" if normalized == "solve" else normalized.title()
    return ACTION_SYSTEM_PROMPT_TEMPLATE.format(tag=tag, contract=ACTION_CONTRACTS[normalized])


def render_action_observation(action: str, user_prompt: str) -> str:
    normalized = "solve" if action == "solver" else action
    expected = "Solve" if normalized == "solve" else normalized.title()
    contract = ACTION_CONTRACTS[normalized]
    prefix = (
        f"Expected action: <{expected}>\n"
        f"Return exactly one <{expected}>...</{expected}> block.\n\n"
    )
    try:
        payload = json.loads(user_prompt)
    except (json.JSONDecodeError, TypeError):
        payload = None

    if normalized == "extract":
        observation = user_prompt.replace("Candidate items:", "Candidate nodes:")
    elif isinstance(payload, dict):
        payload = _without_image_paths(payload)
        if normalized == "navigate":
            title = "Document overview images:" if "overview" in str(payload.get("task", "")) else "Page images:"
            observation = f"{title}\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        elif normalized in {"focus", "solve", "answer"}:
            observation = "Extracted evidence:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            observation = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        observation = user_prompt

    return (
        prefix
        + observation.rstrip()
        + "\n\nUse only page references, candidate IDs, and evidence IDs provided in the current observation.\n"
        + contract
    )


def _single_tag_body(text: str, expected: str) -> str:
    action_names = "Analyze|Navigate|Extract|Focus|Solve|Answer|Execute"
    opened = re.findall(rf"<({action_names})>", text, flags=re.IGNORECASE)
    if not opened or any(tag.lower() != expected.lower() for tag in opened):
        raise ValueError(f"Expected exactly one <{expected}> block.")
    text = _drop_repeated_open_tags(text, expected)
    match = re.fullmatch(
        rf"\s*<{expected}>\s*(.*?)\s*</{expected}>\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Malformed <{expected}> block.")
    return match.group(1)


def _drop_repeated_open_tags(text: str, expected: str) -> str:
    seen = False

    def replace(match: re.Match[str]) -> str:
        nonlocal seen
        if seen:
            return ""
        seen = True
        return match.group(0)

    return re.sub(rf"<{expected}>", replace, text, flags=re.IGNORECASE)


def _boxed_field(body: str, name: str, aliases: tuple[str, ...] = ()) -> str | None:
    values = _boxed_fields(body, name, aliases)
    return values[-1] if values else None


def _boxed_or_plain_field(body: str, name: str, aliases: tuple[str, ...] = ()) -> str | None:
    boxed = _boxed_field(body, name, aliases)
    if boxed is not None:
        return boxed
    labels = (name, *aliases)
    pattern = "|".join(re.escape(label).replace(r"\_", r"[ _]") for label in labels)
    matches = re.findall(rf"(?im)^\s*(?:{pattern})\s*:\s*(.+?)\s*$", body)
    return matches[-1].strip() if matches else None


def _boxed_fields(body: str, name: str, aliases: tuple[str, ...] = ()) -> list[str]:
    labels = (name, *aliases)
    pattern = "|".join(re.escape(label).replace(r"\_", r"[ _]") for label in labels)
    matches = list(re.finditer(rf"(?im)^\s*(?:{pattern})\s*:\s*\\boxed\s*", body))
    return [_balanced_braces(body, match.end()) for match in matches]


def _last_boxed(body: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\s*", body))
    return _balanced_braces(body, matches[-1].end()) if matches else None


def _balanced_braces(text: str, start: int) -> str:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] not in "{[":
        raise ValueError("Expected '{' or '[' after \\boxed.")
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index] if opening == "{" else text[start : index + 1]
    raise ValueError("Unclosed \\boxed value.")


def _split_boxed_values(value: str | None) -> list[str]:
    if value is None:
        return []
    stripped = html.unescape(value).strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            stripped = stripped[1:-1]
        else:
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip().strip("'\"") for item in stripped.strip("[]").split(",") if item.strip()]


def _parse_mapping(value: str) -> dict[str, Any]:
    decoded = html.unescape(value).strip()
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(decoded)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a mapping inside boxed value.")
    return parsed


def _without_image_paths(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_image_paths(item) for item in value]
    if not isinstance(value, dict):
        return value
    hidden = {"image_path", "image_url", "screenshot_path", "overview_image_path", "crop_path", "zoom_path"}
    return {key: _without_image_paths(item) for key, item in value.items() if key not in hidden}
