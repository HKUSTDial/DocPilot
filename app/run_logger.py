from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
from typing import Any


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)
        self._pending_tool_reasonings: dict[int, list[str]] = {}
        self._pending_extract_raw: dict[int, list[str]] = {}

    def info(self, message: str) -> None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        lines = str(message).splitlines() or [""]
        self._fh.write(f"{timestamp} - INFO - {lines[0]}\n")
        for line in lines[1:]:
            self._fh.write(f"  {line}\n")
        self._fh.flush()

    def log_step(self, step: int, action: dict[str, Any]) -> None:
        self.log_tool_start(step, action)

    def log_tool_start(self, step: int, action: dict[str, Any]) -> None:
        tool_name = self._action_tool_name(action)
        args = dict(action.get("args") or {})
        args.pop("tool_name", None)
        action_payload = {
            "name": tool_name,
            "properties": args,
        }
        self.info(f"STEP {step} Executing action: {action_payload}")

    def log_llm_call(
        self,
        *,
        step: int | None,
        call_kind: str,
        call_source: str | None,
        raw_response: str,
        parsed_response: dict[str, Any] | None = None,
    ) -> None:
        if call_source == "react_decision":
            self.info(self._format_react_decision(step, raw_response, parsed_response))
            return
        if step is None:
            return
        if str(call_source or "").startswith("vllm_tags_"):
            action = str(call_source).removeprefix("vllm_tags_")
            self.info(f"STEP {step} vLLM tags {action} raw response:\n{raw_response}")
            return
        if call_source == "extract_candidate_selector":
            raw = self._extract_response_body(raw_response)
            self._pending_extract_raw.setdefault(step, []).append(raw if isinstance(raw, str) else str(raw))
        reasoning = self._format_tool_reasoning(call_source, raw_response, parsed_response)
        if reasoning:
            self._pending_tool_reasonings.setdefault(step, []).append(reasoning)

    def log_result(
        self,
        *,
        step: int | None,
        result: str,
        observation: dict[str, Any] | None = None,
        post_state: dict[str, Any] | None = None,
    ) -> None:
        prefix = f"STEP {step} " if step is not None else ""
        payload = ((observation or {}).get("payload") or {}) if isinstance(observation, dict) else {}
        lines = [f"{prefix}Action execution result:"]
        pending_reasonings = self._pending_tool_reasonings.pop(step, []) if step is not None else []
        if pending_reasonings:
            lines.append("Reasoning:")
            for reasoning in pending_reasonings:
                lines.append(f"  {reasoning}")
        message = result
        if isinstance(observation, dict) and observation.get("message"):
            message = str(observation["message"])
        lines.append("Result:")
        extract_raw = self._pending_extract_raw.pop(step, []) if step is not None else []
        if isinstance(observation, dict) and observation.get("action_type") == "extract" and extract_raw:
            for index, raw in enumerate(extract_raw, start=1):
                title = f"Raw response {index}:" if len(extract_raw) > 1 else "Raw response:"
                lines.append(f"  {title}")
                lines.extend(f"    {line}" for line in raw.splitlines())
            self.info("\n".join(lines))
            return
        lines.append(f"  <message>{message}</message>")
        lines.append(f"  <payload>{json.dumps(self._payload_summary(payload), ensure_ascii=False, indent=2)}</payload>")
        self.info("\n".join(lines))

    def _payload_summary(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return self._sanitize(payload)
        summary: dict[str, Any] = {}
        if "task" in payload:
            summary["task"] = self._sanitize(payload.get("task"))
        if "candidate_pages" in payload:
            summary["candidate_pages"] = self._sanitize(payload.get("candidate_pages"))
        if "reason" in payload:
            summary["reason"] = self._sanitize(payload.get("reason"))
        if "need_more_pages" in payload:
            summary["need_more_pages"] = payload.get("need_more_pages")
        if "pages" in payload:
            summary["pages"] = self._sanitize(payload.get("pages"))
        if "evidence_ids" in payload:
            summary["evidence_ids"] = self._sanitize(payload.get("evidence_ids"))
        if "figure_refs" in payload:
            figure_refs = payload.get("figure_refs") or []
            if isinstance(figure_refs, list):
                summary["figure_refs"] = [
                    {
                        "figure_ref": item.get("figure_ref"),
                        "page": item.get("page"),
                        "evidence_id": item.get("evidence_id"),
                    }
                    for item in figure_refs[:5]
                    if isinstance(item, dict)
                ]
        if "selected_candidate_ids" in payload:
            summary["selected_candidate_ids"] = self._sanitize(payload.get("selected_candidate_ids"))
        if "operations" in payload:
            summary["operations"] = self._sanitize(payload.get("operations"))
        if "result" in payload:
            summary["result"] = self._sanitize(payload.get("result"))
        if "candidate_answer" in payload and isinstance(payload["candidate_answer"], dict):
            answer = payload["candidate_answer"].get("answer")
            if answer:
                summary["answer"] = answer
        if "verification" in payload and isinstance(payload["verification"], dict):
            accepted = payload["verification"].get("accepted")
            if accepted is not None:
                summary["accepted"] = accepted
        if "next_task" in payload:
            summary["next_task"] = self._sanitize(payload.get("next_task"))
        if "analysis" in payload and isinstance(payload["analysis"], dict):
            analysis = payload["analysis"]
            analysis_summary: dict[str, Any] = {}
            if "task" in analysis:
                analysis_summary["task"] = self._sanitize(analysis.get("task"))
            if "summary" in analysis:
                analysis_summary["summary"] = self._sanitize(analysis.get("summary"))
            plan = analysis.get("plan")
            if isinstance(plan, list) and plan:
                analysis_summary["plan"] = [self._sanitize(item) for item in plan[:3]]
                if len(plan) > 3:
                    analysis_summary["plan_truncated"] = True
            if analysis_summary:
                summary["analysis"] = analysis_summary
        return summary

    def _format_react_decision(
        self,
        step: int | None,
        raw_response: str,
        parsed_response: dict[str, Any] | None,
    ) -> str:
        thought = ""
        action_name = ""
        action_args: dict[str, Any] = {}
        if isinstance(parsed_response, dict):
            thought = str(parsed_response.get("thought") or parsed_response.get("content") or "").strip()
            action = parsed_response.get("action") or {}
            if isinstance(action, dict):
                action_name = str(action.get("name") or "").strip()
                action_args = dict(action.get("properties") or {})
        prefix = f"STEP {step} " if step is not None else ""
        action_payload = {"name": action_name, "properties": action_args}
        reasoning = self._extract_reasoning(raw_response)
        reasoning_block = f"<reasoning>{reasoning}</reasoning>\n" if reasoning else ""
        return (
            f"{prefix}LLM Response: {reasoning_block}<thought>{thought}</thought>\n"
            f"<action>{json.dumps(action_payload, ensure_ascii=False, indent=4)}</action>"
        )

    def _format_tool_reasoning(
        self,
        call_source: str | None,
        raw_response: str,
        parsed_response: dict[str, Any] | None,
    ) -> str:
        reasoning = self._extract_reasoning(raw_response)
        if not reasoning:
            return ""
        return f"<reasoning>{reasoning}</reasoning>"

    def _extract_reasoning(self, raw_response: str) -> str:
        start_tag = "<think>"
        end_tag = "</think>"
        start = raw_response.find(start_tag)
        end = raw_response.find(end_tag)
        if start < 0 or end <= start:
            return ""
        reasoning = raw_response[start + len(start_tag):end].strip()
        sanitized = self._sanitize(reasoning)
        return sanitized if isinstance(sanitized, str) else str(sanitized)

    def _extract_response_body(self, raw_response: str) -> str:
        end_tag = "</think>"
        end = raw_response.find(end_tag)
        body = raw_response[end + len(end_tag):] if end >= 0 else raw_response
        sanitized = self._sanitize(body.strip())
        return sanitized if isinstance(sanitized, str) else str(sanitized)

    def _action_tool_name(self, action: dict[str, Any]) -> str:
        action_type = str(action.get("action_type") or "")
        args = dict(action.get("args") or {})
        return str(args.get("tool_name") or action_type or "unknown")

    def _sanitize(self, value: Any, depth: int = 0) -> Any:
        if depth >= 5:
            return "<max-depth-reached>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= 100000 else value[:100000] + "...<truncated>"
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

    def close(self) -> None:
        self._fh.close()
