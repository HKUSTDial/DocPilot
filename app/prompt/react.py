from __future__ import annotations

import json
import re
from typing import Any, Dict

from pydantic import BaseModel, Field


# TODO Agent can choose more than 1 action in some case, but in this version, we only support one action.


class ActOp(BaseModel):
    thought: str = Field(description="your thought on the external observation")
    action: Dict[str, Any] = Field(description="the action you can choose from")


SYSTEM_PROMPT = """
You are given a task, and you need to think on the external observation, and choose the appropriate action to solve the current task.
"""


# - After self-verification rejects an answer, treat the listed issues and missing evidence as constraints for the next action. Keep the question's target and reporting basis stable across actions, do not pre-fill numeric answers into `solver` or `answer` instructions unless the evidence supports the exact metric and scope, and keep related-but-different evidence as context only.
# - If `solver` returns `result = None` or empty inputs, the current evidence is insufficient, ambiguous, or not aligned with the requested metric; it does NOT mean no evidence has been extracted. Continue from existing evidence by refining extraction.

NEXT_STEP_PROMPT = """
Above is your observation.

Now you need to think, and choose the appropriate action to promote the task.
Note that your strategy should adapt to the question type; for example, you must exhaustively explore all relevant content before answering counting questions like "how many".
Avoid meaningless repeated tool calls, but do not hard-block a tool solely because it was used before. A repeated tool call is valid when it has a clear new target, missing evidence slot, unresolved subproblem, or more precise instruction. 
If repeated calls to the same tool produce no new observation, do not get stuck: analyze what the failed observations imply, then either refine the target/instruction or switch to the next appropriate tool.

====================
STRICT ACTION DEPENDENCIES
====================
READ THIS BEFORE CHOOSING THE NEXT ACTION.
1. Tool granularity:
   `navigate` (page-level discovery and candidate page selection)
   `-- `extract` (node/evidence-level extraction of local text, tables, or figures)
       |-- `focus` (local visual reading of extracted figures/regions; do NOT use it on full page screenshots)
       `-- `solver` (local deterministic computation over explicit structuded evidence)
   Prefer the coarsest sufficient tool, and move to the next granularity only when current evidence is too broad, too long, or missing local detail.
2. You must run `navigate` before `extract`, and `extract` before `focus` or `solver`.
3. ALWAYS call `execute` immediately after `solver` if solver generated code.
4. You must run `analyze` before using an `answer` as the final answer.

====================
EVIDENCE STATE GUIDANCE
====================
- State concepts: `page_range` is the available document page range using the same page numbers accepted by navigation/extraction; `candidate_pages` are pages identified as potentially relevant by navigation; `pending_page_sets` are page ranges not yet inspected by navigation; `extraction.extracted_pages` are candidate pages that have already been processed by extract.
- Per-step limits: `max_navigate_pages_per_step` is the maximum number of pages inspected by one `navigate` call; `max_extract_pages_per_step` is the maximum number of candidate pages processed by one `extract` call.
- Page boundaries: when you provide `pages` to `navigate` or `extract`, every requested page MUST be within the available document range shown by `page_range` and/or `pending_page_sets`. Never request pages outside `page_range`, or page ranges that extend beyond the available page range.
- If the question asks for a page outside the available page range (for example page 30 when `page_range` is `1-20`), do NOT call `navigate` for that page. Treat the target page as unavailable evidence and move to downstream reasoning/answering with that constraint.
- If `pending_page_sets` is empty but visited coverage is clearly below `page_range`, navigation is NOT exhausted: continue `navigate` on the remaining page ranges within `page_range` (e.g., after inspecting 69-78 of page_range 69-98, inspect 79-88 next) before settling on incomplete candidate pages.
- If `evidence_state` is `pending`, page navigation is exhausted. Proceed to downstream actions (e.g., `extract`, `solver`, `answer`) and do NOT call `navigate` again unless strictly required by new observations.


You can choose the action from the following actions:
{actions}
"""

MULTIDOC_EVIDENCE_GUIDANCE = """
- Multi-document state uses `document_range`, `pending_page_sets`, and `candidate_pages` with `doc_id:page-range`, plus `overview`.
- Use `pages="overview"` only when `overview` is false and the target page is unknown; once `overview` is true, do not request it again.
- Pages selected by overview are removed from `pending_page_sets`; extract those candidates before sequentially navigating the remaining page sets.
- Treat returned `doc_id:page` candidates as leads for extraction. If no useful candidate or extracted evidence is found, continue ordinary navigation using a `doc_id:page-range` target rather than treating an empty overview result as navigation exhaustion.
- Multi-document `pages` targets use `doc_id:page` or `doc_id:start-end`, and every target must be within `document_range`.
"""

SINGLE_DOCUMENT_OVERVIEW_GUIDANCE = """
- If single-document `overview` is false and the target page is unknown, use `pages="overview"`; it returns integer pages for localization only.
"""

# ACTION_FORMAT = """
# You can execute an action by output a dict following content format:

#     "action": {{
#         "name": "the name of the action you want to execute",
#         "properties": "a list of properties and corresponding values for the action"
#     }}

# Below is the output format regarding the overall output:
# """
ACTION_FORMAT = """
Return ONLY one valid JSON object.
Do NOT output explanations, prose, markdown, code fences, XML, comments, or any extra text.
Your entire response must be parseable by `json.loads`.
The top-level JSON object must contain exactly these keys:
- "thought": a string
- "action": an object with exactly these keys:
  - "name": a string equal to one of the available action names
  - "properties": an object containing the action arguments

Rules:
- Always choose exactly one action from the provided actions.
- Always return an object for "properties". Use `{}` when the chosen action needs no arguments.
- Do not rename keys.
- Do not wrap the JSON in quotes.
- Do not include trailing commas.
- If you are uncertain, still return one valid JSON object and choose the best available action.

Use the following output format exactly:
"""


def output_format() -> str:
    return json.dumps(
        {
            "thought": "your thought on the external observation",
            "action": {
                "name": "the name of the action you want to execute",
                "properties": {"property_name": "property_value"},
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def system_prompt(actions: str) -> str:
    return "\n".join(
        [
            SYSTEM_PROMPT.format(actions=actions).strip(),
            ACTION_FORMAT.strip(),
            output_format(),
        ]
    )


def next_step_prompt(
    actions: str,
    *,
    require_verification: bool = True,
    direct_answer_from_navigation: bool = False,
    available_tool_names: set[str] | None = None,
    multidoc: bool = False,
    single_document_overview: bool = False,
) -> str:
    prompt = NEXT_STEP_PROMPT
    if multidoc:
        prompt = prompt.replace(
            "- State concepts:",
            MULTIDOC_EVIDENCE_GUIDANCE.strip() + "\n- State concepts:",
            1,
        )
    elif single_document_overview:
        prompt = prompt.replace(
            "- State concepts:",
            SINGLE_DOCUMENT_OVERVIEW_GUIDANCE.strip() + "\n- State concepts:",
            1,
        )
    if not require_verification:
        prompt = prompt.replace(
            "4. You must run `analyze` before using an `answer` as the final answer.\n",
            "",
        )
    if available_tool_names is not None:
        if direct_answer_from_navigation:
            prompt = re.sub(
                r"1\. Tool granularity:.*?2\. You must run [^\n]*\n",
                "1. Use `navigate` to select relevant pages, then use `answer` directly from those pages.\n"
                "2. You must run `navigate` before `answer`.\n",
                prompt,
                flags=re.DOTALL,
            )
            prompt = prompt.replace("navigation/extraction", "navigation")
            prompt = prompt.replace(" to `navigate` or `extract`", " to `navigate`")
            prompt = re.sub(r"; `extraction\.extracted_pages`[^.]+\.", ".", prompt)
            prompt = re.sub(r"- Per-step limits:.*?\n", "", prompt)
        else:
            dependency_parts = ["`navigate` before `extract`"]
            downstream_tools = [
                f"`{name}`"
                for name in ("focus", "solver")
                if name in available_tool_names
            ]
            if downstream_tools:
                dependency_parts.append(f"`extract` before {' or '.join(downstream_tools)}")
            prompt = re.sub(
                r"2\. You must run .*\n",
                f"2. You must run {', and '.join(dependency_parts)}.\n",
                prompt,
            )
        pending_examples = [
            f"`{name}`"
            for name in ("extract", "solver", "answer")
            if name in available_tool_names
        ]
        if pending_examples:
            prompt = re.sub(
                r"Proceed to downstream actions \(e\.g\., [^)]+\)",
                f"Proceed to downstream actions (e.g., {', '.join(pending_examples)})",
                prompt,
            )
        if "solver" not in available_tool_names:
            prompt = prompt.replace(
                "       `-- `solver` (local deterministic computation over explicit structuded evidence)\n",
                "",
            )
            prompt = prompt.replace("`solver`, ", "")
        if "focus" not in available_tool_names:
            prompt = prompt.replace(
                "       |-- `focus` (local visual reading of extracted figures/regions; do NOT use it on full page screenshots)\n",
                "",
            )
        if "execute" not in available_tool_names or "solver" not in available_tool_names:
            prompt = prompt.replace(
                "3. ALWAYS call `execute` immediately after `solver` if solver generated code.\n",
                "",
            )
            prompt = prompt.replace("`execute`, ", "")
    return "\n".join(
        [
            prompt.format(actions=actions).strip(),
            ACTION_FORMAT.strip(),
            output_format(),
        ]
    )


DOC_PILOT_REACT_SYSTEM_PROMPT = SYSTEM_PROMPT
