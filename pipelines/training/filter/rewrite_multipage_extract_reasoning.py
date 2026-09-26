#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import load_config
from app.llm import LLMClient


SYSTEM_PROMPT = """You are the Extract Agent in a document question-answering system. Several pages were inspected separately; return the single extraction result that should have been produced for them together.

# Candidate selection
- If `Selected candidate IDs` is provided, treat it as the final target evidence and copy it exactly.
- Otherwise, merge the final selected IDs from all page analyses in first-seen order and remove duplicates. A page selecting nothing does not erase selections from other pages.
- Use only full candidate IDs present in the input, such as `page10::text::10`. Never invent or rename an ID.

# Reason
- Write one concise, natural sentence explaining how the selected candidates support the extraction instruction. If specific candidate IDs make the evidence roles or relationships clearer, then mention them.
- A non-empty `Selected candidate IDs` list is authoritative and must be preserved exactly. Ignore alternative selection recommendations and never discuss unselected candidates or pages.
- Describe specific facts, values, diagrams, or specifications only when the analyses explicitly establish that the selected candidates contain them; never speculate.
- If the selected candidates' exact content or relevance is uncertain, generic, or disputed by the analyses, treat them as a fallback extraction set and say conservatively that they preserve the available text, visual content, and page layout for direct inspection.
- Never give a no-evidence or irrelevance verdict for a non-empty selection. State that no relevant evidence was found only when the final selected ID list is empty.
- State the evidence directly as an Extract Agent. Do not mention analyses, selections, payloads, logs, rewriting, downstream outcomes, or conflicts.
- If `Next step context` is provided, use it only to resolve an inconsistent selection; never cite or mention it as evidence.

# Output
Return only this JSON object, with no reasoning tags, markdown, or extra fields:
{
  "selected_candidate_ids": ["page10::text::10"],
  "reason": "page10::text::10 states the ownership percentage needed to answer the question."
}
"""

FEW_SHOT_EXAMPLES = [
    """Example 1 (real Alaska Trust extraction)
Input:
Question: How much of the Alaska Trust is owned by The Aleut Corporation and its subsidiaries?
Extract instruction: Extract the specific ownership percentage and surrounding relationship text.
Pages: [10, 1]
Page analyses:
page10::text::10 and page10::text::1 state that the Company has 73% in Alaska Trust Company; the remaining candidates contain unrelated revenue, fund, or cover information.
Selected candidate IDs:
["page10::text::10","page10::text::1"]
Output:
{"selected_candidate_ids":["page10::text::10","page10::text::1"],"reason":"page10::text::10 and page10::text::1 explicitly state that the Company has a 73% interest in Alaska Trust Company, directly providing the requested ownership percentage and relationship."}""",

    """Example 2 (real ID-AC rounds extraction)
Input:
Question: How many rounds were discharged by officers during ID-AC incidents in 2014?
Extract instruction: Extract all text, tables, and figure captions related to the number of rounds discharged by officers during ID-AC incidents in 2014. Specifically look for total counts, summary statistics, or data points in Figure 19.
Pages: [33, 34]
Page analyses:
page33::text::6 states that officers discharged 201 rounds during ID-AC incidents in 2014; page34::text::34 discusses rounds fired per incident and references Figure 19, and page34::figure::2 contains the corresponding visualization. The screenshots preserve the surrounding context and figure layout.
Output:
{"selected_candidate_ids":["page33::text::6","page33::screenshot","page34::text::34","page34::figure::2","page34::screenshot"],"reason":"page33::text::6 states that officers discharged 201 rounds during ID-AC incidents in 2014, while page34::text::34 and page34::figure::2 provide the Figure 19 description and visualization, with the screenshots preserving their report context and visual layout."}""",

    """Example 3 (uncertain visual fallback)
Input:
Question: What is the vehicle's braking system?
Extract instruction: Extract braking-system specifications.
Pages: [27]
Page analyses:
The text candidate contains only a page label. The figure and screenshot may preserve content that was not represented in the extracted text, but no specific braking detail is established.
Selected candidate IDs:
["page27::text::27","page27::figure::0","page27::screenshot"]
Output:
{"selected_candidate_ids":["page27::text::27","page27::figure::0","page27::screenshot"],"reason":"The selected text and visual candidates preserve the available page content and layout for direct inspection where the extracted descriptions do not establish the requested specification."}""",
]

NEGATIVE_EVIDENCE_VERDICT = re.compile(
    r"\b(?:no relevant evidence|no evidence|none of (?:the )?(?:provided )?candidates|"
    r"contains? no .*evidence|with no mention of)\b",
    re.IGNORECASE,
)


def result_block(log_path: Path, step: int) -> str:
    start = f"STEP {step} Action execution result:"
    next_step = re.compile(r"^\d{4}-\d{2}-\d{2} .* - INFO - STEP \d+ ")
    lines: list[str] = []
    collecting = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        if not collecting:
            collecting = start in line
            continue
        if next_step.match(line):
            break
        lines.append(line)
    return "".join(lines)


def reasoning_blocks(text: str) -> list[str]:
    return [
        match.strip()
        for match in re.findall(r"<reasoning\b[^>]*>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
        if match.strip()
    ]


def payload_list(text: str, key: str) -> list[str]:
    payloads = re.findall(r"<payload>(.*?)</payload>", text, re.DOTALL | re.IGNORECASE)
    for raw in reversed(payloads):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            return unique_ids([str(item) for item in value])
    return []


def unique_ids(candidate_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in candidate_ids if str(item).strip()))


def fixed_candidate_selections(_pages: list[Any], candidate_ids: list[str]) -> list[dict[str, Any]]:
    return [{"selected_candidate_ids": unique_ids(candidate_ids)}]


def next_step_context(trace_path: Path, step: int) -> dict[str, Any] | None:
    if not trace_path.exists():
        return None
    rows = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if step >= len(rows):
        return None
    action = json.loads(rows[step]).get("action", {})
    action_type = str(action.get("action_type") or "")
    context: dict[str, Any] = {"action_type": action_type}
    if action_type not in {"answer", "analyze", "finish"}:
        context.update({key: action[key] for key in ("intent", "args") if action.get(key)})
    return context


def validated_decision(
    response: object,
    *,
    require_selected_ids: bool,
) -> dict[str, object]:
    if not isinstance(response, dict):
        raise ValueError("model response is not a JSON object")
    selected_ids: list[str] = []
    if require_selected_ids:
        selected = response.get("selected_candidate_ids")
        if not isinstance(selected, list):
            raise ValueError("model response is missing selected_candidate_ids")
        selected_ids = unique_ids([str(item) for item in selected])
    reason = str(response.get("reason") or "").strip()
    if not reason:
        raise ValueError("model response is missing the unified reason")
    return {"selected_candidate_ids": selected_ids, "reason": reason}


def normalized_ids(candidate_ids: list[str]) -> set[str]:
    return {str(item).strip() for item in candidate_ids if str(item).strip()}


def load_completed(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add((str(row["trace_path"]), int(row["step"])))
    return completed


def build_jobs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for sample in payload.get("results", []):
        for step in sample.get("matched_extract_steps", []):
            jobs.append(
                {
                    "trace_path": str(sample["trace_path"]),
                    "log_path": str(sample["log_path"]),
                    "question": str(sample.get("question") or ""),
                    "step": int(step["step"]),
                    "pages": step.get("pages") or [],
                    "instruction": str(step.get("instruction") or ""),
                }
            )
    return jobs


def rewrite_job(
    job: dict[str, Any],
    *,
    llm_config: Any,
    system_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    llm = LLMClient(llm_config)
    block = result_block(Path(job["log_path"]), job["step"])
    analyses = reasoning_blocks(block)
    if len(analyses) < 2:
        raise ValueError(f"expected multiple reasoning blocks, found {len(analyses)}")
    payload_ids = payload_list(block, "selected_candidate_ids")
    payload_evidence_ids = payload_list(block, "evidence_ids")
    payload_decisions = fixed_candidate_selections(job["pages"], payload_ids)
    relevant_analyses = (
        [analysis for analysis in analyses if any(candidate_id in analysis for candidate_id in payload_ids)]
        if payload_ids
        else analyses
    )
    combined_reasoning = "\n\n".join(relevant_analyses or analyses)
    prompt_parts = [
        f"Question:\n{job['question']}",
        f"\nExtract instruction:\n{job['instruction']}",
        f"\nPages:\n{json.dumps(job['pages'], ensure_ascii=False)}",
    ]
    prompt_parts.extend(["\nPage analyses:", combined_reasoning])
    if payload_evidence_ids:
        prompt_parts.append(
            "\nSelected candidate IDs:\n" + json.dumps(payload_ids, ensure_ascii=False)
        )
        prompt_parts.append(
            "\nThe selected list is final and non-empty. If the analyses do not explicitly establish "
            "the selected candidates' query-specific content, describe them conservatively as a "
            "fallback set preserving the available text, visual content, and page layout for direct "
            "inspection. Do not invent a fact, value, diagram, or specification, and do not return "
            "a no-evidence verdict."
        )
    user_prompt = "\n".join(prompt_parts)
    # print(f"\n=== user_prompt first 5000 chars ===\n{user_prompt[:5000]}", flush=True)
    # print(f"\n=== user_prompt last 5000 chars ===\n{user_prompt[-5000:]}", flush=True)
    response = llm.generate_json(
        system_prompt,
        user_prompt,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    # print(f"\n=== LLM response ===\n{json.dumps(response, ensure_ascii=False, indent=2)}", flush=True)
    inferred_decision = validated_decision(
        response,
        require_selected_ids=not payload_evidence_ids,
    )
    if payload_ids and NEGATIVE_EVIDENCE_VERDICT.search(str(inferred_decision["reason"])):
        retry_prompt = (
            user_prompt
            + "\n\nRejected reason:\n"
            + str(inferred_decision["reason"])
            + "\n\nThis reason incorrectly gives a no-evidence verdict despite the non-empty "
            "fixed selection. Rewrite it to describe only the selected nodes' useful evidence or "
            "visual-inspection role, without inventing facts."
        )
        response = llm.generate_json(
            system_prompt,
            retry_prompt,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        inferred_decision = validated_decision(
            response,
            require_selected_ids=not payload_evidence_ids,
        )
        if NEGATIVE_EVIDENCE_VERDICT.search(str(inferred_decision["reason"])):
            raise ValueError("non-empty selected_candidate_ids received a no-evidence reason")
    used_next_step_context = False
    if not payload_evidence_ids:
        reasoning_ids = inferred_decision["selected_candidate_ids"]
        assert isinstance(reasoning_ids, list)
        if normalized_ids(reasoning_ids) != normalized_ids(payload_ids):
            context = next_step_context(Path(job["trace_path"]), job["step"])
            if context:
                retry_prompt = user_prompt + "\n\nNext step context:\n" + json.dumps(context, ensure_ascii=False)
                response = llm.generate_json(
                    system_prompt,
                    retry_prompt,
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                inferred_decision = validated_decision(response, require_selected_ids=True)
                reasoning_ids = inferred_decision["selected_candidate_ids"]
                assert isinstance(reasoning_ids, list)
                used_next_step_context = True
        if normalized_ids(reasoning_ids) != normalized_ids(payload_ids):
            return {
                "trace_path": job["trace_path"],
                "step": job["step"],
                "candidate_selection_decisions": [],
                "drop_extract": True,
                "drop_reason": "reasoning_payload_candidate_selection_mismatch",
                "reasoning_selected_candidate_ids": reasoning_ids,
                "payload_selected_candidate_ids": payload_ids,
                "source_reasoning_count": len(analyses),
                "model": llm_config.model,
                **({"used_next_step_context": True} if used_next_step_context else {}),
            }
    decisions = [{**payload_decisions[0], "reason": inferred_decision["reason"]}]
    row = {
        "trace_path": job["trace_path"],
        "step": job["step"],
        "candidate_selection_decisions": decisions,
        "source_reasoning_count": len(analyses),
        "model": llm_config.model,
    }
    if not payload_evidence_ids:
        row["reasoning_payload_consistent"] = True
    if used_next_step_context:
        row["used_next_step_context"] = True
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Extract reasoning blocks into candidate selection decisions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "pipelines/training/filter/eval_full_true_multipage_extract_multi_reasoning.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "pipelines/training/filter/multipage_extract_reasoning_overrides.jsonl",
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/docpilot.example.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} already exists; use --resume or choose another --output")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    jobs = build_jobs(payload)
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]
    completed = load_completed(args.output) if args.resume else set()
    jobs = [job for job in jobs if (job["trace_path"], job["step"]) not in completed]

    config = load_config(args.config)
    if args.model:
        config.llm.model = args.model
    config.llm.temperature = 0.7
    config.llm.max_tokens = args.max_tokens
    system_prompt = SYSTEM_PROMPT + "\n\n" + "\n\n".join(FEW_SHOT_EXAMPLES)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def handle_result(index: int, job: dict[str, Any], row: dict[str, Any]) -> None:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        print(f"rewritten {index}/{len(jobs)}: {job['trace_path']} step={job['step']}", flush=True)

    def handle_error(index: int, job: dict[str, Any], exc: BaseException) -> None:
        print(
            f"failed {index}/{len(jobs)}: {job['trace_path']} step={job['step']}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    with args.output.open("a", encoding="utf-8") as output:
        if args.workers <= 1:
            for index, job in enumerate(tqdm(jobs, desc="Rewriting extract reasoning", unit="step"), 1):
                try:
                    row = rewrite_job(
                        job,
                        llm_config=config.llm,
                        system_prompt=system_prompt,
                        max_tokens=args.max_tokens,
                    )
                    handle_result(index, job, row)
                except Exception as exc:
                    handle_error(index, job, exc)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        rewrite_job,
                        job,
                        llm_config=config.llm,
                        system_prompt=system_prompt,
                        max_tokens=args.max_tokens,
                    ): (index, job)
                    for index, job in enumerate(jobs, 1)
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Rewriting extract reasoning",
                    unit="step",
                ):
                    index, job = futures[future]
                    try:
                        handle_result(index, job, future.result())
                    except Exception as exc:
                        handle_error(index, job, exc)


# python pipelines/training/filter/rewrite_multipage_extract_reasoning.py \
#     --model qwen3.5-397b-a17b \
#     --resume


# .venv/bin/python \
#     pipelines/training/convert_docpilot_logs_to_agentic_training_official.py \
#     --eval-file pipelines/training/filter/2/eval_full.json \
#     --reasoning-overrides pipelines/training/filter/multipage_extract_reasoning_overrides.jsonl \
#     --output pipelines/training/multi_ability_reasoning_fixed.jsonl

if __name__ == "__main__":
    main()
