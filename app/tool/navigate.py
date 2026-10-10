from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tiktoken

from ..config import NavigateConfig
from ..llm import LLMClient
from ..multimodal import collect_image_inputs, estimate_image_list_tokens
from ..prompt.navigate import (
    MULTIDOC_NAVIGATE_PAGE_RELEVANCE_SYSTEM_PROMPT,
    NAVIGATE_PAGE_RELEVANCE_SYSTEM_PROMPT,
    SINGLE_DOCUMENT_OVERVIEW_SYSTEM_PROMPT,
)
from ..schema import Action, DocVQAState, EvidenceRecord, Observation
from .base import BaseTool


MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 110_000,
    "gpt-4o-mini": 110_000,
    "gpt-5.4": 32_000,
    "qwen": 32_000,
    "default": 32_000,
}
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def token_length(text: str) -> int:
    return len(encoding.encode(text, disallowed_special=()))


class NavigateTool(BaseTool):
    name = "navigate"
    description = (
        "Inspect page-level text and screenshot references to identify relevant pages. "
        "Returns candidate pages and a step-level reason."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language navigation goal or analyzed question.",
            },
            "pages": {
                "type": "string",
                "description": "Optional page list/ranges to inspect directly, e.g. '1', '3-5', or '1,4'.",
            },
            "force_refresh": {
                "type": "boolean",
                "description": "If true, inspect pages even if they were previously verified.",
            },
            "refresh_reason": {
                "type": "string",
                "description": (
                    "Required when force_refresh=true. Must be one of: "
                    "verification_conflict, missing_critical_evidence, "
                    "ocr_or_parsing_failure, wrong_target_binding, "
                    "new_constraint_added, tooling_reconfiguration."
                ),
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        navigate_config: NavigateConfig | None = None,
        workspace_dir: str | Path | None = None,
    ) -> None:
        navigate_config = navigate_config or NavigateConfig()
        self.llm = llm_client
        self.default_top_k = None if navigate_config.top_k is None else max(1, int(navigate_config.top_k))
        self.default_modal = _normalize_modal(navigate_config.modal)
        self.single_document_overview = bool(navigate_config.overview)
        self.page_thumbnail_size = max(1, int(navigate_config.page_thumbnail_size))
        image_type = getattr(getattr(llm_client, "config", None), "image_type", navigate_config.page_thumbnail_type)
        self.page_thumbnail_type = _normalize_thumbnail_type(image_type)
        self.image_page_set_size = max(1, int(navigate_config.image_page_set_size))
        self.max_tokens = None if navigate_config.max_tokens is None else max(1, int(navigate_config.max_tokens))
        self.thumbnail_dir = Path(workspace_dir or "runs/.workspace").resolve() / "navigate_thumbnails"
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.overview_dir = Path(workspace_dir or "runs/.workspace").resolve() / "document_overviews"
        self.overview_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, action: Action, state: DocVQAState, env) -> Observation:
        if self.llm is None:
            state.budget.search_used += 1
            navigation = state.metadata.setdefault("navigation", {})
            navigation["last_error"] = "missing_llm_client"
            return Observation(
                action_type="navigate",
                success=False,
                message="Navigate requires an LLM client for page-level relevance judgment.",
                payload={"error": "missing_llm_client"},
            )

        query = str(action.args.get("query") or state.question).strip()
        top_k = self.default_top_k
        modal = self.default_modal
        with_image = _modal_uses_image(modal)
        force_refresh_requested = bool(action.args.get("force_refresh", False))
        refresh_reason = str(action.args.get("refresh_reason") or "").strip()
        target_pages = _parse_target_pages(action.args)

        navigation = _navigation_state(state, query)
        refresh_checklist = _evaluate_refresh_checklist(
            state=state,
            requested=force_refresh_requested,
            refresh_reason=refresh_reason,
        )
        force_refresh = bool(refresh_checklist["approved"])
        navigation["last_refresh_checklist"] = refresh_checklist

        overview = state.metadata.get("environment_overview")
        if not overview:
            overview = await env.overview(state.document_ids)
            state.metadata["environment_overview"] = overview

        if len(state.document_ids) > 1:
            return await self._execute_multidoc(
                action=action,
                state=state,
                env=env,
                query=query,
                modal=modal,
                with_image=with_image,
                top_k=top_k,
                navigation=navigation,
                overview=overview,
                refresh_checklist=refresh_checklist,
                force_refresh=force_refresh,
            )

        if self.single_document_overview and str(action.args.get("pages") or "").strip().lower() == "overview":
            return await self._execute_single_document_overview(
                state=state,
                env=env,
                query=query,
                navigation=navigation,
                overview=overview,
                top_k=top_k,
                refresh_checklist=refresh_checklist,
            )

        full_page_pool = _resolve_page_pool(overview)
        page_pool = list(full_page_pool)
        verified_pages = set() if force_refresh or target_pages else _verified_page_keys(navigation)
        verified_page_numbers = set() if force_refresh or target_pages else _verified_page_numbers(navigation)
        if target_pages:
            available_pages = set(full_page_pool)
            page_pool = [page for page in target_pages if page in available_pages]
        else:
            page_pool = [page for page in page_pool if page not in verified_page_numbers]

        visited_sets = set(navigation.get("visited_page_sets", []))
        pending_sets = _dedupe_page_sets(navigation.get("pending_page_sets") or [], excluded=visited_sets)
        if target_pages:
            page_set = page_pool
        elif pending_sets:
            page_set = [int(page) for page in pending_sets.pop(0)]
        else:
            page_set = page_pool
        if not target_pages:
            page_set = [page for page in page_set if page not in verified_page_numbers]

        if not page_set:
            payload = {
                "query": query,
                "modal": modal,
                "target_pages": target_pages,
                "candidate_pages": [],
                "verified_pages": navigation.get("verified_pages", []),
                "need_more_pages": False,
                "reason": "Requested target pages were not available." if target_pages else "No unverified pages remain to inspect.",
                "refresh": refresh_checklist,
            }
            state.metadata["navigation"] = navigation
            state.metadata["candidate_pages"] = navigation.get("candidate_pages", [])
            state.budget.search_used += 1
            return Observation(
                action_type="navigate",
                success=True,
                message=payload["reason"],
                payload=payload,
            )

        page_context = await env.fetch_pages(
            page_set,
            document_ids=state.document_ids,
            with_image=with_image,
        )
        pages = list(page_context.get("pages", []))
        if not pages:
            navigation["pending_page_sets"] = pending_sets
            navigation["last_reason"] = "Requested pages were not available in the environment."
            state.metadata["navigation"] = navigation
            state.metadata["candidate_pages"] = navigation.get("candidate_pages", [])
            state.budget.search_used += 1
            return Observation(
                action_type="navigate",
                success=True,
                message="No page context was available for the requested pages.",
                payload={
                    "mode": "page_level",
                    "query": query,
                    "modal": modal,
                    "target_pages": target_pages,
                    "page_set": page_set,
                    "candidate_pages": [],
                    "verified_pages": navigation.get("verified_pages", []),
                    "pending_page_sets": pending_sets,
                    "confidence": None,
                    "reason": navigation["last_reason"],
                    "need_more_pages": bool(pending_sets),
                    "refresh": refresh_checklist,
                },
            )
        model_window = _model_context_window(self.llm)
        token_budget = int(model_window * 0.78)
        model_name = getattr(self.llm.config, "model", "") or ""
        if modal == "image":
            page_sets = _split_page_sets_by_fixed_size(pages, self.image_page_set_size)
        else:
            page_sets = _split_page_sets_by_token_budget(pages, token_budget, model=model_name)

        if len(page_sets) > 1:
            page_set = page_sets[0]
            pending_sets = _dedupe_page_sets(
                page_sets[1:] + pending_sets,
                excluded=visited_sets | {_compact_pages(page_set)},
            )
            page_context = await env.fetch_pages(
                page_set,
                document_ids=state.document_ids,
                with_image=with_image,
            )
            pages = list(page_context.get("pages", []))
        else:
            page_set = [int(page.get("page")) for page in pages if page.get("page") is not None]

        visited = navigation.setdefault("visited_page_sets", [])
        visited_key = _compact_pages(page_set)
        if visited_key and visited_key not in visited:
            visited.append(visited_key)
        pending_sets = _pending_page_sets_from_remaining(
            full_page_pool=full_page_pool,
            visited_sets=visited,
        )
        navigation["pending_page_sets"] = pending_sets
        revisit_keys = _page_visit_keys_for_prompt(pages, navigation)
        pages = _prepare_pages_for_modal(
            pages,
            modal=modal,
            page_thumbnail_size=self.page_thumbnail_size,
            page_thumbnail_type=self.page_thumbnail_type,
            revisit_page_keys=revisit_keys,
            thumbnail_dir=self.thumbnail_dir,
        )
        _increment_page_visit_counts(navigation, pages)

        prompt_payload = _build_prompt_payload(
            query=query,
            question=state.question,
            state=state,
            navigation=navigation,
            pages=pages,
            modal=modal,
            model_window=model_window,
        )
        prompt_images = collect_image_inputs(prompt_payload.get("pages", []), max_images=14)
        max_tokens = self.max_tokens if self.max_tokens is not None else getattr(self.llm.config, "max_tokens", None)
        self.llm.set_call_source("navigate_page_relevance")
        try:
            llm_payload = self.llm.generate_json(
                NAVIGATE_PAGE_RELEVANCE_SYSTEM_PROMPT,
                json.dumps(prompt_payload, ensure_ascii=False),
                images=prompt_images,
                max_tokens=max_tokens,
            )
        finally:
            self.llm.set_call_source(None)

        candidate_pages = _normalize_page_items(llm_payload.get("candidate_pages"), page_set)
        candidate_pages = _enrich_candidate_pages(candidate_pages, pages, state.document_ids)
        navigation["candidate_pages"] = _merge_page_items(
            navigation.get("candidate_pages", []),
            candidate_pages,
            verified_page_keys=verified_pages,
        )
        navigation["candidate_pages"] = _select_top_k_page_items(navigation["candidate_pages"], top_k)
        navigation["last_reason"] = str(llm_payload.get("reason") or "")
        navigation["last_page_set"] = _compact_pages(page_set)
        navigation["need_more_pages"] = bool(llm_payload.get("need_more_pages") or pending_sets)
        navigation["query"] = query
        state.metadata["navigation"] = navigation
        state.metadata["candidate_pages"] = navigation["candidate_pages"]
        state.budget.search_used += 1

        message = (
            f"Inspected pages {_compact_pages(page_set)} and found "
            f"{len(candidate_pages)} candidate pages."
        )
        if pending_sets:
            message += f" {len(pending_sets)} page set(s) remain unvisited."
        return Observation(
            action_type="navigate",
            success=True,
            message=message,
            payload={
                "mode": "page_level",
                "query": query,
                "modal": modal,
                "target_pages": target_pages,
                "page_set": page_set,
                "candidate_pages": _public_candidate_pages(candidate_pages),
                "verified_pages": navigation.get("verified_pages", []),
                "pending_page_sets": pending_sets,
                "reason": navigation.get("last_reason"),
                "need_more_pages": navigation["need_more_pages"],
                "refresh": refresh_checklist,
                "raw_model_payload": llm_payload,
            },
        )

    async def _execute_multidoc(
        self,
        *,
        action: Action,
        state: DocVQAState,
        env,
        query: str,
        modal: str,
        with_image: bool,
        top_k: int | None,
        navigation: dict[str, Any],
        overview: dict[str, Any],
        refresh_checklist: dict[str, Any],
        force_refresh: bool,
    ) -> Observation:
        raw_pages = action.args.get("pages")
        available = _multidoc_page_pool(overview)
        navigation["pending_page_sets"] = _multidoc_pending_page_sets(
            available, navigation, self.image_page_set_size
        )
        if isinstance(raw_pages, str) and raw_pages.strip().lower() == "overview":
            return await self._execute_multidoc_overview(
                state=state,
                env=env,
                query=query,
                navigation=navigation,
                overview=overview,
                top_k=top_k,
                refresh_checklist=refresh_checklist,
            )

        targets = _parse_multidoc_targets(raw_pages, available)
        if raw_pages not in (None, "") and not targets:
            return self._multidoc_empty_observation(
                state,
                navigation,
                query=query,
                modal=modal,
                reason="Requested multi-document page targets were invalid or unavailable.",
                refresh_checklist=refresh_checklist,
            )

        explicit = bool(targets)
        if not targets:
            active_doc = str(navigation.get("active_document") or "").strip()
            if active_doc not in available:
                return self._multidoc_empty_observation(
                    state,
                    navigation,
                    query=query,
                    modal=modal,
                    reason="No active document is selected; specify doc_id:page-range or use overview.",
                    refresh_checklist=refresh_checklist,
                )
            visited_keys = _multidoc_visited_keys(navigation)
            targets = [
                {"doc_id": active_doc, "page": page}
                for page in available[active_doc]
                if _page_key(active_doc, page) not in visited_keys
            ][: self.image_page_set_size]

        targets = targets[: self.image_page_set_size]
        if explicit and len({item["doc_id"] for item in targets}) == 1 and len(targets) > 1:
            navigation["active_document"] = targets[0]["doc_id"]
        if not targets:
            return self._multidoc_empty_observation(
                state,
                navigation,
                query=query,
                modal=modal,
                reason="No unvisited pages remain in the active document.",
                refresh_checklist=refresh_checklist,
            )

        pages: list[dict[str, Any]] = []
        for doc_id, doc_targets in _group_multidoc_targets(targets).items():
            page_context = await env.fetch_pages(
                doc_targets,
                document_ids=[doc_id],
                with_image=with_image,
            )
            pages.extend(page_context.get("pages", []))
        if not pages:
            return self._multidoc_empty_observation(
                state,
                navigation,
                query=query,
                modal=modal,
                reason="No page context was available for the requested pages.",
                refresh_checklist=refresh_checklist,
            )

        visited_key = _compact_doc_targets(targets)
        visited = navigation.setdefault("visited_page_sets", [])
        if visited_key and visited_key not in visited:
            visited.append(visited_key)
        navigation["pending_page_sets"] = _multidoc_pending_page_sets(
            available, navigation, self.image_page_set_size
        )
        revisit_keys = _page_visit_keys_for_prompt(pages, navigation)
        pages = _prepare_pages_for_modal(
            pages,
            modal=modal,
            page_thumbnail_size=self.page_thumbnail_size,
            page_thumbnail_type=self.page_thumbnail_type,
            revisit_page_keys=revisit_keys,
            thumbnail_dir=self.thumbnail_dir,
        )
        _increment_page_visit_counts(navigation, pages)

        prompt_payload = _build_multidoc_prompt_payload(
            query=query,
            question=state.question,
            state=state,
            navigation=navigation,
            pages=pages,
            modal=modal,
            model_window=_model_context_window(self.llm),
        )
        llm_payload = self._call_navigate_llm(prompt_payload, source="navigate_page_relevance")
        inspected_refs = {_doc_page_ref(page.get("doc_id"), page.get("page")) for page in pages}
        candidate_pages = _normalize_multidoc_page_items(
            llm_payload.get("candidate_pages"), inspected_refs
        )
        verified = set() if force_refresh else _verified_page_keys(navigation)
        navigation["candidate_pages"] = _select_top_k_page_items(
            _merge_page_items(
                navigation.get("candidate_pages", []),
                candidate_pages,
                verified_page_keys=verified,
            ),
            top_k,
        )
        navigation["last_reason"] = str(llm_payload.get("reason") or "")
        navigation["last_page_set"] = visited_key
        active_doc = str(navigation.get("active_document") or "")
        visited_keys = _multidoc_visited_keys(navigation)
        has_remaining = bool(
            active_doc
            and any(
                _page_key(active_doc, page) not in visited_keys
                for page in available.get(active_doc, [])
            )
        )
        navigation["need_more_pages"] = bool(llm_payload.get("need_more_pages") or has_remaining)
        navigation["query"] = query
        state.metadata["navigation"] = navigation
        state.metadata["candidate_pages"] = navigation["candidate_pages"]
        state.budget.search_used += 1
        return Observation(
            action_type="navigate",
            success=True,
            message=f"Inspected pages {visited_key} and found {len(candidate_pages)} candidate pages.",
            payload={
                "mode": "page_level",
                "query": query,
                "modal": modal,
                "target_pages": _targets_to_refs(targets),
                "page_set": _targets_to_refs(targets),
                "candidate_pages": _public_multidoc_candidate_pages(candidate_pages),
                "reason": navigation["last_reason"],
                "need_more_pages": navigation["need_more_pages"],
                "refresh": refresh_checklist,
                "raw_model_payload": llm_payload,
            },
        )

    async def _execute_multidoc_overview(
        self,
        *,
        state: DocVQAState,
        env,
        query: str,
        navigation: dict[str, Any],
        overview: dict[str, Any],
        top_k: int | None,
        refresh_checklist: dict[str, Any],
    ) -> Observation:
        document_overviews = []
        titles = state.metadata.get("document_titles", {})
        for document in overview.get("documents", []):
            doc_id = str(document.get("doc_id") or "").strip()
            pages = [int(page) for page in document.get("pages", [])]
            image_path, rendered_pages = await self._build_document_overview(
                env,
                doc_id=doc_id,
                pages=pages,
                title=str(titles.get(doc_id) or doc_id),
            )
            if image_path:
                document_overviews.append(
                    {
                        "doc_id": doc_id,
                        "title": str(titles.get(doc_id) or doc_id),
                        "pages": rendered_pages,
                        "overview_image_path": image_path,
                    }
                )
        if not document_overviews:
            return self._multidoc_empty_observation(
                state,
                navigation,
                query=query,
                modal="image",
                reason="No document overview images could be generated.",
                refresh_checklist=refresh_checklist,
                success=False,
            )

        prompt_payload = {
            "task": "multi_document_overview_navigation",
            "question": state.question,
            "query": query,
            "modal": "image",
            "documents": document_overviews,
            "page_set": [
                f"{item['doc_id']}:{page}"
                for item in document_overviews
                for page in item["pages"]
            ],
            "pages": [],
            "output_contract": {
                "candidate_pages": ["doc_id:page"],
                "reason": "brief localization reason; do not answer the question",
                "need_more_pages": "boolean",
            },
        }
        llm_payload = self._call_navigate_llm(prompt_payload, source="navigate_document_overview")
        inspected_refs = set(prompt_payload["page_set"])
        candidate_pages = _normalize_multidoc_page_items(
            llm_payload.get("candidate_pages"), inspected_refs
        )
        visited_key = _compact_doc_targets(candidate_pages)
        visited = navigation.setdefault("visited_page_sets", [])
        if visited_key and visited_key not in visited:
            visited.append(visited_key)
        navigation["pending_page_sets"] = _multidoc_pending_page_sets(
            _multidoc_page_pool(overview), navigation, self.image_page_set_size
        )
        navigation["candidate_pages"] = _select_top_k_page_items(
            _merge_page_items(
                navigation.get("candidate_pages", []),
                candidate_pages,
                verified_page_keys=_verified_page_keys(navigation),
            ),
            top_k,
        )
        navigation["last_reason"] = str(llm_payload.get("reason") or "")
        navigation["need_more_pages"] = bool(llm_payload.get("need_more_pages"))
        navigation["query"] = query
        state.metadata["document_overviews"] = document_overviews
        state.metadata["overview"] = True
        state.metadata["navigation"] = navigation
        state.metadata["candidate_pages"] = navigation["candidate_pages"]
        state.budget.search_used += 1
        return Observation(
            action_type="navigate",
            success=True,
            message=f"Inspected {len(document_overviews)} document overview image(s) and found {len(candidate_pages)} candidate pages.",
            payload={
                "mode": "document_overview",
                "query": query,
                "modal": "image",
                "page_set": prompt_payload["page_set"],
                "candidate_pages": _public_multidoc_candidate_pages(candidate_pages),
                "reason": navigation["last_reason"],
                "need_more_pages": navigation["need_more_pages"],
                "refresh": refresh_checklist,
                "raw_model_payload": llm_payload,
            },
        )

    async def _execute_single_document_overview(
        self,
        *,
        state: DocVQAState,
        env,
        query: str,
        navigation: dict[str, Any],
        overview: dict[str, Any],
        top_k: int | None,
        refresh_checklist: dict[str, Any],
    ) -> Observation:
        doc_id = state.document_ids[0]
        document = next(
            (
                item
                for item in overview.get("documents", [])
                if str(item.get("doc_id") or "") == doc_id
            ),
            None,
        )
        pages = [int(page) for page in (document or {}).get("pages", [])]
        image_path, rendered_pages = await self._build_document_overview(
            env,
            doc_id=doc_id,
            pages=pages,
            title=doc_id,
        )
        if not image_path:
            state.budget.search_used += 1
            return Observation(
                action_type="navigate",
                success=False,
                message="No single-document overview image could be generated.",
                payload={"mode": "single_document_overview", "candidate_pages": []},
            )

        prompt_payload = {
            "task": "single_document_overview_navigation",
            "question": state.question,
            "query": query,
            "modal": "image",
            "overview_image_path": image_path,
            "page_set": rendered_pages,
            "pages": [],
            "output_contract": {
                "candidate_pages": [{"page": "int", "confidence": "number or string"}],
                "reason": "brief localization reason; do not answer the question",
                "need_more_pages": "boolean",
            },
        }
        llm_payload = self._call_navigate_llm(
            prompt_payload,
            source="navigate_single_document_overview",
            system_prompt=SINGLE_DOCUMENT_OVERVIEW_SYSTEM_PROMPT,
        )
        candidate_pages = _normalize_page_items(llm_payload.get("candidate_pages"), rendered_pages)
        candidate_pages = _enrich_candidate_pages(
            candidate_pages,
            [{"doc_id": doc_id, "page": page} for page in rendered_pages],
            state.document_ids,
        )
        visited = navigation.setdefault("visited_page_sets", [])
        visited_key = _compact_pages([int(item["page"]) for item in candidate_pages])
        if visited_key and visited_key not in visited:
            visited.append(visited_key)
        pending_sets = _pending_page_sets_from_remaining(
            full_page_pool=rendered_pages,
            visited_sets=visited,
        )
        navigation["pending_page_sets"] = pending_sets
        navigation["candidate_pages"] = _select_top_k_page_items(
            _merge_page_items(
                navigation.get("candidate_pages", []),
                candidate_pages,
                verified_page_keys=_verified_page_keys(navigation),
            ),
            top_k,
        )
        navigation["last_reason"] = str(llm_payload.get("reason") or "")
        navigation["need_more_pages"] = bool(llm_payload.get("need_more_pages") or pending_sets)
        navigation["query"] = query
        state.metadata["overview"] = True
        state.metadata["navigation"] = navigation
        state.metadata["candidate_pages"] = navigation["candidate_pages"]
        state.budget.search_used += 1
        return Observation(
            action_type="navigate",
            success=True,
            message=f"Inspected the document overview and found {len(candidate_pages)} candidate pages.",
            payload={
                "mode": "single_document_overview",
                "query": query,
                "modal": "image",
                "page_set": rendered_pages,
                "candidate_pages": _public_candidate_pages(candidate_pages),
                "reason": navigation["last_reason"],
                "need_more_pages": navigation["need_more_pages"],
                "refresh": refresh_checklist,
                "raw_model_payload": llm_payload,
            },
        )

    async def _build_document_overview(
        self,
        env,
        *,
        doc_id: str,
        pages: list[int],
        title: str,
    ) -> tuple[str | None, list[int]]:
        screenshots: list[tuple[int, str]] = []
        for offset in range(0, len(pages), 32):
            result = await env.fetch_pages(
                pages[offset : offset + 32],
                document_ids=[doc_id],
                with_image=True,
            )
            for item in result.get("pages", []):
                screenshot = item.get("screenshot_path")
                if screenshot:
                    screenshots.append((int(item["page"]), str(screenshot)))
        return (
            _build_document_overview_image(
                screenshots,
                doc_id=doc_id,
                title=title,
                cache_dir=self.overview_dir,
                image_type=self.page_thumbnail_type,
            ),
            [page for page, _ in screenshots],
        )

    def _call_navigate_llm(
        self,
        prompt_payload: dict[str, Any],
        *,
        source: str,
        system_prompt: str = MULTIDOC_NAVIGATE_PAGE_RELEVANCE_SYSTEM_PROMPT,
    ) -> dict[str, Any]:
        prompt_images = collect_image_inputs(
            prompt_payload,
            max_images=max(14, len(prompt_payload.get("documents", []))),
        )
        max_tokens = self.max_tokens if self.max_tokens is not None else getattr(self.llm.config, "max_tokens", None)
        self.llm.set_call_source(source)
        try:
            return self.llm.generate_json(
                system_prompt,
                json.dumps(prompt_payload, ensure_ascii=False),
                images=prompt_images,
                max_tokens=max_tokens,
            )
        finally:
            self.llm.set_call_source(None)

    def _multidoc_empty_observation(
        self,
        state: DocVQAState,
        navigation: dict[str, Any],
        *,
        query: str,
        modal: str,
        reason: str,
        refresh_checklist: dict[str, Any],
        success: bool = True,
    ) -> Observation:
        navigation["last_reason"] = reason
        state.metadata["navigation"] = navigation
        state.metadata["candidate_pages"] = navigation.get("candidate_pages", [])
        state.budget.search_used += 1
        return Observation(
            action_type="navigate",
            success=success,
            message=reason,
            payload={
                "mode": "page_level",
                "query": query,
                "modal": modal,
                "candidate_pages": [],
                "reason": reason,
                "need_more_pages": False,
                "refresh": refresh_checklist,
            },
        )


def _multidoc_page_pool(overview: dict[str, Any]) -> dict[str, list[int]]:
    pool: dict[str, list[int]] = {}
    for document in overview.get("documents", []):
        doc_id = str(document.get("doc_id") or "").strip()
        if not doc_id:
            continue
        pages: set[int] = set()
        for raw_page in document.get("pages", []):
            try:
                pages.add(int(raw_page))
            except (TypeError, ValueError):
                continue
        pool[doc_id] = sorted(pages)
    return pool


def _parse_multidoc_targets(raw_pages: Any, available: dict[str, list[int]]) -> list[dict[str, Any]]:
    if not isinstance(raw_pages, str) or not raw_pages.strip():
        return []
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_target in re.split(r"[,;]", raw_pages):
        target = raw_target.strip()
        if ":" not in target:
            continue
        doc_id, page_text = target.rsplit(":", 1)
        doc_id = doc_id.strip()
        if doc_id not in available:
            continue
        for page in _parse_pages(page_text):
            key = _page_key(doc_id, page)
            if page in available[doc_id] and key not in seen:
                seen.add(key)
                targets.append({"doc_id": doc_id, "page": page})
    return targets


def _group_multidoc_targets(targets: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for item in targets:
        grouped.setdefault(str(item["doc_id"]), []).append(int(item["page"]))
    return grouped


def _targets_to_refs(targets: list[dict[str, Any]]) -> list[str]:
    return [_doc_page_ref(item.get("doc_id"), item.get("page")) for item in targets]


def _doc_page_ref(doc_id: Any, page: Any) -> str:
    try:
        page = int(page)
    except (TypeError, ValueError):
        return ""
    return f"{str(doc_id or '').strip()}:{page}"


def _compact_doc_targets(targets: list[dict[str, Any]]) -> str:
    return ",".join(
        f"{doc_id}:{_compact_pages(pages)}"
        for doc_id, pages in _group_multidoc_targets(targets).items()
    )


def _multidoc_visited_keys(navigation: dict[str, Any]) -> set[str]:
    visited: set[str] = set()
    for item in navigation.get("visited_page_sets", []):
        if not isinstance(item, str):
            continue
        for target in item.split(","):
            if ":" not in target:
                continue
            doc_id, page_text = target.rsplit(":", 1)
            for page in _parse_pages(page_text):
                visited.add(_page_key(doc_id, page))
    return visited


def _multidoc_pending_page_sets(
    available: dict[str, list[int]], navigation: dict[str, Any], page_set_size: int
) -> list[str]:
    visited = _multidoc_visited_keys(navigation)
    pending: list[str] = []
    for doc_id, pages in available.items():
        remaining = [page for page in pages if _page_key(doc_id, page) not in visited]
        page_set: list[int] = []
        for page in remaining:
            if page_set and (page != page_set[-1] + 1 or len(page_set) == page_set_size):
                pending.append(f"{doc_id}:{_compact_pages(page_set)}")
                page_set = []
            page_set.append(page)
        if page_set:
            pending.append(f"{doc_id}:{_compact_pages(page_set)}")
    return pending


def _build_multidoc_prompt_payload(
    *,
    query: str,
    question: str,
    state: DocVQAState,
    navigation: dict[str, Any],
    pages: list[dict[str, Any]],
    modal: str,
    model_window: int,
) -> dict[str, Any]:
    return {
        "task": "multi_document_page_navigation",
        "question": question,
        "query": query,
        "modal": modal,
        "model_context_window": model_window,
        "current_evidence": [_evidence_summary(evidence) for evidence in state.evidence_store.values()],
        "understanding": state.metadata.get("understanding", {}),
        "candidate_pages_so_far": [],
        "page_set": [_doc_page_ref(page.get("doc_id"), page.get("page")) for page in pages],
        "pages": [_trim_page_context(page) for page in pages],
        "output_contract": {
            "candidate_pages": ["doc_id:page"],
            "reason": "brief step-level explanation",
            "need_more_pages": "boolean",
        },
    }


def _normalize_multidoc_page_items(raw_items: Any, inspected_refs: set[str]) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        confidence = None
        if isinstance(item, str):
            ref = item.strip()
        elif isinstance(item, dict):
            confidence = item.get("confidence")
            page_value = item.get("page")
            if isinstance(page_value, str) and ":" in page_value:
                ref = page_value.strip()
            else:
                ref = _doc_page_ref(item.get("doc_id"), page_value)
        else:
            continue
        if ref not in inspected_refs or ref in seen or ":" not in ref:
            continue
        doc_id, page_text = ref.rsplit(":", 1)
        try:
            page = int(page_text)
        except ValueError:
            continue
        seen.add(ref)
        normalized.append({"doc_id": doc_id, "page": page, "confidence": confidence})
    return normalized


def _public_multidoc_candidate_pages(items: list[dict[str, Any]]) -> list[str]:
    return [_doc_page_ref(item.get("doc_id"), item.get("page")) for item in items]


def _build_document_overview_image(
    screenshots: list[tuple[int, str]],
    *,
    doc_id: str,
    title: str,
    cache_dir: Path,
    image_type: str,
) -> str | None:
    if not screenshots:
        return None
    cache_parts = []
    for page, image_path in screenshots:
        path = Path(image_path).expanduser().resolve()
        try:
            stat = path.stat()
        except OSError:
            continue
        cache_parts.append(f"{page}:{path}:{stat.st_mtime_ns}:{stat.st_size}")
    if not cache_parts:
        return None
    image_type = _normalize_thumbnail_type(image_type)
    extension = ".jpg" if image_type == "jpeg" else ".png"
    cache_key = hashlib.sha1(
        f"{doc_id}|{title}|{'|'.join(cache_parts)}".encode("utf-8")
    ).hexdigest()[:16]
    safe_doc_id = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_id).strip("_") or "document"
    output_path = cache_dir / f"{safe_doc_id}.{cache_key}.overview{extension}"
    if output_path.exists():
        return str(output_path)

    thumb_size = 160
    thumbnails = []
    for page, image_path in sorted(screenshots):
        thumb_path = _resize_page_screenshot_to_thumbnail(
            image_path,
            size=thumb_size,
            image_type=image_type,
            page=page,
            cache_dir=cache_dir,
        )
        thumb = cv2.imread(str(thumb_path), cv2.IMREAD_COLOR) if thumb_path else None
        if thumb is not None:
            thumbnails.append(thumb)
    if not thumbnails:
        return None

    columns = min(8, len(thumbnails))
    rows = (len(thumbnails) + columns - 1) // columns
    header_height = 44
    canvas = np.full((header_height + rows * thumb_size, columns * thumb_size, 3), 255, dtype=np.uint8)
    header = f"{title} [{doc_id}]".encode("ascii", errors="replace").decode("ascii")
    cv2.putText(canvas, header[:100], (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    for index, thumb in enumerate(thumbnails):
        row, column = divmod(index, columns)
        y = header_height + row * thumb_size
        x = column * thumb_size
        canvas[y : y + thumb_size, x : x + thumb_size] = thumb
    write_params = [cv2.IMWRITE_JPEG_QUALITY, 80] if image_type == "jpeg" else []
    return str(output_path) if cv2.imwrite(str(output_path), canvas, write_params) else None

def _navigation_state(state: DocVQAState, query: str) -> dict[str, Any]:
    navigation = state.metadata.setdefault("navigation", {})
    navigation.setdefault("query", query)
    navigation.setdefault("candidate_pages", state.metadata.get("candidate_pages", []))
    navigation.setdefault("verified_pages", [])
    navigation.setdefault("pending_page_sets", [])
    navigation.setdefault("visited_page_sets", [])
    navigation.setdefault("page_visit_counts", {})
    return navigation


def _normalize_modal(value: Any) -> str:
    text = str(value or "image").strip().lower()
    text = text.replace(" ", "").replace("_", "").replace("-", "")
    if text in {"text", "文本"}:
        return "text"
    if text in {"image", "images", "视觉", "图像", "图片"}:
        return "image"
    if text in {
        "[text,image]",
        "text,image",
        "image,text",
        "text+image",
        "image+text",
        "both",
        "all",
        "textandimage",
        "imageandtext",
    }:
        return "text,image"
    return "image"


def _modal_uses_image(modal: str) -> bool:
    return modal in {"image", "text,image", "[text,image]"}


def _modal_uses_text(modal: str) -> bool:
    return modal in {"text", "text,image", "[text,image]"}


def _resolve_page_pool(overview: dict[str, Any]) -> list[int]:
    pages: set[int] = set()
    for doc in overview.get("documents", []):
        for page in doc.get("pages", []):
            try:
                pages.add(int(page))
            except (TypeError, ValueError):
                continue
    return sorted(pages)


def _parse_pages(value: str) -> list[int]:
    pages: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
            continue
        try:
            pages.add(int(item))
        except ValueError:
            continue
    return sorted(pages)


def _parse_target_pages(args: dict[str, Any]) -> list[int]:
    raw_pages = args.get("pages")
    if isinstance(raw_pages, str) and raw_pages.strip():
        return _parse_pages(raw_pages)
    if isinstance(raw_pages, int):
        return [raw_pages]
    if isinstance(raw_pages, list):
        pages: set[int] = set()
        for item in raw_pages:
            try:
                pages.add(int(item))
            except (TypeError, ValueError):
                continue
        if pages:
            return sorted(pages)

    return []


def _model_context_window(llm: LLMClient) -> int:
    model = (getattr(llm.config, "model", "") or "").lower()
    for key, window in MODEL_CONTEXT_WINDOWS.items():
        if key != "default" and key in model:
            return window
    return MODEL_CONTEXT_WINDOWS["default"]


def _split_page_sets_by_token_budget(
    pages: list[dict[str, Any]],
    token_budget: int,
    *,
    model: str = "",
) -> list[list[int]]:
    page_sets: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    usable_budget = max(4_000, token_budget - 3_000)
    for page in pages:
        page_num = page.get("page")
        if page_num is None:
            continue
        page_tokens = _estimate_page_tokens(page, model=model)
        if current and current_tokens + page_tokens > usable_budget:
            page_sets.append(current)
            current = []
            current_tokens = 0
        current.append(int(page_num))
        current_tokens += page_tokens
    if current:
        page_sets.append(current)
    return page_sets or [[]]


def _split_page_sets_by_fixed_size(pages: list[dict[str, Any]], set_size: int) -> list[list[int]]:
    size = max(1, int(set_size))
    page_nums = [int(page.get("page")) for page in pages if page.get("page") is not None]
    page_sets: list[list[int]] = []
    for i in range(0, len(page_nums), size):
        chunk = page_nums[i : i + size]
        if chunk:
            page_sets.append(chunk)
    return page_sets or [[]]


def _estimate_page_tokens(page: dict[str, Any], *, model: str = "") -> int:
    trimmed = _trim_page_context(page)
    text = json.dumps(trimmed, ensure_ascii=False)
    images = collect_image_inputs(trimmed)
    image_cost = estimate_image_list_tokens(images, model=model)
    return max(250, token_length(text) + image_cost)


def _build_prompt_payload(
    *,
    query: str,
    question: str,
    state: DocVQAState,
    navigation: dict[str, Any],
    pages: list[dict[str, Any]],
    modal: str,
    model_window: int,
) -> dict[str, Any]:
    # NOTE: Do not pass historical candidate pages into navigate LLM input.
    # verified_page_keys = _verified_page_keys(navigation)
    # candidate_pages_so_far = [
    #     item
    #     for item in navigation.get("candidate_pages", [])
    #     if _page_key(item.get("doc_id"), item.get("page")) not in verified_page_keys
    # ]
    return {
        "task": "page_level_navigation",
        "question": question,
        "query": query,
        "modal": modal,
        "model_context_window": model_window,
        "current_evidence": [_evidence_summary(evidence) for evidence in state.evidence_store.values()],
        "understanding": state.metadata.get("understanding", {}),
        "candidate_pages_so_far": [],
        "verified_pages": navigation.get("verified_pages", []),
        "page_set": [page.get("page") for page in pages],
        "pages": [_trim_page_context(page) for page in pages],
        "output_contract": {
            "candidate_pages": [
                {
                    "page": "int",
                    "confidence": "number or string, no threshold is applied",
                }
            ],
            "reason": "brief step-level explanation for why these pages were selected",
            "need_more_pages": "boolean",
        },
    }


def _trim_page_context(page: dict[str, Any]) -> dict[str, Any]:
    trimmed = dict(page)
    trimmed["text"] = str(trimmed.get("text") or "")
    nodes = []
    for node in trimmed.get("nodes", []) or []:
        item = dict(node)
        item["text"] = str(item.get("text") or "")
        nodes.append(item)
    trimmed["nodes"] = nodes[:60]
    trimmed["figures"] = (trimmed.get("figures", []) or [])[:20]
    trimmed["tables"] = (trimmed.get("tables", []) or [])[:20]
    return trimmed


def _prepare_pages_for_modal(
    pages: list[dict[str, Any]],
    *,
    modal: str,
    page_thumbnail_size: int,
    page_thumbnail_type: str,
    revisit_page_keys: set[str] | None = None,
    thumbnail_dir: Path | None = None,
) -> list[dict[str, Any]]:
    use_text = _modal_uses_text(modal)
    use_image = _modal_uses_image(modal)
    revisit_page_keys = revisit_page_keys or set()
    prepared = []
    for page in pages:
        item = dict(page)
        if not use_text:
            item["text"] = ""
            item["nodes"] = []
            item["figures"] = []
            item["tables"] = []
        else:
            item["nodes"] = _strip_nested_image_refs(item.get("nodes", []))
            item["figures"] = _strip_nested_image_refs(item.get("figures", []))
            item["tables"] = _strip_nested_image_refs(item.get("tables", []))
        if use_image:
            screenshot_path = item.get("screenshot_path")
            thumbnail_path = _resize_page_screenshot_to_thumbnail(
                str(screenshot_path),
                size=page_thumbnail_size,
                image_type=page_thumbnail_type,
                page=item.get("page"),
                cache_dir=thumbnail_dir,
            ) if screenshot_path else None
            if thumbnail_path is not None:
                item["screenshot_path"] = thumbnail_path
                item["thumbnail_size"] = [page_thumbnail_size, page_thumbnail_size]
            page_key = _page_key(item.get("doc_id"), item.get("page"))
            if screenshot_path and page_key in revisit_page_keys:
                item["grid_crops"] = _build_page_grid_crops(
                    str(screenshot_path),
                    page=item.get("page"),
                    cache_dir=thumbnail_dir,
                )
        else:
            item.pop("screenshot_path", None)
        prepared.append(item)
    return prepared


def _strip_nested_image_refs(items: Any) -> list[Any]:
    if not isinstance(items, list):
        return []
    stripped = []
    for item in items:
        if not isinstance(item, dict):
            stripped.append(item)
            continue
        cleaned = dict(item)
        for key in ("image_path", "img_path", "page_image_path", "screenshot_path", "crop_path", "zoom_path"):
            cleaned.pop(key, None)
        stripped.append(cleaned)
    return stripped


def _resize_page_screenshot_to_thumbnail(
    image_path: str,
    *,
    size: int,
    image_type: str = "png",
    page: Any = None,
    cache_dir: Path | None = None,
) -> str | None:
    try:
        size = max(1, int(size))
        image_type = _normalize_thumbnail_type(image_type)
        extension = ".jpg" if image_type == "jpeg" else ".png"
        page_label = _page_label(page)
        src = Path(image_path).expanduser()
        if not src.is_absolute():
            src = src.resolve()
        if not src.exists() or not src.is_file():
            return None
        stat = src.stat()
        cache_key = hashlib.sha1(
            f"{src}:{stat.st_mtime_ns}:{stat.st_size}:{size}:{image_type}:{page_label}".encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = cache_dir or Path("runs/.workspace/navigate_thumbnails").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / f"{src.stem}.{cache_key}.thumb{extension}"
        if dst.exists():
            return str(dst)

        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return None

        scale = min(size / width, size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

        top = (size - resized_height) // 2
        bottom = size - resized_height - top
        left = (size - resized_width) // 2
        right = size - resized_width - left
        thumbnail = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        if page_label:
            _annotate_thumbnail_page_label(thumbnail, page_label)
        write_params = [cv2.IMWRITE_JPEG_QUALITY, 75] if image_type == "jpeg" else []
        if not cv2.imwrite(str(dst), thumbnail, write_params):
            return None
        return str(dst)
    except Exception:
        return None


def _normalize_thumbnail_type(value: Any) -> str:
    text = str(value or "png").strip().lower()
    if text in {"jpg", "jpeg", "jepg"}:
        return "jpeg"
    return "png"


def _page_label(page: Any) -> str:
    try:
        return f"Page {int(page)}"
    except (TypeError, ValueError):
        return ""


def _page_visit_keys_for_prompt(pages: list[dict[str, Any]], navigation: dict[str, Any]) -> set[str]:
    visit_counts = navigation.get("page_visit_counts", {})
    if not isinstance(visit_counts, dict):
        return set()
    revisit_keys: set[str] = set()
    for page in pages:
        key = _page_key(page.get("doc_id"), page.get("page"))
        try:
            count = int(visit_counts.get(key, 0))
        except (TypeError, ValueError):
            count = 0
        if count >= 1:
            revisit_keys.add(key)
    return revisit_keys


def _increment_page_visit_counts(navigation: dict[str, Any], pages: list[dict[str, Any]]) -> None:
    visit_counts = navigation.setdefault("page_visit_counts", {})
    if not isinstance(visit_counts, dict):
        visit_counts = {}
        navigation["page_visit_counts"] = visit_counts
    for page in pages:
        key = _page_key(page.get("doc_id"), page.get("page"))
        try:
            current = int(visit_counts.get(key, 0))
        except (TypeError, ValueError):
            current = 0
        visit_counts[key] = current + 1


def _build_page_grid_crops(image_path: str, *, page: Any, cache_dir: Path | None = None) -> list[dict[str, Any]]:
    try:
        src = Path(image_path).expanduser()
        if not src.is_absolute():
            src = src.resolve()
        if not src.exists() or not src.is_file():
            return []
        stat = src.stat()
        page_label = _page_label(page)
        cache_key = hashlib.sha1(
            f"{src}:{stat.st_mtime_ns}:{stat.st_size}:grid2x2:{page_label}".encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = cache_dir or Path("runs/.workspace/navigate_thumbnails").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        if height <= 1 or width <= 1:
            return []

        mid_x = max(1, width // 2)
        mid_y = max(1, height // 2)
        crops = [
            ("top_left", (0, 0, mid_x, mid_y)),
            ("top_right", (mid_x, 0, width, mid_y)),
            ("bottom_left", (0, mid_y, mid_x, height)),
            ("bottom_right", (mid_x, mid_y, width, height)),
        ]
        outputs: list[dict[str, Any]] = []
        for name, (x1, y1, x2, y2) in crops:
            if x2 <= x1 or y2 <= y1:
                continue
            dst = cache_dir / f"{src.stem}.{cache_key}.{name}.png"
            if not dst.exists():
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                if not cv2.imwrite(str(dst), crop):
                    continue
            outputs.append(
                {
                    "view": name,
                    "image_path": str(dst),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return outputs
    except Exception:
        return []


def _annotate_thumbnail_page_label(image: Any, label: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, min(0.8, image.shape[1] / 420))
    thickness = max(1, int(round(image.shape[1] / 220)))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    padding_x = max(4, int(round(image.shape[1] * 0.025)))
    padding_y = max(4, int(round(image.shape[0] * 0.02)))
    x1, y1 = 0, 0
    x2 = min(image.shape[1], text_width + padding_x * 2)
    y2 = min(image.shape[0], text_height + baseline + padding_y * 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)
    cv2.putText(
        image,
        label,
        (padding_x, padding_y + text_height),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _evidence_summary(evidence: EvidenceRecord) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "node_id": evidence.node_id,
        "page": evidence.page,
        "modality": evidence.modality,
        "content_preview": str(evidence.content),
        "provenance": evidence.provenance,
    }


def _normalize_page_items(raw_items: Any, inspected_pages: list[int]) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    inspected = set(int(page) for page in inspected_pages)
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            continue
        if inspected and page not in inspected:
            continue
        normalized_item = {
            "page": page,
            "confidence": item.get("confidence"),
        }
        normalized.append(normalized_item)
    return normalized


def _enrich_candidate_pages(
    candidate_pages: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    document_ids: list[str],
) -> list[dict[str, Any]]:
    page_to_doc_ids: dict[int, set[str]] = {}
    for page in pages:
        try:
            page_num = int(page.get("page"))
        except (TypeError, ValueError):
            continue
        doc_id = str(page.get("doc_id") or "")
        if doc_id:
            page_to_doc_ids.setdefault(page_num, set()).add(doc_id)

    enriched: list[dict[str, Any]] = []
    for item in candidate_pages:
        enriched_item = dict(item)
        page_doc_ids = page_to_doc_ids.get(int(item["page"]), set())
        if len(page_doc_ids) == 1:
            enriched_item["doc_id"] = next(iter(page_doc_ids))
        elif len(document_ids) == 1:
            enriched_item["doc_id"] = document_ids[0]
        enriched.append(enriched_item)
    return enriched


def _public_candidate_pages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_items: list[dict[str, Any]] = []
    for item in items:
        public_items.append(
            {
                "page": item.get("page"),
                "confidence": item.get("confidence"),
            }
        )
    return public_items


def _merge_page_items(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    verified_page_keys: set[str],
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in existing:
        key = _page_key(item.get("doc_id"), item.get("page"))
        if key not in verified_page_keys:
            by_key[key] = item
    for item in incoming:
        key = _page_key(item.get("doc_id"), item.get("page"))
        if key not in verified_page_keys:
            by_key[key] = {**by_key.get(key, {}), **item}
    return sorted(by_key.values(), key=_candidate_priority)


def _select_top_k_page_items(items: list[dict[str, Any]], top_k: int | None) -> list[dict[str, Any]]:
    if top_k is None:
        return sorted(items, key=_candidate_priority)
    k = max(1, int(top_k))
    return sorted(items, key=_candidate_priority)[:k]


def _verified_page_keys(navigation: dict[str, Any]) -> set[str]:
    return {_page_key(item.get("doc_id"), item.get("page")) for item in navigation.get("verified_pages", []) if isinstance(item, dict)}


def _evaluate_refresh_checklist(
    *,
    state: DocVQAState,
    requested: bool,
    refresh_reason: str,
) -> dict[str, Any]:
    checklist_items = {
        "verification_conflict": _check_verification_conflict(state),
        "missing_critical_evidence": _check_missing_critical_evidence(state),
        "ocr_or_parsing_failure": _check_ocr_or_parsing_failure(state),
        "wrong_target_binding": _check_wrong_target_binding(state),
        "new_constraint_added": _check_new_constraint_added(state),
        "tooling_reconfiguration": _check_tooling_reconfiguration(state),
    }
    allowed_reasons = set(checklist_items.keys())
    normalized_reason = refresh_reason.strip()
    approved = False
    message = "force_refresh not requested."
    if requested:
        if not normalized_reason:
            message = "force_refresh denied: missing refresh_reason."
        elif normalized_reason not in allowed_reasons:
            message = "force_refresh denied: refresh_reason not in checklist."
        elif not checklist_items[normalized_reason]:
            message = f"force_refresh denied: checklist condition '{normalized_reason}' not met."
        else:
            approved = True
            message = f"force_refresh approved with reason '{normalized_reason}'."
    return {
        "requested": requested,
        "approved": approved,
        "refresh_reason": normalized_reason or None,
        "checklist": checklist_items,
        "message": message,
    }


def _check_verification_conflict(state: DocVQAState) -> bool:
    verification = state.verification
    return bool(verification is not None and not verification.accepted and verification.issues)


def _check_missing_critical_evidence(state: DocVQAState) -> bool:
    verification = state.verification
    return bool(
        verification is not None
        and not verification.accepted
        and verification.missing_evidence
    )


def _check_ocr_or_parsing_failure(state: DocVQAState) -> bool:
    for item in reversed(state.history[-8:]):
        action = item.get("action", {})
        observation = item.get("observation", {})
        if action.get("action_type") not in {"extract", "answer"}:
            continue
        if not observation.get("success", False):
            return True
        payload = observation.get("payload", {})
        evidence = payload.get("evidence", {})
        content = evidence.get("content")
        if isinstance(content, str) and not content.strip():
            return True
    return False


def _check_wrong_target_binding(state: DocVQAState) -> bool:
    for item in reversed(state.history[-12:]):
        action = item.get("action", {})
        observation = item.get("observation", {})
        if action.get("action_type") != "extract":
            continue
        args = action.get("args", {})
        target = args.get("target", {})
        if not isinstance(target, dict) or target.get("page") is None:
            continue
        evidence = (observation.get("payload") or {}).get("evidence") or {}
        observed_page = evidence.get("page")
        if observed_page is None:
            continue
        try:
            target_page = int(target.get("page"))
            extracted_page = int(observed_page)
        except (TypeError, ValueError):
            continue
        if target_page != extracted_page:
            return True
    return False


def _check_new_constraint_added(state: DocVQAState) -> bool:
    if bool(state.metadata.get("constraint_revision")):
        return True
    verification = state.verification
    if verification is None:
        return False
    text = " ".join(str(issue) for issue in verification.issues).lower()
    return any(term in text for term in ["only", "instead", "constraint", "scope"])


def _check_tooling_reconfiguration(state: DocVQAState) -> bool:
    return bool(state.metadata.get("tooling_reconfigured"))


def _candidate_priority(item: dict[str, Any]) -> tuple[float, str, int]:
    return (
        -_confidence_value(item.get("confidence")),
        str(item.get("doc_id") or ""),
        _page_number(item.get("page")),
    )


def _confidence_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            parsed = float(match.group(0))
            return parsed / 100.0 if parsed > 1.0 and parsed <= 100.0 else parsed
    return -1.0


def _page_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pending_page_sets_from_remaining(
    *,
    full_page_pool: list[int],
    visited_sets: list[Any],
) -> list[list[int]]:
    visited_pages: set[int] = set()
    for item in visited_sets or []:
        visited_pages.update(_normalize_page_set(item))
    remaining = [page for page in full_page_pool if page not in visited_pages]
    return _dedupe_page_sets([remaining])


def _dedupe_page_sets(
    page_sets: list[Any],
    *,
    excluded: set[str] | None = None,
) -> list[list[int]]:
    excluded = excluded or set()
    seen: set[str] = set()
    deduped: list[list[int]] = []
    for raw_set in page_sets:
        page_set = _normalize_page_set(raw_set)
        key = _compact_pages(page_set)
        if not key or key in excluded or key in seen:
            continue
        seen.add(key)
        deduped.append(page_set)
    return deduped


def _normalize_page_set(raw_set: Any) -> list[int]:
    if isinstance(raw_set, str):
        return _parse_pages(raw_set)
    if isinstance(raw_set, (list, tuple, set)):
        pages: list[int] = []
        for page in raw_set:
            try:
                pages.append(int(page))
            except (TypeError, ValueError):
                continue
        return sorted(set(pages))
    try:
        return [int(raw_set)]
    except (TypeError, ValueError):
        return []


def _verified_page_numbers(navigation: dict[str, Any]) -> set[int]:
    numbers: set[int] = set()
    for item in navigation.get("verified_pages", []):
        if not isinstance(item, dict):
            continue
        try:
            numbers.add(int(item.get("page")))
        except (TypeError, ValueError):
            continue
    return numbers


def _page_key(doc_id: Any, page: Any) -> str:
    try:
        page_int = int(page)
    except (TypeError, ValueError):
        page_int = -1
    return f"{doc_id or '*'}::{page_int}"


def _compact_pages(pages: list[int]) -> str:
    sorted_pages = sorted({int(page) for page in pages})
    if not sorted_pages:
        return ""
    ranges: list[str] = []
    start = prev = sorted_pages[0]
    for page in sorted_pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = page
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)
