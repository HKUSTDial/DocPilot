from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal


AgentRole = Literal[
    "analyze",
    "navigate",
    "extract",
    "focus",
    "solver",
    "execute",
    "answer",
    "finish",
]

NodeType = Literal[
    "document",
    "section",
    "paragraph",
    "table",
    "figure",
    "chart",
    "formula",
    "footnote",
    "cell",
    "region",
]

Modality = Literal["text", "table", "chart", "image", "mixed"]
EvidenceState = Literal["in_progress", "pending", "finish"]

ActionType = Literal[
    "analyze",
    "navigate",
    "extract",
    "focus",
    "solver",
    "execute",
    "answer",
    "finish",
]

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class DocumentNode:
    node_id: str
    doc_id: str
    page: int
    node_type: NodeType
    modality: Modality
    title: str = ""
    text: str = ""
    summary: str = ""
    content: Any = None
    bbox: list[float] | None = None
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [self.title, self.summary, self.text]
        if isinstance(self.content, str):
            parts.append(self.content)
        elif isinstance(self.content, dict):
            parts.extend(str(v) for v in self.content.values())
        elif isinstance(self.content, list):
            parts.extend(str(v) for v in self.content)
        return "\n".join(part for part in parts if part).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRecord:
    evidence_id: str
    node_id: str
    page: int
    modality: Modality
    content: Any
    provenance: dict[str, Any]
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetState:
    max_steps: int = 12
    step_used: int = 0
    max_search_calls: int | None = None
    search_used: int = 0
    max_extract_calls: int | None = None
    extract_used: int = 0
    max_compute_calls: int | None = None
    compute_used: int = 0
    image_type: str = "png"
    disabled_tools: list[str] = field(default_factory=list)

    def can_search(self) -> bool:
        return self.max_search_calls is None or self.search_used < self.max_search_calls

    def can_extract(self) -> bool:
        return self.max_extract_calls is None or self.extract_used < self.max_extract_calls

    def can_compute(self) -> bool:
        return self.max_compute_calls is None or self.compute_used < self.max_compute_calls


@dataclass
class CandidateAnswer:
    answer: str
    rationale: str | None = None
    supporting_evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    accepted: bool
    issues: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    suggestion: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Action:
    action_type: ActionType
    agent_role: AgentRole
    intent: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    action_type: ActionType
    success: bool
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Message:
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    @classmethod
    def system_message(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user_message(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant_message(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)

    @classmethod
    def tool_message(cls, content: str, name: str | None = None) -> "Message":
        return cls(role="tool", content=content, name=name)

    def to_dict(self) -> dict[str, Any]:
        payload = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass
class Memory:
    messages: list[Message] = field(default_factory=list)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.messages]


@dataclass
class DocVQAState:
    question: str
    document_ids: list[str]
    current_plan: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    evidence_store: dict[str, EvidenceRecord] = field(default_factory=dict)
    evidence_state: EvidenceState = "in_progress"
    candidate_answer: CandidateAnswer | None = None
    verification: VerificationResult | None = None
    budget: BudgetState = field(default_factory=BudgetState)
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_history(self, action: Action, observation: Observation) -> None:
        self.history.append(
            {
                "action": action.to_dict(),
                "observation": observation.to_dict(),
            }
        )
        self.budget.step_used += 1
        self.sync_evidence_state()

    def add_evidence(self, evidence: EvidenceRecord) -> EvidenceRecord:
        fingerprint = _evidence_fingerprint(evidence)
        for existing in self.evidence_store.values():
            if _evidence_fingerprint(existing) == fingerprint:
                return existing
        self.evidence_store[evidence.evidence_id] = evidence
        self.sync_evidence_state()
        return evidence

    def sync_evidence_state(self) -> EvidenceState:
        if self.done or (self.verification is not None and self.verification.accepted):
            self.evidence_state = "finish"
            return self.evidence_state

        navigation = self.metadata.get("navigation", {}) if isinstance(self.metadata, dict) else {}
        need_more_pages = bool(navigation.get("need_more_pages"))
        pending_page_sets = navigation.get("pending_page_sets") or []
        has_unfinished_navigation = need_more_pages or bool(pending_page_sets)

        if has_unfinished_navigation:
            self.evidence_state = "in_progress"
        else:
            self.evidence_state = "pending"
        return self.evidence_state

    def last_observation(self, action_type: ActionType | None = None) -> Observation | None:
        for item in reversed(self.history):
            observation = item.get("observation")
            if not isinstance(observation, dict):
                continue
            if action_type is not None and observation.get("action_type") != action_type:
                continue
            return Observation(
                action_type=observation["action_type"],
                success=bool(observation["success"]),
                message=str(observation["message"]),
                payload=observation.get("payload", {}) or {},
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "document_ids": self.document_ids,
            "current_plan": self.current_plan,
            "history": self.history,
            "evidence_store": {
                key: value.to_dict() for key, value in self.evidence_store.items()
            },
            "evidence_state": self.evidence_state,
            "candidate_answer": (
                self.candidate_answer.to_dict() if self.candidate_answer else None
            ),
            "verification": (
                self.verification.to_dict() if self.verification else None
            ),
            "budget": asdict(self.budget),
            "done": self.done,
            "metadata": self.metadata,
        }


def _evidence_fingerprint(evidence: EvidenceRecord) -> tuple[Any, ...]:
    return (
        evidence.node_id,
        evidence.page,
        evidence.modality,
        _stable_serialize(evidence.content),
    )


def _stable_serialize(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)
