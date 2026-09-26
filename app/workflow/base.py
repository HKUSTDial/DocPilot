from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import CandidateAnswer, DocVQAState, VerificationResult


class DocumentEnvironment(ABC):
    @abstractmethod
    async def overview(self, document_ids: list[str]) -> dict:
        pass

    @abstractmethod
    async def search(self, query: str, filters: dict | None = None) -> dict:
        pass

    @abstractmethod
    async def fetch_pages(
        self,
        pages: list[int],
        *,
        document_ids: list[str] | None = None,
        with_image: bool = True,
        zoom: float = 1.0,
    ) -> dict:
        pass

    @abstractmethod
    async def open_node(self, node_id: str) -> dict:
        pass

    @abstractmethod
    async def extract_text(self, node_id: str, span: dict | None = None) -> dict:
        pass

    @abstractmethod
    async def extract_table(self, node_id: str, cell_query: dict | None = None) -> dict:
        pass

    @abstractmethod
    async def extract_figure(self, node_id: str, target: dict | None = None) -> dict:
        pass

    @abstractmethod
    async def extract_image_region(
        self, node_id: str, bbox: list[float] | None = None
    ) -> dict:
        pass

    @abstractmethod
    async def crop(self, node_id: str, bbox: list[float] | None = None) -> dict:
        pass

    @abstractmethod
    async def subfigure_split(
        self,
        node_id: str,
        grid_hint: dict | None = None,
    ) -> dict:
        pass

    @abstractmethod
    async def subfigure_crop(
        self,
        node_id: str,
        panel_query: str,
        split_hint: dict | None = None,
    ) -> dict:
        pass

    @abstractmethod
    async def zoom(
        self,
        node_id: str,
        scale: float,
        bbox: list[float] | None = None,
    ) -> dict:
        pass

    @abstractmethod
    async def run_python(self, code: str, inputs: dict | None = None) -> dict:
        pass

    @abstractmethod
    async def verify_answer(
        self, answer: CandidateAnswer, state: DocVQAState
    ) -> VerificationResult:
        pass
