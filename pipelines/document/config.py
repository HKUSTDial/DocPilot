from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import read_json


@dataclass
class IncludeSpec:
    dataset: str
    doc_id: str


@dataclass
class DatasetConfig:
    mmdocrag_dir: Path = Path("data/DocVQA/MMDocRAG")
    mmdocrag_sample_files: list[Path] = field(default_factory=lambda: [
        Path("data/DocVQA/MMDocRAG/dev_20.jsonl"),
        Path("data/DocVQA/MMDocRAG/dev_15.jsonl"),
    ])
    longdocurl_dir: Path = Path("data/DocVQA/LongDocURL")
    longdocurl_sample_jsonl: Path = Path("./data/datasets/LongDocURL/LongDocURL_public_multi_evidence_sources.jsonl")
    mmlongbench_dir: Path = Path("data/DocVQA/MMLongBench-Doc")
    mmlongbench_documents_dir: Path = Path("data/DocVQA/MMLongBench-Doc/data/documents")
    mmlongbench_sample_file: Path = Path("data/DocVQA/MMLongBench-Doc/data/samples.json")
    madqa_dir: Path = Path("data/DocVQA/MADQA/dataset")
    madqa_pdf_dir: Path = Path("data/DocVQA/MADQA/dataset/data/pdfs")
    madqa_sample_file: Path = Path("data/DocVQA/MADQA/dataset/data/dev.json")
    dude_dir: Path = Path("data/DocVQA/DUDE")
    dude_pdf_dir: Path = Path("data/DocVQA/DUDE/extracted/DUDE_train-val-test_binaries/PDF/train")
    dude_sample_file: Path = Path("./data/datasets/DUDE/calculation.json")
    finragbench_dir: Path = Path("data/DocVQA/FinRAGBench-V")
    finragbench_image_dir: Path = Path("data/DocVQA/FinRAGBench-V/data/corpus/en/img_origin")
    finragbench_sample_file: Path = Path("data/DocVQA/FinRAGBench-V/data/queries/queries_en.json")
    docbench_dir: Path = Path("data/DocVQA/DocBench")
    docbench_data_dir: Path = Path("data/DocVQA/DocBench/data")
    mmdocrag_max_samples: int | None = None
    longdocurl_max_samples: int | None = None
    longdocurl_max_docs: int | None = None
    mmlongbench_max_samples: int | None = None
    madqa_max_samples: int | None = None
    dude_max_samples: int | None = None
    finragbench_max_samples: int | None = None
    docbench_max_samples: int | None = None
    include: list[IncludeSpec] = field(default_factory=list)


@dataclass
class MinerUConfig:
    enabled: bool = False
    backend: str | None = "pipeline"
    model_source: str | None = None
    executable: str = "mineru"
    max_docs: int | None = None


@dataclass
class DocumentPipelineConfig:
    output_dir: Path = Path("runs/environment")
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    mineru: MinerUConfig = field(default_factory=MinerUConfig)


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _as_dict(config: DocumentPipelineConfig) -> dict[str, Any]:
    return {
        "output_dir": str(config.output_dir),
        "datasets": {
            "mmdocrag_dir": str(config.datasets.mmdocrag_dir),
            "mmdocrag_sample_files": [str(path) for path in config.datasets.mmdocrag_sample_files],
            "longdocurl_dir": str(config.datasets.longdocurl_dir),
            "longdocurl_sample_jsonl": str(config.datasets.longdocurl_sample_jsonl),
            "mmlongbench_dir": str(config.datasets.mmlongbench_dir),
            "mmlongbench_documents_dir": str(config.datasets.mmlongbench_documents_dir),
            "mmlongbench_sample_file": str(config.datasets.mmlongbench_sample_file),
            "madqa_dir": str(config.datasets.madqa_dir),
            "madqa_pdf_dir": str(config.datasets.madqa_pdf_dir),
            "madqa_sample_file": str(config.datasets.madqa_sample_file),
            "dude_dir": str(config.datasets.dude_dir),
            "dude_pdf_dir": str(config.datasets.dude_pdf_dir),
            "dude_sample_file": str(config.datasets.dude_sample_file),
            "finragbench_dir": str(config.datasets.finragbench_dir),
            "finragbench_image_dir": str(config.datasets.finragbench_image_dir),
            "finragbench_sample_file": str(config.datasets.finragbench_sample_file),
            "docbench_dir": str(config.datasets.docbench_dir),
            "docbench_data_dir": str(config.datasets.docbench_data_dir),
            "mmdocrag_max_samples": config.datasets.mmdocrag_max_samples,
            "longdocurl_max_samples": config.datasets.longdocurl_max_samples,
            "longdocurl_max_docs": config.datasets.longdocurl_max_docs,
            "mmlongbench_max_samples": config.datasets.mmlongbench_max_samples,
            "madqa_max_samples": config.datasets.madqa_max_samples,
            "dude_max_samples": config.datasets.dude_max_samples,
            "finragbench_max_samples": config.datasets.finragbench_max_samples,
            "docbench_max_samples": config.datasets.docbench_max_samples,
            "include": [
                {"dataset": item.dataset, "doc_id": item.doc_id}
                for item in config.datasets.include
            ],
        },
        "mineru": {
            "enabled": config.mineru.enabled,
            "backend": config.mineru.backend,
            "model_source": config.mineru.model_source,
            "executable": config.mineru.executable,
            "max_docs": config.mineru.max_docs,
        },
    }


def load_config(path: str | Path | None = None) -> DocumentPipelineConfig:
    base = _as_dict(DocumentPipelineConfig())
    if path:
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required for YAML config files.") from exc
            with path.open("r", encoding="utf-8") as f:
                patch = yaml.safe_load(f) or {}
        else:
            patch = read_json(path, default={}) or {}
        _deep_update(base, patch)

    return DocumentPipelineConfig(
        output_dir=Path(base["output_dir"]),
        datasets=DatasetConfig(
            mmdocrag_dir=Path(base["datasets"]["mmdocrag_dir"]),
            mmdocrag_sample_files=[Path(path) for path in base["datasets"].get("mmdocrag_sample_files", [])],
            longdocurl_dir=Path(base["datasets"]["longdocurl_dir"]),
            longdocurl_sample_jsonl=Path(base["datasets"].get("longdocurl_sample_jsonl", "./data/datasets/LongDocURL/LongDocURL_public_multi_evidence_sources.jsonl")),
            mmlongbench_dir=Path(base["datasets"].get("mmlongbench_dir", "data/DocVQA/MMLongBench-Doc")),
            mmlongbench_documents_dir=Path(base["datasets"].get("mmlongbench_documents_dir", "data/DocVQA/MMLongBench-Doc/data/documents")),
            mmlongbench_sample_file=Path(base["datasets"].get("mmlongbench_sample_file", "data/DocVQA/MMLongBench-Doc/data/samples.json")),
            madqa_dir=Path(base["datasets"].get("madqa_dir", "data/DocVQA/MADQA/dataset")),
            madqa_pdf_dir=Path(base["datasets"].get("madqa_pdf_dir", "data/DocVQA/MADQA/dataset/data/pdfs")),
            madqa_sample_file=Path(base["datasets"].get("madqa_sample_file", "data/DocVQA/MADQA/dataset/data/dev.json")),
            dude_dir=Path(base["datasets"].get("dude_dir", "data/DocVQA/DUDE")),
            dude_pdf_dir=Path(base["datasets"].get("dude_pdf_dir", "data/DocVQA/DUDE/extracted/DUDE_train-val-test_binaries/PDF/train")),
            dude_sample_file=Path(base["datasets"].get("dude_sample_file", "./data/datasets/DUDE/calculation.json")),
            finragbench_dir=Path(base["datasets"].get("finragbench_dir", "data/DocVQA/FinRAGBench-V")),
            finragbench_image_dir=Path(base["datasets"].get("finragbench_image_dir", "data/DocVQA/FinRAGBench-V/data/corpus/en/img_origin")),
            finragbench_sample_file=Path(base["datasets"].get("finragbench_sample_file", "data/DocVQA/FinRAGBench-V/data/queries/queries_en.json")),
            docbench_dir=Path(base["datasets"].get("docbench_dir", "data/DocVQA/DocBench")),
            docbench_data_dir=Path(base["datasets"].get("docbench_data_dir", "data/DocVQA/DocBench/data")),
            mmdocrag_max_samples=base["datasets"].get("mmdocrag_max_samples"),
            longdocurl_max_samples=base["datasets"].get("longdocurl_max_samples"),
            longdocurl_max_docs=base["datasets"].get("longdocurl_max_docs"),
            mmlongbench_max_samples=base["datasets"].get("mmlongbench_max_samples"),
            madqa_max_samples=base["datasets"].get("madqa_max_samples"),
            dude_max_samples=base["datasets"].get("dude_max_samples"),
            finragbench_max_samples=base["datasets"].get("finragbench_max_samples"),
            docbench_max_samples=base["datasets"].get("docbench_max_samples"),
            include=[
                IncludeSpec(
                    dataset=str(item["dataset"]),
                    doc_id=str(item["doc_id"]),
                )
                for item in base["datasets"].get("include", []) or []
            ],
        ),
        mineru=MinerUConfig(**base["mineru"]),
    )
