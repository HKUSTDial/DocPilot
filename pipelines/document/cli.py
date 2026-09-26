from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import DEFAULT_STAGES, STAGES, run_stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Document preprocessing pipeline")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run one or more pipeline stages")
    run_parser.add_argument(
        "--stages",
        type=str,
        default=",".join(DEFAULT_STAGES),
        help=f"Comma-separated stages. Valid: {','.join(STAGES)}",
    )

    subparsers.add_parser("list-stages", help="List available pipeline stages")
    subparsers.add_parser("show-config", help="Print resolved config")

    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "list-stages":
        print("\n".join(STAGES))
        return
    if args.command == "show-config":
        print(
            json.dumps(
                {
                    "output_dir": str(config.output_dir),
                    "datasets": {
                        "mmdocrag_dir": str(config.datasets.mmdocrag_dir),
                        "mmdocrag_sample_files": [
                            str(path) for path in config.datasets.mmdocrag_sample_files
                        ],
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
                        "mmdocrag_max_samples": config.datasets.mmdocrag_max_samples,
                        "longdocurl_max_samples": config.datasets.longdocurl_max_samples,
                        "longdocurl_max_docs": config.datasets.longdocurl_max_docs,
                        "mmlongbench_max_samples": config.datasets.mmlongbench_max_samples,
                        "madqa_max_samples": config.datasets.madqa_max_samples,
                        "dude_max_samples": config.datasets.dude_max_samples,
                        "finragbench_max_samples": config.datasets.finragbench_max_samples,
                        "include": [
                            {"dataset": item.dataset, "doc_id": item.doc_id}
                            for item in config.datasets.include
                        ],
                    },
                    "mineru": config.mineru.__dict__,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command in {None, "run"}:
        stages = [s.strip() for s in getattr(args, "stages", "").split(",") if s.strip()]
        results = run_stages(config, stages)
        print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
        return
    parser.error(f"Unknown command: {args.command}")
