import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from demo.chat.backend import main as chat_backend


class ParsedOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / "runs"
        self.output = self.runs / "mineru" / "report" / "auto"
        self.output.mkdir(parents=True)
        self.json_path = self.output / "structured_content.json"
        self.json_path.write_text(json.dumps({"pages": [
            {"page_idx": 0, "blocks": [
                {"type": "text", "bbox": [0.1, 0.2, 0.8, 0.3]},
                {"type": "table", "bbox": [0.1, 0.4, 0.9, 0.8], "captions": [
                    {"content": "Table caption", "bbox": [0.1, 0.35, 0.9, 0.39]},
                ]},
                {"type": "text", "bbox": [0, 0, 10, 20]},
                {"type": "text", "bbox": [0.8, 0.2, 0.1, 0.3]},
                {"type": "text", "bbox": [False, 0, 1, 1]},
                {"type": "text", "bbox": [0, 0, float("nan"), 1]},
                {"type": "text"},
            ]},
            {"page_idx": 1, "blocks": [{"type": "chart", "bbox": [0, 0, 1, 1]}]},
        ]}), encoding="utf-8")
        (self.output / "markdown.md").write_text("# Report\n\n![](images/chart.png)", encoding="utf-8")
        self.record = {"processing_status": "ready", "mineru_output_dir": str(self.output.parent)}
        for name, value in (
            ("BACKEND_RUNS_ROOT", self.runs),
            ("WORKSPACE_ROOT", self.root / "workspace"),
        ):
            patcher = patch.object(chat_backend, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(chat_backend, "get_file_record", side_effect=(
            lambda session_id, file_name: self.record
            if session_id == "default" and file_name == "report.pdf" else None
        ))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_layout_matches_physical_page_and_omits_invalid_boxes(self) -> None:
        layout = chat_backend.get_file_layout("report.pdf", "default", 1)
        self.assertEqual(layout["page"], 1)
        self.assertEqual([block["type"] for block in layout["blocks"]], ["text", "table", "caption"])
        self.assertEqual(layout["blocks"][0]["bbox"], [0.1, 0.2, 0.8, 0.3])
        second = chat_backend.get_file_layout("report.pdf", "default", 2)
        self.assertEqual([block["type"] for block in second["blocks"]], ["chart"])
        with self.assertRaises(HTTPException) as raised:
            chat_backend.get_file_layout("report.pdf", "default", 3)
        self.assertEqual(raised.exception.status_code, 404)

    def test_outputs_use_structured_json_and_document_image_base(self) -> None:
        markdown = chat_backend.get_file_parsed_output("report.pdf", "default", "markdown")
        self.assertEqual(markdown["content"], "# Report\n\n![](images/chart.png)")
        self.assertEqual(markdown["image_base_url"], "/api/action-images/report/auto/")
        output = chat_backend.get_file_parsed_output("report.pdf", "default", "json")
        self.assertEqual(len(json.loads(output["content"])["pages"]), 2)
        self.assertNotIn(str(self.runs), output["image_base_url"])

    def test_nonready_and_unknown_documents_are_not_served(self) -> None:
        self.record["processing_status"] = "processing"
        with self.assertRaises(HTTPException) as raised:
            chat_backend.get_file_layout("report.pdf", "default", 1)
        self.assertEqual(raised.exception.status_code, 409)
        self.record["processing_status"] = "ready"
        for session_id, file_name in (("other", "report.pdf"), ("..", "report.pdf"),
                                      ("default", "missing.pdf"), ("default", "../report.pdf")):
            with self.subTest(session_id=session_id, file_name=file_name):
                with self.assertRaises(HTTPException) as raised:
                    chat_backend.get_file_parsed_output(file_name, session_id, "json")
                self.assertEqual(raised.exception.status_code, 404)

    def test_paths_and_symlinks_cannot_escape_the_document_output(self) -> None:
        private = self.root / "private.json"
        private.write_text('{"pages": []}', encoding="utf-8")
        self.json_path.unlink()
        self.json_path.symlink_to(private)
        with self.assertRaises(HTTPException) as raised:
            chat_backend.get_file_layout("report.pdf", "default", 1)
        self.assertEqual(raised.exception.status_code, 404)
        self.record["mineru_output_dir"] = str(self.root)
        with self.assertRaises(HTTPException) as raised:
            chat_backend.get_file_parsed_output("report.pdf", "default", "json")
        self.assertEqual(raised.exception.status_code, 404)

    def test_missing_and_malformed_outputs_fail_without_affecting_ready_state(self) -> None:
        self.json_path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(HTTPException) as raised:
            chat_backend.get_file_layout("report.pdf", "default", 1)
        self.assertEqual(raised.exception.status_code, 422)
        self.json_path.unlink()
        with self.assertRaises(HTTPException) as raised:
            chat_backend.get_file_layout("report.pdf", "default", 1)
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(self.record["processing_status"], "ready")


if __name__ == "__main__":
    unittest.main()
