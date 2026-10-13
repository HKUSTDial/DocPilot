import json
import tempfile
import unittest
from pathlib import Path

from app.workflow.mineru_loader import load_nodes_from_mineru_output


class MinerULoaderTest(unittest.TestCase):
    def test_loads_mineru_4_middle_json(self) -> None:
        payload = {
            "schema": "docvortex.middle",
            "schema_version": "2.0",
            "extensions": {},
            "metadata": {
                "file_suffix": "pdf",
                "producer": {"name": "MinerU", "version": "4.0.12"},
            },
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "text",
                            "index": 0,
                            "bbox": [0.1, 0.2, 0.8, 0.3],
                            "content": [{"type": "text", "content": "Hello MinerU 4", "styles": []}],
                        }
                    ],
                }
            ],
            "is_full_document": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory, "images", "screenshots", "page1.png")
            screenshot.parent.mkdir(parents=True)
            screenshot.touch()
            Path(directory, "middle_json.json").write_text(json.dumps(payload), encoding="utf-8")
            nodes = load_nodes_from_mineru_output(directory, doc_id="sample")

        self.assertEqual([node.node_type for node in nodes], ["paragraph", "document"])
        self.assertEqual(nodes[0].text, "Hello MinerU 4")
        self.assertEqual(nodes[0].bbox, [100, 200, 800, 300])
        self.assertEqual(nodes[-1].page, 1)
        self.assertEqual(nodes[-1].metadata["page_image_path"], str(screenshot.resolve()))


if __name__ == "__main__":
    unittest.main()
