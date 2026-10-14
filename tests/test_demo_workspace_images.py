import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from starlette.routing import Mount

from demo.chat.backend import main as chat_backend


class WorkspaceImagesTest(unittest.TestCase):
    def test_only_session_images_are_served(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory, "workspace")
            session = workspace / "default"
            session.mkdir(parents=True)
            image = session / "page 1.png"
            image.write_bytes(b"test image")
            (session / ".docpilot_state.json").write_text("{}", encoding="utf-8")
            logs = session / ".chat_runs"
            logs.mkdir()
            (logs / "run.log").write_text("private", encoding="utf-8")
            other = workspace / "other"
            other.mkdir()
            (other / "private.png").write_bytes(b"other image")
            (session / "linked.png").symlink_to(other / "private.png")

            with patch.object(chat_backend, "WORKSPACE_ROOT", workspace):
                payload = chat_backend._workspace_image_payload(str(image), "default")
                self.assertEqual(payload["url"], "/api/workspace-images/default/page%201.png")

                response = asyncio.run(chat_backend.get_workspace_image("default", "page 1.png"))
                self.assertEqual(response.headers["content-type"], "image/png")
                self.assertEqual(Path(response.path).read_bytes(), b"test image")
                self.assertIn(
                    "/api/workspace-images/{session_id}/{image_path:path}",
                    [route.path for route in chat_backend.app.routes],
                )
                self.assertNotIn(
                    "/workspace",
                    [route.path for route in chat_backend.app.routes if isinstance(route, Mount)],
                )

                for session_id, image_path in (
                    ("default", ".docpilot_state.json"),
                    ("default", ".chat_runs/run.log"),
                    ("default", "linked.png"),
                    ("..", "other/private.png"),
                    ("default", "../other/private.png"),
                ):
                    with self.subTest(session_id=session_id, image_path=image_path):
                        with self.assertRaises(HTTPException) as raised:
                            asyncio.run(chat_backend.get_workspace_image(session_id, image_path))
                        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
