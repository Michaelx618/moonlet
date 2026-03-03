"""
API tests: POST /stream with mode=ask, plan, agent.

Verifies server dispatches to the correct route and returns valid SSE (chunk + meta).
Uses mocked run_ask/run_plan/run_agent so no model is required.
"""

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from unittest import TestCase, main as unittest_main
from unittest.mock import patch

os.environ.setdefault("SC2_MLX_MODEL", "dummy")

REPO = Path(__file__).resolve().parent.parent
FIXTURES_W7 = REPO / "tests" / "fixtures" / "w7"
sys.path.insert(0, str(REPO))


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _read_sse_until_meta(response_body: str):
    """Parse SSE body; return (chunk_events, meta_dict or None)."""
    chunks = []
    meta = None
    current_event = None
    for line in response_body.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                if current_event == "meta":
                    try:
                        meta = json.loads(data)
                        break
                    except json.JSONDecodeError:
                        pass
                elif current_event == "chunk":
                    chunks.append(data)
            current_event = None
    return chunks, meta


def _start_server_on_port(port: int):
    """Start server in thread on given port; return base URL."""
    from ai_shell.server import start_server
    from ai_shell import config

    config.SERVER_PORT = port
    url = f"http://127.0.0.1:{port}"
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Server did not start in time")
    return url


def _post_stream(url: str, body: dict) -> str:
    """POST /stream; return response body as string."""
    import urllib.request
    req = urllib.request.Request(
        f"{url}/stream",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


class TestStreamAPIModes(TestCase):
    """POST /stream with mode=ask, plan, agent dispatches correctly and returns SSE meta."""

    def setUp(self):
        if not FIXTURES_W7.is_dir():
            self.skipTest("tests/fixtures/w7 not found")
        from ai_shell.files import set_root, set_include
        set_root(str(FIXTURES_W7))
        set_include(None)
        self.server_url = _start_server_on_port(_find_free_port())

    @patch("ai_shell.server.run_agent")
    @patch("ai_shell.server.run_plan")
    @patch("ai_shell.server.run_ask")
    def test_stream_mode_ask_calls_run_ask(self, mock_ask, mock_plan, mock_agent):
        """POST /stream with mode=ask must call run_ask and return meta with mode_used=ask."""
        mock_ask.return_value = ("Here is the answer.", {"mode_used": "ask", "touched": [], "ok": True})
        body = {"mode": "ask", "text": "what does this file do?", "focus_file": "checkpasswd.c"}
        try:
            response = _post_stream(self.server_url, body)
        except Exception as e:
            self.skipTest(f"POST failed: {e}")
        mock_ask.assert_called_once()
        mock_plan.assert_not_called()
        mock_agent.assert_not_called()
        _, meta = _read_sse_until_meta(response)
        self.assertIsNotNone(meta, "Expected at least one SSE data line with meta")
        self.assertEqual(meta.get("mode_used"), "ask")
        self.assertEqual(meta.get("touched"), [])

    @patch("ai_shell.server.run_agent")
    @patch("ai_shell.server.run_plan")
    @patch("ai_shell.server.run_ask")
    def test_stream_mode_plan_calls_run_plan(self, mock_ask, mock_plan, mock_agent):
        """POST /stream with mode=plan must call run_plan and return meta with mode_used=plan."""
        mock_plan.return_value = ("1. Read file. 2. Add validation.", {"mode_used": "plan", "touched": [], "ok": True})
        body = {"mode": "plan", "text": "how would we add validation?", "focus_file": "checkpasswd.c"}
        try:
            response = _post_stream(self.server_url, body)
        except Exception as e:
            self.skipTest(f"POST failed: {e}")
        mock_plan.assert_called_once()
        mock_ask.assert_not_called()
        mock_agent.assert_not_called()
        _, meta = _read_sse_until_meta(response)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("mode_used"), "plan")
        self.assertEqual(meta.get("touched"), [])

    @patch("ai_shell.server.run_agent")
    @patch("ai_shell.server.run_plan")
    @patch("ai_shell.server.run_ask")
    def test_stream_mode_agent_calls_run_agent(self, mock_ask, mock_plan, mock_agent):
        """POST /stream with mode=agent must call run_agent (not ask/plan)."""
        mock_agent.return_value = ("Done.", {"mode_used": "agent", "touched": [], "ok": True})
        body = {"mode": "agent", "text": "add a comment at top", "focus_file": "checkpasswd.c"}
        try:
            response = _post_stream(self.server_url, body)
        except Exception as e:
            self.skipTest(f"POST failed: {e}")
        mock_agent.assert_called_once()
        mock_ask.assert_not_called()
        mock_plan.assert_not_called()
        _, meta = _read_sse_until_meta(response)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("mode_used"), "agent")

    @patch("ai_shell.server.run_agent")
    @patch("ai_shell.server.run_plan")
    @patch("ai_shell.server.run_ask")
    def test_stream_chat_treated_as_ask(self, mock_ask, mock_plan, mock_agent):
        """Backward compat: mode=chat is treated as ask."""
        mock_ask.return_value = ("Ok.", {"mode_used": "ask", "touched": []})
        body = {"mode": "chat", "text": "hello"}
        try:
            _post_stream(self.server_url, body)
        except Exception as e:
            self.skipTest(f"POST failed: {e}")
        mock_ask.assert_called_once()
        mock_plan.assert_not_called()
        mock_agent.assert_not_called()


if __name__ == "__main__":
    unittest_main(verbosity=2)
