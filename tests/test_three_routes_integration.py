"""
Integration tests: Ask, Plan, and Agent routes with real model and human-like requests.

To avoid high RAM use, run one test at a time in a separate process:

  python tests/run_integration_one_at_a_time.py

To run this module alone (all tests in one process, more RAM):

  python3 -m unittest tests.test_three_routes_integration -v
"""

import sys
from pathlib import Path
from unittest import TestCase, main as unittest_main

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Use same model and env as app; disable debug during tests to save RAM
import tests._load_app_model_env as _model_env
_model_env.load_app_model_env(reduce_debug_for_tests=True)

FIXTURES_W7 = REPO / "tests" / "fixtures" / "w7"


def _workspace_setup():
    if not FIXTURES_W7.is_dir():
        raise FileNotFoundError("tests/fixtures/w7 not found")
    for name in ("checkpasswd.c", "validate.c"):
        if not (FIXTURES_W7 / name).exists():
            raise FileNotFoundError(f"tests/fixtures/w7/{name} missing; needed for integration tests")
    from ai_shell.files import set_root, set_include
    set_root(str(FIXTURES_W7))
    set_include(None)


class TestAskRouteIntegration(TestCase):
    """Ask route: human-like questions; must return text and never touch files."""

    def setUp(self):
        _workspace_setup()

    def test_vague_question_what_does_this_do(self):
        """Vague: 'what does the code in here do?'"""
        from ai_shell.ask_plan import run_ask

        output, meta = run_ask(
            "what does the code in here do?",
            focus_file="checkpasswd.c",
            extra_read_files=None,
            context_folders=None,
        )
        self.assertEqual(meta["mode_used"], "ask")
        self.assertEqual(meta["touched"], [])
        self.assertIn(meta.get("per_file_staged"), (None, {}))
        self.assertIsInstance(output, str)

    def test_slightly_ambiguous_question(self):
        """Slightly ambiguous: 'can you explain this file?'"""
        from ai_shell.ask_plan import run_ask

        output, meta = run_ask(
            "can you explain this file?",
            focus_file="checkpasswd.c",
        )
        self.assertEqual(meta["mode_used"], "ask")
        self.assertEqual(meta["touched"], [])
        self.assertIsInstance(output, str)

    def test_ask_no_focus_file(self):
        """Ask with no focus file (discovery or generic answer)."""
        from ai_shell.ask_plan import run_ask

        output, meta = run_ask(
            "what's in this project?",
            focus_file=None,
        )
        self.assertEqual(meta["mode_used"], "ask")
        self.assertEqual(meta["touched"], [])
        self.assertIsInstance(output, str)


class TestPlanRouteIntegration(TestCase):
    """Plan route: human-like planning requests; must return plan text and never touch files."""

    def setUp(self):
        _workspace_setup()

    def test_vague_plan_request(self):
        """Vague: 'how would we add password validation?'"""
        from ai_shell.ask_plan import run_plan

        output, meta = run_plan(
            "how would we add password validation?",
            focus_file="checkpasswd.c",
        )
        self.assertEqual(meta["mode_used"], "plan")
        self.assertEqual(meta["touched"], [])
        self.assertIn(meta.get("per_file_staged"), (None, {}))
        self.assertIsInstance(output, str)

    def test_plan_explore_codebase(self):
        """Natural: 'figure out what needs to change to make it work'"""
        from ai_shell.ask_plan import run_plan

        output, meta = run_plan(
            "figure out what needs to change to make it work",
            focus_file="checkpasswd.c",
        )
        self.assertEqual(meta["mode_used"], "plan")
        self.assertEqual(meta["touched"], [])
        self.assertIsInstance(output, str)

    def test_plan_no_focus(self):
        """Plan with no focus file."""
        from ai_shell.ask_plan import run_plan

        output, meta = run_plan(
            "what would we need to do to get the build passing?",
            focus_file=None,
        )
        self.assertEqual(meta["mode_used"], "plan")
        self.assertEqual(meta["touched"], [])
        self.assertIsInstance(output, str)


class TestAgentRouteIntegration(TestCase):
    """Agent route: real edits; may touch files. We only assert it runs and returns valid meta."""

    def setUp(self):
        _workspace_setup()

    def test_agent_vague_edit_request(self):
        """Vague: 'add a comment at the top of the file' (small, safe edit)."""
        from ai_shell.agent_loop import run_agent

        output, meta = run_agent(
            "add a comment at the top of the file saying // checkpasswd",
            focus_file="checkpasswd.c",
            mode="agent",
            extra_read_files=None,
            context_folders=None,
            max_rounds=5,
        )
        self.assertIn("mode_used", meta)
        self.assertIsInstance(output, str)
        self.assertIsInstance(meta.get("touched", []), list)
        self.assertIsInstance(meta.get("per_file_staged", {}), dict)

    def test_agent_explicit_small_edit(self):
        """More explicit but still natural: 'replace the blank lines before return 0 with a single comment'."""
        from ai_shell.agent_loop import run_agent

        output, meta = run_agent(
            "replace the blank lines before return 0 with a single comment: // TODO",
            focus_file="checkpasswd.c",
            mode="agent",
            max_rounds=5,
        )
        self.assertIn("mode_used", meta)
        self.assertIsInstance(output, str)
        self.assertIsInstance(meta.get("touched", []), list)


if __name__ == "__main__":
    unittest_main(verbosity=2)
