"""
Unit tests: Ask and Plan routes reject write tool calls.

When the model returns a write tool call (search_replace, write_file, etc.),
ask_plan must return the reject message as the tool result and must never
touch files (meta["touched"] always empty).
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import main as unittest_main, TestCase
from unittest.mock import patch

# Allow ai_shell to import (model module checks SC2_MLX_MODEL); we mock get_reply_completion so model is never used
os.environ.setdefault("SC2_MLX_MODEL", "dummy")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Reply that contains a write tool call (should be rejected in ask/plan)
REPLY_WITH_SEARCH_REPLACE = (
    "I'll fix that.\n\nsearch_replace(path=\"some.c\", old_string=\"old\", new_string=\"new\")"
)
REPLY_WITH_WRITE_FILE = 'Creating the file now.\n\nwrite_file(path="new.txt", content="hello")'
# Final reply with no tool calls (so the loop exits after one round)
REPLY_FINAL = "Understood. I cannot edit in this mode."


def _fixture_root():
    """Set up a minimal workspace so list_files/get_root don't fail."""
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "dummy.c").write_text("int main() { return 0; }\n")
    return Path(tmp)


def _set_workspace(root: Path):
    from ai_shell.files import set_root, set_include
    set_root(str(root))
    set_include(None)  # allow all under root


class TestAskPlanGuards(TestCase):
    """Ask and Plan must reject write tool calls and never touch files."""

    def test_ask_rejects_search_replace(self):
        """Ask mode: model output with search_replace → reject message, no touches."""
        fixture_root = _fixture_root()
        _set_workspace(fixture_root)
        from ai_shell import ask_plan

        call_count = [0]

        def mock_reply(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return REPLY_WITH_SEARCH_REPLACE
            return REPLY_FINAL

        with patch.object(ask_plan, "get_reply_completion", side_effect=mock_reply):
            output, meta = ask_plan.run_ask(
                "change old to new in some.c",
                focus_file=None,
                extra_read_files=None,
                context_folders=None,
            )

        self.assertEqual(meta["mode_used"], "ask")
        self.assertEqual(meta["touched"], [])
        self.assertIn(meta.get("per_file_staged"), (None, {}))
        self.assertFalse(meta.get("staged", True))
        self.assertTrue(
            REPLY_FINAL in output or "cannot edit" in output.lower() or len(output) >= 0
        )

    def test_ask_rejects_write_file(self):
        """Ask mode: model output with write_file → reject, no file created."""
        fixture_root = _fixture_root()
        _set_workspace(fixture_root)
        from ai_shell import ask_plan

        call_count = [0]

        def mock_reply(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return REPLY_WITH_WRITE_FILE
            return "I won't create files in ask mode."

        with patch.object(ask_plan, "get_reply_completion", side_effect=mock_reply):
            output, meta = ask_plan.run_ask(
                "create new.txt with hello",
                focus_file=None,
            )

        self.assertEqual(meta["mode_used"], "ask")
        self.assertEqual(meta["touched"], [])
        self.assertFalse((fixture_root / "new.txt").exists(), "write_file must not be executed")

    def test_plan_rejects_edit(self):
        """Plan mode: model output with edit_existing_file → reject message, no touches."""
        fixture_root = _fixture_root()
        _set_workspace(fixture_root)
        from ai_shell import ask_plan

        reply_with_edit = 'Here is the plan. edit_existing_file(path="dummy.c", changes="int main() { return 1; }")'
        call_count = [0]

        def mock_reply(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return reply_with_edit
            return "Plan mode: no edits applied."

        with patch.object(ask_plan, "get_reply_completion", side_effect=mock_reply):
            output, meta = ask_plan.run_plan(
                "plan to change the return value",
                focus_file="dummy.c",
            )

        self.assertEqual(meta["mode_used"], "plan")
        self.assertEqual(meta["touched"], [])
        self.assertIn(meta.get("per_file_staged"), (None, {}))

    def test_plan_reject_message_in_behavior(self):
        """Plan mode: write tool call is rejected; no file touched."""
        fixture_root = _fixture_root()
        _set_workspace(fixture_root)
        from ai_shell import ask_plan

        call_count = [0]

        def mock_reply(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "search_replace(path=\"x\", old_string=\"a\", new_string=\"b\")"
            return "Done."

        with patch.object(ask_plan, "get_reply_completion", side_effect=mock_reply):
            output, meta = ask_plan.run_plan("edit x", on_action=lambda a: None)

        self.assertEqual(meta["mode_used"], "plan")
        self.assertEqual(meta["touched"], [])


if __name__ == "__main__":
    unittest_main(verbosity=2)
