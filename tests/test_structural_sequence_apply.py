import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import types

# Prevent model backend boot during agent import in unit tests.
if "ai_shell.model" not in sys.modules:
    _model_stub = types.ModuleType("ai_shell.model")
    _model_stub.backend_name = lambda: "test-backend"
    _model_stub.stream_reply = lambda *_a, **_k: ""
    sys.modules["ai_shell.model"] = _model_stub

from ai_shell import agent


class StructuralSequenceApplyTests(unittest.TestCase):
    def test_kv_cache_key_stable_within_same_tuple(self) -> None:
        agent.reset_structural_kv_cache(reason="test_init")
        with (
            patch.object(agent.config, "STRUCTURAL_PACKED_CONTEXT_VERSION", "1"),
            patch.object(agent.config, "STRUCTURAL_RULES_VERSION", "1"),
        ):
            key_1 = agent._build_structural_cache_key_base("sample.py", "request one")
            key_2 = agent._build_structural_cache_key_base("sample.py", "request two")
        self.assertEqual(key_1, key_2)

    def test_kv_cache_key_rotates_when_tuple_changes(self) -> None:
        agent.reset_structural_kv_cache(reason="test_init")
        with (
            patch.object(agent.config, "STRUCTURAL_PACKED_CONTEXT_VERSION", "1"),
            patch.object(agent.config, "STRUCTURAL_RULES_VERSION", "1"),
        ):
            key_a_1 = agent._build_structural_cache_key_base("a.py", "request")
            key_b = agent._build_structural_cache_key_base("b.py", "request")
            key_a_2 = agent._build_structural_cache_key_base("a.py", "request")
        self.assertNotEqual(key_a_1, key_b)
        self.assertNotEqual(key_a_1, key_a_2)

    def test_kv_cache_key_rotates_on_new_chat_reset(self) -> None:
        agent.reset_structural_kv_cache(reason="test_init")
        with (
            patch.object(agent.config, "STRUCTURAL_PACKED_CONTEXT_VERSION", "1"),
            patch.object(agent.config, "STRUCTURAL_RULES_VERSION", "1"),
        ):
            key_before = agent._build_structural_cache_key_base("sample.py", "request")
            agent.reset_structural_kv_cache(reason="new_chat")
            key_after = agent._build_structural_cache_key_base("sample.py", "request")
        self.assertNotEqual(key_before, key_after)

    def test_sequence_applies_when_last_step_noop_but_net_changed(self) -> None:
        focus_file = "sample.py"
        baseline = "def a():\n    return 1\n\ndef b():\n    return 2\n"
        changed = "def a():\n    return 10\n\ndef b():\n    return 2\n"

        step_1 = {
            "output": "[Candidate file_edit]",
            "meta": {},
            "candidate_content": changed,
            "candidate_diff": "",
        }
        step_2 = {
            "output": "[No-op file_edit: target already up to date]",
            "meta": {"noop": True, "already_up_to_date": True},
            "candidate_content": changed,
            "candidate_diff": "",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / focus_file
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(baseline, encoding="utf-8")

            with (
                patch.object(agent, "read_single_file_for_context", return_value={focus_file: baseline}),
                patch.object(
                    agent,
                    "build_symbol_index",
                    return_value=[
                        SimpleNamespace(name="a", kind="function"),
                        SimpleNamespace(name="b", kind="function"),
                    ],
                ),
                patch.object(agent, "extract_target_snippet", return_value=""),
                patch.object(agent, "_run_structural_edit", side_effect=[step_1, step_2]),
                patch.object(agent, "get_root", return_value=root),
                patch.object(agent.config, "STAGE_EDITS", False),
            ):
                result = agent._run_structural_multi_symbol_edit(
                    user_text="update a and b",
                    focus_file=focus_file,
                    silent=True,
                    full_context=False,
                    analysis_packet="",
                    sliced_request="update a and b",
                    target_names=["a", "b"],
                    cache_key_base="x",
                )

            self.assertEqual(result["output"], "[Applied file_edit]")
            meta = result.get("meta", {}) or {}
            self.assertTrue(meta.get("structural_sequence_net_changed"))
            self.assertNotIn("noop", meta)
            self.assertIn("--- a/sample.py", str(meta.get("diff") or ""))
            self.assertEqual(result.get("candidate_content"), changed)
            self.assertEqual(target_path.read_text(encoding="utf-8"), changed)

    def test_sequence_keeps_partial_progress_on_late_failure(self) -> None:
        focus_file = "sample.py"
        baseline = "def a():\n    return 1\n\ndef b():\n    return 2\n"
        changed = "def a():\n    return 10\n\ndef b():\n    return 2\n"

        step_1 = {
            "output": "[Candidate file_edit]",
            "meta": {},
            "candidate_content": changed,
            "candidate_diff": "",
        }
        step_2 = {
            "output": "[File edit failed: STRUCTURAL_OUTPUT_INVALID: bad format]",
            "meta": {"failure_kind": "format", "failure_reason": "bad format"},
            "candidate_content": changed,
            "candidate_diff": "",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / focus_file
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(baseline, encoding="utf-8")

            with (
                patch.object(agent, "read_single_file_for_context", return_value={focus_file: baseline}),
                patch.object(
                    agent,
                    "build_symbol_index",
                    return_value=[
                        SimpleNamespace(name="a", kind="function"),
                        SimpleNamespace(name="b", kind="function"),
                    ],
                ),
                patch.object(agent, "extract_target_snippet", return_value=""),
                patch.object(agent, "_run_structural_edit", side_effect=[step_1, step_2]),
                patch.object(agent, "get_root", return_value=root),
                patch.object(agent.config, "STAGE_EDITS", False),
            ):
                result = agent._run_structural_multi_symbol_edit(
                    user_text="update a and b",
                    focus_file=focus_file,
                    silent=True,
                    full_context=False,
                    analysis_packet="",
                    sliced_request="update a and b",
                    target_names=["a", "b"],
                    cache_key_base="x",
                )

            self.assertEqual(result["output"], "[Applied file_edit]")
            meta = result.get("meta", {}) or {}
            self.assertTrue(meta.get("structural_sequence_partial_failure"))
            self.assertTrue(meta.get("non_blocking_failure"))
            self.assertEqual(meta.get("failure_kind"), "structural")
            self.assertIn("step 2/2", str(meta.get("failure_reason") or ""))
            self.assertEqual(result.get("candidate_content"), changed)
            self.assertEqual(target_path.read_text(encoding="utf-8"), changed)


if __name__ == "__main__":
    unittest.main()
