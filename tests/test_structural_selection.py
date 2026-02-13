import json
import unittest

from ai_shell.structural import select_target_symbol, select_target_symbols


class StructuralSelectionTests(unittest.TestCase):
    def test_single_symbol_request_is_eligible(self) -> None:
        content = (
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "int mul(int a, int b) {\n"
            "  return a * b;\n"
            "}\n"
        )
        decision = select_target_symbol(
            user_text="fix add to handle overflow checks",
            focus_file="sample.c",
            content=content,
            analysis_packet="{}",
        )
        self.assertTrue(decision.eligible)
        self.assertIsNotNone(decision.target)
        self.assertEqual(decision.target.name, "add")

    def test_multi_symbol_touchpoints_is_ineligible(self) -> None:
        content = (
            "int add(int a, int b) { return a + b; }\n"
            "int mul(int a, int b) { return a * b; }\n"
        )
        analysis_packet = json.dumps(
            {
                "touch_points": [
                    {"symbol": "add"},
                    {"symbol": "mul"},
                ]
            }
        )
        decision = select_target_symbol(
            user_text="fix math helpers",
            focus_file="sample.c",
            content=content,
            analysis_packet=analysis_packet,
        )
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason, "multiple_touch_points")

    def test_disambiguates_by_in_symbol_phrase(self) -> None:
        content = (
            "int add(int a, int b) { return a + b; }\n"
            "int mul(int a, int b) { return a * b; }\n"
        )
        decision = select_target_symbol(
            user_text="fix leak in mul and keep behavior",
            focus_file="sample.c",
            content=content,
            analysis_packet="{}",
        )
        self.assertTrue(decision.eligible)
        self.assertIsNotNone(decision.target)
        self.assertEqual(decision.target.name, "mul")

    def test_select_target_symbols_orders_multiple_mentions(self) -> None:
        content = (
            "int add(int a, int b) { return a + b; }\n"
            "int mul(int a, int b) { return a * b; }\n"
            "int sub(int a, int b) { return a - b; }\n"
        )
        targets = select_target_symbols(
            user_text="update mul and add only",
            focus_file="sample.c",
            content=content,
            analysis_packet="{}",
        )
        self.assertEqual([t.name for t in targets], ["mul", "add"])


if __name__ == "__main__":
    unittest.main()
