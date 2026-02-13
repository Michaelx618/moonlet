import unittest

from ai_shell.structural import (
    StructuralTarget,
    apply_symbol_replacement,
    build_symbol_index,
    extract_target_snippet,
    validate_replacement_symbol_unit,
    validate_structural_candidate,
)


class StructuralReplaceTests(unittest.TestCase):
    def test_replaces_target_symbol_only(self) -> None:
        original = (
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "int mul(int a, int b) {\n"
            "  return a * b;\n"
            "}\n"
        )
        index = build_symbol_index("sample.c", original)
        target = [s for s in index if s.name == "add"][0]
        replacement = (
            "int add(int a, int b) {\n"
            "  if (a == 0) {\n"
            "    return b;\n"
            "  }\n"
            "  return a + b;\n"
            "}\n"
        )
        candidate = apply_symbol_replacement(original, target, replacement)
        self.assertIn("if (a == 0)", candidate)
        self.assertIn("int mul(int a, int b)", candidate)
        ok, reason = validate_structural_candidate(
            focus_file="sample.c",
            original_content=original,
            candidate_content=candidate,
            target=target,
        )
        self.assertTrue(ok, msg=reason)

    def test_extract_target_snippet_includes_definition(self) -> None:
        original = (
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
        )
        index = build_symbol_index("sample.c", original)
        target = [s for s in index if s.name == "add"][0]
        snippet = extract_target_snippet(original, target, padding_lines=0)
        self.assertIn("int add(int a, int b)", snippet)
        self.assertIn("return a + b;", snippet)

    def test_signature_change_is_rejected(self) -> None:
        original = (
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "int mul(int a, int b) {\n"
            "  return a * b;\n"
            "}\n"
        )
        index = build_symbol_index("sample.c", original)
        target = [s for s in index if s.name == "add"][0]
        replacement = (
            "long add(int a, int b) {\n"
            "  return (long)a + (long)b;\n"
            "}\n"
        )
        candidate = apply_symbol_replacement(original, target, replacement)
        ok, reason = validate_structural_candidate(
            focus_file="sample.c",
            original_content=original,
            candidate_content=candidate,
            target=target,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "signature_modified")

    def test_rejects_changes_outside_target_span(self) -> None:
        original = (
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "int mul(int a, int b) {\n"
            "  return a * b;\n"
            "}\n"
        )
        index = build_symbol_index("sample.c", original)
        target = [s for s in index if s.name == "add"][0]
        candidate = original.replace("return a * b;", "return a * b + 1;")
        ok, reason = validate_structural_candidate(
            focus_file="sample.c",
            original_content=original,
            candidate_content=candidate,
            target=target,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "replacement_outside_target_span")

    def test_replacement_unit_requires_single_matching_symbol(self) -> None:
        bad = (
            "int add(int a, int b) { return a + b; }\n"
            "int mul(int a, int b) { return a * b; }\n"
        )
        ok, reason = validate_replacement_symbol_unit(
            focus_file="sample.c",
            replacement_text=bad,
            target_name="add",
            target_kind="function",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "multiple_top_level_symbols_detected")

    def test_python_replacement_uses_line_span_when_byte_end_is_wrong(self) -> None:
        original = (
            "def apply_discount(product, percent):\n"
            "    return product\n"
            "\n"
            "def restock(product, amount):\n"
            "    return amount\n"
        )
        target = StructuralTarget(
            name="apply_discount",
            kind="function",
            line_start=1,
            line_end=2,
            byte_start=0,
            byte_end=5,  # intentionally wrong byte_end
            parent="",
        )
        replacement = (
            "def apply_discount(product, percent):\n"
            "    if percent < 0:\n"
            "        return product\n"
            "    return product\n"
        )
        candidate = apply_symbol_replacement(
            original,
            target,
            replacement,
            focus_file="sample.py",
        )
        self.assertIn("if percent < 0", candidate)
        self.assertIn("def restock(product, amount):", candidate)

    def test_replacement_unit_rejects_forbidden_symbol_defs_in_step(self) -> None:
        replacement = (
            "class Product:\n"
            "    def __post_init__(self):\n"
            "        return None\n"
            "    def apply_discount(self, percent):\n"
            "        return None\n"
        )
        ok, reason = validate_replacement_symbol_unit(
            focus_file="sample.py",
            replacement_text=replacement,
            target_name="Product",
            target_kind="class",
            forbidden_symbol_names=["apply_discount", "restock"],
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "forbidden_symbol_definition_in_step")


if __name__ == "__main__":
    unittest.main()
