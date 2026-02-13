import unittest

from ai_shell.normalizer import normalize_symbol


class NormalizerTests(unittest.TestCase):
    def test_accepts_marker_wrapped_symbol(self) -> None:
        raw = (
            "BEGIN_SYMBOL\n"
            "def apply_discount(product, percent):\n"
            "    return product\n"
            "END_SYMBOL\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertEqual(result.error_code, "")
        self.assertEqual(result.confidence, "green")
        self.assertTrue(result.used_markers)
        self.assertIn("def apply_discount", result.text)

    def test_salvages_malformed_markers(self) -> None:
        raw = "BEGIN_SYMBOL\ndef apply_discount(product, percent): return product\n"
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertEqual(result.error_code, "")
        self.assertEqual(result.confidence, "yellow")
        self.assertIn("python_expand_one_liner", result.repairs)
        self.assertIn("def apply_discount", result.text)

    def test_markerless_salvage_from_prose(self) -> None:
        raw = (
            "Sure, here is the update:\n\n"
            "def apply_discount(product, percent):\n"
            "    return product\n"
            "\nThanks.\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("def apply_discount", result.text)

    def test_extracts_target_symbol_from_multi_symbol_candidate(self) -> None:
        raw = (
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "int mul(int a, int b) {\n"
            "  return a * b;\n"
            "}\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.c",
            target_name="mul",
            target_kind="function",
            original_symbol_text="",
            request_text="update mul",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("int mul", result.text)
        self.assertNotIn("int add", result.text)

    def test_applies_python_trailing_newline_and_indent_normalization(self) -> None:
        raw = (
            "def apply_discount(product, percent):\n"
            "  if percent < 0:\n"
            "   return product\n"
            "  return product"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertEqual(result.error_code, "")
        self.assertTrue(result.text.endswith("\n"))
        self.assertEqual(result.confidence, "yellow")

    def test_repairs_single_missing_brace(self) -> None:
        raw = (
            "int add(int a, int b) {\n"
            "  return a + b;\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.c",
            target_name="add",
            target_kind="function",
            original_symbol_text="",
            request_text="update add",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("append_1_missing_closing_brace", result.repairs)
        self.assertEqual(result.confidence, "yellow")

    def test_repairs_two_missing_braces(self) -> None:
        raw = (
            "int add(int a, int b) {\n"
            "  if (a > b) {\n"
            "    return a;\n"
            "  return b;\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.c",
            target_name="add",
            target_kind="function",
            original_symbol_text="",
            request_text="update add",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("append_2_missing_closing_brace", result.repairs)
        self.assertEqual(result.confidence, "yellow")

    def test_recovers_from_nested_missing_braces(self) -> None:
        raw = (
            "int add(int a, int b) {\n"
            "  if (a > b) {\n"
            "    return a;\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.c",
            target_name="add",
            target_kind="function",
            original_symbol_text="",
            request_text="update add",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("append_2_missing_closing_brace", result.repairs)
        self.assertEqual(result.confidence, "yellow")

    def test_stage_a_cleanup_empty(self) -> None:
        result = normalize_symbol(
            raw_output="```\n```",
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertEqual(result.error_code, "norm_a_cleanup_empty")
        self.assertEqual(result.stage, "A")

    def test_stage_b_no_candidate(self) -> None:
        result = normalize_symbol(
            raw_output="BEGIN_SYMBOL\nEND_SYMBOL\n",
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertEqual(result.error_code, "norm_b_no_candidate")
        self.assertEqual(result.stage, "B")

    def test_stage_c_symbol_not_found(self) -> None:
        raw = (
            "int add(int a, int b) { return a + b; }\n"
            "int mul(int a, int b) { return a * b; }\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.c",
            target_name="sub",
            target_kind="function",
            original_symbol_text="",
            request_text="update sub",
        )
        self.assertEqual(result.error_code, "norm_c_symbol_not_found")
        self.assertEqual(result.stage, "C")

    def test_stage_c_ambiguous_target(self) -> None:
        raw = (
            "int add(int a, int b) { return a + b; }\n"
            "int add(int a, int b, int c) { return a + b + c; }\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.c",
            target_name="add",
            target_kind="function",
            original_symbol_text="",
            request_text="update add",
        )
        self.assertEqual(result.error_code, "norm_c_ambiguous_target")
        self.assertEqual(result.stage, "C")

    def test_stage_e_reparse_failed(self) -> None:
        result = normalize_symbol(
            raw_output="def broken(\n",
            focus_file="sample.py",
            target_name="broken",
            target_kind="function",
            original_symbol_text="",
            request_text="update broken",
        )
        self.assertEqual(result.error_code, "norm_e_reparse_failed")
        self.assertEqual(result.stage, "E")

    def test_stage_e_scope_violation(self) -> None:
        raw = (
            "def other_symbol(x):\n"
            "    return x\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertEqual(result.error_code, "norm_e_scope_violation")
        self.assertEqual(result.stage, "E")

    def test_strips_diff_headers_for_cpp(self) -> None:
        raw = (
            "diff --git a/a.cpp b/a.cpp\n"
            "--- a/a.cpp\n"
            "+++ b/a.cpp\n"
            "@@ -1,3 +1,3 @@\n"
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="a.cpp",
            target_name="add",
            target_kind="function",
            original_symbol_text="",
            request_text="update add",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("int add", result.text)

    def test_accepts_javascript_class_with_nested_method(self) -> None:
        raw = (
            "BEGIN_SYMBOL\n"
            "class Product {\n"
            "  applyDiscount(percent) {\n"
            "    return percent;\n"
            "  }\n"
            "}\n"
            "END_SYMBOL\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.js",
            target_name="Product",
            target_kind="class",
            original_symbol_text="",
            request_text="update Product",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("class Product", result.text)

    def test_extracts_javascript_class_target_from_extra_top_level(self) -> None:
        raw = (
            "class Product {\n"
            "  applyDiscount(percent) {\n"
            "    return percent;\n"
            "  }\n"
            "}\n"
            "function rogue() { return 1; }\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.js",
            target_name="Product",
            target_kind="class",
            original_symbol_text="",
            request_text="update Product",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("class Product", result.text)
        self.assertNotIn("function rogue", result.text)

    def test_accepts_typescript_function(self) -> None:
        raw = (
            "function applyDiscount(value: number): number {\n"
            "  return value;\n"
            "}\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.ts",
            target_name="applyDiscount",
            target_kind="function",
            original_symbol_text="",
            request_text="update applyDiscount",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("function applyDiscount", result.text)

    def test_accepts_go_function(self) -> None:
        raw = (
            "func applyDiscount(value int) int {\n"
            "  return value\n"
            "}\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.go",
            target_name="applyDiscount",
            target_kind="function",
            original_symbol_text="",
            request_text="update applyDiscount",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("func applyDiscount", result.text)

    def test_accepts_java_class_target(self) -> None:
        raw = (
            "class Product {\n"
            "  int applyDiscount(int value) {\n"
            "    return value;\n"
            "  }\n"
            "}\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="Product.java",
            target_name="Product",
            target_kind="class",
            original_symbol_text="",
            request_text="update Product",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("class Product", result.text)

    def test_removes_trailing_extra_brace_noise(self) -> None:
        raw = (
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "}\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.c",
            target_name="add",
            target_kind="function",
            original_symbol_text="",
            request_text="update add",
        )
        self.assertEqual(result.error_code, "")
        self.assertIn("int add", result.text)
        self.assertNotIn("}\n}\n", result.text)

    def test_marker_fallback_handles_outer_junk(self) -> None:
        raw = (
            "Here is the update\n"
            "BEGIN_SYMBOL\n"
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "END_SYMBOL\n"
            "No explanation\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.c",
            target_name="add",
            target_kind="function",
            original_symbol_text="",
            request_text="update add",
        )
        self.assertEqual(result.error_code, "")
        self.assertTrue(result.used_markers)
        self.assertIn("int add", result.text)

    def test_rejects_unrecoverable_plain_text(self) -> None:
        result = normalize_symbol(
            raw_output="this is not code and has no symbols",
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertIn(result.error_code, {"norm_c_parse_failed", "norm_e_reparse_failed"})
        self.assertEqual(result.confidence, "red")

    def test_language_target_extraction_salvages_broken_prefix(self) -> None:
        raw = (
            "RESPONSE:\n"
            "def broken(:\n"
            "    pass\n\n"
            "def apply_discount(product, percent):\n"
            "    return product\n"
        )
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertEqual(result.error_code, "")
        self.assertEqual(result.salvage_mode, "language_extract")
        self.assertIn("def apply_discount", result.text)

    def test_junk_low_entropy_lines_rejected(self) -> None:
        raw = ("s\n" * 40) + "BEGIN_SYMBOL\nEND_SYMBOL\n"
        result = normalize_symbol(
            raw_output=raw,
            focus_file="sample.py",
            target_name="apply_discount",
            target_kind="function",
            original_symbol_text="",
            request_text="update apply_discount",
        )
        self.assertEqual(result.error_code, "norm_a_cleanup_empty")
        self.assertEqual(result.confidence, "red")


if __name__ == "__main__":
    unittest.main()
