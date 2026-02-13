import unittest

from ai_shell.structural import normalize_structural_output


class StructuralOutputContractTests(unittest.TestCase):
    def test_accepts_wrapped_target_symbol(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="add",
            target_kind="function",
            focus_file="sample.c",
        )
        self.assertIsNone(err)
        self.assertIn("int add(int a, int b)", normalized)

    def test_rejects_partial_markers(self) -> None:
        normalized, err = normalize_structural_output(
            "BEGIN_SYMBOL\nint add(int a, int b) { return a + b; }\n",
            target_symbol="add",
            target_kind="function",
            focus_file="sample.c",
        )
        self.assertEqual(normalized, "")
        self.assertEqual(err, "missing_symbol_markers")

    def test_accepts_markerless_valid_single_symbol(self) -> None:
        output = (
            "def apply_discount(product, percent):\n"
            "    return product\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="apply_discount",
            target_kind="function",
            focus_file="sample.py",
        )
        self.assertIsNone(err)
        self.assertIn("def apply_discount", normalized)

    def test_rejects_markerless_symbol_mismatch(self) -> None:
        output = (
            "def restock(product, amount):\n"
            "    return amount\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="apply_discount",
            target_kind="function",
            focus_file="sample.py",
        )
        self.assertEqual(normalized, "")
        self.assertEqual(err, "markerless_output_symbol_mismatch")

    def test_rejects_markerless_parse_failed(self) -> None:
        normalized, err = normalize_structural_output(
            "totally invalid output",
            target_symbol="apply_discount",
            target_kind="function",
            focus_file="sample.py",
        )
        self.assertEqual(normalized, "")
        self.assertEqual(err, "markerless_output_parse_failed")

    def test_rejects_low_entropy_output(self) -> None:
        output = "s\n" * 80
        normalized, err = normalize_structural_output(
            output,
            target_symbol="apply_discount",
            target_kind="function",
            focus_file="sample.py",
        )
        self.assertEqual(normalized, "")
        self.assertEqual(err, "degenerate_low_entropy_output")

    def test_accepts_python_class_with_nested_methods(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "class Product:\n"
            "    def __post_init__(self) -> None:\n"
            "        if self.price < 0:\n"
            "            raise ValueError(\"Price must be non-negative\")\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="Product",
            target_kind="class",
            focus_file="sample.py",
        )
        self.assertIsNone(err)
        self.assertIn("class Product", normalized)

    def test_rejects_python_class_with_extra_top_level_def(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "class Product:\n"
            "    def __post_init__(self) -> None:\n"
            "        return\n"
            "\n"
            "def rogue():\n"
            "    return 1\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="Product",
            target_kind="class",
            focus_file="sample.py",
        )
        self.assertEqual(normalized, "")
        self.assertEqual(err, "multiple_top_level_symbols_detected")

    def test_accepts_js_class_with_nested_method(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "class Product {\n"
            "  restock(amount) {\n"
            "    if (amount < 0) throw new Error('bad');\n"
            "    this.inStock += amount;\n"
            "  }\n"
            "}\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="Product",
            target_kind="class",
            focus_file="sample.js",
        )
        self.assertIsNone(err)
        self.assertIn("class Product", normalized)

    def test_rejects_js_class_with_extra_top_level_function(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "class Product {\n"
            "  restock(amount) {\n"
            "    this.inStock += amount;\n"
            "  }\n"
            "}\n"
            "function rogue() { return 1; }\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="Product",
            target_kind="class",
            focus_file="sample.js",
        )
        self.assertEqual(normalized, "")
        self.assertIn(err, {"multiple_top_level_symbols_detected", "multiple_root_level_blocks"})

    def test_rejects_forbidden_content(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "#include <stdio.h>\n"
            "int add(int a, int b) { return a + b; }\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="add",
            target_kind="function",
            focus_file="sample.c",
        )
        self.assertEqual(normalized, "")
        self.assertEqual(err, "includes_not_allowed")

    def test_rejects_line_prefix_and_multiple_symbols(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "| int add(int a, int b) { return a + b; }\n"
            "int mul(int a, int b) { return a * b; }\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(output)
        self.assertEqual(normalized, "")
        self.assertEqual(err, "line_number_prefix_detected")

    def test_rejects_wrong_target(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "int mul(int a, int b) { return a * b; }\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="add",
            target_kind="function",
            focus_file="sample.c",
        )
        self.assertEqual(normalized, "")
        self.assertEqual(err, "target_symbol_missing_in_output")

    def test_allows_balanced_brace_growth(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "int add(int a, int b) {\n"
            "  if (a > 0) {\n"
            "    return a + b;\n"
            "  }\n"
            "  return b;\n"
            "}\n"
            "END_SYMBOL\n"
        )
        original_symbol = (
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="add",
            original_symbol_text=original_symbol,
            target_kind="function",
            focus_file="sample.c",
        )
        self.assertIsNone(err)
        self.assertIn("if (a > 0)", normalized)

    def test_rejects_multiple_root_level_blocks(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "int add(int a, int b) { return a + b; }\n"
            "{ int x = 1; }\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="add",
            target_kind="function",
            focus_file="sample.c",
        )
        self.assertEqual(normalized, "")
        self.assertEqual(err, "multiple_root_level_blocks")

    def test_rejects_python_one_liner_def(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "def apply_discount(product, percent): return product\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="apply_discount",
            target_kind="function",
            focus_file="sample.py",
        )
        self.assertEqual(normalized, "")
        self.assertEqual(err, "python_one_liner_def_not_allowed")

    def test_accepts_python_multiline_with_non_four_space_indent(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "def apply_discount(product, percent):\n"
            "  return product\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="apply_discount",
            focus_file="sample.py",
        )
        self.assertIsNone(err)
        self.assertIn("def apply_discount", normalized)

    def test_accepts_python_typed_multiline_header(self) -> None:
        output = (
            "BEGIN_SYMBOL\n"
            "def apply_discount(product: Product, percent: float) -> None:\n"
            "    return\n"
            "END_SYMBOL\n"
        )
        normalized, err = normalize_structural_output(
            output,
            target_symbol="apply_discount",
            focus_file="sample.py",
        )
        self.assertIsNone(err)
        self.assertIn("def apply_discount", normalized)


if __name__ == "__main__":
    unittest.main()
