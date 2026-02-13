import unittest

from ai_shell.structural import build_symbol_index


class StructuralSymbolIndexTests(unittest.TestCase):
    def test_extracts_symbols_with_byte_ranges(self) -> None:
        content = (
            "#include <stdio.h>\n"
            "int add(int a, int b) {\n"
            "  return a + b;\n"
            "}\n"
            "int mul(int a, int b) {\n"
            "  return a * b;\n"
            "}\n"
        )
        symbols = build_symbol_index("sample.c", content)
        names = {s.name for s in symbols}
        self.assertIn("add", names)
        self.assertIn("mul", names)
        add = [s for s in symbols if s.name == "add"][0]
        self.assertGreater(add.byte_end, add.byte_start)
        self.assertGreaterEqual(add.line_start, 1)
        self.assertGreaterEqual(add.line_end, add.line_start)

    def test_python_class_methods_are_nested_under_parent(self) -> None:
        content = (
            "class Product:\n"
            "    def apply_discount(self, percent):\n"
            "        return None\n"
            "\n"
            "    def restock(self, amount):\n"
            "        return None\n"
        )
        symbols = build_symbol_index("sample.py", content)
        top = [s for s in symbols if not (s.parent or "").strip()]
        nested = [s for s in symbols if (s.parent or "").strip()]
        self.assertEqual(len([s for s in top if s.kind == "class" and s.name == "Product"]), 1)
        self.assertEqual(len([s for s in nested if s.parent == "Product" and s.kind == "method"]), 2)

    def test_js_class_methods_are_nested_under_parent(self) -> None:
        content = (
            "class Product {\n"
            "  applyDiscount(percent) {\n"
            "    return percent;\n"
            "  }\n"
            "  restock(amount) {\n"
            "    return amount;\n"
            "  }\n"
            "}\n"
        )
        symbols = build_symbol_index("sample.js", content)
        top = [s for s in symbols if not (s.parent or "").strip()]
        nested = [s for s in symbols if (s.parent or "").strip()]
        self.assertEqual(len([s for s in top if s.kind == "class" and s.name == "Product"]), 1)
        self.assertEqual(len([s for s in nested if s.parent == "Product" and s.kind == "method"]), 2)


if __name__ == "__main__":
    unittest.main()
