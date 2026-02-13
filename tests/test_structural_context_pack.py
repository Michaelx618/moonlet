import unittest

from ai_shell.structural import build_packed_context, build_symbol_index


class StructuralContextPackTests(unittest.TestCase):
    def _sample_content(self) -> str:
        return (
            "from dataclasses import dataclass\n"
            "MAX_DISCOUNT = 0.25\n"
            "\n"
            "@dataclass\n"
            "class Product:\n"
            "    name: str\n"
            "    price: float\n"
            "\n"
            "def helper_round(value: float) -> float:\n"
            "    return round(value, 2)\n"
            "\n"
            "def apply_discount(product: Product, percent: float) -> float:\n"
            "    rate = min(percent, MAX_DISCOUNT)\n"
            "    return helper_round(product.price * (1 - rate))\n"
            "\n"
            "def restock(product: Product, amount: int) -> None:\n"
            "    product.price += amount * 0.0\n"
            "\n"
            "def run_checkout() -> float:\n"
            "    p = Product('Keyboard', 100.0)\n"
            "    return apply_discount(p, 0.1)\n"
        )

    def test_builds_target_focused_context(self) -> None:
        content = self._sample_content()
        index = build_symbol_index("sample.py", content)
        target = [s for s in index if s.name == "apply_discount"][0]

        packed = build_packed_context(
            target=target,
            content=content,
            index=index,
            user_text="fix behavior for discount rounding",
            focus_file="sample.py",
            max_lines=120,
            max_bytes=4096,
        )

        self.assertIn("TARGET_SYMBOL: function apply_discount", packed)
        self.assertIn("def apply_discount(product: Product, percent: float) -> float:", packed)
        self.assertIn("OUTPUT CONTRACT:", packed)
        self.assertIn("helper_round", packed)

    def test_respects_context_budget(self) -> None:
        content = self._sample_content()
        index = build_symbol_index("sample.py", content)
        target = [s for s in index if s.name == "apply_discount"][0]

        packed = build_packed_context(
            target=target,
            content=content,
            index=index,
            user_text="fix behavior",
            focus_file="sample.py",
            max_lines=40,
            max_bytes=700,
        )

        self.assertLessEqual(len(packed.splitlines()), 40)
        self.assertLessEqual(len(packed.encode("utf-8")), 700)
        self.assertIn("def apply_discount", packed)

    def test_packed_context_smaller_than_full_file_for_large_files(self) -> None:
        content = self._sample_content() + "\n" + "\n".join(
            f"# filler line {i}" for i in range(300)
        )
        index = build_symbol_index("sample.py", content)
        target = [s for s in index if s.name == "apply_discount"][0]

        packed = build_packed_context(
            target=target,
            content=content,
            index=index,
            user_text="fix behavior",
            focus_file="sample.py",
            max_lines=120,
            max_bytes=4096,
        )

        self.assertLess(len(packed), len(content))


if __name__ == "__main__":
    unittest.main()
