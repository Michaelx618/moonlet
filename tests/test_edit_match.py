"""Tests for Continue-style edit match strategies and search_replace validation."""

import unittest

from ai_shell.edit_match import find_search_match, find_search_matches
from ai_shell.search_replace import (
    SearchReplaceError,
    validate_single_edit,
    execute_find_and_replace,
    execute_multi_find_and_replace,
)


class TestEditMatch(unittest.TestCase):
    def test_find_search_matches_exact(self):
        self.assertEqual(find_search_matches("hello world", "world"), [(6, 11)])
        self.assertEqual(find_search_matches("a b a b", "a"), [(0, 1), (4, 5)])

    def test_find_search_matches_empty(self):
        self.assertEqual(find_search_matches("x", ""), [(0, 0)])
        self.assertEqual(find_search_matches("x", "   "), [(0, 0)])

    def test_find_search_match_trimmed(self):
        # Exact "hello" not in file; trimmed "  hello  " -> "hello" matches
        m = find_search_match(" \thello\t ", "  hello  ")
        self.assertIsNotNone(m)
        start, end, name = m
        self.assertEqual(name, "trimmed")
        self.assertIn("hello", " \thello\t "[start:end])


class TestValidateSingleEdit(unittest.TestCase):
    def test_ok(self):
        o, n, r = validate_single_edit("old", "new", False)
        self.assertEqual(o, "old")
        self.assertEqual(n, "new")
        self.assertFalse(r)

    def test_reject_same(self):
        with self.assertRaises(SearchReplaceError):
            validate_single_edit("same", "same", False)

    def test_reject_empty_old(self):
        with self.assertRaises(SearchReplaceError):
            validate_single_edit(None, "new", False)


class TestExecuteFindAndReplace(unittest.TestCase):
    def test_single(self):
        out = execute_find_and_replace("foo bar baz", "bar", "qux", replace_all=False)
        self.assertEqual(out, "foo qux baz")

    def test_replace_all(self):
        out = execute_find_and_replace("a b a b", "a", "x", replace_all=True)
        self.assertEqual(out, "x b x b")

    def test_not_found(self):
        with self.assertRaises(SearchReplaceError):
            execute_find_and_replace("hello", "xyz", "a", replace_all=False)

    def test_multi(self):
        out = execute_multi_find_and_replace(
            "a b c",
            [
                {"old_string": "a", "new_string": "A"},
                {"old_string": "c", "new_string": "C"},
            ],
        )
        self.assertEqual(out, "A b C")


if __name__ == "__main__":
    unittest.main()
