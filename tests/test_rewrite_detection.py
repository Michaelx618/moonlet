import unittest

from ai_shell.router import wants_file_rewrite


class RewriteDetectionTests(unittest.TestCase):
    def test_explicit_rewrite_phrases(self) -> None:
        self.assertTrue(wants_file_rewrite("refactor whole file to improve style"))
        self.assertTrue(wants_file_rewrite("replace entire file with a new version"))

    def test_non_rewrite_request(self) -> None:
        self.assertFalse(wants_file_rewrite("fix add function memory leak"))


if __name__ == "__main__":
    unittest.main()
