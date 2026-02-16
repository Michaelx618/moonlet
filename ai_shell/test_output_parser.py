#!/usr/bin/env python3
"""Tests for output_parser: one-liner code blocks, diffs, path inference."""
import sys
import os

# Ensure ai_shell is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable debug logging during tests
os.environ.setdefault("SC2_DEBUG", "0")

from ai_shell.output_parser import (
    _reformat_oneliner_code_block,
    _reformat_oneliner_diff,
    _strip_diff_markdown_fences,
    extract_markdown_lang_only_blocks,
    _infer_path_from_reply_text,
    parse_flexible_output,
)


def test_reformat_oneliner_code_block():
    """One-liner C code should be split into lines."""
    block = "#include <stdio.h> #include <stdlib.h> int main(void) { return 0; }"
    out = _reformat_oneliner_code_block(block)
    assert "#include <stdio.h>" in out
    assert "#include <stdlib.h>" in out
    assert "int main(void)" in out
    assert out.count("\n") >= 2
    print("  ok _reformat_oneliner_code_block")


def test_reformat_oneliner_diff():
    """One-liner diff should be split into lines."""
    block = "--- a/foo.c +++ b/foo.c @@ -1,3 +1,4 @@ #include <stdio.h> #include <stdlib.h> +int x;"
    out = _reformat_oneliner_diff(block)
    assert "--- a/foo.c" in out
    assert "+++ b/foo.c" in out
    assert "@@ -1,3 +1,4 @@" in out
    assert out.count("\n") >= 4
    print("  ok _reformat_oneliner_diff")


def test_strip_diff_one_liner():
    """Diff fence with no newline after ```diff should still extract."""
    text = "```diff --- a/checkpasswd.c +++ b/checkpasswd.c @@ -1,5 +1,6 @@ #include <stdio.h>```"
    out = _strip_diff_markdown_fences(text)
    assert "--- a/checkpasswd.c" in out
    assert "+++ b/checkpasswd.c" in out
    assert "```" not in out
    print("  ok _strip_diff_one_liner")


def test_extract_oneliner_code_block():
    """One-liner ```c block (no newline after fence) should be extracted and reformatted."""
    output = (
        "Here is the updated `checkpasswd.c`: "
        "```c #include <stdio.h> #include <stdlib.h> int main(void) { return 0; } ```"
    )
    blocks = extract_markdown_lang_only_blocks(output, candidate_paths=["w7/checkpasswd.c"])
    assert len(blocks) == 1
    path, body = blocks[0]
    assert path == "w7/checkpasswd.c"
    assert "#include <stdio.h>" in body
    assert body.count("\n") >= 2
    print("  ok extract_oneliner_code_block")


def test_infer_path_from_reply_text():
    """Path should be inferred from 'the `checkpasswd.c` file' in surrounding text."""
    # Block is in the middle; before/after contain the mention
    output = "To complete the `checkpasswd.c` file we need to add: ```c int x; ```"
    block_start = output.find("```c")
    block_end = output.find("```", block_start + 5) + 3
    path = _infer_path_from_reply_text(output, block_start, block_end, ["w7/checkpasswd.c", "Makefile"])
    assert path == "w7/checkpasswd.c"
    print("  ok _infer_path_from_reply_text")


def test_parse_flexible_output_oneliner():
    """parse_flexible_output should handle one-liner code block and return blocks."""
    output = (
        "Here is the updated code: ```c #include <stdio.h> int main(void) { return 0; } ```"
    )
    kind, diff_data, blocks = parse_flexible_output(output, candidate_paths=["checkpasswd.c"])
    assert kind == "blocks"
    assert blocks is not None
    assert len(blocks) >= 1
    path, body = blocks[0]
    assert "checkpasswd.c" in path or path == "checkpasswd.c"
    assert "#include" in body
    assert "int main" in body
    print("  ok parse_flexible_output_oneliner")


def main():
    print("Running output_parser tests...")
    test_reformat_oneliner_code_block()
    test_reformat_oneliner_diff()
    test_strip_diff_one_liner()
    test_extract_oneliner_code_block()
    test_infer_path_from_reply_text()
    test_parse_flexible_output_oneliner()
    print("All tests passed.")


if __name__ == "__main__":
    main()
