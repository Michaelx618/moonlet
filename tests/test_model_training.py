#!/usr/bin/env python3
"""
Test script to verify model training for search_replace / tool use.

Runs a few coding tasks similar to training and checks:
1. Does the model produce tool calls (search_replace)?
2. Are the edits semantically correct?
3. Does the output match the instruction?

Usage:
  cd /Users/michael/moonlet
  SC2_GGUF=/path/to/ds67b.gguf SC2_USE_CHAT_TOOLS=1 python tests/test_model_training.py

Or: python -m tests.test_model_training
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("SC2_DEBUG", "1")
os.environ.setdefault("SC2_DEBUG_LOG", str(REPO / "runtime-debug.log"))


def run_test(name: str, spec: str, test_file: Path, expected_contains: list) -> str | None:
    """Run agent, return error message or None if passed."""
    from ai_shell.files import set_root
    from ai_shell.agent_loop import run_agent

    root = test_file.parent
    set_root(str(root))

    try:
        output, meta = run_agent(spec, silent=True, focus_file=test_file.name, mode="agent")
    except Exception as e:
        return f"Agent error: {e}"

    touched = meta.get("touched", [])
    if not touched:
        return f"No files touched. Output: {output[:300]}"

    content = (root / test_file.name).read_text()
    for needle in expected_contains:
        if needle not in content:
            return f"Expected {needle!r} not found. Got:\n{content[:500]}"

    return None


def test_tools_direct():
    """Test read_file, grep, search_replace implementations directly (no model)."""
    from ai_shell.files import set_root, set_include
    from ai_shell.tool_executor import execute_tool_from_kwargs
    from ai_shell.search_replace import apply_search_replace_edits
    from ai_shell.index import rebuild_index

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        set_root(str(tmp))

        # Setup
        (tmp / "a.c").write_text('int foo() { return 42; }\n')
        (tmp / "b.c").write_text('int bar() { return foo(); }\n')
        rebuild_index()
        set_include(["a.c", "b.c"])

        # read_file
        out = execute_tool_from_kwargs("read_file", {"path": "a.c"})
        assert "42" in out, f"read_file: {out}"
        print("  read_file: OK")

        # grep
        out = execute_tool_from_kwargs("grep", {"pattern": "foo", "path": "."})
        assert "No matches" not in out and "foo" in out, f"grep: {out}"
        print("  grep: OK")

        # search_replace (direct apply)
        edits = [{"old_string": "return 42", "new_string": "return 99", "path": "a.c"}]
        touched, _, staged, _ = apply_search_replace_edits(edits, stage_only=False)
        assert "a.c" in touched
        assert "99" in (tmp / "a.c").read_text()
        print("  search_replace: OK")


def main():
    print("=== Tool Usage Tests (direct, no model) ===\n")
    try:
        test_tools_direct()
        print("  All tool impl tests PASS\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")

    print("=== Model Training Verification Tests ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Test 1: Simple string replacement
        print("Test 1: Simple search_replace (replace HELLO with WORLD)")
        f1 = tmp / "hello.c"
        f1.write_text('''#include <stdio.h>
int main() {
  printf("HELLO\\n");
  return 0;
}
''')
        err = run_test(
            "simple_replace",
            'Replace "HELLO" with "WORLD" in the printf.',
            f1,
            expected_contains=["WORLD"],
        )
        if err:
            print(f"  FAIL: {err}\n")
        else:
            print("  PASS\n")

        # Test 2: Add a line of code
        print("Test 2: Add a variable declaration")
        f2 = tmp / "add_var.c"
        f2.write_text('''#include <stdio.h>
int main() {
  return 0;
}
''')
        err = run_test(
            "add_var",
            'Add a variable "int x = 42;" at the start of main, before the return.',
            f2,
            expected_contains=["int x = 42", "return 0"],
        )
        if err:
            print(f"  FAIL: {err}\n")
        else:
            print("  PASS\n")

        # Test 3: read_file — model must read a file NOT in prompt to make edit
        print("Test 3: read_file (read config.h, use MAX_BUF in main.c)")
        (tmp / "config.h").write_text("#define MAX_BUF 256\n")
        f3a = tmp / "main.c"
        f3a.write_text('''#include <stdio.h>
int main() {
  char buf[128];
  return 0;
}
''')
        err = run_test(
            "read_file",
            "Read config.h to get MAX_BUF. Then change buf[128] to buf[MAX_BUF] in main.c.",
            f3a,
            expected_contains=["MAX_BUF"],
        )
        if err:
            print(f"  FAIL: {err}\n")
        else:
            print("  PASS\n")

        # Test 4: grep — model must grep to find, then search_replace
        print("Test 4: grep (find legacy_func, replace with new_func)")
        f4a = tmp / "utils.c"
        f4a.write_text('''int legacy_func(int x) { return x + 1; }
''')
        f4b = tmp / "app.c"
        f4b.write_text('''#include "utils.h"
int main() { return legacy_func(0); }
''')
        err = run_test(
            "grep",
            "Use grep to find legacy_func. Replace it with new_func everywhere.",
            f4b,
            expected_contains=["new_func"],
        )
        if err:
            print(f"  FAIL: {err}\n")
        else:
            print("  PASS\n")

        # Test 5: Fork/exec style (like checkpasswd training)
        print("Test 5: Add fork/exec logic (like checkpasswd training)")
        f5 = tmp / "call_validate.c"
        f5.write_text('''#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
  char user_id[256];
  char password[256];

  if (fgets(user_id, 256, stdin) == NULL) { perror("fgets"); exit(1); }
  if (fgets(password, 256, stdin) == NULL) { perror("fgets"); exit(1); }

  return 0;
}
''')
        err = run_test(
            "fork_exec",
            "Add code to fork a child process and call validate. Pass user_id and password to validate over stdin. "
            "Use pipe() or dup2() to connect stdin. After the fgets calls, add the fork/exec logic before return.",
            f5,
            expected_contains=["fork", "exec"],
        )
        if err:
            print(f"  FAIL: {err}\n")
        else:
            print("  PASS\n")

    print("=== Done ===")


if __name__ == "__main__":
    main()
