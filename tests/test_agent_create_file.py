"""
Hard create-file tasks: same routing and flow as the user (run_agent → model → tool loop).

Uses an in-repo test fixture only: tests/fixtures/w7 (not your machine's directories).
Root is set to that path for the run; replace or add fixtures under tests/fixtures/ as needed.

To avoid high RAM use, run one test at a time in a separate process:

  python tests/run_integration_one_at_a_time.py

To run this module alone (all tests in one process, more RAM):

  python3 -m unittest tests.test_agent_create_file -v
"""

import sys
from pathlib import Path
from unittest import TestCase, main as unittest_main

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Use same model and env as app; disable debug during tests to save RAM
import tests._load_app_model_env as _model_env
_model_env.load_app_model_env(reduce_debug_for_tests=True)

# In-repo fixture path only (not the user's CS lab or home directory)
FIXTURES_W7 = REPO / "tests" / "fixtures" / "w7"


def _setup_workspace():
    if not FIXTURES_W7.is_dir():
        raise FileNotFoundError("tests/fixtures/w7 not found")
    # Fail fast if key fixture files are missing (avoids confusing agent errors)
    for name in ("checkpasswd.c", "validate.c", "Makefile"):
        if not (FIXTURES_W7 / name).exists():
            raise FileNotFoundError(f"tests/fixtures/w7/{name} missing; needed for integration tests")
    from ai_shell.files import set_root, set_include
    set_root(str(FIXTURES_W7))
    set_include(None)


class TestAgentCreateFile(TestCase):
    """Hard tasks: create files (and optionally edit existing). Same flow as user."""

    def setUp(self):
        _setup_workspace()

    def test_create_readme_from_source(self):
        """Read the two C files, then create README.md describing what they do and how they work together."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Read checkpasswd.c and validate.c. Create a new file README.md that describes "
            "what each program does and how they work together. Use only what you see in the code and comments."
        )
        output, meta = run_agent(spec, focus_file="checkpasswd.c", mode="agent", extra_read_files=["validate.c"], max_rounds=8)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create or edit at least one file; touched={touched!r}")
        readme = FIXTURES_W7 / "README.md"
        self.assertTrue(readme.exists(), "README.md must exist")
        content = readme.read_text().lower()
        self.assertIn("checkpasswd", content, "README must mention checkpasswd")
        self.assertIn("validate", content, "README must mention validate")

    def test_create_common_header_and_include_in_checkpasswd(self):
        """Create common.h with MAXLINE and MAX_PASSWORD; edit checkpasswd.c to include it and remove local defines."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Create common.h with #define MAXLINE 256 and #define MAX_PASSWORD 10. "
            "Then edit checkpasswd.c to #include \"common.h\" and remove its own #define MAXLINE and #define MAX_PASSWORD."
        )
        output, meta = run_agent(spec, focus_file="checkpasswd.c", mode="agent", max_rounds=8)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create or edit files; touched={touched!r}")
        common_h = FIXTURES_W7 / "common.h"
        self.assertTrue(common_h.exists(), "common.h must exist")
        h_text = common_h.read_text()
        self.assertIn("MAXLINE", h_text)
        self.assertIn("MAX_PASSWORD", h_text)
        checkpasswd = (FIXTURES_W7 / "checkpasswd.c").read_text()
        self.assertIn("common.h", checkpasswd, "checkpasswd.c must include common.h")

    def test_create_run_script_that_builds_and_runs(self):
        """Create run.sh: run make, then run ./checkpasswd with stdin from test_input.txt. Shebang required."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Look at the Makefile and files here. Create run.sh that runs make, then runs ./checkpasswd "
            "with stdin from test_input.txt (if it exists). Script must start with #!/bin/sh."
        )
        output, meta = run_agent(spec, focus_file=None, mode="agent", max_rounds=8)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create run.sh; touched={touched!r}")
        run_sh = FIXTURES_W7 / "run.sh"
        self.assertTrue(run_sh.exists(), "run.sh must exist")
        text = run_sh.read_text()
        self.assertIn("make", text.lower())
        self.assertTrue(text.lstrip().startswith("#!"), "run.sh must have shebang")

    def test_create_helper_c_and_header_with_trim_newline(self):
        """Create helper.c with trim_newline(char *buf, int size) and helper.h declaring it; header must have include guards."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Create helper.c with a function void trim_newline(char *buf, int size) that sets the trailing newline to '\\0'. "
            "Create helper.h that declares it with include guards (#ifndef HELPER_H etc)."
        )
        output, meta = run_agent(spec, focus_file="checkpasswd.c", mode="agent", max_rounds=8)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create helper.c and/or helper.h; touched={touched!r}")
        helper_c = FIXTURES_W7 / "helper.c"
        helper_h = FIXTURES_W7 / "helper.h"
        self.assertTrue(helper_c.exists(), "helper.c must exist")
        self.assertTrue(helper_h.exists(), "helper.h must exist")
        self.assertIn("trim_newline", helper_c.read_text())
        self.assertIn("trim_newline", helper_h.read_text())
        h_text = helper_h.read_text()
        self.assertIn("#ifndef", h_text.replace(" ", ""), "helper.h must have include guards")

    def test_create_build_doc_from_makefile(self):
        """Read Makefile and create BUILD.md that explains how to build and what targets exist."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Read the Makefile. Create a new file BUILD.md that explains how to build this project, "
            "what targets exist (e.g. all, validate, checkpasswd, clean), and what each does."
        )
        output, meta = run_agent(spec, focus_file="Makefile", mode="agent", max_rounds=6)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create BUILD.md; touched={touched!r}")
        build_md = FIXTURES_W7 / "BUILD.md"
        self.assertTrue(build_md.exists(), "BUILD.md must exist")
        text = build_md.read_text().lower()
        self.assertIn("make", text)
        self.assertTrue("checkpasswd" in text or "validate" in text, "BUILD.md must mention at least one target")

    def test_create_gitignore_for_build_artifacts(self):
        """Create .gitignore that ignores the built binaries and common C artifacts."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Look at the Makefile to see what gets built. Create .gitignore so that the built executables "
            "and common C build artifacts (e.g. *.o, *.d) are ignored."
        )
        output, meta = run_agent(spec, focus_file="Makefile", mode="agent", max_rounds=6)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create .gitignore; touched={touched!r}")
        gitignore = FIXTURES_W7 / ".gitignore"
        self.assertTrue(gitignore.exists(), ".gitignore must exist")
        text = gitignore.read_text()
        self.assertTrue("checkpasswd" in text or "validate" in text or "*" in text, ".gitignore must list binaries or a pattern")

    def test_create_pass_example_from_format(self):
        """Read pass.txt format and create pass_example.txt with one valid example line."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Read pass.txt to see the format (user and password lines). Create pass_example.txt "
            "with one example user/password pair in the same format."
        )
        output, meta = run_agent(spec, focus_file="pass.txt", mode="agent", max_rounds=6)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create pass_example.txt; touched={touched!r}")
        example = FIXTURES_W7 / "pass_example.txt"
        self.assertTrue(example.exists(), "pass_example.txt must exist")
        self.assertGreater(len(example.read_text().strip()), 0, "pass_example.txt must not be empty")

    def test_create_usage_from_comments(self):
        """Create USAGE.md from the comments in validate.c (how stdin is used, exit codes)."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Read validate.c and its comments. Create USAGE.md that documents: what validate reads from stdin, "
            "what the exit codes mean (0, 1, 2, 3), and how to run it."
        )
        output, meta = run_agent(spec, focus_file="validate.c", mode="agent", max_rounds=6)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create USAGE.md; touched={touched!r}")
        usage = FIXTURES_W7 / "USAGE.md"
        self.assertTrue(usage.exists(), "USAGE.md must exist")
        text = usage.read_text().lower()
        self.assertIn("stdin", text)
        self.assertTrue("exit" in text or "0" in text or "2" in text or "3" in text, "USAGE must mention exit codes or values")

    def test_create_clean_script(self):
        """Create clean.sh that runs make clean."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Look at the Makefile. Create clean.sh that runs 'make clean'. Use #!/bin/sh."
        )
        output, meta = run_agent(spec, focus_file="Makefile", mode="agent", max_rounds=4)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create clean.sh; touched={touched!r}")
        clean_sh = FIXTURES_W7 / "clean.sh"
        self.assertTrue(clean_sh.exists(), "clean.sh must exist")
        text = clean_sh.read_text()
        self.assertIn("clean", text.lower())
        self.assertTrue(text.lstrip().startswith("#!"), "clean.sh must have shebang")

    def test_create_sources_list(self):
        """Create SOURCES.txt listing the C source files in this project (from Makefile or directory)."""
        from ai_shell.agent_loop import run_agent

        spec = (
            "Figure out which C source files this project has (read Makefile or list files). "
            "Create SOURCES.txt with one filename per line listing those .c files."
        )
        output, meta = run_agent(spec, focus_file=None, mode="agent", max_rounds=6)
        touched = meta.get("touched", [])
        self.assertGreater(len(touched), 0, f"Agent should create SOURCES.txt; touched={touched!r}")
        sources = FIXTURES_W7 / "SOURCES.txt"
        self.assertTrue(sources.exists(), "SOURCES.txt must exist")
        text = sources.read_text()
        self.assertIn("validate", text)
        self.assertIn("checkpasswd", text)


if __name__ == "__main__":
    unittest_main(verbosity=2)
