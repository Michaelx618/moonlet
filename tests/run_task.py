#!/usr/bin/env python3
"""
Generic task runner: run pipeline on any spec + project, with verify and repair loop.

Usage:
  cd /Users/michael/moonlet
  python3 tests/run_task.py tests/fixtures/w7
  # Or with env:
  SC2_GGUF=... SC2_USE_CHAT_TOOLS=1 python3 tests/run_task.py /path/to/task_dir

Task directory should contain:
  - spec.txt (or spec.md): the instruction for the model
  - task.json (optional): { "focus_file": "...", "extra_read_files": [...] }
  - test_input.txt (optional): stdin for the built executable
  - Makefile, etc.: build system (auto-detected)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("SC2_DEBUG", "1")
os.environ.setdefault("SC2_DEBUG_LOG", str(REPO / "runtime-debug.log"))
os.environ.setdefault("SC2_STAGE_EDITS", "0")


def load_task_config(task_dir: Path) -> dict:
    """Load task.json if present."""
    cfg_path = task_dir / "task.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def load_spec(task_dir: Path) -> str:
    """Load spec from spec.txt or spec.md."""
    for name in ("spec.txt", "spec.md"):
        p = task_dir / name
        if p.exists():
            return p.read_text().strip()
    raise FileNotFoundError(f"No spec.txt or spec.md in {task_dir}")


def run_verify(root: Path, staged_paths: list[str]) -> tuple[int, str, str, str]:
    """Build and run tests. Returns (exit_code, stdout, stderr, error_summary)."""
    from ai_shell.verify import run_verify as _run_verify

    return _run_verify(root=root, staged_paths=staged_paths)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tests/run_task.py <task_dir>")
        print("  task_dir: path to task (must contain spec.txt)")
        sys.exit(1)

    task_dir = Path(sys.argv[1]).resolve()
    if not task_dir.is_dir():
        print(f"Task dir not found: {task_dir}")
        sys.exit(1)

    try:
        spec = load_spec(task_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    config = load_task_config(task_dir)
    focus_file = config.get("focus_file")
    extra_read_files = config.get("extra_read_files") or []

    from ai_shell.files import set_root
    from ai_shell.agent_loop import run_agent

    set_root(str(task_dir))
    print(f"Task: {task_dir}")
    print("Running agent...")

    output, meta = run_agent(
        spec,
        focus_file=focus_file,
        mode="agent",
        silent=False,
        extra_read_files=extra_read_files if extra_read_files else None,
    )
    touched = meta.get("touched", [])
    if not touched:
        print(f"\nFAIL: No edits. Output: {output[:500]}")
        sys.exit(1)

    # Verify: build + run
    rc, stdout, stderr, err_summary = run_verify(task_dir, touched)
    success_criterion = config.get("success_criterion", {})

    def passed(exit_code: int, out: str, _err: str) -> bool:
        if exit_code != 0:
            return False
        if "stdout_contains" in success_criterion:
            return success_criterion["stdout_contains"] in (out or "")
        return True

    if passed(rc, stdout, stderr):
        print("\nBuild OK")
        if stdout:
            print(stdout.strip())
        print("\nTask PASSED")
        return

    print(f"\nVerify failed (exit {rc}):")
    if err_summary:
        print(err_summary[:800])
    else:
        print(stderr[:500] if stderr else stdout[:500])

    # Repair loop
    max_repairs = int(os.environ.get("SC2_MAX_REPAIR_ITERATIONS", "3"))
    for i in range(max_repairs):
        print(f"\n--- Repair attempt {i + 1}/{max_repairs} ---")
        last_error = err_summary or f"Build/runtime failed. stdout: {stdout[:300]!r} stderr: {stderr[:300]!r}"
        retry_text = f"{spec}\n\nRepair target:\n{last_error}"
        output, meta = run_agent(
            retry_text,
            focus_file=(touched[0] if touched else focus_file),
            mode="repair",
            silent=False,
            extra_read_files=extra_read_files or list(touched) or None,
        )
        touched = meta.get("touched", [])
        if not touched:
            print(f"Repair produced no edits: {output[:300]}")
            continue

        rc, stdout, stderr, err_summary = run_verify(task_dir, touched)
        if passed(rc, stdout, stderr):
            print("\nBuild OK")
            if stdout:
                print(stdout.strip())
            print("\nTask PASSED (after repair)")
            return

        print(f"Verify still failed: {err_summary[:400] if err_summary else stderr[:200]}")

    print("\nTask FAILED after all repair attempts")
    sys.exit(1)


if __name__ == "__main__":
    main()
