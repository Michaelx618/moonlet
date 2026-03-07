#!/usr/bin/env python3
"""Run exactly one eval task, then exit. No model processes left running.

Usage:
  python tests/evals/run_single_task.py <task_file> <task_index> [--timeout N]
  python tests/evals/run_single_task.py plan_tasks.json 1 --timeout 600

Rule: Run one task per process. Process exits when done (success or fail).
For sequential suite with hard kill on timeout: use run_suite_sequential.sh
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

# Ensure repo root is on path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tests.evals.eval_harness import run_single_task

# Default timeout (seconds). 0 = no timeout. Override with --timeout N or EVAL_TASK_TIMEOUT.
_DEFAULT_TIMEOUT = int(os.environ.get("EVAL_TASK_TIMEOUT", "600"))


def _timeout_handler(signum: int, frame: object) -> None:
    raise TimeoutError("Task timed out (hard kill recommended via run_suite_sequential.sh)")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    timeout = _DEFAULT_TIMEOUT
    if "--timeout" in sys.argv:
        idx = sys.argv.index("--timeout")
        if idx + 1 < len(sys.argv):
            try:
                timeout = int(sys.argv[idx + 1])
            except ValueError:
                pass

    if len(args) < 2:
        print("Usage: run_single_task.py <task_file> <task_index> [--timeout N]", file=sys.stderr)
        print("  task_file: e.g. tests/evals/plan_tasks.json, tests/evals/agent_tasks.json", file=sys.stderr)
        print("  task_index: 1-based index", file=sys.stderr)
        return 2

    task_file = args[0]
    # Resolve relative to repo root
    if not Path(task_file).is_absolute():
        task_file = str(_repo_root / task_file)
    try:
        task_index = int(args[1])
    except ValueError:
        print(f"Invalid task_index: {args[1]}", file=sys.stderr)
        return 2

    if timeout > 0:
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
        except (ValueError, OSError):
            pass  # signal not available on this platform

    try:
        result = run_single_task(task_file, task_index)
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1
    except TimeoutError as e:
        print(json.dumps({"ok": False, "error": str(e), "task_index": task_index}, indent=2))
        return 124
    finally:
        if timeout > 0:
            try:
                signal.alarm(0)
            except (ValueError, OSError):
                pass


if __name__ == "__main__":
    sys.exit(main())
