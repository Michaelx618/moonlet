"""Regression eval harness for Moonlet (agent_loop and ask_plan).

Supports two modes:
- agent: run_agent() - full tool loop, edits allowed. Pass = meta.ok and (optionally) touched.
- plan: run_plan() - read-only loop, produces plan text. Pass = non-empty output with plan-like content.

Rule: Run tests one at a time. Each test runs in its own process and exits when done.
Use run_single_task.py to run a single task (process exits after).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_shell.agent_loop import run_agent
from ai_shell.ask_plan import run_ask, run_plan
from ai_shell.files import get_root, set_root, set_include


def _load_tasks(task_file: str) -> List[Dict[str, Any]]:
    path = Path(task_file)
    if not path.is_absolute():
        path = (get_root() / task_file).resolve()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        return []
    return [t for t in tasks if isinstance(t, dict)]


def _apply_task_setup(task: Dict[str, Any]) -> None:
    """Apply root/include from task if present. Restores agent context for fixture."""
    root = task.get("root")
    if root:
        root_path = Path(root)
        if not root_path.is_absolute():
            # Resolve relative to repo root (parent of tests/)
            repo_root = Path(__file__).resolve().parent.parent.parent
            root_path = (repo_root / root).resolve()
        set_root(str(root_path))
    include = task.get("include")
    if include is not None:
        set_include(include if isinstance(include, list) else [])


def _check_plan_output(output: str, task: Dict[str, Any]) -> bool:
    """Plan mode pass: output should be non-empty and contain plan-like structure."""
    if not (output or "").strip():
        return False
    # Require some structure: steps, numbers, or file references
    has_structure = (
        bool(re.search(r"\b(step|plan|1\.|2\.|first|second|then)\b", output, re.I))
        or bool(re.search(r"\.(c|py|h|txt)\b", output))
        or len(output.strip()) >= 100
    )
    return has_structure


def run_single_task(
    task_file: str,
    task_index: int,
    *,
    stop_on_failure: bool = False,
) -> Dict[str, Any]:
    """Run exactly one task by index (1-based). Process should exit after.
    Returns result dict. Use for sequential one-at-a-time runs."""
    tasks = _load_tasks(task_file)
    if not tasks:
        return {"ok": False, "error": "no_tasks", "task_file": task_file}
    if task_index < 1 or task_index > len(tasks):
        return {
            "ok": False,
            "error": "invalid_index",
            "task_index": task_index,
            "total": len(tasks),
        }
    task = tasks[task_index - 1]
    spec = str(task.get("spec") or "").strip()
    if not spec:
        return {"ok": False, "error": "empty_spec", "task_index": task_index}

    mode = str(task.get("mode") or "agent").strip().lower()
    focus = str(task.get("focus_file") or "") or None
    extra = list(task.get("extra_read_files") or [])
    context_folders = list(task.get("context_folders") or [])

    _apply_task_setup(task)

    start = time.time()
    try:
        if mode == "plan":
            output, meta = run_plan(
                spec,
                focus_file=focus,
                silent=True,
                extra_read_files=extra if extra else None,
                context_folders=context_folders if context_folders else None,
                max_rounds=8,
            )
            passed = _check_plan_output(output or "", task)
            has_touched = False
        elif mode == "ask":
            output, meta = run_ask(
                spec,
                focus_file=focus,
                silent=True,
                extra_read_files=extra if extra else None,
                context_folders=context_folders if context_folders else None,
                max_rounds=8,
            )
            passed = _check_plan_output(output or "", task)  # same: non-empty, structured
            has_touched = False
        else:
            output, meta = run_agent(
                spec,
                focus_file=focus,
                mode="agent",
                silent=True,
                extra_read_files=extra if extra else None,
                context_folders=context_folders if context_folders else None,
            )
            has_touched = bool(meta.get("touched"))
            passed = bool(meta.get("ok"))
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "ok": False,
            "task_index": task_index,
            "name": task.get("name") or f"task_{task_index}",
            "mode": mode,
            "error": str(e),
            "latency_ms": elapsed_ms,
        }

    elapsed_ms = int((time.time() - start) * 1000)
    return {
        "ok": passed,
        "task_index": task_index,
        "name": task.get("name") or f"task_{task_index}",
        "mode": mode,
        "touched": has_touched,
        "latency_ms": elapsed_ms,
        "summary": meta.get("summary") or (output[:300] if output else ""),
    }


def run_eval_suite(
    task_file: str,
    stop_on_failure: bool = False,
    task_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run eval suite. If task_indices given, run only those (1-based).
    For one-at-a-time runs, prefer run_single_task() in separate processes."""
    tasks = _load_tasks(task_file)
    if not tasks:
        return {"ok": False, "error": "no_tasks", "task_file": task_file}

    indices = task_indices if task_indices is not None else list(range(1, len(tasks) + 1))
    rows: List[Dict[str, Any]] = []
    touched_ok = 0
    verify_ok = 0
    latencies: List[int] = []

    for i in indices:
        if i < 1 or i > len(tasks):
            continue
        task = tasks[i - 1]
        spec = str(task.get("spec") or "").strip()
        if not spec:
            continue

        mode = str(task.get("mode") or "agent").strip().lower()
        focus = str(task.get("focus_file") or "") or None
        extra = list(task.get("extra_read_files") or [])
        context_folders = list(task.get("context_folders") or [])

        _apply_task_setup(task)

        start = time.time()
        try:
            if mode == "plan":
                output, meta = run_plan(
                    spec,
                    focus_file=focus,
                    silent=True,
                    extra_read_files=extra if extra else None,
                    context_folders=context_folders if context_folders else None,
                    max_rounds=8,
                )
                passed = _check_plan_output(output or "", task)
                has_touched = False
            elif mode == "ask":
                output, meta = run_ask(
                    spec,
                    focus_file=focus,
                    silent=True,
                    extra_read_files=extra if extra else None,
                    context_folders=context_folders if context_folders else None,
                    max_rounds=8,
                )
                passed = _check_plan_output(output or "", task)
                has_touched = False
            else:
                output, meta = run_agent(
                    spec,
                    focus_file=focus,
                    mode="agent",
                    silent=True,
                    extra_read_files=extra if extra else None,
                    context_folders=context_folders if context_folders else None,
                )
                has_touched = bool(meta.get("touched"))
                passed = bool(meta.get("ok"))
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            rows.append(
                {
                    "index": i,
                    "name": task.get("name") or f"task_{i}",
                    "ok": False,
                    "touched": False,
                    "latency_ms": elapsed_ms,
                    "summary": str(e),
                }
            )
            if stop_on_failure:
                break
            continue

        elapsed_ms = int((time.time() - start) * 1000)
        latencies.append(elapsed_ms)
        if has_touched:
            touched_ok += 1
        if passed:
            verify_ok += 1
        rows.append(
            {
                "index": i,
                "name": task.get("name") or f"task_{i}",
                "ok": passed,
                "touched": has_touched,
                "latency_ms": elapsed_ms,
                "summary": meta.get("summary") or "",
            }
        )
        if stop_on_failure and not passed:
            break

    total = max(1, len(rows))
    return {
        "ok": verify_ok == len(rows),
        "task_file": task_file,
        "total": len(rows),
        "touched_rate": round(touched_ok / total, 3),
        "verify_pass_rate": round(verify_ok / total, 3),
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "results": rows,
    }

