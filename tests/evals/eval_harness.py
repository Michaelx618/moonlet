"""Regression eval harness for Moonlet (agent_loop)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ai_shell.agent_loop import run_agent
from ai_shell.files import get_root


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


def run_eval_suite(task_file: str, stop_on_failure: bool = False) -> Dict[str, Any]:
    tasks = _load_tasks(task_file)
    if not tasks:
        return {"ok": False, "error": "no_tasks", "task_file": task_file}
    rows: List[Dict[str, Any]] = []
    touched_ok = 0
    verify_ok = 0
    latencies: List[int] = []
    for i, task in enumerate(tasks, start=1):
        spec = str(task.get("spec") or "").strip()
        if not spec:
            continue
        focus = str(task.get("focus_file") or "") or None
        extra = list(task.get("extra_read_files") or [])
        start = time.time()
        _output, meta = run_agent(
            spec,
            focus_file=focus,
            mode="agent",
            silent=True,
            extra_read_files=extra if extra else None,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        latencies.append(elapsed_ms)
        has_touched = bool(meta.get("touched"))
        passed = bool(meta.get("ok"))
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

