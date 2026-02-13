import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

PLAN_FILENAME = "moonlet_smoke.json"


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    cmd: List[str]
    cwd: str


@dataclass
class SmokeFailure:
    kind: str
    message: str
    command: str = ""
    snippet: str = ""


@dataclass
class SmokeResult:
    passed: bool
    failures: List[SmokeFailure]


_SAFE_EXECUTABLES = {
    "make",
    "cc",
    "gcc",
    "clang",
    "clang++",
    "c++",
    "python",
    "python3",
    "node",
    "npm",
    "npx",
    "bash",
    "sh",
    "go",
    "cargo",
}


def tail_lines(text: str, n: int = 30) -> str:
    lines = (text or "").splitlines()
    if n <= 0:
        return ""
    return "\n".join(lines[-n:])


def _sanitize_timeout(timeout_s: int, fallback: Optional[int] = None) -> int:
    if fallback is None:
        fallback = int(getattr(config, "SMOKE_TIMEOUT_DEFAULT", 20) or 20)
    try:
        value = int(timeout_s)
    except Exception:
        value = fallback
    return max(1, min(300, value))


def _is_cmd_safe(cmd: List[str]) -> bool:
    if not cmd:
        return False
    exe = str(cmd[0] or "").strip()
    if not exe:
        return False
    for tok in cmd:
        tok_s = str(tok or "")
        if any(op in tok_s for op in (";", "&&", "||", "|", "$(", "`")):
            return False
    if exe.startswith("./"):
        norm = os.path.normpath(exe)
        return not norm.startswith("../")
    if "/" in exe:
        return False
    return exe in _SAFE_EXECUTABLES


def run_command(cmd: List[str], cwd: str, timeout_s: int) -> CommandResult:
    if not _is_cmd_safe(cmd):
        return CommandResult(
            exit_code=126,
            stdout="",
            stderr=f"unsafe or disallowed command: {' '.join(cmd)}",
            timed_out=False,
            cmd=cmd,
            cwd=cwd,
        )
    timeout_val = _sanitize_timeout(timeout_s)
    try:
        res = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_val,
        )
        return CommandResult(
            exit_code=int(res.returncode),
            stdout=res.stdout or "",
            stderr=res.stderr or "",
            timed_out=False,
            cmd=cmd,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            exit_code=124,
            stdout=(exc.stdout or ""),
            stderr=((exc.stderr or "") + "\nCommand timed out.").strip(),
            timed_out=True,
            cmd=cmd,
            cwd=cwd,
        )
    except Exception as exc:
        return CommandResult(
            exit_code=127,
            stdout="",
            stderr=str(exc),
            timed_out=False,
            cmd=cmd,
            cwd=cwd,
        )


def _normalize_steps(raw_steps: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in list(raw_steps or []):
        cmd = []
        timeout_s = int(getattr(config, "SMOKE_TIMEOUT_DEFAULT", 20) or 20)
        if isinstance(item, dict):
            cmd = list(item.get("cmd") or [])
            timeout_s = item.get("timeout_s", timeout_s)
        elif isinstance(item, list):
            cmd = list(item)
        if not cmd:
            continue
        out.append(
            {
                "cmd": [str(tok) for tok in cmd],
                "timeout_s": _sanitize_timeout(int(timeout_s)),
            }
        )
    return out


def load_smoke_plan(root: Path) -> Optional[Dict[str, Any]]:
    plan_path = Path(root) / PLAN_FILENAME
    if not plan_path.exists():
        return None
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return {
        "pre_clean": [str(p) for p in list(raw.get("pre_clean") or [])],
        "build": _normalize_steps(raw.get("build")),
        "run": _normalize_steps(raw.get("run")),
        "assert_stdout": list(raw.get("assert_stdout") or []),
        "assert_files": [str(p) for p in list(raw.get("assert_files") or [])],
        "assert_file_magic": list(raw.get("assert_file_magic") or []),
    }


def _failure(
    kind: str,
    message: str,
    command: str = "",
    snippet: str = "",
) -> SmokeResult:
    return SmokeResult(
        passed=False,
        failures=[
            SmokeFailure(
                kind=kind,
                message=message,
                command=command,
                snippet=snippet,
            )
        ],
    )


def _assert_stdout(stdout_text: str, checks: List[Any]) -> Optional[SmokeFailure]:
    for check in checks:
        if isinstance(check, dict) and check.get("regex"):
            pattern = str(check.get("regex") or "")
            if not re.search(pattern, stdout_text, flags=re.MULTILINE):
                return SmokeFailure(
                    kind="ASSERT_STDOUT",
                    message=f"stdout missing regex: {pattern}",
                    snippet=tail_lines(stdout_text, 10),
                )
            continue
        needle = str(check or "")
        if needle and needle not in stdout_text:
            return SmokeFailure(
                kind="ASSERT_STDOUT",
                message=f"stdout missing substring: {needle}",
                snippet=tail_lines(stdout_text, 10),
            )
    return None


def _expected_magic_bytes(spec: Dict[str, Any]) -> bytes:
    if isinstance(spec.get("expected_bytes"), list):
        return bytes([int(v) & 0xFF for v in spec.get("expected_bytes")])
    if isinstance(spec.get("expected_hex"), str):
        clean = spec.get("expected_hex").replace(" ", "")
        return bytes.fromhex(clean)
    return str(spec.get("expected") or "").encode("ascii", errors="ignore")


def run_smoke(plan: Dict[str, Any], cwd: str) -> SmokeResult:
    cwd_path = Path(cwd).resolve()
    # pre-clean artifacts
    for rel in list(plan.get("pre_clean") or []):
        target = (cwd_path / rel).resolve()
        if str(target).startswith(str(cwd_path)) and target.exists():
            try:
                target.unlink()
            except Exception:
                pass

    # build steps
    for step in list(plan.get("build") or []):
        cmd = list(step.get("cmd") or [])
        timeout_s = int(step.get("timeout_s") or 20)
        res = run_command(cmd, str(cwd_path), timeout_s)
        if res.timed_out:
            return _failure(
                "BUILD",
                "build command timed out",
                command=" ".join(cmd),
                snippet=tail_lines(
                    res.stderr or res.stdout,
                    int(getattr(config, "SMOKE_MAX_SNIPPET_LINES", 30) or 30),
                ),
            )
        if res.exit_code != 0:
            return _failure(
                "BUILD",
                f"build command exited with code {res.exit_code}",
                command=" ".join(cmd),
                snippet=tail_lines(
                    res.stderr or res.stdout,
                    int(getattr(config, "SMOKE_MAX_SNIPPET_LINES", 30) or 30),
                ),
            )

    # run steps
    run_stdout = ""
    for step in list(plan.get("run") or []):
        cmd = list(step.get("cmd") or [])
        timeout_s = int(step.get("timeout_s") or 20)
        res = run_command(cmd, str(cwd_path), timeout_s)
        run_stdout += (res.stdout or "")
        if res.timed_out:
            return _failure(
                "RUN",
                "run command timed out",
                command=" ".join(cmd),
                snippet=tail_lines(
                    res.stderr or res.stdout,
                    int(getattr(config, "SMOKE_MAX_SNIPPET_LINES", 30) or 30),
                ),
            )
        if res.exit_code != 0:
            return _failure(
                "RUN",
                f"run command exited with code {res.exit_code}",
                command=" ".join(cmd),
                snippet=tail_lines(
                    res.stderr or res.stdout,
                    int(getattr(config, "SMOKE_MAX_SNIPPET_LINES", 30) or 30),
                ),
            )

    # stdout checks
    stdout_fail = _assert_stdout(run_stdout, list(plan.get("assert_stdout") or []))
    if stdout_fail:
        return SmokeResult(passed=False, failures=[stdout_fail])

    # file existence checks
    for rel in list(plan.get("assert_files") or []):
        target = (cwd_path / rel).resolve()
        if not str(target).startswith(str(cwd_path)) or not target.exists():
            return _failure(
                "ASSERT_FILE",
                f"required file missing: {rel}",
            )

    # magic-byte checks
    for spec in list(plan.get("assert_file_magic") or []):
        if not isinstance(spec, dict):
            continue
        rel = str(spec.get("path") or "")
        offset = int(spec.get("offset") or 0)
        expected = _expected_magic_bytes(spec)
        if not rel or not expected:
            continue
        target = (cwd_path / rel).resolve()
        if not target.exists():
            return _failure(
                "ASSERT_MAGIC",
                f"magic check file missing: {rel}",
            )
        try:
            with open(target, "rb") as handle:
                handle.seek(max(0, offset))
                got = handle.read(len(expected))
        except Exception as exc:
            return _failure(
                "ASSERT_MAGIC",
                f"magic read failed for {rel}: {exc}",
            )
        if got != expected:
            return _failure(
                "ASSERT_MAGIC",
                f"magic mismatch at {rel}:{offset}",
                snippet=f"expected={expected!r} got={got!r}",
            )

    return SmokeResult(passed=True, failures=[])


def serialize_failures(result: SmokeResult, max_items: int = 3) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for failure in list(result.failures or [])[:max(1, int(max_items))]:
        out.append(
            {
                "kind": str(failure.kind or ""),
                "message": str(failure.message or ""),
                "command": str(failure.command or ""),
                "snippet": str(failure.snippet or ""),
            }
        )
    return out


def build_smoke_report(result: SmokeResult, max_lines: int = 80) -> str:
    if result.passed:
        return "SMOKE_REPORT:\n- status: pass"
    failure = (result.failures or [SmokeFailure(kind="RUN", message="unknown smoke failure")])[0]
    lines: List[str] = [
        "SMOKE_REPORT:",
        f"- cmd: {failure.command or '<assertion>'}",
        "- exit: nonzero_or_timeout",
        f"- failure: {failure.message}",
        "- details:",
        f"  - kind={failure.kind}",
    ]
    snippet = (failure.snippet or "").strip()
    if snippet:
        lines.append("- snippet:")
        for ln in snippet.splitlines()[:30]:
            lines.append(f"  - {ln}")
    return "\n".join(lines[:max(1, int(max_lines))]).strip()
