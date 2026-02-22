"""Build/test verify and runtime test. Extracted from pipeline for use by server and CLI."""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from . import config
from .files import get_root, _norm_rel_path
from .utils import dbg, dbg_dump


def detect_verify_command(root: Optional[Path] = None) -> Optional[Tuple[str, List[str]]]:
    """Auto-detect build/test command. Returns (cmd, args) or None."""
    root = root or get_root()
    root = Path(root).resolve()

    override = getattr(config, "VERIFY_CMD", None) or ""
    if override:
        parts = override.split()
        if parts:
            return (parts[0], parts[1:])
        return None

    for name in ("Makefile", "makefile", "GNUmakefile"):
        if (root / name).exists():
            return ("make", [])

    if (root / "CMakeLists.txt").exists():
        for d in ("build", "cmake-build-debug", "cmake-build-release"):
            if (root / d).exists():
                return ("cmake", ["--build", d])
        return ("cmake", ["--build", "."])

    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text())
            scripts = data.get("scripts") or {}
            if "test" in scripts:
                return ("npm", ["test"])
            if "build" in scripts:
                return ("npm", ["run", "build"])
        except Exception:
            pass

    if (root / "Cargo.toml").exists():
        return ("cargo", ["build"])

    if (root / "go.mod").exists():
        return ("go", ["build", "./..."])

    if (root / "pyproject.toml").exists():
        return ("python", ["-m", "pytest", "-x"])
    if (root / "setup.py").exists():
        return ("python", ["setup.py", "build"])

    return None


def _extract_first_error(stderr: str, stdout: str) -> str:
    """Extract first error block from stderr/stdout."""
    combined = (stderr or "") + "\n" + (stdout or "")
    lines = combined.splitlines()
    out: List[str] = []
    in_error = False
    for line in lines:
        if re.search(r"error:|Error:|undefined|syntax error|fatal error", line, re.I):
            in_error = True
        if in_error:
            out.append(line)
            if len(out) >= 15:
                break
    return "\n".join(out)[:800] if out else combined[:500]


def run_syntax_check(
    paths: List[str],
    root: Optional[Path] = None,
) -> Tuple[int, str]:
    """Run syntax-only compile check on C/C++ files. Returns (exit_code, stderr)."""
    root = root or get_root()
    root = Path(root).resolve()
    c_exts = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hxx"}
    to_check = [p for p in paths if Path(p).suffix.lower() in c_exts]
    if not to_check:
        return (0, "")
    cc = None
    for c in ("gcc", "clang"):
        try:
            subprocess.run([c, "--version"], capture_output=True, timeout=2)
            cc = c
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    if not cc:
        dbg("verify: syntax_check skipped (no gcc/clang)")
        return (0, "")
    full_paths = [str(root / _norm_rel_path(p)) for p in to_check]
    args = ["-fsyntax-only", "-c"] + full_paths
    try:
        res = subprocess.run(
            [cc] + args,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        stderr = res.stderr or ""
        dbg(f"verify: syntax_check exit={res.returncode} paths={to_check}")
        return (res.returncode, stderr)
    except subprocess.TimeoutExpired:
        return (124, "Syntax check timed out")
    except Exception as e:
        return (1, str(e))


_SIGNAL_NAMES = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 4: "SIGILL",
    6: "SIGABRT", 8: "SIGFPE", 9: "SIGKILL", 11: "SIGSEGV",
    13: "SIGPIPE", 14: "SIGALRM", 15: "SIGTERM",
}


def _find_test_input(root: Path) -> Optional[Path]:
    """Search for a test input file to pipe as stdin."""
    patterns = [
        "test_input.txt", "test_input", "input.txt",
        "tests/input.txt", "tests/test_input.txt",
    ]
    for pat in patterns:
        candidate = root / pat
        if candidate.is_file():
            return candidate
    import glob as glob_mod
    for g in ("tests/*.in", "tests/*.txt", "test_input_*.txt", "*.input"):
        matches = sorted(glob_mod.glob(str(root / g)))
        if matches:
            return Path(matches[0])
    return None


def _has_make_test_target(root: Path) -> bool:
    """Check if Makefile has a 'test' target."""
    for name in ("Makefile", "makefile", "GNUmakefile"):
        if (root / name).exists():
            try:
                res = subprocess.run(
                    ["make", "-n", "test"],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return res.returncode == 0
            except Exception:
                return False
    return False


def _detect_executables(staged_paths: List[str], root: Path) -> List[Path]:
    """Derive executable paths from staged source files."""
    import os as _os
    exes: List[Path] = []
    compiled_exts = {".c", ".cc", ".cpp", ".cxx", ".go", ".rs"}
    for p in staged_paths:
        src = Path(p)
        if src.suffix.lower() in compiled_exts:
            candidate = root / src.stem
            if candidate.exists() and _os.access(str(candidate), _os.X_OK):
                exes.append(candidate)
    return exes


def run_executable_test(
    staged_paths: List[str],
    root: Optional[Path] = None,
) -> Tuple[int, str, str, str]:
    """Run the built executable(s) after a successful build.
    Returns (exit_code, stdout, stderr, error_summary). exit_code 0 means all tests passed."""
    root = root or get_root()
    root = Path(root).resolve()
    run_timeout = getattr(config, "RUN_TIMEOUT", 0) or None

    run_cmd_override = getattr(config, "RUN_CMD", None) or ""
    if run_cmd_override:
        dbg(f"verify: run_test using RUN_CMD override: {run_cmd_override}")
        try:
            res = subprocess.run(
                run_cmd_override,
                shell=True,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=run_timeout,
            )
            if res.returncode != 0:
                err = _extract_first_error(res.stderr or "", res.stdout or "")
                summary = f"RUNTIME ERROR (exit {res.returncode}):\n{err}" if err else f"RUNTIME ERROR: exit code {res.returncode}"
                dbg(f"verify: RUN_CMD failed exit={res.returncode}")
                return (res.returncode, res.stdout or "", res.stderr or "", summary)
            dbg("verify: RUN_CMD passed")
            return (0, res.stdout or "", res.stderr or "", "")
        except subprocess.TimeoutExpired:
            dbg(f"verify: RUN_CMD timed out ({run_timeout}s)")
            return (124, "", "", f"RUNTIME ERROR: program timed out after {run_timeout}s (possible hang or infinite loop)")
        except Exception as e:
            return (1, "", str(e), f"RUNTIME ERROR: {e}")

    if _has_make_test_target(root):
        dbg("verify: run_test found 'make test' target, running it")
        try:
            res = subprocess.run(
                ["make", "test"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=run_timeout,
            )
            stdout = res.stdout or ""
            stderr = res.stderr or ""
            if res.returncode != 0:
                err = _extract_first_error(stderr, stdout)
                summary = f"RUNTIME ERROR from 'make test' (exit {res.returncode}):\n{err}"
                dbg(f"verify: make test failed exit={res.returncode}")
                return (res.returncode, stdout, stderr, summary)
            dbg("verify: make test passed")
            return (0, stdout, stderr, "")
        except subprocess.TimeoutExpired:
            dbg(f"verify: make test timed out ({run_timeout}s)")
            return (124, "", "", f"RUNTIME ERROR: 'make test' timed out after {run_timeout}s")
        except Exception as e:
            return (1, "", str(e), f"RUNTIME ERROR: {e}")

    exes = _detect_executables(staged_paths or [], root)
    if not exes:
        dbg("verify: run_test no executables detected, skipping runtime test")
        return (0, "", "", "")

    test_input_file = _find_test_input(root)
    if test_input_file:
        dbg(f"verify: run_test found test input file: {test_input_file}")

    _stdin_timeout = run_timeout if test_input_file else (run_timeout or 5)

    for exe in exes:
        exe_rel = exe.relative_to(root) if exe.is_relative_to(root) else exe
        dbg(f"verify: run_test running ./{exe_rel}" + (f" < {test_input_file}" if test_input_file else " (no stdin)"))
        try:
            stdin_data = ""
            if test_input_file:
                stdin_data = test_input_file.read_text()
            res = subprocess.run(
                [str(exe)],
                cwd=str(root),
                capture_output=True,
                text=True,
                input=stdin_data,
                timeout=_stdin_timeout,
            )
            stdout = res.stdout or ""
            stderr = res.stderr or ""
            rc = res.returncode

            if rc < 0:
                sig_num = abs(rc)
                sig_name = _SIGNAL_NAMES.get(sig_num, f"signal {sig_num}")
                summary = f"RUNTIME ERROR: ./{exe_rel} crashed with {sig_name} (killed by signal)"
                if stderr:
                    summary += f"\nProgram stderr:\n{stderr[:500]}"
                dbg(f"verify: {exe_rel} killed by {sig_name}")
                dbg_dump("run_test_stdout", stdout)
                dbg_dump("run_test_stderr", stderr)
                return (rc, stdout, stderr, summary)
            elif rc != 0:
                err = _extract_first_error(stderr, stdout)
                summary = f"RUNTIME ERROR: ./{exe_rel} exited with code {rc}"
                if not test_input_file:
                    summary += (
                        " (no test input was piped — program received empty stdin). "
                        "Add test_input.txt in project root or set SC2_RUN_CMD."
                    )
                if err:
                    summary += f"\nProgram output:\n{err}"
                elif stderr:
                    summary += f"\nProgram stderr:\n{stderr[:500]}"
                if stdout:
                    summary += f"\nProgram stdout:\n{stdout[:300]}"
                dbg(f"verify: {exe_rel} exited with code {rc}")
                dbg_dump("run_test_stdout", stdout)
                dbg_dump("run_test_stderr", stderr)
                return (rc, stdout, stderr, summary)
            else:
                dbg(f"verify: {exe_rel} passed (exit 0)")
                dbg_dump("run_test_stdout", stdout)
                return (0, stdout, stderr, "")
        except subprocess.TimeoutExpired:
            if not test_input_file:
                summary = (
                    f"RUNTIME ERROR: ./{exe_rel} timed out after {_stdin_timeout}s with empty stdin. "
                    "Program likely needs input; add test_input.txt in project root or set SC2_RUN_CMD."
                )
                dbg(f"verify: {exe_rel} timed out without stdin")
                return (124, "", "", summary)
            summary = f"RUNTIME ERROR: ./{exe_rel} timed out after {_stdin_timeout}s (possible infinite loop or deadlock)"
            dbg(f"verify: {exe_rel} timed out ({_stdin_timeout}s)")
            return (124, "", "", summary)
        except Exception as e:
            return (1, "", str(e), f"RUNTIME ERROR running ./{exe_rel}: {e}")

    return (0, "", "", "")


def run_verify(
    root: Optional[Path] = None,
    staged_paths: Optional[List[str]] = None,
) -> Tuple[int, str, str, str]:
    """Run verify command. If staged_paths has C files, run syntax-only check first.
    Returns (exit_code, stdout, stderr, first_error_block)."""
    root = root or get_root()
    root = Path(root).resolve()

    if staged_paths:
        syn_code, syn_stderr = run_syntax_check(staged_paths, root)
        if syn_code != 0 and syn_stderr:
            first_error = _extract_first_error(syn_stderr, "")
            dbg(f"verify: syntax_check failed exit={syn_code}")
            return (syn_code, "", syn_stderr, first_error)

    cmd_info = detect_verify_command(root)
    if not cmd_info:
        dbg("verify: no cmd exit=0")
        return (0, "", "", "")

    cmd, args = cmd_info
    full_cmd = [cmd] + args
    timeout = getattr(config, "VERIFY_TIMEOUT", 60)
    dbg(f"verify: running {' '.join(full_cmd)} in {root}")

    try:
        res = subprocess.run(
            full_cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stderr = res.stderr or ""
        stdout = res.stdout or ""
        first_error = _extract_first_error(stderr, stdout)
        dbg(f"verify: build exit={res.returncode} first_error_len={len(first_error)}")
        dbg_dump("verify_stdout", stdout)
        dbg_dump("verify_stderr", stderr)
        dbg_dump("verify_first_error", first_error)
        if res.returncode != 0:
            return (res.returncode, stdout, stderr, first_error)
        if staged_paths:
            run_rc, run_out, run_err, run_summary = run_executable_test(staged_paths, root)
            if run_rc != 0:
                dbg(f"verify: runtime test failed exit={run_rc}")
                return (run_rc, run_out, run_err, run_summary)
            dbg("verify: runtime test passed")
            return (0, run_out, run_err, "")
        dbg("verify: done exit=0")
        return (0, stdout, stderr, "")
    except subprocess.TimeoutExpired:
        dbg("verify: timeout exit=124")
        return (124, "", "Command timed out", "Command timed out")
    except Exception as e:
        dbg(f"verify: exception exit=1 err={e}")
        return (1, "", str(e), str(e))
