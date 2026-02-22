"""Semantic operation executor for RAIL v3.

This layer attempts language-server style structural edits first, then returns
structured capability errors when unsupported.
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _tool_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _rg_refs(root: Path, symbol: str) -> List[Dict[str, Any]]:
    if not symbol:
        return []
    try:
        res = subprocess.run(
            ["rg", "-n", r"\b" + re.escape(symbol) + r"\b", "."],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for ln in (res.stdout or "").splitlines():
        m = re.match(r"^(.+?):(\d+):(.*)$", ln)
        if m:
            out.append({"path": m.group(1), "line": int(m.group(2)), "text": m.group(3)})
    return out


def execute_semantic_op(
    name: str,
    args: Dict[str, Any],
    root: Path,
    whitelist: List[str],
) -> Dict[str, Any]:
    name = (name or "").strip()
    whitelist_set = set(whitelist or [])
    if name == "find_references":
        symbol = str(args.get("symbol") or "")
        refs = _rg_refs(root, symbol)
        return {"ok": True, "op": name, "references": refs}

    if name == "rename_symbol":
        symbol = str(args.get("symbol") or "")
        new_name = str(args.get("new_name") or "")
        if not symbol or not new_name:
            return {"ok": False, "op": name, "reason": "missing_symbol_or_new_name"}
        refs = _rg_refs(root, symbol)
        touched = {}
        for r in refs:
            p = r["path"]
            if whitelist_set and p not in whitelist_set:
                continue
            fp = (root / p).resolve()
            if not fp.exists():
                continue
            old = fp.read_text()
            new = re.sub(r"\b" + re.escape(symbol) + r"\b", new_name, old)
            if new != old:
                fp.write_text(new)
                touched[p] = True
        return {"ok": True, "op": name, "touched": sorted(touched.keys())}

    if name == "add_parameter":
        function = str(args.get("function") or "")
        param = str(args.get("param") or "")
        path = str(args.get("path") or "")
        if not function or not param or not path:
            return {"ok": False, "op": name, "reason": "missing_function_param_or_path"}
        if whitelist_set and path not in whitelist_set:
            return {"ok": False, "op": name, "reason": "path_not_allowed"}
        fp = (root / path).resolve()
        if not fp.exists():
            return {"ok": False, "op": name, "reason": "path_missing"}
        old = fp.read_text()
        # conservative function signature tweak
        pat = re.compile(r"(\b" + re.escape(function) + r"\s*\()([^)]*)(\))")
        m = pat.search(old)
        if not m:
            return {"ok": False, "op": name, "reason": "function_signature_not_found"}
        middle = m.group(2).strip()
        repl_mid = (middle + ", " + param) if middle else param
        new = old[: m.start()] + m.group(1) + repl_mid + m.group(3) + old[m.end() :]
        fp.write_text(new)
        return {"ok": True, "op": name, "touched": [path]}

    if name == "organize_imports":
        path = str(args.get("file") or args.get("path") or "")
        if not path:
            return {"ok": False, "op": name, "reason": "missing_path"}
        if whitelist_set and path not in whitelist_set:
            return {"ok": False, "op": name, "reason": "path_not_allowed"}
        fp = (root / path).resolve()
        if not fp.exists():
            return {"ok": False, "op": name, "reason": "path_missing"}
        # Python: try isort first
        if path.endswith(".py") and _tool_exists("isort"):
            rc = subprocess.run(["isort", str(fp)], cwd=str(root), capture_output=True, text=True)
            return {"ok": rc.returncode == 0, "op": name, "touched": [path] if rc.returncode == 0 else []}
        # fallback: deterministic no-op with capability notice
        return {"ok": False, "op": name, "reason": "organize_imports_backend_unavailable"}

    if name == "move_function":
        src = str(args.get("src") or "")
        dest = str(args.get("dest") or "")
        function = str(args.get("function") or "")
        if not src or not dest or not function:
            return {"ok": False, "op": name, "reason": "missing_src_dest_function"}
        if whitelist_set and (src not in whitelist_set or dest not in whitelist_set):
            return {"ok": False, "op": name, "reason": "path_not_allowed"}
        src_fp = (root / src).resolve()
        dst_fp = (root / dest).resolve()
        if not src_fp.exists() or not dst_fp.exists():
            return {"ok": False, "op": name, "reason": "src_or_dest_missing"}
        src_text = src_fp.read_text()
        # naive C/Python function block matcher
        m = re.search(r"(^\s*.*\b" + re.escape(function) + r"\s*\([^)]*\)\s*\{)", src_text, re.M)
        if not m:
            m = re.search(r"(^\s*def\s+" + re.escape(function) + r"\s*\([^)]*\)\s*:)", src_text, re.M)
        if not m:
            return {"ok": False, "op": name, "reason": "function_not_found"}
        start = m.start()
        end = src_text.find("\n\n", start)
        if end == -1:
            end = len(src_text)
        block = src_text[start:end].rstrip() + "\n"
        new_src = src_text[:start] + src_text[end:]
        src_fp.write_text(new_src)
        dst_fp.write_text(dst_fp.read_text().rstrip() + "\n\n" + block)
        return {"ok": True, "op": name, "touched": [src, dest]}

    return {"ok": False, "op": name, "reason": "semantic_op_unknown"}
