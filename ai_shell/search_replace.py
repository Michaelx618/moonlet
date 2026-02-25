"""Parse and execute search_replace(old_string, new_string, path) from model output.

Simple algorithm: literal find old_string in file, replace with new_string (all occurrences).
Continue parity: replace_all, multi_edit, validate_single_edit.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .files import (
    _norm_rel_path,
    generate_diff,
    get_include,
    get_root,
    is_edit_allowed,
    resolve_path,
    write_file_text,
)
from .utils import dbg, dbg_dump

try:
    from file_utils import is_security_concern
except ImportError:
    def is_security_concern(*args, **kwargs):
        return False


class SearchReplaceError(Exception):
    """Continue-style validation error for find-and-replace."""
    pass


def validate_single_edit(
    old_string: Any,
    new_string: Any,
    replace_all: Any = None,
    index: Optional[int] = None,
) -> Tuple[str, str, bool]:
    """Validate a single edit (Continue findAndReplaceUtils). Returns (old_string, new_string, replace_all)."""
    ctx = f"edit at index {index}: " if index is not None else ""
    if old_string is None or not isinstance(old_string, str):
        raise SearchReplaceError(f"{ctx}old_string is required")
    if new_string is None or not isinstance(new_string, str):
        raise SearchReplaceError(f"{ctx}new_string is required")
    if old_string == new_string:
        raise SearchReplaceError(f"{ctx}old_string and new_string must be different")
    if replace_all is not None and not isinstance(replace_all, bool):
        raise SearchReplaceError(f"{ctx}replace_all must be a valid boolean")
    return (old_string, new_string, bool(replace_all) if replace_all else False)


def _unescape_string(s: str) -> str:
    """Unescape \\n, \\t, \\\", etc. in a string."""
    if not s:
        return s
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _parse_python_string(
    text: str, start: int, quote_char: str = '"'
) -> Tuple[Optional[str], int]:
    """Parse a Python string from text starting at start. Returns (value, end_pos) or (None, start)."""
    if start >= len(text) or text[start] != quote_char:
        return None, start
    pos = start + 1
    triple = text[start : start + 3] == quote_char * 3
    if triple:
        pos = start + 3
    result: List[str] = []
    while pos < len(text):
        ch = text[pos]
        if triple:
            if text[pos : pos + 3] == quote_char * 3:
                return "".join(result), pos + 3
            if ch == "\\" and pos + 1 < len(text):
                next_ch = text[pos + 1]
                if next_ch == "n":
                    result.append("\n")
                    pos += 2
                    continue
                if next_ch == "t":
                    result.append("\t")
                    pos += 2
                    continue
                if next_ch == "r":
                    result.append("\r")
                    pos += 2
                    continue
                if next_ch == quote_char:
                    result.append(quote_char)
                    pos += 2
                    continue
                if next_ch == "\\":
                    result.append("\\")
                    pos += 2
                    continue
            result.append(ch)
            pos += 1
        else:
            if ch == quote_char:
                return "".join(result), pos + 1
            if ch == "\\" and pos + 1 < len(text):
                next_ch = text[pos + 1]
                if next_ch == "n":
                    result.append("\n")
                    pos += 2
                    continue
                if next_ch == "t":
                    result.append("\t")
                    pos += 2
                    continue
                if next_ch == "r":
                    result.append("\r")
                    pos += 2
                    continue
                if next_ch == quote_char:
                    result.append(quote_char)
                    pos += 2
                    continue
                if next_ch == "\\":
                    result.append("\\")
                    pos += 2
                    continue
            result.append(ch)
            pos += 1
    return None, start


def parse_search_replace_calls(
    raw: str,
    *,
    filter_noop: bool = True,
) -> List[Dict[str, str]]:
    """Extract search_replace(old_string=..., new_string=..., path=...) from model output.

    Returns [{"old_string": str, "new_string": str, "path": str}, ...].
    Supports "..." and \"\"\"...\"\"\" for multi-line strings.
    When filter_noop=True, skips edits where old_string == new_string.
    """
    if not raw or not isinstance(raw, str):
        return []
    text = raw.strip()
    results: List[Dict[str, str]] = []
    pattern = re.compile(r"\bsearch_replace\s*\(", re.IGNORECASE)
    for m in pattern.finditer(text):
        start = m.end()
        pos = start
        kwargs: Dict[str, str] = {}
        while pos < len(text):
            # Skip whitespace and commas
            while pos < len(text) and text[pos] in " \t\n\r,":
                pos += 1
            if pos >= len(text):
                break
            if text[pos] == ")":
                pos += 1
                break
            # Parse key=
            key_match = re.match(r"(old_string|new_string|path|replace_all)\s*=\s*", text[pos:], re.IGNORECASE)
            if not key_match:
                pos += 1
                continue
            key = key_match.group(1).lower()
            pos += key_match.end()
            if pos >= len(text):
                break
            if key == "replace_all":
                # boolean: true/false
                rest = text[pos:].strip()
                if rest.startswith("true"):
                    kwargs["replace_all"] = "true"
                    pos += 4
                elif rest.startswith("false"):
                    kwargs["replace_all"] = "false"
                    pos += 5
                continue
            # Parse string value
            if text[pos] == '"':
                triple = text[pos : pos + 3] == '"""'
                quote = '"""' if triple else '"'
                val, end_pos = _parse_python_string(text, pos, quote[0])
                if triple and val is not None:
                    pass  # already unescaped in parser
                elif val is not None:
                    val = _unescape_string(val)
                if val is not None:
                    kwargs[key] = val
                pos = end_pos
            else:
                break
        if kwargs.get("old_string") is not None and kwargs.get("new_string") is not None and kwargs.get("path"):
            old_s = kwargs["old_string"]
            new_s = kwargs["new_string"]
            path_s = kwargs["path"].strip().strip('"')
            if filter_noop and old_s == new_s:
                continue
            replace_all = kwargs.get("replace_all", "").lower() in ("true", "1", "yes")
            results.append(
                {
                    "old_string": old_s,
                    "new_string": new_s,
                    "path": path_s,
                    "replace_all": replace_all,
                }
            )
    return results


def parse_write_file_calls(raw: str) -> List[Dict[str, str]]:
    """Extract write_file(path=..., content=...) from model output.

    Returns [{"path": str, "content": str}, ...].
    Supports "..." and \"\"\"...\"\"\" for multi-line content.
    """
    if not raw or not isinstance(raw, str):
        return []
    text = raw.strip()
    results: List[Dict[str, str]] = []
    pattern = re.compile(r"\bwrite_file\s*\(", re.IGNORECASE)
    for m in pattern.finditer(text):
        start = m.end()
        pos = start
        kwargs: Dict[str, str] = {}
        while pos < len(text):
            while pos < len(text) and text[pos] in " \t\n\r,":
                pos += 1
            if pos >= len(text):
                break
            if text[pos] == ")":
                pos += 1
                break
            key_match = re.match(r"(path|content)\s*=\s*", text[pos:], re.IGNORECASE)
            if not key_match:
                pos += 1
                continue
            key = key_match.group(1).lower()
            pos += key_match.end()
            if pos >= len(text):
                break
            if text[pos] == '"':
                triple = text[pos : pos + 3] == '"""'
                quote = '"""' if triple else '"'
                val, end_pos = _parse_python_string(text, pos, quote[0])
                if triple and val is not None:
                    pass
                elif val is not None:
                    val = _unescape_string(val)
                if val is not None:
                    kwargs[key] = val
                pos = end_pos
            else:
                break
        if kwargs.get("path") is not None and kwargs.get("content") is not None:
            results.append(
                {
                    "path": kwargs["path"].strip().strip('"'),
                    "content": kwargs["content"],
                }
            )
    return results


def parse_multi_edit_calls(raw: str) -> List[Dict[str, Any]]:
    """Extract multi_edit(path=..., edits=[...]) from model output.

    Returns [{"path": str, "edits": [{"old_string", "new_string", "replace_all?"}, ...]}, ...].
    """
    import json
    if not raw or not isinstance(raw, str):
        return []
    text = raw.strip()
    results: List[Dict[str, Any]] = []
    pattern = re.compile(r"\bmulti_edit\s*\(", re.IGNORECASE)
    for m in pattern.finditer(text):
        start = m.end()
        pos = start
        path_val: Optional[str] = None
        edits_val: Optional[str] = None
        while pos < len(text):
            while pos < len(text) and text[pos] in " \t\n\r,":
                pos += 1
            if pos >= len(text) or text[pos] == ")":
                break
            key_match = re.match(r"(path|filepath|edits)\s*=\s*", text[pos:], re.IGNORECASE)
            if not key_match:
                pos += 1
                continue
            key = key_match.group(1).lower()
            pos += key_match.end()
            if key == "path" or key == "filepath":
                if pos < len(text) and text[pos] == '"':
                    triple = text[pos : pos + 3] == '"""'
                    quote = '"""' if triple else '"'
                    val, end_pos = _parse_python_string(text, pos, quote[0])
                    if val is not None and not triple:
                        val = _unescape_string(val)
                    if val is not None:
                        path_val = val.strip().strip('"')
                    pos = end_pos
            elif key == "edits":
                # JSON array: [ { "old_string": "...", "new_string": "..." }, ... ]
                if pos < len(text) and text[pos] == "[":
                    depth = 0
                    jstart = pos
                    for i in range(pos, len(text)):
                        if text[i] == "[":
                            depth += 1
                        elif text[i] == "]":
                            depth -= 1
                            if depth == 0:
                                edits_val = text[jstart : i + 1]
                                pos = i + 1
                                break
                    else:
                        pos += 1
            else:
                pos += 1
        if path_val and edits_val:
            try:
                arr = json.loads(edits_val)
            except json.JSONDecodeError:
                continue
            if not isinstance(arr, list):
                continue
            out_edits: List[Dict[str, Any]] = []
            for item in arr:
                if not isinstance(item, dict):
                    continue
                o = (item.get("old_string") or "").strip()
                n = (item.get("new_string") or "").strip()
                if o == n:
                    continue
                out_edits.append({
                    "old_string": o,
                    "new_string": n,
                    "replace_all": item.get("replace_all") is True,
                })
            if out_edits:
                results.append({"path": path_val, "edits": out_edits})
    return results


def _strip_markdown_code_block(content: str) -> str:
    """Strip ```lang and ``` wrappers if present (model sometimes outputs markdown)."""
    if not content or not isinstance(content, str):
        return content
    s = content.strip()
    if s.startswith("```"):
        first = s.find("\n")
        if first >= 0:
            s = s[first + 1 :]
        if s.endswith("```"):
            s = s[:-3].rstrip()
    return s


def apply_write_file_edits(
    edits: List[Dict[str, str]],
    allowed_edit_files: Optional[List[str]] = None,
    root: Optional[Path] = None,
    stage_only: bool = True,
) -> Tuple[List[str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Apply write_file edits. Returns (touched, per_file_diffs, per_file_staged, per_file_before)."""
    root = root or get_root()
    root = Path(root).resolve()
    touched: List[str] = []
    per_file_diffs: Dict[str, str] = {}
    per_file_staged: Dict[str, str] = {}
    per_file_before: Dict[str, str] = {}

    if not edits:
        return touched, per_file_diffs, per_file_staged, per_file_before

    for e in edits:
        path_str = (e.get("path") or "").strip().strip('"\'')
        content = _strip_markdown_code_block(e.get("content") or "")
        if not path_str:
            continue
        try:
            rel_path = _norm_rel_path(path_str)
            abs_path = (root / rel_path).resolve()
            abs_path.relative_to(root.resolve())
        except Exception:
            continue
        if allowed_edit_files:
            allowed_set = {_norm_rel_path(p) for p in allowed_edit_files}
            if rel_path not in allowed_set and not any(
                rel_path == a or rel_path.startswith(a.rstrip("/") + "/")
                for a in allowed_set
            ):
                continue
        if not is_edit_allowed(rel_path, allow_new=True):
            continue
        if is_security_concern(abs_path):
            continue
        old_content = abs_path.read_text() if abs_path.exists() else ""
        new_content = content
        diff = generate_diff(old_content, new_content, str(abs_path))
        touched.append(rel_path)
        per_file_before[rel_path] = old_content
        per_file_staged[rel_path] = new_content
        per_file_diffs[rel_path] = diff
        if not stage_only:
            write_file_text(rel_path, new_content)
            dbg(f"write_file: wrote {rel_path}")

    return touched, per_file_diffs, per_file_staged, per_file_before


def _resolve_edit_path(path_str: str, root: Path) -> Tuple[str, Path]:
    """Resolve path to (rel_path, abs_path). Path can be relative or absolute under root."""
    path_str = (path_str or "").strip().strip('"\'')
    if not path_str:
        raise ValueError("path is empty")
    # Absolute path - must be under root
    if path_str.startswith("/") or (len(path_str) > 1 and path_str[1] == ":"):
        abs_path = Path(path_str).resolve()
        try:
            rel = abs_path.relative_to(root.resolve())
            return str(rel.as_posix()), abs_path
        except ValueError:
            raise ValueError(f"path outside root: {path_str}")
    # Relative path
    rel = _norm_rel_path(path_str)
    abs_path = (root / rel).resolve()
    try:
        abs_path.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"path outside root: {path_str}")
    return rel, abs_path


def _norm_line_endings(s: str) -> str:
    """Normalize line endings to \\n so file (e.g. \\r\\n) and model string (\\n) match."""
    if not s:
        return s
    return s.replace("\r\n", "\n").replace("\r", "\n")


def execute_search_replace(
    file_content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    edit_index: int = 0,
) -> str:
    """Replace every occurrence of old_string with new_string in file_content.

    Simple literal replace. Line endings are normalized so \\r\\n and \\n match.
    Fails when old_string does not appear in the file (e.g. already applied).
    """
    content_norm = _norm_line_endings(file_content)
    old_norm = _norm_line_endings(old_string)
    new_norm = _norm_line_endings(new_string)
    if old_norm not in content_norm:
        raise SearchReplaceError(
            f"Edit at index {edit_index}: string not found in file: {repr(old_string[:80])}..."
        )
    if replace_all:
        return content_norm.replace(old_norm, new_norm)
    return content_norm.replace(old_norm, new_norm, 1)


def execute_multi_find_and_replace(
    file_content: str,
    edits: List[Dict[str, Any]],
) -> str:
    """Apply a list of edits in sequence (Continue performReplace.executeMultiFindAndReplace)."""
    result = file_content
    for i, edit in enumerate(edits):
        old_s = edit.get("old_string") or ""
        new_s = edit.get("new_string", "")
        replace_all = edit.get("replace_all") is True
        result = execute_search_replace(result, old_s, new_s, replace_all=replace_all, edit_index=i)
    return result


def apply_single_multi_edit(
    path_str: str,
    edits: List[Dict[str, Any]],
    allowed_edit_files: Optional[List[str]] = None,
    root: Optional[Path] = None,
    stage_only: bool = True,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Apply one multi_edit (one file, list of edits). Returns (success, message, meta)."""
    root = root or get_root()
    root = Path(root).resolve()
    if not edits:
        return False, "multi_edit: edits array is required", None
    try:
        rel_path, abs_path = _resolve_edit_path(path_str, root)
    except ValueError as e:
        return False, f"multi_edit: {e}", None
    if allowed_edit_files:
        allowed_set = {_norm_rel_path(p) for p in allowed_edit_files}
        if rel_path not in allowed_set and not any(
            rel_path == a or rel_path.startswith(a.rstrip("/") + "/") for a in allowed_set
        ):
            return False, f"multi_edit: path {rel_path} not in allowed files", None
    if not abs_path.exists() or not abs_path.is_file():
        return False, f"multi_edit: file not found {rel_path}", None
    if not is_edit_allowed(rel_path, allow_new=False):
        return False, (
            f"multi_edit: edit not allowed for {rel_path} "
            "(file not in imported/allow list; add this file to imported files to edit it)"
        ), None
    if is_security_concern(abs_path):
        return False, f"multi_edit: security concern for {rel_path}", None
    content = abs_path.read_text()
    try:
        new_content = execute_multi_find_and_replace(content, edits)
    except SearchReplaceError as e:
        snippet_len = 600
        snippet = (content[:snippet_len] + "..." if len(content) > snippet_len else content)
        if snippet.strip():
            snippet = "\nCurrent file content (snippet):\n" + snippet.replace("\r\n", "\n")
        else:
            snippet = "\n(File is empty.)"
        return False, f"multi_edit: {e}{snippet}", None
    diff = generate_diff(content, new_content, str(abs_path))
    if not stage_only:
        write_file_text(rel_path, new_content)
        dbg(f"multi_edit: wrote {rel_path}")
    return True, f"Successfully edited {rel_path}", {
        "path": rel_path,
        "old_content": content,
        "new_content": new_content,
        "diff": diff,
    }


def apply_search_replace(
    edit: Dict[str, Any],
    allowed_edit_files: Optional[List[str]] = None,
    root: Optional[Path] = None,
    stage_only: bool = True,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Execute one search_replace edit. Returns (success, message, meta_or_none).

    meta contains: path, old_content, new_content, diff (for staging).
    """
    import json as _json
    import time as _time
    # #region agent log
    def _dlog(loc: str, msg: str, data: dict, hid: str) -> None:
        try:
            with open("/Users/michael/moonlet/.cursor/debug-c7e239.log", "a") as _f:
                _f.write(_json.dumps({"sessionId": "c7e239", "hypothesisId": hid, "location": loc, "message": msg, "data": data, "timestamp": int(_time.time() * 1000)}) + "\n")
        except Exception:
            pass
    # #endregion
    root = root or get_root()
    root = Path(root).resolve()
    old_str = edit.get("old_string") or ""
    new_str = edit.get("new_string", "")
    path_str = edit.get("path") or ""
    replace_all = edit.get("replace_all") is True
    # #region agent log
    _dlog("search_replace.apply_search_replace:entry", "entry", {"path_str": path_str, "root": str(root), "stage_only": stage_only}, "H1")
    # #endregion

    if not old_str:
        return False, "search_replace: old_string is empty", None
    if not path_str:
        return False, "search_replace: path is empty", None
    try:
        validate_single_edit(old_str, new_str, replace_all)
    except SearchReplaceError as e:
        return False, f"search_replace: {e}", None

    try:
        rel_path, abs_path = _resolve_edit_path(path_str, root)
    except ValueError as e:
        # #region agent log
        _dlog("search_replace.apply_search_replace:resolve_fail", "resolve_fail", {"path_str": path_str, "error": str(e)}, "H1")
        # #endregion
        return False, f"search_replace: {e}", None

    # #region agent log
    _dlog("search_replace.apply_search_replace:resolved", "resolved", {"rel_path": rel_path, "abs_path": str(abs_path), "exists": abs_path.exists() and abs_path.is_file()}, "H1")
    # #endregion

    if allowed_edit_files:
        allowed_set = {_norm_rel_path(p) for p in allowed_edit_files}
        path_allowed = (
            rel_path in allowed_set
            or any(
                rel_path == a or rel_path.endswith("/" + a) or a.endswith("/" + rel_path.split("/")[-1])
                for a in allowed_set
            )
        )
        if not path_allowed:
            return False, f"search_replace: path {rel_path} not in allowed files", None

    allow_new = not (abs_path.exists() and abs_path.is_file())
    edit_allowed = is_edit_allowed(rel_path, allow_new=allow_new)
    # #region agent log
    _dlog("search_replace.apply_search_replace:edit_allowed", "edit_allowed", {"rel_path": rel_path, "allow_new": allow_new, "edit_allowed": edit_allowed, "include_paths": get_include() or None}, "H2")
    # #endregion
    if not edit_allowed:
        return False, (
            f"search_replace: edit not allowed for {rel_path} "
            "(file not in imported/allow list; add this file to imported files to edit it)"
        ), None

    if is_security_concern(abs_path):
        return False, f"search_replace: security concern for {rel_path}", None

    if not abs_path.exists() or not abs_path.is_file():
        return False, f"search_replace: file not found: {rel_path}", None

    content = abs_path.read_text()
    old_in_content = old_str in content
    new_in_content = new_str in content
    # #region agent log
    _dlog("search_replace.apply_search_replace:before_replace", "before_replace", {"rel_path": rel_path, "old_in_content": old_in_content, "new_in_content": new_in_content, "content_len": len(content)}, "H3")
    # #endregion
    try:
        new_content = execute_search_replace(
            content, old_str, new_str, replace_all=replace_all, edit_index=0
        )
    except SearchReplaceError as e:
        err_msg = str(e)
        # #region agent log
        _dlog("search_replace.apply_search_replace:string_not_found", "string_not_found", {"rel_path": rel_path, "err_msg": err_msg[:200]}, "H3")
        # #endregion
        # Continue-style: short failure message only (no file snippet)
        fail_msg = (
            f"Failed to edit {rel_path}. To continue working with the file, read it again to see the most up-to-date contents."
        )
        return False, fail_msg, None

    diff = generate_diff(content, new_content, str(abs_path))

    if stage_only:
        # #region agent log
        _dlog("search_replace.apply_search_replace:return_ok_stage", "return_ok", {"rel_path": rel_path, "stage_only": True}, "H4")
        # #endregion
        return True, f"Successfully edited {rel_path}", {
            "path": rel_path,
            "old_content": content,
            "new_content": new_content,
            "diff": diff,
        }
    abs_path.write_text(new_content)
    # #region agent log
    _dlog("search_replace.apply_search_replace:return_ok_write", "return_ok", {"rel_path": rel_path, "stage_only": False}, "H4")
    # #endregion
    return True, f"Successfully edited {rel_path}", {
        "path": rel_path,
        "old_content": content,
        "new_content": new_content,
        "diff": diff,
    }


def apply_search_replace_edits(
    edits: List[Dict[str, str]],
    allowed_edit_files: Optional[List[str]] = None,
    root: Optional[Path] = None,
    stage_only: bool = True,
) -> Tuple[List[str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Apply a list of search_replace edits. Returns (touched, per_file_diffs, per_file_staged, per_file_before).

    Edits to the same file are applied in reverse order by old_string position to avoid offset shifts.
    """
    root = root or get_root()
    root = Path(root).resolve()
    touched: List[str] = []
    per_file_diffs: Dict[str, str] = {}
    per_file_staged: Dict[str, str] = {}
    per_file_before: Dict[str, str] = {}

    if not edits:
        return touched, per_file_diffs, per_file_staged, per_file_before

    # Filter by allowed_edit_files
    if allowed_edit_files:
        allowed_set = {_norm_rel_path(p) for p in allowed_edit_files}
        filtered: List[Dict[str, str]] = []
        for e in edits:
            try:
                rel_path, _ = _resolve_edit_path(e.get("path", ""), root)
            except ValueError:
                continue
            path_ok = (
                rel_path in allowed_set
                or any(
                    rel_path == a or rel_path.endswith("/" + a) or a.endswith("/" + rel_path.split("/")[-1])
                    for a in allowed_set
                )
            )
            if path_ok:
                filtered.append(e)
        edits = filtered
        if not edits:
            return touched, per_file_diffs, per_file_staged, per_file_before

    # Group by file and sort by position (desc) so we apply from end to start
    by_file: Dict[str, List[Dict[str, str]]] = {}
    for e in edits:
        try:
            rel_path, _ = _resolve_edit_path(e.get("path", ""), root)
        except ValueError:
            continue
        by_file.setdefault(rel_path, []).append(e)

    for rel_path, file_edits in by_file.items():
        abs_path = root / rel_path
        if not abs_path.exists() or not abs_path.is_file():
            continue
        content = abs_path.read_text()
        try:
            # Continue: apply edits in sequence (execute_multi_find_and_replace)
            edits_payload = [
                {
                    "old_string": e.get("old_string") or "",
                    "new_string": e.get("new_string", ""),
                    "replace_all": e.get("replace_all") is True,
                }
                for e in file_edits
            ]
            content = execute_multi_find_and_replace(content, edits_payload)
        except SearchReplaceError as err:
            dbg(f"search_replace: {rel_path} {err}")
            continue
        touched.append(rel_path)
        original = abs_path.read_text()
        per_file_before[rel_path] = original
        per_file_staged[rel_path] = content
        per_file_diffs[rel_path] = generate_diff(original, content, str(abs_path))

        if not stage_only:
            write_file_text(rel_path, content)
            dbg(f"search_replace: wrote {rel_path}")

    dbg(f"search_replace: applied {len(edits)} edit(s), touched {touched}")
    return touched, per_file_diffs, per_file_staged, per_file_before
