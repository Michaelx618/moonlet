import json
import ast
from collections import Counter
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from . import config
from .tools import extract_symbols_treesitter
from .utils import dbg


@dataclass
class StructuralTarget:
    name: str
    kind: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    parent: str = ""


@dataclass
class StructuralDecision:
    eligible: bool
    reason: str
    target: Optional[StructuralTarget] = None


def _line_char_offsets(content: str) -> List[int]:
    starts = [0]
    for line in content.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    if not starts:
        starts = [0]
    return starts


def _line_byte_offsets(content: str) -> List[int]:
    starts = [0]
    for line in content.splitlines(keepends=True):
        starts.append(starts[-1] + len(line.encode("utf-8")))
    if not starts:
        starts = [0]
    return starts


def _line_to_char_range(content: str, line_start: int, line_end: int) -> Tuple[int, int]:
    starts = _line_char_offsets(content)
    total_lines = max(1, len(starts) - 1)
    s_line = max(1, min(total_lines, int(line_start or 1)))
    e_line = max(s_line, min(total_lines, int(line_end or s_line)))
    start_char = starts[s_line - 1]
    end_char = starts[e_line]
    return start_char, end_char


def _line_to_byte_range(content: str, line_start: int, line_end: int) -> Tuple[int, int]:
    starts = _line_byte_offsets(content)
    total_lines = max(1, len(starts) - 1)
    s_line = max(1, min(total_lines, int(line_start or 1)))
    e_line = max(s_line, min(total_lines, int(line_end or s_line)))
    start_byte = starts[s_line - 1]
    end_byte = starts[e_line]
    return start_byte, end_byte


def _replacement_byte_span(
    focus_file: str,
    content: str,
    target: StructuralTarget,
) -> Tuple[int, int]:
    data = (content or "").encode("utf-8")
    data_len = len(data)
    ext = Path(focus_file or "").suffix.lower()

    # Python safety: prefer line-derived range over tree-sitter byte_end.
    if ext == ".py":
        s, e = _line_to_byte_range(content or "", int(target.line_start or 1), int(target.line_end or 1))
        s = max(0, min(data_len, int(s)))
        e = max(s, min(data_len, int(e)))
        if e > s:
            return s, e

    start = max(0, int(target.byte_start or 0))
    end = max(start, int(target.byte_end or start))
    if end > data_len:
        end = data_len
    if end <= start:
        s, e = _line_to_byte_range(content or "", int(target.line_start or 1), int(target.line_end or 1))
        start = max(0, min(data_len, int(s)))
        end = max(start, min(data_len, int(e)))
    return start, end


def _byte_to_line(content: str, byte_pos: int) -> int:
    starts = _line_byte_offsets(content)
    total_lines = max(1, len(starts) - 1)
    b = max(0, int(byte_pos or 0))
    # starts is monotonic; find greatest line whose start <= b
    line = 1
    for i in range(1, len(starts)):
        if starts[i] > b:
            line = i
            break
    else:
        line = total_lines
    return max(1, min(total_lines, int(line)))


def _slice_by_byte_range(content: str, start: int, end: int) -> str:
    data = (content or "").encode("utf-8")
    s = max(0, int(start or 0))
    e = max(s, int(end or s))
    if e > len(data):
        e = len(data)
    return data[s:e].decode("utf-8", errors="replace")


def line_to_byte_range(content: str, line_start: int, line_end: int) -> Tuple[int, int]:
    return _line_to_byte_range(content, line_start, line_end)


def byte_to_line(content: str, byte_pos: int) -> int:
    return _byte_to_line(content, byte_pos)


def slice_by_byte_range(content: str, start: int, end: int) -> str:
    return _slice_by_byte_range(content, start, end)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_c_params(params: str) -> str:
    raw = (params or "").strip()
    if not raw:
        return ""
    parts = [p.strip() for p in raw.split(",")]
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        x = _normalize_space(p)
        x = re.sub(r"\s*,\s*", ",", x)
        x = re.sub(r"\s*\*\s*", "*", x)
        out.append(x)
    return ",".join(out)


def _extract_c_signature_parts(body: str) -> Optional[Tuple[str, str, str]]:
    text = (body or "").strip()
    if not text:
        return None
    m = re.search(
        r"(?s)^\s*([A-Za-z_][A-Za-z0-9_\s\*\[\]]*?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*\{",
        text,
    )
    if not m:
        return None
    ret = _normalize_space(m.group(1) or "")
    name = str(m.group(2) or "").strip()
    params = _normalize_c_params(m.group(3) or "")
    if not ret or not name:
        return None
    return ret, name, params


def _is_c_family_file(path: str) -> bool:
    ext = Path(path or "").suffix.lower()
    return ext in {".c", ".h", ".cc", ".cpp", ".hpp"}


def _is_brace_language_file(path: str) -> bool:
    ext = Path(path or "").suffix.lower()
    return ext in {
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cxx", ".hh",
        ".js", ".jsx", ".ts", ".tsx",
        ".java", ".go", ".rs", ".cs", ".php", ".swift", ".kt",
    }


def _find_matching_symbol(index: List[StructuralTarget], target: StructuralTarget) -> Optional[StructuralTarget]:
    hits = [s for s in index if s.name == target.name and s.kind == target.kind]
    if not hits:
        return None
    hits.sort(
        key=lambda s: (
            abs(int(s.byte_start) - int(target.byte_start)),
            abs(int(s.line_start) - int(target.line_start)),
            abs((int(s.byte_end) - int(s.byte_start)) - (int(target.byte_end) - int(target.byte_start))),
        )
    )
    return hits[0]


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    items = sorted((int(s), int(e)) for s, e in ranges)
    merged: List[Tuple[int, int]] = []
    for s, e in items:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _changed_regions(original_bytes: bytes, candidate_bytes: bytes) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    import difflib
    sm = difflib.SequenceMatcher(a=original_bytes, b=candidate_bytes, autojunk=False)
    original_ranges: List[Tuple[int, int]] = []
    candidate_ranges: List[Tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        original_ranges.append((int(i1), int(i2)))
        candidate_ranges.append((int(j1), int(j2)))
    return _merge_ranges(original_ranges), _merge_ranges(candidate_ranges)


def _range_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(int(a_start), int(b_start)) < min(int(a_end), int(b_end))


def _param_count(params: str) -> int:
    p = (params or "").strip()
    if not p:
        return 0
    if p.lower() == "void":
        return 0
    return len([x for x in p.split(",") if x.strip()])


def _extract_signature_info(body: str) -> Optional[Tuple[str, int, Optional[str]]]:
    c_sig = _extract_c_signature_parts(body)
    if c_sig:
        ret, name, params = c_sig
        return name, _param_count(params), ret or None

    text = (body or "").strip()
    if not text:
        return None
    py = re.search(r"(?m)^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:->\s*([^:]+))?:", text)
    if py:
        name = str(py.group(1) or "").strip()
        params = str(py.group(2) or "")
        ret = (py.group(3) or "").strip() or None
        return name, _param_count(params), ret

    js = re.search(r"(?m)^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*\{", text)
    if js:
        name = str(js.group(1) or "").strip()
        params = str(js.group(2) or "")
        return name, _param_count(params), None

    return None


def _first_nonblank_line(text: str) -> str:
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln
    return ""


def _indent_style(symbol_text: str) -> str:
    for ln in (symbol_text or "").splitlines():
        if not ln.strip():
            continue
        m = re.match(r"^(\s+)\S", ln)
        if not m:
            continue
        ws = m.group(1)
        if "\t" in ws:
            return "tabs"
        return "spaces"
    return "unknown"


def _js_ts_decl_style(symbol_text: str) -> str:
    line = _first_nonblank_line(symbol_text).lstrip()
    if re.match(r"^(?:const|let|var)\s+[A-Za-z_][A-Za-z0-9_]*\s*=", line):
        return "arrow"
    if re.match(r"^(?:export\s+)?function\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", line):
        return "function"
    return "function"


_PY_HEADER_LINE_RE = re.compile(r"^\s*(?:async\s+def|def|class)\b.*:\s*$")


def _python_find_header_line(text: str) -> str:
    for ln in (text or "").splitlines():
        if _PY_HEADER_LINE_RE.match(ln or ""):
            return (ln or "").rstrip()
    return ""


def _python_line_has_inline_header_body(line: str) -> bool:
    ln = str(line or "")
    if not re.match(r"^\s*(?:async\s+def|def|class)\b", ln):
        return False
    colon = ln.rfind(":")
    if colon < 0:
        return False
    tail = (ln[colon + 1 :] or "").strip()
    if not tail:
        return False
    if tail.startswith("#"):
        return False
    return True


def _python_has_one_line_header_body(text: str) -> bool:
    for ln in (text or "").splitlines():
        if _python_line_has_inline_header_body(ln):
            return True
    return False


def _python_extract_header_line(
    original_signature_line: str,
    original_symbol_text: str,
    target_name: str,
    target_kind: str,
) -> str:
    header = _python_find_header_line(original_signature_line or "")
    if header:
        return header
    header = _python_find_header_line(original_symbol_text or "")
    if header:
        return header
    name = (target_name or "target_symbol").strip() or "target_symbol"
    return f"def {name}(...):"


def _python_header_indent(header_line: str) -> str:
    m = re.match(r"^([ \t]*)", header_line or "")
    return str(m.group(1) if m else "")


def _python_body_indent(header_ws: str, original_symbol_text: str) -> str:
    if _indent_style(original_symbol_text or "") == "tabs":
        return f"{header_ws}\t"
    return f"{header_ws}    "


def _normalize_header_for_compare(line: str) -> str:
    raw = (line or "").rstrip()
    if not raw:
        return ""
    m = re.match(r"^([ \t]*)(.*)$", raw)
    ws = str(m.group(1) if m else "")
    body = str(m.group(2) if m else raw).strip()
    body = re.sub(r"[ \t]+", " ", body)
    return f"{ws}{body}"


def structural_output_skeleton(
    focus_file: str,
    target_name: str = "",
    target_kind: str = "",
    original_symbol_text: str = "",
    original_signature_line: str = "",
) -> List[str]:
    _ = original_symbol_text
    _ = original_signature_line
    _ = focus_file
    name = (target_name or "target").strip() or "target"
    kind = (target_kind or "symbol").strip() or "symbol"
    out: List[str] = [
        "OUTPUT CONTRACT:",
        f"- Return the updated {kind} `{name}` only.",
        "- Keep the same signature unless the request explicitly asks to change it.",
        "- No other top-level symbols.",
    ]
    return out


def structural_structure_rules(
    target_kind: str = "",
    target_name: str = "",
    original_signature_line: str = "",
    original_symbol_text: str = "",
) -> List[str]:
    _ = original_symbol_text
    sig = (original_signature_line or "").strip()
    name = (target_name or "target").strip() or "target"
    kind = (target_kind or "symbol").strip() or "symbol"
    rules: List[str] = [
        "STRUCTURE RULES:",
        f"- Edit only `{name}` ({kind}).",
        "- No unrelated top-level changes.",
    ]
    if sig:
        rules.append(f"- Keep signature/header unless explicitly requested: {sig}")
    return rules


def structural_format_retry_rules(
    focus_file: str,
    target_name: str = "",
    target_kind: str = "",
) -> List[str]:
    _ = focus_file
    _ = target_name
    _ = target_kind
    return [
        "RETRY RULES:",
        "- You must output exactly one symbol. Start with BEGIN_SYMBOL and end with END_SYMBOL. No other text.",
        "BEGIN_SYMBOL",
        "END_SYMBOL",
    ]


def structural_general_retry_rules() -> List[str]:
    return [
        "RETRY RULES:",
        "- Keep structural mode and return only the requested target symbol.",
        "- No explanation and no unrelated top-level symbols.",
        "- Keep signature unless explicitly requested.",
    ]


def _starts_with_token(text: str) -> str:
    line = _first_nonblank_line(text).lstrip()
    if not line:
        return ""
    m_async_def = re.match(r"async\s+def\b", line)
    if m_async_def:
        return "def"
    m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", line)
    return str(m.group(0) if m else "").strip().lower()


def _has_signature_line(text: str) -> bool:
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if "(" in s and ")" in s and (":" in s or "{" in s):
            return True
    return False


def _python_header_line_index(text: str) -> int:
    lines = (text or "").splitlines()
    for idx, ln in enumerate(lines):
        if _PY_HEADER_LINE_RE.match(ln or ""):
            return idx
    for idx, ln in enumerate(lines):
        if _python_line_has_inline_header_body(ln or ""):
            return idx
    return -1


def _python_has_indented_body(text: str) -> bool:
    lines = (text or "").splitlines()
    hidx = _python_header_line_index(text)
    if hidx < 0 or hidx + 1 >= len(lines):
        return False
    header_ws = _python_header_indent(lines[hidx] if hidx < len(lines) else "")
    for ln in lines[hidx + 1 :]:
        if not ln.strip():
            continue
        m = re.match(r"^([ \t]+)\S", ln)
        if not m:
            return False
        body_ws = str(m.group(1) or "")
        if len(body_ws) > len(header_ws) and (not header_ws or body_ws.startswith(header_ws)):
            return True
        return False
    return False


def _shape_consistency_error(
    focus_file: str,
    original_symbol_text: str,
    replacement_text: str,
) -> Optional[str]:
    original = str(original_symbol_text or "")
    replacement = str(replacement_text or "")
    if not original.strip() or not replacement.strip():
        return None

    o_tok = _starts_with_token(original)
    c_tok = _starts_with_token(replacement)
    if o_tok and c_tok and o_tok != c_tok:
        return "replacement_shape_startswith_mismatch"

    if _has_signature_line(original) and not _has_signature_line(replacement):
        return "replacement_shape_signature_missing"

    if _is_brace_language_file(focus_file):
        if not _has_balanced_braces(replacement):
            return "replacement_shape_brace_mismatch"
        if "{" in original and "}" in original and ("{" not in replacement or "}" not in replacement):
            return "replacement_shape_brace_mismatch"

    if Path(focus_file or "").suffix.lower() == ".py":
        original_header = _python_find_header_line(original)
        replacement_header = _python_find_header_line(replacement)
        if original_header and replacement_header:
            if _normalize_header_for_compare(original_header) != _normalize_header_for_compare(replacement_header):
                return "replacement_shape_header_mismatch"

    return None


def _nested_symbols(index: List[StructuralTarget], owner: StructuralTarget) -> Set[Tuple[str, str, str]]:
    out: Set[Tuple[str, str, str]] = set()
    for sym in index:
        if (
            sym.name == owner.name
            and sym.kind == owner.kind
            and int(sym.byte_start) == int(owner.byte_start)
            and int(sym.byte_end) == int(owner.byte_end)
        ):
            continue
        if int(sym.byte_start) >= int(owner.byte_start) and int(sym.byte_end) <= int(owner.byte_end):
            out.add((sym.name, sym.kind, sym.parent or ""))
    return out


def _root_level_open_brace_count(text: str) -> int:
    depth = 0
    root_open = 0
    for ch in text or "":
        if ch == "{":
            if depth == 0:
                root_open += 1
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
    return root_open


def _has_balanced_braces(text: str) -> bool:
    depth = 0
    for ch in text or "":
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _extract_symbol_mentions(user_text: str) -> Set[str]:
    text = user_text or ""
    out: Set[str] = set()
    for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]*)`", text):
        out.add(m.group(1))
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        out.add(m.group(1))
    for m in re.finditer(r"\b(?:function|method|class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b", text, re.IGNORECASE):
        out.add(m.group(1))
    return {x for x in out if x}


def _analysis_touchpoint_symbols(analysis_packet: str) -> List[str]:
    try:
        parsed = json.loads(analysis_packet or "")
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    points = parsed.get("touch_points")
    if not isinstance(points, list):
        return []
    symbols: List[str] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        sym = str(point.get("symbol") or "").strip()
        if sym and sym != "<file>":
            symbols.append(sym)
    return symbols


def _candidate_name_scores(user_text: str, candidate_names: List[str]) -> Dict[str, int]:
    text = user_text or ""
    low = text.lower()
    scores: Dict[str, int] = {name.lower(): 0 for name in candidate_names}
    for name in candidate_names:
        n = name.lower()
        if not n:
            continue
        scores[n] += len(re.findall(rf"`{re.escape(n)}`", low)) * 5
        scores[n] += len(re.findall(rf"\b{re.escape(n)}\s*\(", low)) * 4
        scores[n] += len(re.findall(rf"\b(?:in|inside|within|for|on)\s+{re.escape(n)}\b", low)) * 3
        scores[n] += len(re.findall(rf"\b{re.escape(n)}\b", low))
    return scores


def _disambiguate_candidates(
    user_text: str,
    candidates: List[StructuralTarget],
    unique_touch: List[str],
) -> Optional[StructuralTarget]:
    if not candidates:
        return None
    by_name: Dict[str, List[StructuralTarget]] = {}
    for c in candidates:
        by_name.setdefault(c.name.lower(), []).append(c)

    if len(unique_touch) == 1:
        one = unique_touch[0].lower().strip()
        if one in by_name:
            return by_name[one][0]

    scores = _candidate_name_scores(user_text or "", list(by_name.keys()))
    if not scores:
        return None
    best_score = max(scores.values())
    if best_score <= 0:
        return None
    best_names = [name for name, score in scores.items() if score == best_score]
    if len(best_names) != 1:
        return None
    chosen = best_names[0]
    return by_name.get(chosen, [None])[0]


def _looks_cross_scope_request(user_text: str) -> bool:
    low = (user_text or "").lower()
    cross_signals = (
        "across file",
        "across files",
        "whole file",
        "entire file",
        "all functions",
        "all methods",
        "imports and",
        "include and",
        "cross-function",
        "multiple functions",
        "multiple methods",
        "plus import",
        "add import",
        "update import",
    )
    return any(sig in low for sig in cross_signals)


def build_symbol_index(focus_file: str, content: str) -> List[StructuralTarget]:
    symbols = extract_symbols_treesitter(focus_file, content=content) or []
    out: List[StructuralTarget] = []
    for sym in symbols:
        name = str(getattr(sym, "name", "") or "").strip()
        kind = str(getattr(sym, "kind", "") or "").strip()
        if not name or kind not in {"function", "method", "class"}:
            continue
        line_start = int(getattr(sym, "line", 1) or 1)
        line_end = int(getattr(sym, "end_line", line_start) or line_start)
        if line_end < line_start:
            line_end = line_start
        byte_start = int(getattr(sym, "start_byte", 0) or 0)
        byte_end = int(getattr(sym, "end_byte", 0) or 0)
        if byte_end <= byte_start:
            byte_start, byte_end = _line_to_byte_range(content, line_start, line_end)
        if byte_end > byte_start:
            line_start_from_byte = _byte_to_line(content, byte_start)
            line_end_from_byte = _byte_to_line(content, max(byte_start, byte_end - 1))
            line_start = max(1, min(line_start, line_start_from_byte))
            line_end = max(line_end, line_end_from_byte)
        parent = str(getattr(sym, "parent", "") or "").strip()
        out.append(
            StructuralTarget(
                name=name,
                kind=kind,
                line_start=line_start,
                line_end=line_end,
                byte_start=max(0, byte_start),
                byte_end=max(0, byte_end),
                parent=parent,
            )
        )
    return out


def select_target_symbol(
    user_text: str,
    focus_file: str,
    content: str,
    analysis_packet: str,
) -> StructuralDecision:
    if _looks_cross_scope_request(user_text):
        return StructuralDecision(False, "cross_scope_request")

    symbols = build_symbol_index(focus_file, content)
    if not symbols:
        return StructuralDecision(False, "treesitter_parse_failed")

    names_to_symbols: Dict[str, List[StructuralTarget]] = {}
    for sym in symbols:
        names_to_symbols.setdefault(sym.name.lower(), []).append(sym)

    touchpoint_symbols = _analysis_touchpoint_symbols(analysis_packet)
    unique_touch = sorted({s.lower() for s in touchpoint_symbols if s})
    if len(unique_touch) > 1:
        matched = [x for x in unique_touch if x in names_to_symbols]
        if len(matched) > 1:
            return StructuralDecision(False, "multiple_touch_points")

    mentions = {m.lower() for m in _extract_symbol_mentions(user_text)}
    low_text = (user_text or "").lower()
    if low_text:
        for low_name in names_to_symbols.keys():
            if re.search(rf"\b{re.escape(low_name)}\b", low_text):
                mentions.add(low_name)
    candidates: List[StructuralTarget] = []

    for low_name in mentions:
        hits = names_to_symbols.get(low_name) or []
        candidates.extend(hits)

    if not candidates and unique_touch:
        for low_name in unique_touch:
            hits = names_to_symbols.get(low_name) or []
            candidates.extend(hits)

    dedup: Dict[Tuple[str, int, int, str], StructuralTarget] = {}
    for c in candidates:
        dedup[(c.name, c.line_start, c.line_end, c.kind)] = c
    unique_candidates = list(dedup.values())

    if len(unique_candidates) == 1:
        return StructuralDecision(True, "ok", unique_candidates[0])
    if len(unique_candidates) > 1:
        chosen = _disambiguate_candidates(
            user_text=user_text,
            candidates=unique_candidates,
            unique_touch=unique_touch,
        )
        if chosen is not None:
            return StructuralDecision(True, "disambiguated", chosen)
        return StructuralDecision(False, "ambiguous_symbols")

    return StructuralDecision(False, "target_symbol_not_identified")


def select_target_symbols(
    user_text: str,
    focus_file: str,
    content: str,
    analysis_packet: str,
    max_targets: int = 6,
) -> List[StructuralTarget]:
    if _looks_cross_scope_request(user_text):
        return []
    symbols = build_symbol_index(focus_file, content)
    if not symbols:
        return []

    names_to_symbols: Dict[str, List[StructuralTarget]] = {}
    for sym in symbols:
        names_to_symbols.setdefault(sym.name.lower(), []).append(sym)

    low_text = (user_text or "").lower()
    ordered_names: List[str] = []
    seen_names: Set[str] = set()

    pos_hits: List[Tuple[int, str]] = []
    for low_name in names_to_symbols.keys():
        m = re.search(rf"\b{re.escape(low_name)}\b", low_text)
        if m:
            pos_hits.append((int(m.start()), low_name))
    pos_hits.sort(key=lambda x: x[0])
    for _pos, low_name in pos_hits:
        if low_name in seen_names:
            continue
        seen_names.add(low_name)
        ordered_names.append(low_name)

    for touch in _analysis_touchpoint_symbols(analysis_packet):
        low_name = str(touch or "").strip().lower()
        if not low_name or low_name in seen_names:
            continue
        if low_name in names_to_symbols:
            seen_names.add(low_name)
            ordered_names.append(low_name)

    targets: List[StructuralTarget] = []
    for low_name in ordered_names[: max(1, int(max_targets))]:
        hits = names_to_symbols.get(low_name) or []
        if hits:
            targets.append(hits[0])

    if targets:
        return targets[: max(1, int(max_targets))]

    single = select_target_symbol(user_text, focus_file, content, analysis_packet)
    if single.eligible and single.target:
        return [single.target]
    return []


def apply_symbol_replacement(
    content: str,
    target: StructuralTarget,
    replacement_text: str,
    focus_file: str = "",
) -> str:
    original_bytes = (content or "").encode("utf-8")
    start, end = _replacement_byte_span(
        focus_file=focus_file or "",
        content=content or "",
        target=target,
    )
    if start == end:
        start = max(0, int(target.byte_start or 0))
        end = max(start, int(target.byte_end or start))
        if end > len(original_bytes):
            end = len(original_bytes)
    replacement_bytes = (replacement_text or "").encode("utf-8")
    out_bytes = original_bytes[:start] + replacement_bytes + original_bytes[end:]
    return out_bytes.decode("utf-8", errors="replace")


def validate_structural_candidate(
    focus_file: str,
    original_content: str,
    candidate_content: str,
    target: StructuralTarget,
    allow_signature_change: bool = False,
) -> Tuple[bool, str]:
    if (candidate_content or "") == (original_content or ""):
        return False, "no_effective_change"

    original_bytes = (original_content or "").encode("utf-8")
    candidate_bytes = (candidate_content or "").encode("utf-8")
    target_start, target_end = _replacement_byte_span(
        focus_file=focus_file,
        content=original_content or "",
        target=target,
    )
    target_start = max(0, int(target_start))
    target_end = max(target_start, int(target_end))
    changed_original, _changed_candidate = _changed_regions(original_bytes, candidate_bytes)
    if not changed_original:
        return False, "no_effective_change"

    for s, e in changed_original:
        if s < target_start or e > target_end:
            return False, "replacement_outside_target_span"

    change_start = min(s for s, _ in changed_original)
    change_end = max(e for _, e in changed_original)
    if not _range_overlaps(change_start, max(change_start + 1, change_end), target_start, max(target_start + 1, target_end)):
        return False, "replacement_outside_target_span"

    original_index = build_symbol_index(focus_file, original_content or "")
    candidate_index = build_symbol_index(focus_file, candidate_content or "")
    if not candidate_index:
        return False, "candidate_parse_failed"
    if len(original_index) > 1 and len(candidate_index) < len(original_index):
        return False, "top_level_symbol_count_collapsed"

    original_target = _find_matching_symbol(original_index, target) or target

    original_symbol_text = _slice_by_byte_range(
        original_content or "",
        original_target.byte_start,
        original_target.byte_end,
    )
    candidate_symbol_text = _slice_by_byte_range(
        candidate_content or "",
        target_start,
        target_end,
    )
    if not original_symbol_text.strip() or not candidate_symbol_text.strip():
        return False, "target_symbol_missing_after_replacement"

    if target.kind in {"function", "method"} and not allow_signature_change:
        orig_sig = _extract_signature_info(original_symbol_text)
        cand_sig = _extract_signature_info(candidate_symbol_text)
        if not orig_sig or not cand_sig:
            return False, "signature_modified"
        if orig_sig[0] != cand_sig[0]:
            return False, "signature_modified"
        if int(orig_sig[1]) != int(cand_sig[1]):
            return False, "signature_modified"
        o_ret = (orig_sig[2] or "").strip()
        c_ret = (cand_sig[2] or "").strip()
        if o_ret and c_ret and _normalize_space(o_ret) != _normalize_space(c_ret):
            return False, "signature_modified"

    delta = len(candidate_bytes) - len(original_bytes)
    modified_start = target_start
    modified_end = max(modified_start, target_end + delta)
    symbols_in_original_region = [
        s
        for s in candidate_index
        if int(s.byte_start) >= target_start and int(s.byte_end) <= target_end
    ]
    symbols_in_modified_region = [
        s
        for s in candidate_index
        if int(s.byte_start) >= modified_start and int(s.byte_end) <= modified_end
    ]
    chosen: List[StructuralTarget]
    if len(symbols_in_original_region) == 1:
        chosen = symbols_in_original_region
    elif len(symbols_in_modified_region) == 1:
        chosen = symbols_in_modified_region
    else:
        return False, "symbol_region_corrupted"

    candidate_target = chosen[0]
    if candidate_target.name != target.name or candidate_target.kind != target.kind:
        return False, "symbol_region_corrupted"

    if (candidate_target.parent or "") != (target.parent or ""):
        return False, "symbol_region_corrupted"

    if len(candidate_index) != len(original_index):
        return False, "unexpected_top_level_symbol"

    original_counter = Counter((s.name, s.kind, s.parent or "") for s in original_index)
    candidate_counter = Counter((s.name, s.kind, s.parent or "") for s in candidate_index)
    if candidate_counter != original_counter:
        added = candidate_counter - original_counter
        for ident, count in added.items():
            if count <= 0:
                continue
            outside = [
                sym
                for sym in candidate_index
                if (sym.name, sym.kind, sym.parent or "") == ident
                and not _range_overlaps(
                    int(sym.byte_start),
                    int(sym.byte_end),
                    modified_start,
                    modified_end,
                )
            ]
            if outside:
                return False, "unexpected_top_level_symbol"
        return False, "unexpected_top_level_symbol"

    return True, ""


def validate_replacement_symbol_unit(
    focus_file: str,
    replacement_text: str,
    target_name: str,
    target_kind: str,
    original_symbol_text: str = "",
    forbidden_symbol_names: Optional[List[str]] = None,
    allow_signature_change: bool = False,
    enforce_shape_guard: bool = True,
    normalized_mode: bool = False,
) -> Tuple[bool, str]:
    if normalized_mode:
        if not allow_signature_change:
            sig_err = _signature_change_error(
                target_kind=target_kind,
                original_symbol_text=original_symbol_text,
                replacement_text=replacement_text,
            )
            if sig_err:
                return False, sig_err
    else:
        if Path(focus_file or "").suffix.lower() == ".py":
            if not _python_parses_as_module(replacement_text or ""):
                return False, "replacement_symbol_syntax_invalid"
        scope_ok, scope_err = _validate_replacement_symbol_scope(
            focus_file=focus_file,
            text=replacement_text or "",
            target_symbol=target_name,
            target_kind=target_kind,
            markerless=False,
        )
        if not scope_ok:
            return False, scope_err or "replacement_symbol_shape_invalid"
    offender = _contains_forbidden_symbol_defs(
        focus_file=focus_file,
        text=replacement_text or "",
        forbidden_symbol_names=list(forbidden_symbol_names or []),
    )
    if offender:
        return False, "forbidden_symbol_definition_in_step"
    if enforce_shape_guard and not allow_signature_change:
        shape_err = _shape_consistency_error(
            focus_file=focus_file,
            original_symbol_text=original_symbol_text,
            replacement_text=replacement_text,
        )
        if shape_err:
            return False, shape_err
    return True, ""


def _signature_change_error(
    target_kind: str,
    original_symbol_text: str,
    replacement_text: str,
) -> str:
    kind = str(target_kind or "").strip().lower()
    original = str(original_symbol_text or "")
    replacement = str(replacement_text or "")
    if not original.strip() or not replacement.strip():
        return ""
    if kind in {"function", "method"}:
        orig_sig = _extract_signature_info(original)
        cand_sig = _extract_signature_info(replacement)
        if not orig_sig or not cand_sig:
            return "signature_modified"
        if orig_sig[0] != cand_sig[0]:
            return "signature_modified"
        if int(orig_sig[1]) != int(cand_sig[1]):
            return "signature_modified"
        o_ret = (orig_sig[2] or "").strip()
        c_ret = (cand_sig[2] or "").strip()
        if o_ret and c_ret and _normalize_space(o_ret) != _normalize_space(c_ret):
            return "signature_modified"
        return ""
    if kind == "class":
        original_header = _first_nonblank_line(original)
        replacement_header = _first_nonblank_line(replacement)
        if (
            original_header
            and replacement_header
            and _normalize_header_for_compare(original_header)
            != _normalize_header_for_compare(replacement_header)
        ):
            return "signature_modified"
    return ""


def extract_target_snippet(content: str, target: StructuralTarget, padding_lines: int = 1) -> str:
    lines = (content or "").splitlines()
    if not lines:
        return ""
    start = max(1, int(target.line_start) - max(0, int(padding_lines)))
    end = min(len(lines), int(target.line_end) + max(0, int(padding_lines)))
    return "\n".join(lines[start - 1 : end]).strip()


def _target_symbol_text(content: str, target: StructuralTarget) -> str:
    try:
        return _slice_by_byte_range(content or "", int(target.byte_start), int(target.byte_end)).strip()
    except Exception:
        return extract_target_snippet(content or "", target, padding_lines=0).strip()


def _symbol_signature_line(text: str) -> str:
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln.rstrip()
    return ""


def _request_wants_behavior_context(user_text: str) -> bool:
    low = (user_text or "").lower()
    if not low:
        return False
    signals = (
        "behavior",
        "logic",
        "bug",
        "edge case",
        "correctness",
        "wrong",
        "fix",
        "should",
        "regression",
        "unexpected",
    )
    return any(tok in low for tok in signals)


def _identifier_counts(text: str) -> Counter:
    ids = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text or "")
    return Counter(ids)


def _direct_call_names(text: str) -> Set[str]:
    keywords = {
        "if", "for", "while", "switch", "return", "catch", "throw", "sizeof",
        "print", "assert", "await", "yield", "new", "lambda",
    }
    calls = set(re.findall(r"(?<!\.)\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text or ""))
    return {c for c in calls if c not in keywords}


def _annotation_type_names(text: str) -> Set[str]:
    out: Set[str] = set()
    for m in re.finditer(r":\s*([A-Za-z_][A-Za-z0-9_\.]*)", text or ""):
        out.add(str(m.group(1) or "").split(".")[-1])
    for m in re.finditer(r"->\s*([A-Za-z_][A-Za-z0-9_\.]*)", text or ""):
        out.add(str(m.group(1) or "").split(".")[-1])
    return {x for x in out if x}


def _top_level_constant_map(content: str, index: List[StructuralTarget]) -> Dict[str, Tuple[int, str]]:
    occupied: List[Tuple[int, int]] = []
    for sym in index:
        if str(sym.parent or "").strip():
            continue
        occupied.append((int(sym.line_start), int(sym.line_end)))
    lines = (content or "").splitlines()
    out: Dict[str, Tuple[int, str]] = {}
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        inside_symbol = any(s <= i <= e for s, e in occupied)
        if inside_symbol:
            continue
        m = re.match(r"^\s*([A-Z][A-Z0-9_]{1,})\s*=\s*(.+)$", line)
        if not m:
            continue
        name = str(m.group(1) or "").strip()
        out[name] = (i, line.rstrip())
    return out


def _closest_callsite_blocks(
    content: str,
    target: StructuralTarget,
    target_name: str,
    max_sites: int = 2,
) -> List[str]:
    lines = (content or "").splitlines()
    if not lines or not target_name:
        return []
    pat = re.compile(rf"\b{re.escape(target_name)}\s*\(")
    hits: List[Tuple[int, int]] = []
    for i, line in enumerate(lines, start=1):
        if i >= int(target.line_start) and i <= int(target.line_end):
            continue
        if not pat.search(line):
            continue
        if re.search(rf"\b(?:def|class|function)\s+{re.escape(target_name)}\b", line):
            continue
        prev = "\n".join(lines[max(0, i - 3) : i]).lower()
        if "__name__" in prev and "__main__" in prev:
            continue
        dist = abs(i - int(target.line_start))
        hits.append((dist, i))
    hits.sort(key=lambda t: t[0])
    blocks: List[str] = []
    for _dist, line_no in hits[: max(0, int(max_sites))]:
        s = max(1, line_no - 1)
        e = min(len(lines), line_no + 1)
        block = "\n".join(f"{ln}| {lines[ln - 1]}" for ln in range(s, e + 1)).strip()
        if not block:
            continue
        blocks.append(f"CALLSITE L{line_no}:\n{block}")
    return blocks


def _pack_within_budget(text: str, max_lines: int, max_bytes: int) -> bool:
    line_count = len((text or "").splitlines())
    byte_count = len((text or "").encode("utf-8"))
    return line_count <= max(1, int(max_lines)) and byte_count <= max(1, int(max_bytes))


def build_packed_context(
    target: StructuralTarget,
    content: str,
    index: List[StructuralTarget],
    user_text: str,
    focus_file: str = "",
    max_lines: int = 200,
    max_bytes: int = 8192,
) -> str:
    lines = (content or "").splitlines()
    if not lines:
        return ""
    ext = Path(focus_file or "").suffix.lower()
    lang = "python" if ext == ".py" else (ext.lstrip(".") or "source")
    target_text = _target_symbol_text(content or "", target)
    if not target_text:
        target_text = extract_target_snippet(content or "", target, padding_lines=0)
    target_text = target_text.strip()
    if not target_text:
        return ""

    top_level = [s for s in (index or []) if not str(s.parent or "").strip()]
    id_counts = _identifier_counts(target_text)
    dep_names: Set[str] = set()
    dep_names.update(_direct_call_names(target_text))
    dep_names.update(_annotation_type_names(target_text))
    dep_names.discard(target.name)

    name_map: Dict[str, StructuralTarget] = {}
    for sym in top_level:
        low = str(sym.name or "").strip().lower()
        if low and low not in name_map:
            name_map[low] = sym

    dep_symbols: List[Tuple[int, StructuralTarget]] = []
    for dep in dep_names:
        sym = name_map.get(str(dep).lower())
        if not sym:
            continue
        score = int(id_counts.get(sym.name, 0))
        dep_symbols.append((score, sym))
    dep_symbols.sort(key=lambda t: (-t[0], int(t[1].line_start), t[1].name))

    const_map = _top_level_constant_map(content or "", top_level)
    const_hits: List[Tuple[int, str, int, str]] = []
    for name, (line_no, line_text) in const_map.items():
        score = int(id_counts.get(name, 0))
        if score <= 0:
            continue
        const_hits.append((score, name, line_no, line_text))
    const_hits.sort(key=lambda t: (-t[0], t[2], t[1]))

    header_block = (
        f"FILE: {focus_file}\n"
        f"LANGUAGE: {lang}\n"
        f"TARGET_SYMBOL: {target.kind} {target.name}\n"
        "NOTE: edit ONLY this target symbol."
    )
    target_block = (
        "TARGET SYMBOL (CURRENT):\n"
        f"{target_text}"
    )

    dep_full: List[str] = []
    dep_sig: List[str] = []
    dep_cap = max(1, int(getattr(config, "STRUCTURAL_PACKED_MAX_DEPS", 6)))
    for _score, sym in dep_symbols[:dep_cap]:
        text = _target_symbol_text(content or "", sym)
        if not text:
            continue
        sig = _symbol_signature_line(text)
        dep_full.append(f"DEP {sym.kind} `{sym.name}`:\n{text}")
        if sig:
            dep_sig.append(f"DEP SIG {sym.kind} `{sym.name}`: {sig}")
    const_blocks = [
        f"DEP CONST L{line_no} `{name}`: {line_text}"
        for _score, name, line_no, line_text in const_hits[:dep_cap]
    ]

    callsite_blocks: List[str] = []
    if _request_wants_behavior_context(user_text):
        callsite_blocks = _closest_callsite_blocks(
            content=content or "",
            target=target,
            target_name=target.name,
            max_sites=max(0, int(getattr(config, "STRUCTURAL_PACKED_MAX_CALLSITES", 2))),
        )

    contract_lines = [
        "OUTPUT CONTRACT:",
        f"- Return updated `{target.kind} {target.name}` only.",
        "- No explanation.",
        "- Keep same signature unless explicitly requested.",
    ]
    if ext == ".py":
        contract_lines.append("- Python: use normal multi-line def/class.")
    contract_block = "\n".join(contract_lines)

    def compose(
        include_callsites: bool,
        use_dep_signatures: bool,
        dep_limit: int,
    ) -> str:
        sections: List[str] = [header_block, target_block]
        if dep_full or const_blocks:
            dep_sections: List[str] = []
            if use_dep_signatures:
                dep_sections.extend(dep_sig[:dep_limit])
            else:
                dep_sections.extend(dep_full[:dep_limit])
            dep_sections.extend(const_blocks[:dep_limit])
            if dep_sections:
                sections.append("MINIMAL DEPENDENCIES:\n" + "\n\n".join(dep_sections))
        if include_callsites and callsite_blocks:
            sections.append("CALL SITES:\n" + "\n\n".join(callsite_blocks))
        sections.append(contract_block)
        return "\n\n".join(s for s in sections if s).strip()

    dep_count = max(len(dep_full), len(dep_sig), len(const_blocks))
    text = compose(include_callsites=True, use_dep_signatures=False, dep_limit=dep_count)
    if _pack_within_budget(text, max_lines=max_lines, max_bytes=max_bytes):
        return text

    text = compose(include_callsites=False, use_dep_signatures=False, dep_limit=dep_count)
    if _pack_within_budget(text, max_lines=max_lines, max_bytes=max_bytes):
        return text

    text = compose(include_callsites=False, use_dep_signatures=True, dep_limit=dep_count)
    if _pack_within_budget(text, max_lines=max_lines, max_bytes=max_bytes):
        return text

    top_dep_limit = min(3, max(1, dep_count))
    text = compose(include_callsites=False, use_dep_signatures=True, dep_limit=top_dep_limit)
    if _pack_within_budget(text, max_lines=max_lines, max_bytes=max_bytes):
        return text

    hard_bytes = max(256, int(max_bytes))
    hard_lines = max(12, int(max_lines))
    trimmed_lines = text.splitlines()[:hard_lines]
    trimmed = "\n".join(trimmed_lines)
    if len(trimmed.encode("utf-8")) > hard_bytes:
        data = trimmed.encode("utf-8")[:hard_bytes]
        trimmed = data.decode("utf-8", errors="ignore").rstrip()
    return trimmed.strip()


def _contains_forbidden_symbol_defs(
    focus_file: str,
    text: str,
    forbidden_symbol_names: List[str],
) -> Optional[str]:
    forbidden = {
        str(n or "").strip().lower()
        for n in (forbidden_symbol_names or [])
        if str(n or "").strip()
    }
    if not forbidden:
        return None

    # Primary: language-aware parser extraction.
    for sym in build_symbol_index(focus_file, text or ""):
        low = str(sym.name or "").strip().lower()
        if low in forbidden:
            return sym.name
    return None


def _python_parses_as_module(text: str) -> bool:
    try:
        ast.parse(text or "")
        return True
    except Exception:
        return False


def _is_low_entropy_structural_output(text: str) -> bool:
    if not bool(getattr(config, "STRUCTURAL_LOW_ENTROPY_ENABLED", True)):
        return False
    src = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    stripped = src.strip()
    min_len = max(1, int(getattr(config, "STRUCTURAL_LOW_ENTROPY_MIN_LEN", 80)))
    dominance = float(getattr(config, "STRUCTURAL_LOW_ENTROPY_DOMINANCE", 0.72))
    if len(stripped) < min_len:
        return False

    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) >= 8:
        singleish = [ln for ln in lines if re.fullmatch(r"[A-Za-z_]{1,3}", ln or "")]
        if len(singleish) >= max(6, int(len(lines) * 0.65)):
            top_line_count = Counter(singleish).most_common(1)[0][1]
            if top_line_count >= max(6, int(len(lines) * 0.5)):
                return True

    compact = re.sub(r"\s+", "", stripped)
    if len(compact) >= min_len:
        uniq_ratio = len(set(compact)) / max(1, len(compact))
        if uniq_ratio <= 0.08:
            return True
        top_char_ratio = Counter(compact).most_common(1)[0][1] / max(1, len(compact))
        if top_char_ratio >= dominance:
            return True

    tokens = re.findall(r"[A-Za-z_]+|\S", stripped.lower())
    if len(tokens) >= 30:
        short_tokens = [t for t in tokens if len(t) <= 2]
        if short_tokens:
            top_short = Counter(short_tokens).most_common(1)[0][1]
            if top_short >= max(12, int(len(tokens) * 0.45)):
                return True
    return False


def _kind_matches_target(target_kind: str, candidate_kind: str) -> bool:
    t = str(target_kind or "").strip().lower()
    c = str(candidate_kind or "").strip().lower()
    if not t:
        return True
    if t == c:
        return True
    if t in {"function", "method"} and c in {"function", "method"}:
        return True
    return False


def _is_parent_under_target(parent: str, target_name: str) -> bool:
    p = str(parent or "").strip()
    t = str(target_name or "").strip()
    if not p or not t:
        return False
    if p == t:
        return True
    parts = [seg.strip() for seg in re.split(r"[.:]", p) if seg.strip()]
    return t in parts


def _validate_replacement_symbol_scope(
    focus_file: str,
    text: str,
    target_symbol: str = "",
    target_kind: str = "",
    markerless: bool = False,
) -> Tuple[bool, str]:
    src = str(text or "").strip()
    if not src:
        return False, ("markerless_output_parse_failed" if markerless else "replacement_symbol_shape_invalid")
    if Path(focus_file or "").suffix.lower() == ".py" and not _python_parses_as_module(src):
        return False, ("markerless_output_parse_failed" if markerless else "replacement_symbol_syntax_invalid")

    index = build_symbol_index(focus_file, src)
    if not index:
        if markerless:
            return False, "markerless_output_parse_failed"
        return False, "replacement_symbol_shape_invalid"

    top_level = [s for s in index if not str(s.parent or "").strip()]
    if not top_level:
        return False, ("markerless_output_parse_failed" if markerless else "replacement_symbol_shape_invalid")

    want_name = str(target_symbol or "").strip()
    want_kind = str(target_kind or "").strip()
    top_matches = [
        s for s in top_level
        if (not want_name or s.name == want_name)
        and _kind_matches_target(want_kind, s.kind)
    ]

    if len(top_matches) != 1:
        if markerless:
            return False, "markerless_output_symbol_mismatch"
        if want_name:
            return False, "target_symbol_missing_in_output"
        return False, "multiple_top_level_symbols_detected"

    if len(top_level) != 1:
        return False, ("markerless_output_symbol_mismatch" if markerless else "multiple_top_level_symbols_detected")

    target_top = top_matches[0]
    extras = [
        s for s in index
        if not (
            s.name == target_top.name
            and _kind_matches_target(target_top.kind, s.kind)
            and int(s.byte_start) == int(target_top.byte_start)
            and int(s.byte_end) == int(target_top.byte_end)
        )
    ]
    if not extras:
        return True, ""

    if str(want_kind or "").strip().lower() == "class":
        allowed_nested_kinds = {"method", "function"}
        for sym in extras:
            if not _is_parent_under_target(sym.parent or "", target_top.name):
                return False, ("markerless_output_symbol_mismatch" if markerless else "multiple_top_level_symbols_detected")
            if str(sym.kind or "").strip().lower() not in allowed_nested_kinds:
                return False, ("markerless_output_symbol_mismatch" if markerless else "multiple_top_level_symbols_detected")
        return True, ""

    return False, ("markerless_output_symbol_mismatch" if markerless else "multiple_top_level_symbols_detected")


def _matches_single_target_symbol(
    focus_file: str,
    text: str,
    target_symbol: str = "",
    target_kind: str = "",
) -> Tuple[bool, str]:
    return _validate_replacement_symbol_scope(
        focus_file=focus_file,
        text=text,
        target_symbol=target_symbol,
        target_kind=target_kind,
        markerless=True,
    )


def _repair_python_compact_def(text: str) -> str:
    src = (text or "").strip()
    if not src or "\n" in src:
        return src

    m = re.match(
        r"^\s*((?:async\s+def|def)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*?\)\s*(?:->\s*[^:]+)?):\s*(.+)$",
        src,
        re.DOTALL,
    )
    if not m:
        return src

    header = (m.group(1) or "").strip()
    body = (m.group(2) or "").strip()
    if not header or not body:
        return src

    # First: simple newline + indent.
    candidate = f"{header}:\n    {body}"
    if _python_parses_as_module(candidate):
        return candidate

    # Next: split semicolon statements into separate lines.
    if ";" in body:
        stmts = [x.strip() for x in body.split(";") if x.strip()]
        if stmts:
            candidate = f"{header}:\n" + "\n".join(f"    {stmt}" for stmt in stmts)
            if _python_parses_as_module(candidate):
                return candidate

    # Heuristic: inline if + trailing assignment/update statement.
    m_if = re.match(r"^\s*if\s+(.+?)\s*:\s*(.+)$", body, re.DOTALL)
    if m_if:
        cond = (m_if.group(1) or "").strip()
        tail = (m_if.group(2) or "").strip()
        split = re.search(
            r"\s+([A-Za-z_][A-Za-z0-9_\.]*\s*(?:\+=|-=|\*=|/=|//=|%=|=)\s*.+)$",
            tail,
        )
        if split:
            cons = tail[: split.start()].strip()
            trail = (split.group(1) or "").strip()
            if cond and cons and trail:
                candidate = (
                    f"{header}:\n"
                    f"    if {cond}:\n"
                    f"        {cons}\n"
                    f"    {trail}"
                )
                if _python_parses_as_module(candidate):
                    return candidate

    # Heuristic: chained inline if statements followed by trailing assignment/update.
    m_chain = re.match(
        r"^\s*if\s+(.+?)\s*:\s*(.+?)\s+if\s+(.+?)\s*:\s*(.+?)\s+([A-Za-z_][A-Za-z0-9_\.]*\s*(?:\+=|-=|\*=|/=|//=|%=|=)\s*.+)$",
        body,
        re.DOTALL,
    )
    if m_chain:
        cond1 = (m_chain.group(1) or "").strip()
        then1 = (m_chain.group(2) or "").strip()
        cond2 = (m_chain.group(3) or "").strip()
        then2 = (m_chain.group(4) or "").strip()
        trail = (m_chain.group(5) or "").strip()
        if cond1 and then1 and cond2 and then2 and trail:
            candidate = (
                f"{header}:\n"
                f"    if {cond1}:\n"
                f"        {then1}\n"
                f"    if {cond2}:\n"
                f"        {then2}\n"
                f"    {trail}"
            )
            if _python_parses_as_module(candidate):
                return candidate

    # Heuristic: if/if/else one-liner sequence.
    m_if_if_else = re.match(
        r"^\s*if\s+(.+?)\s*:\s*(.+?)\s+if\s+(.+?)\s*:\s*(.+?)\s+else\s*:\s*(.+)$",
        body,
        re.DOTALL,
    )
    if m_if_if_else:
        cond1 = (m_if_if_else.group(1) or "").strip()
        then1 = (m_if_if_else.group(2) or "").strip()
        cond2 = (m_if_if_else.group(3) or "").strip()
        then2 = (m_if_if_else.group(4) or "").strip()
        else2 = (m_if_if_else.group(5) or "").strip()
        if cond1 and then1 and cond2 and then2 and else2:
            candidate = (
                f"{header}:\n"
                f"    if {cond1}:\n"
                f"        {then1}\n"
                f"    if {cond2}:\n"
                f"        {then2}\n"
                f"    else:\n"
                f"        {else2}"
            )
            if _python_parses_as_module(candidate):
                return candidate

    return src


def normalize_structural_output(
    output: str,
    target_symbol: str = "",
    original_symbol_text: str = "",
    target_kind: str = "",
    focus_file: str = "",
) -> Tuple[str, Optional[str]]:
    text = (output or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```[A-Za-z0-9_-]*", "", text)
    text = text.replace("```", "").strip()
    if _is_low_entropy_structural_output(text):
        dbg("structural.low_entropy_reject.count=1")
        return "", "degenerate_low_entropy_output"
    has_begin = "BEGIN_SYMBOL" in text
    has_end = "END_SYMBOL" in text
    if has_begin or has_end:
        if text.count("BEGIN_SYMBOL") == 1 and text.count("END_SYMBOL") == 1:
            m = re.search(r"BEGIN_SYMBOL\s*(.*?)\s*END_SYMBOL", text, re.DOTALL)
            if m:
                text = (m.group(1) or "").strip()
            else:
                text = text.replace("BEGIN_SYMBOL", "").replace("END_SYMBOL", "").strip()
                dbg("structural.marker_salvage=malformed_markers")
        else:
            text = text.replace("BEGIN_SYMBOL", "").replace("END_SYMBOL", "").strip()
            dbg("structural.marker_salvage=partial_markers")
    else:
        markerless = text.strip()
        marker_ok, marker_reason = _matches_single_target_symbol(
            focus_file=focus_file,
            text=markerless,
            target_symbol=target_symbol,
            target_kind=target_kind,
        )
        if not marker_ok:
            return "", (marker_reason or "markerless_output_parse_failed")
        dbg("structural.markerless_salvage.accepted=1")
        text = markerless
    forbidden = (
        "[[[file:",
        "[[[end]]]",
        "diff --git ",
        "--- a/",
        "+++ b/",
        "@@ -",
    )
    for token in forbidden:
        if token in text:
            return "", "forbidden_artifact_in_structural_output"
    if re.search(r"(?m)^\s*\d+\s*\|", text) or re.search(r"(?m)^\s*\|", text):
        return "", "line_number_prefix_detected"
    if "#include" in text:
        return "", "includes_not_allowed"
    if re.search(r"\bint\s+main\s*\(", text):
        return "", "main_not_allowed"
    is_python_file = Path(focus_file or "").suffix.lower() == ".py"
    if is_python_file:
        if _python_has_one_line_header_body(text):
            repaired = _repair_python_compact_def(text)
            if repaired and repaired != text:
                text = repaired
        if _python_has_one_line_header_body(text):
            return "", "python_one_liner_def_not_allowed"
        repaired = _repair_python_compact_def(text)
        if (
            repaired
            and repaired != text
            and "\n" in repaired
            and _python_has_indented_body(repaired)
            and _python_parses_as_module(repaired)
        ):
            text = repaired
        if not _python_has_indented_body(text):
            return "", "replacement_shape_indentation_invalid"
    if not text.endswith("\n"):
        text = f"{text}\n"
    if _is_brace_language_file(focus_file):
        if not _has_balanced_braces(text):
            return "", "brace_mismatch"
        root_braces = _root_level_open_brace_count(text)
        if root_braces > 1:
            return "", "multiple_root_level_blocks"
        if original_symbol_text:
            original = str(original_symbol_text or "")
            if (target_kind or "").lower() in {"function", "method"} and "{" in original and root_braces == 0:
                return "", "missing_root_brace"
    if target_symbol:
        scope_ok, scope_err = _validate_replacement_symbol_scope(
            focus_file=focus_file,
            text=text,
            target_symbol=target_symbol,
            target_kind=target_kind,
            markerless=False,
        )
        if not scope_ok:
            return "", (scope_err or "replacement_symbol_shape_invalid")
    if not text:
        return "", "empty_structural_output"
    return text, None
