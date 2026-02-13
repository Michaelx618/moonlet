import hashlib
import re
import json
from typing import Dict, List, Optional, Set, Tuple

from . import config
from . import state
from .files import get_include, read_file_text, read_single_file_for_context
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    STRUCTURAL_MINIMALITY_PROMPT,
    _ext,
    _language_name_for_ext,
)
from .structural import (
    structural_output_skeleton,
    structural_structure_rules,
    line_to_byte_range,
    byte_to_line,
    slice_by_byte_range,
)
from .utils import dbg, dbg_dump

def _render_section(title: str, content: str) -> str:
    if not content:
        return ""
    header = f"{title}\n" if title else ""
    return f"{header}{content}\n\n"


def trim_to_budget(sections: List[Tuple[str, str]], budget: int) -> str:
    """Join sections and trim from the end to fit budget."""
    rendered = [_render_section(title, content) for title, content in sections]
    total = sum(len(chunk) for chunk in rendered)
    if total <= budget:
        return "".join(rendered).strip()
    updated = list(sections)
    for idx in range(len(updated) - 1, -1, -1):
        if total <= budget:
            break
        title, content = updated[idx]
        if not content:
            continue
        header = f"{title}\n" if title else ""
        trailer = "\n\n"
        section_len = len(header) + len(content) + len(trailer)
        allowed = budget - (total - section_len)
        if allowed <= len(header) + len(trailer):
            updated[idx] = (title, "")
            total -= section_len
            continue
        keep_len = max(0, allowed - len(header) - len(trailer))
        trimmed = content[:keep_len].rstrip()
        updated[idx] = (title, trimmed)
        total = total - section_len + len(header) + len(trimmed) + len(trailer)
        break
    rendered = [_render_section(title, content) for title, content in updated]
    combined = "".join(rendered).strip()
    return combined[:budget]


def _score_snippet(text: str, keywords: List[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(word) for word in keywords if word)


def _looks_like_task_card(text: str) -> bool:
    return str(text or "").lstrip().startswith("TASK_CARD")


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "but", "not", "are",
    "was", "were", "has", "have", "had", "been", "will", "can", "would",
    "could", "should", "may", "might", "its", "also", "into", "than",
    "then", "when", "where", "which", "what", "who", "how", "all", "each",
    "every", "some", "any", "most", "more", "other", "only", "very",
    "just", "about", "after", "before", "between", "through", "during",
    "without", "again", "here", "there", "because", "does", "did",
    "doing", "being", "having", "use", "using", "used", "make", "made",
    "file", "code", "line", "function", "please", "want", "need",
    "create", "introduction", "overview", "assignment", "lab", "week",
    "starter", "submit", "requirements", "goal", "therefore",
})


def _keywords_from_text(text: str) -> List[str]:
    """Extract keywords: remove stopwords, prefer longer/frequent tokens, cap at 25."""
    words = re.findall(r"[a-zA-Z_]\w+", text.lower())
    # Count frequency
    freq: Dict[str, int] = {}
    for w in words:
        if len(w) > 2 and w not in _STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    # Sort by frequency desc, then length desc
    ranked = sorted(freq.keys(), key=lambda w: (freq[w], len(w)), reverse=True)
    return ranked[:25]


def _head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    head = text[:half].rstrip()
    tail = text[-half:].lstrip()
    return f"{head}\n...\n{tail}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _parse_diag_lines_for_file(diag: str, file_path: str) -> List[int]:
    if not diag:
        return []
    rel = re.escape((file_path or "").replace("\\", "/"))
    out: List[int] = []
    for m in re.finditer(rf"{rel}:(\d+):\d+:", diag):
        try:
            out.append(int(m.group(1)))
        except Exception:
            continue
    for m in re.finditer(r"\bL(\d+)\b", diag):
        try:
            out.append(int(m.group(1)))
        except Exception:
            continue
    uniq: List[int] = []
    seen = set()
    for n in out:
        if n <= 0 or n in seen:
            continue
        seen.add(n)
        uniq.append(n)
    return uniq[:12]


def _merge_line_windows(
    windows: List[Tuple[int, int]],
    max_lines: int,
) -> List[Tuple[int, int]]:
    if not windows:
        return []
    pairs = sorted((max(1, s), max(1, e)) for s, e in windows if e >= s)
    merged: List[Tuple[int, int]] = []
    for s, e in pairs:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            merged[-1] = (ps, max(pe, e))
    # keep deterministic small set
    return merged[:max_lines]


def _search_anchor_lines(focus_file: str, keywords: List[str]) -> List[int]:
    if not keywords:
        return []
    try:
        from .tools import grep_search
    except Exception:
        return []
    out: List[int] = []
    seen = set()
    rel = focus_file.replace("\\", "/")
    for kw in keywords[:6]:
        try:
            res = grep_search(kw, max_results=30, literal=True)
        except Exception:
            continue
        for m in res.matches:
            if str(m.file).replace("\\", "/") != rel:
                continue
            ln = int(m.line)
            if ln in seen:
                continue
            seen.add(ln)
            out.append(ln)
            if len(out) >= 10:
                return out
    return out


def _symbol_anchor_lines(focus_file: str, content: str, keywords: List[str]) -> List[int]:
    try:
        from .tools import extract_symbols_treesitter
    except Exception:
        return []
    try:
        syms = extract_symbols_treesitter(focus_file, content=content) or []
    except Exception:
        syms = []
    if not syms:
        return []
    out: List[int] = []
    seen = set()
    for sym in syms[:100]:
        name = str(getattr(sym, "name", "") or "").lower()
        line = int(getattr(sym, "line", 1) or 1)
        if not name:
            continue
        if keywords and not any((k in name or name in k) for k in keywords[:12]):
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
        if len(out) >= 10:
            break
    if out:
        return out
    # fallback: first symbols
    for sym in syms[:6]:
        line = int(getattr(sym, "line", 1) or 1)
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _imports_window(lines: List[str]) -> Optional[Tuple[int, int]]:
    if not lines:
        return None
    patterns = (
        re.compile(r"^\s*#include\s+[<\"]"),
        re.compile(r"^\s*import\s+\S+"),
        re.compile(r"^\s*from\s+\S+\s+import\s+"),
        re.compile(r"^\s*package\s+\S+"),
        re.compile(r"^\s*using\s+\S+"),
    )
    end = 0
    for i, line in enumerate(lines[:160], start=1):
        if not line.strip():
            if i <= 40:
                end = max(end, i)
            continue
        if any(p.match(line) for p in patterns):
            end = i
            continue
        if end > 0 and i - end > 8:
            break
    if end <= 0:
        return None
    return (1, min(len(lines), max(24, end + 12)))


def _structural_is_tiny(content: str) -> bool:
    line_count = len((content or "").splitlines())
    byte_count = len((content or "").encode("utf-8"))
    return (
        line_count <= max(1, int(getattr(config, "STRUCTURAL_TINY_MAX_LINES", 30)))
        or byte_count <= max(1, int(getattr(config, "STRUCTURAL_TINY_MAX_BYTES", 700)))
    )


def _line_to_byte_range_local(content: str, line_start: int, line_end: int) -> Tuple[int, int]:
    return line_to_byte_range(content, line_start, line_end)


def _byte_to_line_local(content: str, byte_pos: int) -> int:
    return byte_to_line(content, byte_pos)


def _slice_bytes_local(content: str, start: int, end: int) -> str:
    return slice_by_byte_range(content, start, end)


def _resolve_symbol_span(
    focus_file: str,
    content: str,
    symbol_name: str,
    symbol_kind: str,
    line_start: int,
    line_end: int,
    byte_start: int,
    byte_end: int,
) -> Tuple[int, int, int, int]:
    data_len = len((content or "").encode("utf-8"))
    s_byte = int(byte_start or 0)
    e_byte = int(byte_end or 0)
    s_line = int(line_start or 1)
    e_line = int(line_end or s_line)
    if e_line < s_line:
        e_line = s_line

    if not (0 <= s_byte < e_byte <= data_len):
        try:
            syms = extract_symbols_treesitter(focus_file, content=content) or []
        except Exception:
            syms = []
        low_name = str(symbol_name or "").strip().lower()
        low_kind = str(symbol_kind or "").strip().lower()
        for sym in syms:
            name = str(getattr(sym, "name", "") or "").strip().lower()
            kind = str(getattr(sym, "kind", "") or "").strip().lower()
            if low_name and name != low_name:
                continue
            if low_kind and kind != low_kind:
                continue
            cand_s = int(getattr(sym, "start_byte", 0) or 0)
            cand_e = int(getattr(sym, "end_byte", 0) or 0)
            if 0 <= cand_s < cand_e <= data_len:
                s_byte, e_byte = cand_s, cand_e
                s_line = int(getattr(sym, "line", s_line) or s_line)
                e_line = int(getattr(sym, "end_line", e_line) or e_line)
                if e_line < s_line:
                    e_line = s_line
                break

    if not (0 <= s_byte < e_byte <= data_len):
        s_byte, e_byte = _line_to_byte_range_local(content, s_line, e_line)

    if e_byte <= s_byte:
        s_byte = max(0, min(data_len, s_byte))
        e_byte = max(s_byte, min(data_len, e_byte))
    if e_byte <= s_byte:
        s_byte, e_byte = _line_to_byte_range_local(content, max(1, s_line), max(1, e_line))

    s_line = _byte_to_line_local(content, s_byte)
    e_line = _byte_to_line_local(content, max(s_byte, e_byte - 1))
    if e_line < s_line:
        e_line = s_line
    return s_line, e_line, s_byte, e_byte


def _is_comment_or_blank(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return True
    return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*") or stripped.startswith("*/")


def _structural_prelude_window(lines: List[str]) -> Tuple[int, int]:
    if not lines:
        return (1, 1)
    total = len(lines)
    max_scan = min(total, 120)
    include_re = re.compile(
        r"^\s*(#\s*include\b|import\b|from\b.+\bimport\b|using\b|package\b|namespace\b|module\b)"
    )
    decl_re = re.compile(
        r"^\s*(#\s*define\b|typedef\b|const\b|static\s+const\b|enum\b|struct\b|class\b|type\b|using\b)"
    )

    saw_include = False
    end = 0
    for idx in range(1, max_scan + 1):
        line = lines[idx - 1]
        if include_re.match(line):
            saw_include = True
            end = idx
            continue
        if not saw_include:
            if _is_comment_or_blank(line):
                continue
            break
        if _is_comment_or_blank(line) or decl_re.match(line):
            end = idx
            continue
        break

    if saw_include and end > 0:
        return (1, min(total, end))
    return (1, min(total, 80))


def _language_needs_neighborhood(focus_file: str) -> bool:
    ext = _ext(focus_file)
    return ext in {".c", ".h", ".cc", ".cpp", ".hpp", ".rs", ".go", ".java", ".ts", ".tsx", ".js", ".jsx"}


def _request_mentions_neighborhood(user_text: str) -> bool:
    low = (user_text or "").lower()
    phrases = (
        "nearby",
        "neighbor",
        "context around",
        "around this",
        "around that",
        "above",
        "below",
        "adjacent",
        "surrounding",
        "depends on nearby",
    )
    return any(p in low for p in phrases)


def _target_refs_nearby_identifiers(
    target_text: str,
    lines: List[str],
    start_line: int,
    end_line: int,
) -> bool:
    if not target_text or not lines:
        return False
    ids = {
        tok
        for tok in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", target_text)
        if len(tok) >= 4
    }
    if not ids:
        return False
    low_ids = {x.lower() for x in ids}
    total = len(lines)
    before_s = max(1, int(start_line) - 20)
    before_e = max(0, int(start_line) - 1)
    after_s = min(total + 1, int(end_line) + 1)
    after_e = min(total, int(end_line) + 20)
    nearby_lines: List[str] = []
    if before_e >= before_s:
        nearby_lines.extend(lines[before_s - 1 : before_e])
    if after_e >= after_s:
        nearby_lines.extend(lines[after_s - 1 : after_e])
    if not nearby_lines:
        return False
    nearby_text = "\n".join(nearby_lines).lower()
    hits = 0
    for ident in low_ids:
        if re.search(rf"\b{re.escape(ident)}\b", nearby_text):
            hits += 1
            if hits >= 2:
                return True
    return False


def _symbol_spans_for_file(
    focus_file: str,
    content: str,
) -> List[Tuple[str, int, int, str]]:
    spans: List[Tuple[str, int, int, str]] = []
    if not content:
        return spans
    try:
        from .tools import extract_symbols_treesitter

        syms = extract_symbols_treesitter(focus_file, content=content) or []
        for sym in syms:
            kind = str(getattr(sym, "kind", "") or "")
            if kind not in {"function", "method", "class"}:
                continue
            name = str(getattr(sym, "name", "") or "").strip()
            if not name:
                continue
            start = int(getattr(sym, "line", 1) or 1)
            end = int(getattr(sym, "end_line", start) or start)
            if end < start:
                end = start
            spans.append((name, start, end, kind))
    except Exception:
        spans = []
    return spans


def _target_function_windows(
    focus_file: str,
    content: str,
    symbol_names: List[str],
) -> Tuple[List[Tuple[int, int, str]], Set[str], List[Tuple[str, int, int, str]]]:
    lines = content.splitlines()
    total = len(lines) or 1
    spans = _symbol_spans_for_file(focus_file, content)
    found: Set[str] = set()
    windows: List[Tuple[int, int, str]] = []
    if not spans or not symbol_names:
        return windows, found, spans
    by_name: Dict[str, List[Tuple[str, int, int, str]]] = {}
    for item in spans:
        by_name.setdefault(item[0].lower(), []).append(item)
    for raw in symbol_names:
        key = str(raw or "").strip().lower()
        if not key:
            continue
        hit = by_name.get(key) or []
        if not hit:
            continue
        name, start, end, _kind = hit[0]
        start = max(1, start - 2)
        end = min(total, end + 2)
        windows.append((start, end, f"function:{name}"))
        found.add(name)
    return windows, found, spans


def _helper_windows_for_calls(
    lines: List[str],
    target_windows: List[Tuple[int, int, str]],
    spans: List[Tuple[str, int, int, str]],
    excluded_symbols: Set[str],
    max_helpers: int = 2,
) -> List[Tuple[int, int, str]]:
    if not lines or not target_windows or not spans:
        return []
    excluded = {s.lower() for s in excluded_symbols}
    keywords = {
        "if", "for", "while", "switch", "return", "sizeof", "printf", "fprintf",
        "fopen", "fread", "fwrite", "fseek", "ftell", "fclose", "malloc", "free",
    }
    calls: List[str] = []
    seen_calls = set()
    call_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for start, end, _anchor in target_windows:
        body = "\n".join(lines[start - 1 : end])
        for m in call_re.finditer(body):
            name = m.group(1).strip()
            key = name.lower()
            if not name or key in seen_calls or key in excluded or key in keywords:
                continue
            seen_calls.add(key)
            calls.append(name)
            if len(calls) >= 20:
                break
        if len(calls) >= 20:
            break
    if not calls:
        return []
    by_name: Dict[str, Tuple[str, int, int, str]] = {}
    for item in spans:
        nm = item[0].lower()
        if nm not in by_name:
            by_name[nm] = item
    out: List[Tuple[int, int, str]] = []
    added = set()
    total = len(lines)
    for call in calls:
        item = by_name.get(call.lower())
        if not item:
            continue
        name, start, end, kind = item
        if kind not in {"function", "method"}:
            continue
        key = name.lower()
        if key in added:
            continue
        added.add(key)
        out.append((max(1, start - 1), min(total, end + 1), f"helper:{name}"))
        if len(out) >= max_helpers:
            break
    return out


def _build_context_pack_text(
    user_text: str,
    candidate_files: List[str],
    diagnostics_text: str = "",
    max_chars: int = 14000,
    required_symbols: Optional[List[str]] = None,
    primary_file: str = "",
) -> str:
    constraints = [
        "Return only unified diff output.",
        "No markdown fences and no prose outside diff.",
        "Keep edits minimal; do not reformat unrelated lines.",
    ]
    parts: List[str] = ["CONTEXT PACK:", f"REQUEST SUMMARY: {(user_text or '').strip()}"]
    parts.append("CONSTRAINTS:")
    for c in constraints:
        parts.append(f"- {c}")

    keywords = _keywords_from_text(user_text or "")
    for focus_file in candidate_files[: max(1, config.PATCH_MAX_FILES_PER_DIFF)]:
        content = read_single_file_for_context(focus_file).get(focus_file, "")
        if content is None:
            content = ""
        lines = content.splitlines()
        required = [str(s or "").strip() for s in (required_symbols or []) if str(s or "").strip()]
        parts.append(f"\nFILE: {focus_file}")
        parts.append(f"FILE_HASH: {_sha256_text(content)}")
        summary = _build_file_summary(
            focus_file,
            content,
            max_preview=0,
            max_defs=24,
            skip_preview=True,
        )
        if summary:
            parts.append(summary)

        # For small files, include full content to stabilize hunk generation.
        if len(lines) <= max(1, int(config.PATCH_FULL_FILE_MAX_LINES)):
            parts.append("FULL FILE:")
            parts.append(content)
            dbg(
                "context_pack.full_file_small "
                f"file={focus_file} lines={len(lines)}"
            )
            continue

        # Stub-focused shaping: include target function bodies + tiny imports/header section.
        if required and (not primary_file or focus_file == primary_file):
            anchored_windows: List[Tuple[int, int, str]] = []
            imp = _imports_window(lines)
            if imp:
                anchored_windows.append((imp[0], imp[1], "imports"))
            fn_windows, found_symbols, all_spans = _target_function_windows(
                focus_file,
                content,
                required,
            )
            anchored_windows.extend(fn_windows)
            helper_windows = _helper_windows_for_calls(
                lines,
                fn_windows,
                all_spans,
                excluded_symbols=found_symbols | set(required),
                max_helpers=2,
            )
            anchored_windows.extend(helper_windows)
            if fn_windows:
                merged = _merge_line_windows(
                    [(s, e) for s, e, _a in anchored_windows],
                    max_lines=8,
                )
                parts.append("SNIPPETS:")
                for start, end in merged:
                    anchor_type = "mixed"
                    for s, e, a in anchored_windows:
                        if s <= end and start <= e:
                            anchor_type = a
                            break
                    snippet = "\n".join(lines[start - 1 : end]) if lines else ""
                    parts.append(f"@@ SNIPPET {anchor_type} {start}-{end}")
                    parts.append(snippet)
                dbg(
                    "context_pack.stub_focus "
                    f"file={focus_file} functions={len(fn_windows)} helpers={len(helper_windows)}"
                )
                continue

        diag_anchors = _parse_diag_lines_for_file(diagnostics_text, focus_file)
        sym_anchors = _symbol_anchor_lines(focus_file, content, keywords)
        search_anchors = _search_anchor_lines(focus_file, keywords)
        anchor_rows: List[Tuple[int, str]] = []
        anchor_rows.extend((ln, "diagnostic") for ln in diag_anchors)
        anchor_rows.extend((ln, "symbol") for ln in sym_anchors)
        anchor_rows.extend((ln, "search") for ln in search_anchors)

        win = max(config.PATCH_CONTEXT_WINDOW_MIN, min(config.PATCH_CONTEXT_WINDOW_MAX, 140))
        half = max(10, win // 2)
        windows: List[Tuple[int, int]] = []
        for line_no, _anchor in anchor_rows[:16]:
            windows.append((max(1, line_no - half), min(len(lines) or 1, line_no + half)))
        imp = _imports_window(lines)
        if imp:
            windows.append(imp)
        if not windows and lines:
            end = min(len(lines), max(40, config.PATCH_CONTEXT_WINDOW_MIN))
            windows.append((1, end))
        merged = _merge_line_windows(windows, max_lines=6)
        parts.append("SNIPPETS:")
        for start, end in merged:
            anchor_type = "mixed"
            for ln, typ in anchor_rows:
                if start <= ln <= end:
                    anchor_type = typ
                    break
            snippet = "\n".join(lines[start - 1 : end]) if lines else ""
            parts.append(f"@@ SNIPPET {anchor_type} {start}-{end}")
            parts.append(snippet)

    text = "\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n...[context pack truncated]"
    return text


def _is_existing_nontiny(content: str) -> bool:
    """Return True if file content is substantial (not new/empty/tiny).
    Uses either char count OR line count — whichever signals "nontiny" first."""
    if not content:
        return False
    stripped = content.strip()
    if len(stripped) >= 200:
        return True
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    return len(lines) >= 10


def _build_file_summary(
    focus_file: str, content: str,
    max_preview: int = 80, max_defs: int = 30,
    skip_preview: bool = False,
) -> str:
    """Create a lightweight summary: defs/classes + optional preview lines."""
    lines = content.splitlines()
    sigs: List[str] = []
    try:
        from .tools import extract_symbols_treesitter

        symbols = extract_symbols_treesitter(focus_file, content=content) or []
        for sym in symbols[:max_defs]:
            line_no = int(getattr(sym, "line", 1))
            kind = str(getattr(sym, "kind", "") or "symbol")
            name = str(getattr(sym, "name", "") or "<unnamed>")
            sigs.append(f"  L{line_no}: {kind} {name}")
    except Exception:
        symbols = []
    if not sigs:
        generic = re.compile(
            r"^\s*(class|def|function|fn|func|interface|type|struct|enum)\b|^\s*(?:export\s+)?(?:const|let|var)\s+[A-Za-z_][A-Za-z0-9_]*\s*=",
            re.IGNORECASE,
        )
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if generic.match(stripped):
                sigs.append(f"  L{i+1}: {stripped.rstrip()}")
    lang = _language_name_for_ext(_ext(focus_file))
    parts = [f"FILE: {focus_file}", f"LANGUAGE: {lang}"]
    total_lines = len(lines)
    parts.append(f"LINES: {total_lines}")
    if sigs:
        parts.append("DEFINITIONS:\n" + "\n".join(sigs[:max_defs]))
    if not skip_preview and max_preview > 0:
        preview = "\n".join(lines[:max_preview])
        if total_lines > max_preview:
            preview += f"\n... ({total_lines - max_preview} more lines)"
        parts.append(f"PREVIEW:\n{preview}")
    return "\n".join(parts)


def _extract_focus_snippets(
    content: str,
    user_text: str,
    focus_file: str = "",
    context_lines: int = 20,
) -> Tuple[str, List[str]]:
    """Extract relevant portions of a file.

    For large files, prefer symbol blocks (tree-sitter/LSP-like) and ripgrep line
    anchors over plain keyword line scoring.
    Returns (snippets_text, list_of_anchor_names)."""
    keywords = _keywords_from_text(user_text)
    if not content:
        return "", []
    lines = content.splitlines()
    if not lines:
        return "", []

    total = len(lines)
    anchors: List[str] = []
    windows: List[Tuple[int, int, str]] = []
    seen_lines: Set[int] = set()

    # 1) Preferred anchors for big files: symbol starts + ripgrep hits.
    if keywords and focus_file:
        symbol_lines = _symbol_anchor_lines(focus_file, content, keywords)
        search_lines = _search_anchor_lines(focus_file, keywords)
        for ln in symbol_lines:
            if ln <= 0 or ln in seen_lines:
                continue
            seen_lines.add(ln)
            windows.append((max(1, ln - context_lines), min(total, ln + context_lines), "symbol"))
            anchors.append(f"symbol@L{ln}")
        for ln in search_lines:
            if ln <= 0 or ln in seen_lines:
                continue
            seen_lines.add(ln)
            windows.append((max(1, ln - context_lines), min(total, ln + context_lines), "rg"))
            anchors.append(f"rg@L{ln}")

    # 2) Expand anchor lines to full symbol/function bodies when possible.
    if windows and focus_file:
        spans = _symbol_spans_for_file(focus_file, content)
        expanded: List[Tuple[int, int, str]] = []
        for s, e, tag in windows:
            matched = False
            for name, ss, ee, kind in spans:
                if ss <= s <= ee or ss <= e <= ee:
                    expanded.append((max(1, ss - 2), min(total, ee + 2), f"{tag}:{kind}:{name}"))
                    matched = True
                    break
            if not matched:
                expanded.append((s, e, tag))
        windows = expanded

    # 3) Fallback when symbol/rg found nothing: keyword scoring windows.
    if not windows and keywords:
        scored_lines: List[Tuple[int, int]] = []  # (score, 1-based line)
        for i, line in enumerate(lines, start=1):
            score = _score_snippet(line, keywords)
            if score > 0:
                scored_lines.append((score, i))
        scored_lines.sort(reverse=True)
        for _score, ln in scored_lines[:6]:
            windows.append((max(1, ln - context_lines), min(total, ln + context_lines), "keyword"))
            anchors.append(f"kw@L{ln}")

    if not windows:
        return "", []

    merged = _merge_line_windows([(s, e) for s, e, _ in windows], max_lines=8)
    snippet_parts: List[str] = []
    for s, e in merged:
        label = "mixed"
        for ws, we, wt in windows:
            if ws <= e and s <= we:
                label = wt
                break
        snippet_parts.append(f"--- {label} L{s}-{e} ---")
        for ln in range(s, e + 1):
            snippet_parts.append(f"{ln}| {lines[ln - 1]}")
    return "\n".join(snippet_parts), anchors[:12]


def extract_relevant_context(
    file_content: str,
    user_text: str,
    max_lines: int = 120,
    numbered: bool = True,
    focus_file: str = "",
) -> str:
    """Language-agnostic context slicer.

    Prioritizes tree-sitter symbol ranges when available, then generic regex blocks.
    """
    if not file_content:
        return ""
    lines = file_content.splitlines()
    if len(lines) <= max_lines:
        if numbered:
            return "\n".join(f"{i+1}| {line}" for i, line in enumerate(lines))
        return "\n".join(lines)

    user_lower = (user_text or "").lower()
    tokens = set(re.findall(r"\b[a-zA-Z_]\w{2,}\b", user_lower))
    ranges: List[Tuple[int, int, int]] = []  # (score, start_1_idx, end_1_idx)

    try:
        from .tools import extract_symbols_treesitter

        if focus_file:
            syms = extract_symbols_treesitter(focus_file, content=file_content)
        else:
            syms = []
        for sym in syms or []:
            name = str(getattr(sym, "name", "") or "").lower()
            start = int(getattr(sym, "line", 1))
            end = int(getattr(sym, "end_line", start))
            if end < start:
                end = start
            score = 1
            if name and name in tokens:
                score += 6
            if name and any(name in t or t in name for t in tokens):
                score += 2
            ranges.append((score, start, end))
    except Exception:
        pass

    if not ranges:
        # Generic regex-based symbol starts across common language families.
        patterns = [
            r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
            r"^\s*class\s+[A-Za-z_][A-Za-z0-9_]*",
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
            r"^\s*(?:export\s+)?(?:const|let|var)\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
            r"^\s*(?:fn|func)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
            r"^\s*(?:public|private|protected|static|final|async|\w[\w:<>\[\]]*\s+)+[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*\{",
            r"^\s*(?:[A-Za-z_][A-Za-z0-9_\s\*:&<>\[\]]+\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*\{",
        ]
        starts: List[int] = []
        for idx, line in enumerate(lines, start=1):
            if any(re.match(p, line) for p in patterns):
                starts.append(idx)
        for i, s in enumerate(starts):
            e = starts[i + 1] - 1 if i + 1 < len(starts) else len(lines)
            block_text = "\n".join(lines[s - 1:e]).lower()
            score = 1 + sum(1 for tok in tokens if tok in block_text)
            ranges.append((score, s, e))

    if not ranges:
        clipped = lines[:max_lines]
        if numbered:
            return "\n".join(f"{i+1}| {line}" for i, line in enumerate(clipped))
        return "\n".join(clipped)

    ranges.sort(key=lambda x: (x[0], x[2] - x[1]), reverse=True)
    selected: List[Tuple[int, int]] = []
    budget = 0
    margin = 6
    for _score, s, e in ranges:
        rs = max(1, s - margin)
        re_ = min(len(lines), e + margin)
        size = re_ - rs + 1
        if budget + size > max_lines and selected:
            continue
        selected.append((rs, re_))
        budget += size
        if budget >= max_lines:
            break
        if len(selected) >= 3:
            break

    selected.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in selected:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    out: List[str] = []
    for s, e in merged:
        if out:
            out.append("...")
        for i in range(s, e + 1):
            out.append(f"{i}| {lines[i - 1]}" if numbered else lines[i - 1])
    return "\n".join(out)


def _build_file_context(
    focus_file: Optional[str],
    full_context: bool,
    user_text: str,
    focus_content_override: Optional[str] = None,
    symbol_focus: bool = False,
    symbol_name: str = "",
    symbol_kind: str = "",
    symbol_line_start: int = 0,
    symbol_line_end: int = 0,
    symbol_byte_start: int = 0,
    symbol_byte_end: int = 0,
) -> Tuple[str, Dict[str, object]]:
    """Build file context. Returns (context_text, context_meta).
    context_meta has: mode_used, context_bytes, snippets_used."""
    sections: List[str] = []
    ctx_meta: Dict[str, object] = {
        "mode_used": "full_file",
        "context_bytes": 0,
        "snippets_used": [],
    }
    if focus_file:
        try:
            if focus_content_override is not None:
                content = focus_content_override
            else:
                content_map = read_single_file_for_context(focus_file)
                content = content_map.get(focus_file, "")
        except Exception:
            content = ""
        ctx_meta["file_has_todos"] = bool(re.search(r"\bTODO\b", content, re.IGNORECASE))
        if content:
            lang = _language_name_for_ext(_ext(focus_file))
            line_count = len(content.splitlines())
            small_file_lines = max(1, int(config.PATCH_TINY_SINGLE_HUNK_MAX_LINES))
            is_small_file = line_count <= small_file_lines
            if symbol_focus:
                byte_count = len((content or "").encode("utf-8"))
                is_tiny_structural = _structural_is_tiny(content)
                if is_tiny_structural:
                    section = content
                    sections.append(section)
                    ctx_meta["mode_used"] = "full_file"
                    ctx_meta["context_bytes"] = len(section)
                    ctx_meta["snippets_used"] = [f"full_file@{focus_file}"]
                    dbg(
                        "context_policy=full reason=structural_tiny_file_full_context "
                        f"lines={line_count} bytes={byte_count}"
                    )
                    return "\n\n".join(sections), ctx_meta
                lines = content.splitlines()
                target_name = str(symbol_name or "").strip()
                target_kind = str(symbol_kind or "").strip()
                start, end, s_byte, e_byte = _resolve_symbol_span(
                    focus_file=focus_file,
                    content=content,
                    symbol_name=target_name,
                    symbol_kind=target_kind,
                    line_start=int(symbol_line_start or 0),
                    line_end=int(symbol_line_end or 0),
                    byte_start=int(symbol_byte_start or 0),
                    byte_end=int(symbol_byte_end or 0),
                )
                target_text = _slice_bytes_local(content, s_byte, e_byte).strip()
                if not target_text:
                    target_text = "\n".join(lines[max(1, start) - 1 : min(line_count, end)]).strip()

                prelude_s, prelude_e = _structural_prelude_window(lines)
                prelude_text = "\n".join(lines[prelude_s - 1 : prelude_e]).strip()

                include_neighborhood = (
                    _language_needs_neighborhood(focus_file)
                    or _request_mentions_neighborhood(user_text)
                    or _target_refs_nearby_identifiers(target_text, lines, start, end)
                )
                neighborhood_text = ""
                if include_neighborhood:
                    before_s = max(1, start - 20)
                    before_e = max(0, start - 1)
                    after_s = min(line_count + 1, end + 1)
                    after_e = min(line_count, end + 20)
                    parts: List[str] = []
                    if before_e >= before_s:
                        parts.append("BEFORE_TARGET:")
                        parts.append("\n".join(lines[before_s - 1 : before_e]).strip())
                    if after_e >= after_s:
                        parts.append("AFTER_TARGET:")
                        parts.append("\n".join(lines[after_s - 1 : after_e]).strip())
                    neighborhood_text = "\n\n".join([p for p in parts if p]).strip()

                target_label = target_name or "<symbol>"
                section = (
                    f"FILE: {focus_file}\n"
                    f"LANGUAGE: {lang}\n"
                    "PRELUDE:\n"
                    f"{prelude_text}\n\n"
                    "TARGET_SYMBOL:\n"
                    f"{target_text}"
                )
                if neighborhood_text:
                    section += f"\n\nNEIGHBORHOOD:\n{neighborhood_text}"
                sections.append(section)
                ctx_meta["mode_used"] = "focused_symbol"
                ctx_meta["context_bytes"] = len(section)
                snippet_labels = [f"prelude@L{prelude_s}-{prelude_e}", f"symbol@L{start}-{end}"]
                if neighborhood_text:
                    snippet_labels.append(f"neighborhood@L{max(1, start-20)}-{min(line_count, end+20)}")
                ctx_meta["snippets_used"] = snippet_labels
                dbg(
                    "context_policy=focused reason=structural_symbol_context_pack "
                    f"lines={line_count} bytes={byte_count} symbol={target_label} "
                    f"neighborhood={1 if bool(neighborhood_text) else 0}"
                )
                # Symbol-focused context is intentionally lean; skip full-file logic.
                return "\n\n".join(sections), ctx_meta
            if full_context or is_small_file or not _is_existing_nontiny(content):
                # Explicit full context, small files, and tiny/new files use full content.
                body = content
                label = "CONTENT"
                section = f"FILE: {focus_file}\nLANGUAGE: {lang}\n{label}:\n{body}"
                sections.append(section)
                ctx_meta["mode_used"] = "full_file"
                ctx_meta["context_bytes"] = len(section)
                if full_context:
                    dbg("context_policy=full reason=explicit_full_context")
                elif is_small_file:
                    dbg(
                        "context_policy=full reason=small_file_full_context "
                        f"lines={line_count}"
                    )
                else:
                    dbg("context_policy=full reason=tiny_or_new_file")
            else:
                # Existing non-tiny file: summary + focused snippets only
                snippets_text, anchors = _extract_focus_snippets(
                    content,
                    user_text,
                    focus_file=focus_file or "",
                )
                # If snippets exist, minimal preview; otherwise 40-line preview
                has_snippets = bool(snippets_text)
                summary = _build_file_summary(
                    focus_file, content,
                    max_preview=30 if has_snippets else 60,
                    max_defs=25,
                    skip_preview=False,
                )
                parts = [summary]
                if snippets_text:
                    parts.append(f"FOCUS SNIPPETS:\n{snippets_text}")
                section = "\n\n".join(parts)
                sections.append(section)
                ctx_meta["mode_used"] = "line_edits"
                ctx_meta["context_bytes"] = len(section)
                ctx_meta["snippets_used"] = anchors
                dbg("context_policy=focused reason=existing_non_tiny")
                dbg(f"prompt_buffer: using summary+snippets for {focus_file} "
                    f"({ctx_meta['context_bytes']} chars, {len(anchors)} anchors)")
    # Include files: only top 2 with score >= 2, smaller snippets
    includes = []
    try:
        includes = list(get_include() or [])
    except Exception:
        includes = []
    include_count = 0
    include_chars = 0
    if includes:
        keywords = _keywords_from_text(user_text)
        scored = []
        for path in includes:
            if path == focus_file:
                continue  # Don't re-include the focus file
            try:
                text = read_file_text(path)
            except Exception:
                continue
            score = _score_snippet(text, keywords)
            if score >= 2:  # Only include if meaningfully relevant
                scored.append((score, path, text))
        scored.sort(reverse=True, key=lambda item: item[0])
        snippets = []
        for _, path, text in scored[:2]:  # Max 2 includes
            snippet = _head_tail(text, 400)  # Smaller snippets
            snippets.append(f"INCLUDE: {path}\n{snippet}")
            include_count += 1
            include_chars += len(snippet)
        if snippets:
            sections.append("\n\n".join(snippets))
    ctx_meta["include_count"] = include_count
    ctx_meta["include_chars"] = include_chars
    return "\n\n".join(sections), ctx_meta


def _build_chat_history(mode: str = "chat") -> str:
    """Build shared chat history for both chat and agent modes."""
    if config.DISABLE_HISTORY:
        dbg("prompt_buffer: history disabled by config")
        return ""

    def _looks_corrupted_output(text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return False
        low = s.lower()
        if re.search(r"(?:^|\n)(?:[a-z]\s*){20,}(?:\n|$)", low):
            return True
        compact = re.sub(r"\s+", "", low)
        if len(compact) >= 40:
            uniq = len(set(compact))
            if uniq <= 3:
                return True
            most_common = max(compact.count(ch) for ch in set(compact))
            if most_common / max(1, len(compact)) >= 0.75:
                return True
        return False

    def _short(s: str, n: int = 220) -> str:
        s = re.sub(r"\s+", " ", (s or "")).strip()
        if len(s) <= n:
            return s
        return s[: n - 3] + "..."

    max_turns = max(1, int(getattr(state, "MAX_CHAT_HISTORY", 7)))
    pairs = state.get_recent_chat(max_turns)
    filtered: List[Tuple[str, str]] = []
    dropped = 0
    for user_text, assistant_text in pairs:
        if _looks_corrupted_output(assistant_text or ""):
            dropped += 1
            continue
        filtered.append((user_text, assistant_text))
    if dropped:
        dbg(f"prompt_buffer: history dropped_corrupted_turns={dropped}")
    pairs = filtered[-max_turns:]
    if not pairs:
        return ""
    lines = []
    for user_text, assistant_text in pairs:
        if user_text:
            lines.append(f"User: {_short(user_text, 220)}")
        if assistant_text:
            # Cap assistant responses in chat history to 200 chars
            trimmed = _short(assistant_text, 200)
            lines.append(f"Assistant: {trimmed}")
    try:
        change_notes = list(state.get_change_notes() or [])
    except Exception:
        change_notes = []
    if change_notes:
        lines.append("Recent code changes:")
        for note in change_notes[-2:]:
            note_text = _short(str(note or ""), 200)
            if note_text:
                lines.append(f"- {note_text}")
    return "\n".join(lines)


def _task_card_required_symbols(task_card_text: str) -> List[str]:
    """Extract required symbol names from TASK_CARD work items."""
    if not task_card_text:
        return []
    out: List[str] = []
    seen = set()
    for line in task_card_text.splitlines():
        m = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", line)
        if not m:
            continue
        name = m.group(1).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out[:10]


def build_prompt(
    mode: str,
    user_text: str,
    focus_file: Optional[str] = None,
    full_context: bool = False,
    force_file_block: bool = False,
    output_contract: str = "",
    structural_target: str = "",
    structural_kind: str = "",
    structural_line_start: int = 0,
    structural_line_end: int = 0,
    structural_byte_start: int = 0,
    structural_byte_end: int = 0,
    structural_original_symbol: str = "",
    structural_signature_line: str = "",
    error_message: str = "",
    pre_sliced_request: Optional[str] = None,
    focus_content_override: Optional[str] = None,
    analysis_packet: str = "",
    plan_contract: str = "",
) -> Tuple[str, Dict[str, object]]:
    """Build a prompt from prioritized sections and budget.
    Returns (prompt_text, context_meta) where context_meta has:
    mode_used, context_bytes, snippets_used."""
    use_diff_contract = mode == "agent" and (output_contract or "").strip().lower() == "diff"
    use_structural_contract = mode == "agent" and (output_contract or "").strip().lower() == "structural"
    sliced_for_context = pre_sliced_request or (user_text or "")
    file_context, ctx_meta = _build_file_context(
        focus_file,
        full_context,
        sliced_for_context or (user_text or ""),
        focus_content_override=focus_content_override,
        symbol_focus=use_structural_contract and bool(focus_file),
        symbol_name=structural_target,
        symbol_kind=structural_kind,
        symbol_line_start=int(structural_line_start or 0),
        symbol_line_end=int(structural_line_end or 0),
        symbol_byte_start=int(structural_byte_start or 0),
        symbol_byte_end=int(structural_byte_end or 0),
    )

    if mode == "agent":
        # Agent mode: keep semantic essentials, trim low-priority context first.
        history = ""
        if not force_file_block and not use_diff_contract and not use_structural_contract:
            history = _build_chat_history("agent")
            if history:
                dbg("prompt_buffer: history included")
            else:
                dbg("prompt_buffer: history skipped")
        req_text = (pre_sliced_request or user_text or "").strip()
        dbg(f"prompt_buffer: request_slice_len={len(req_text)}")
        if not req_text:
            req_text = (user_text or "").strip()
        required_symbols = _task_card_required_symbols(req_text)

        if use_diff_contract and focus_file:
            candidate_files: List[str] = [focus_file]
            try:
                include_files = list(get_include() or [])
            except Exception:
                include_files = []
            for rel in include_files:
                if rel and rel != focus_file:
                    candidate_files.append(rel)
            file_context = _build_context_pack_text(
                req_text,
                candidate_files=candidate_files,
                diagnostics_text=error_message or "",
                max_chars=max(2000, int(config.CTX_CHAR_BUDGET * 0.8)),
                required_symbols=required_symbols,
                primary_file=focus_file,
            )
            ctx_meta["mode_used"] = "context_pack"
            ctx_meta["context_bytes"] = len(file_context)
            ctx_meta["snippets_used"] = candidate_files[: config.PATCH_MAX_FILES_PER_DIFF]
            dbg(
                "context_pack.metrics "
                f"files={len(ctx_meta['snippets_used'])} "
                f"bytes={ctx_meta['context_bytes']}"
            )

        instruction_block = [
            AGENT_SYSTEM_PROMPT,
            "Work from TASK_CARD when present; otherwise work from USER_REQUEST and file context below.",
            "You may reason internally in chat style, but output only the required artifact.",
        ]
        if use_structural_contract:
            instruction_block.append(STRUCTURAL_MINIMALITY_PROMPT)
        if focus_file:
            instruction_block.append(f"Target file: {focus_file}")
        if error_message:
            instruction_block.append(f"Previous error: {error_message}")
        output_rules = []
        structure_block = ""
        if use_diff_contract:
            tiny_single_hunk = False
            if focus_file:
                try:
                    if focus_content_override is not None:
                        _tiny_content = focus_content_override
                    else:
                        _tiny_content = read_single_file_for_context(focus_file).get(focus_file, "")
                    _tiny_lines = len((_tiny_content or "").splitlines())
                    tiny_single_hunk = _tiny_lines < max(
                        1, int(config.PATCH_TINY_SINGLE_HUNK_MAX_LINES)
                    )
                except Exception:
                    tiny_single_hunk = False
            output_rules = [
                "OUTPUT RULES:",
                "- Return ONLY a unified diff patch.",
                "- No markdown fences and no prose/explanations.",
                "- Include --- / +++ headers and valid @@ hunks.",
                "- The diff must apply to the provided CONTEXT PACK.",
                *(
                    [
                        "- Tiny-file policy: this file is under 200 lines; prefer a single hunk.",
                        "- If needed, multiple non-overlapping hunks are allowed for tiny files.",
                    ]
                    if tiny_single_hunk
                    else [
                        "- Use one hunk per function; if editing two functions, output two hunks.",
                        "- Each hunk must include strong unchanged anchors (signature/context lines).",
                        "- Do not span one hunk across two functions.",
                    ]
                ),
                "- Keep edits minimal and do not reformat unrelated lines.",
                "RESPONSE:",
            ]
        elif use_structural_contract and focus_file:
            original_symbol = (structural_original_symbol or "").strip()
            if original_symbol:
                sym_lines = original_symbol.splitlines()
                if len(sym_lines) > 120:
                    original_symbol = "\n".join(sym_lines[:120]).rstrip() + "\n# ... truncated ..."
                symbol_already_in_context = original_symbol in (file_context or "")
                structure_lines = []
                if symbol_already_in_context:
                    structure_lines.append(
                        "ORIGINAL_TARGET_SYMBOL is already present in CONTEXT PACK above; do not duplicate it."
                    )
                else:
                    structure_lines.extend(
                        [
                            "ORIGINAL_TARGET_SYMBOL (verbatim):",
                            original_symbol,
                        ]
                    )
                structure_lines.extend(
                    structural_structure_rules(
                        target_kind=structural_kind,
                        target_name=structural_target,
                        original_signature_line=structural_signature_line,
                        original_symbol_text=original_symbol,
                    )
                )
                structure_block = "\n".join(structure_lines).strip()
            output_rules = [
                *structural_output_skeleton(
                    focus_file=focus_file,
                    target_name=structural_target,
                    target_kind=structural_kind,
                    original_symbol_text=structural_original_symbol,
                    original_signature_line=structural_signature_line,
                ),
                "RESPONSE:",
            ]
        elif force_file_block and focus_file:
            output_rules = [
                "OUTPUT RULES:",
                f"- Return exactly ONE block: [[[file: {focus_file}]]] ... [[[end]]].",
                "- Output the FULL updated file content (not a snippet).",
                "- Modify only the target file; preserve unrelated code/structure.",
                "- Satisfy TASK_CARD work items with minimal edits.",
                "- Keep code buildable/syntactically valid.",
                "- Do not modify main unless explicitly requested.",
                *(
                    [f"- Required symbols to complete in this pass: {', '.join(required_symbols)}."]
                    if required_symbols else []
                ),
                "- Never assume file/stream cursor position; set/restore cursor before reads/writes.",
                "RESPONSE:",
            ]
        else:
            output_rules = ["RESPONSE:"]
        request_label = "TASK_CARD" if _looks_like_task_card(req_text) else "USER_REQUEST"
        mandatory_parts = [
            "\n".join(instruction_block),
            f"{request_label}:\n{req_text}",
        ]
        if structure_block:
            mandatory_parts.append(structure_block)
        mandatory_parts.append("\n".join(output_rules))
        dependency_block = ""
        if analysis_packet:
            try:
                parsed = json.loads(analysis_packet)
            except Exception:
                parsed = {}
            dep_items = parsed.get("dependency_context") if isinstance(parsed, dict) else []
            if isinstance(dep_items, list) and dep_items:
                compact_rows: List[str] = []
                for item in dep_items[:2]:
                    if not isinstance(item, dict):
                        continue
                    f = str(item.get("file") or "").strip()
                    sym = str(item.get("symbol") or "").strip()
                    snip = re.sub(r"\s+", " ", str(item.get("snippet") or "")).strip()[:220]
                    if not snip:
                        continue
                    lead = f"{f}:{sym}" if (f or sym) else "dependency"
                    compact_rows.append(f"- {lead} :: {snip}")
                if compact_rows:
                    dependency_block = "DEPENDENCY CONTEXT:\n" + "\n".join(compact_rows)
        rule_lines = [ln.strip() for ln in AGENT_SYSTEM_PROMPT.splitlines() if ln.strip()]
        if focus_file:
            rule_lines.append(f"Target file: {focus_file}")
        if use_diff_contract:
            rule_lines.append("Output contract: unified diff only.")
        elif use_structural_contract:
            rule_lines.append("Output contract: structural symbol replacement only.")
            if structural_target:
                rule_lines.append(f"Target symbol: {structural_target}.")
        elif force_file_block and focus_file:
            rule_lines.append(f"Return one file block for {focus_file}.")
            rule_lines.append("Output full updated file content, not snippets.")
            if required_symbols:
                rule_lines.append(f"Required symbols: {', '.join(required_symbols)}.")
        if error_message:
            rule_lines.append("Fix previous error.")
        dbg(f"agent_rules_count={len(rule_lines)}")
        dbg_dump("agent_rules_effective", "\n".join(rule_lines))

        optional_parts: List[str] = []
        if dependency_block:
            optional_parts.append(dependency_block)
        if file_context:
            optional_parts.append(file_context)
        if history:
            optional_parts.append(f"Recent commands:\n{history}")

        prompt = "\n\n".join(mandatory_parts[:-1] + optional_parts + [mandatory_parts[-1]])
        budget = config.CTX_CHAR_BUDGET
        if len(prompt) <= budget:
            return prompt, ctx_meta

        # Trim low-priority sections in order:
        # include snippets -> history -> previews/examples -> mandatory only.
        dbg(f"agent_prompt_trim: over_budget initial_len={len(prompt)} budget={budget}")
        context_no_includes = file_context
        if context_no_includes:
            context_no_includes = re.sub(
                r"(?:^|\n)INCLUDE:\s[^\n]+(?:\n.*?)*(?=\nINCLUDE:|\Z)",
                "",
                context_no_includes,
                flags=re.DOTALL,
            ).strip()

        prompt = "\n\n".join(
            mandatory_parts[:-1]
            + ([context_no_includes] if context_no_includes else [])
            + ([f"Recent commands:\n{history}"] if history else [])
            + [mandatory_parts[-1]]
        )
        if len(prompt) <= budget:
            dbg("agent_prompt_trim: removed include snippets")
            return prompt, ctx_meta

        prompt = "\n\n".join(
            mandatory_parts[:-1]
            + ([context_no_includes] if context_no_includes else [])
            + [mandatory_parts[-1]]
        )
        if len(prompt) <= budget:
            dbg("agent_prompt_trim: removed history")
            return prompt, ctx_meta

        compact_context = _head_tail(context_no_includes, 1000) if context_no_includes else ""
        prompt = "\n\n".join(
            mandatory_parts[:-1]
            + ([compact_context] if compact_context else [])
            + [mandatory_parts[-1]]
        )
        if len(prompt) <= budget:
            dbg("agent_prompt_trim: compacted context head+tail")
            return prompt, ctx_meta

        # Last resort: shrink task card text, keep output rules intact.
        req_text = _head_tail(req_text, 700)
        mandatory_parts[1] = f"TASK_CARD:\n{req_text}"
        prompt = "\n\n".join(mandatory_parts[:-1] + [mandatory_parts[-1]])
        if len(prompt) > budget:
            dbg("agent_prompt_trim: hard_cap_applied keep OUTPUT RULES+RESPONSE")
            tail = mandatory_parts[-1]
            room = max(0, budget - len(tail) - 2)
            head = "\n\n".join(mandatory_parts[:-1])[:room]
            prompt = f"{head}\n\n{tail}"
        return prompt, ctx_meta

    # Chat mode: use section-based template.
    # Keep short conversational turns lean to avoid history/context bleed.
    system = (
        "You are a helpful assistant. "
        "Default to concise responses: short answer first, then only essential details. "
        "Avoid long preambles and avoid repeating the user's prompt."
    )
    user_trimmed = (user_text or "").strip()
    low = user_trimmed.lower()
    short_chat = (
        len(user_trimmed) <= 24
        and not re.search(r"[{}()\[\];=]", user_trimmed)
        and low in {"hi", "hey", "hello", "yo", "sup", "hola", "ping", "test"}
    )
    history = "" if short_chat else _build_chat_history(mode)
    context_block = "" if short_chat else file_context
    user_section = f"User: {user_text}\nAssistant:"
    sections = [
        ("SYSTEM:", system),
        ("CONTEXT:", context_block),
        ("HISTORY:", history),
        ("", user_section),
    ]
    prompt = trim_to_budget(sections, config.CTX_CHAR_BUDGET)
    return prompt, ctx_meta
