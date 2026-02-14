import hashlib
import re
import json
from typing import Dict, List, Optional, Set, Tuple

from . import config
from . import state
from .files import get_include, get_root, read_file_text, read_single_file_for_context
from .relevance import find_relevant_files
from .prompts import (
    _ext,
    _language_name_for_ext,
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


def build_freedom_context(
    user_text: str,
    candidate_files: List[str],
    max_chars: int = 14000,
) -> str:
    """Build multi-file context for freedom mode. Uses tree-sitter + grep for snippets.
    Model decides which file(s) to edit; no single-file bias."""
    if not candidate_files:
        return ""
    parts: List[str] = [
        "FILES (edit any or all as needed):",
        f"REQUEST: {(user_text or '').strip()}",
    ]
    keywords = _keywords_from_text(user_text or "")
    max_files = min(len(candidate_files), config.PATCH_MAX_FILES_PER_DIFF)
    for focus_file in candidate_files[:max_files]:
        content = read_single_file_for_context(focus_file).get(focus_file, "")
        if content is None:
            content = ""
        lines = content.splitlines()
        parts.append(f"\nFILE: {focus_file}")
        summary = _build_file_summary(
            focus_file,
            content,
            max_preview=0,
            max_defs=24,
            skip_preview=True,
        )
        if summary:
            parts.append(summary)

        if len(lines) <= max(1, int(config.PATCH_FULL_FILE_MAX_LINES)):
            parts.append("FULL FILE:")
            parts.append(content)
            dbg(f"freedom_context: full file {focus_file} lines={len(lines)}")
            continue

        sym_anchors = _symbol_anchor_lines(focus_file, content, keywords)
        search_anchors = _search_anchor_lines(focus_file, keywords)
        anchor_rows: List[Tuple[int, str]] = []
        anchor_rows.extend((ln, "symbol") for ln in sym_anchors)
        anchor_rows.extend((ln, "search") for ln in search_anchors)

        win = max(config.PATCH_CONTEXT_WINDOW_MIN, min(config.PATCH_CONTEXT_WINDOW_MAX, 140))
        half = max(10, win // 2)
        windows: List[Tuple[int, int]] = []
        for line_no, _ in anchor_rows[:16]:
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
            snippet = "\n".join(lines[start - 1 : end]) if lines else ""
            parts.append(f"@@ {start}-{end}")
            parts.append(snippet)

    text = "\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n...[truncated]"
    dbg(f"freedom_context: {max_files} files, {len(text)} chars")
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

    # Per-turn char caps for history (no hard cap; these control prompt size)
    HISTORY_USER_CHARS = 450
    HISTORY_ASSISTANT_CHARS = 400
    HISTORY_NOTE_CHARS = 350

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
            lines.append(f"User: {_short(user_text, HISTORY_USER_CHARS)}")
        if assistant_text:
            trimmed = _short(assistant_text, HISTORY_ASSISTANT_CHARS)
            lines.append(f"Assistant: {trimmed}")
    try:
        change_notes = list(state.get_change_notes() or [])
    except Exception:
        change_notes = []
    if change_notes:
        lines.append("Recent code changes:")
        for note in change_notes[-2:]:
            note_text = _short(str(note or ""), HISTORY_NOTE_CHARS)
            if note_text:
                lines.append(f"- {note_text}")
    return "\n".join(lines)


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
    context_override: str = "",
    analysis_packet: str = "",
    plan_contract: str = "",
) -> Tuple[str, Dict[str, object]]:
    """Build a prompt from prioritized sections and budget.
    Returns (prompt_text, context_meta) where context_meta has:
    mode_used, context_bytes, snippets_used."""
    sliced_for_context = pre_sliced_request or (user_text or "")
    if context_override:
        file_context = str(context_override or "").strip()
        ctx_meta = {
            "mode_used": "packed_structural_context",
            "context_bytes": len(file_context),
            "snippets_used": ["packed_context_override"],
            "include_count": 0,
            "include_chars": 0,
        }
        dbg(
            "context_policy=packed reason=structural_packed_context_override "
            f"bytes={ctx_meta['context_bytes']}"
        )
    else:
        file_context, ctx_meta = _build_file_context(
            focus_file,
            full_context,
            sliced_for_context or (user_text or ""),
            focus_content_override=focus_content_override,
        )

    # Chat mode: use section-based template.
    # When user asks to read a file but no focus_file, discover it for context.
    effective_focus = focus_file
    if not effective_focus and user_text:
        low = (user_text or "").lower()
        read_intent = bool(
            re.search(r"\b(?:read|show|open|display|what'?s?\s+in)\s+(?:the\s+)?", low)
            or re.search(r"\b(?:makefile|readme|dockerfile)\b", low)
        )
        if read_intent:
            discovered = find_relevant_files(user_text, open_file=None)
            if discovered:
                effective_focus = discovered[0]
                dbg(f"chat: discovered focus_file for read intent: {effective_focus}")
    if effective_focus and not file_context:
        file_context, ctx_meta = _build_file_context(
            effective_focus,
            full_context=False,
            user_text=sliced_for_context or (user_text or ""),
            focus_content_override=focus_content_override,
        )
    # Keep short conversational turns lean to avoid history/context bleed.
    if mode == "chat_for_agent":
        system = (
            "You are a code-editing assistant.Do what the user says and nothing more "
            "When the user asks for code: give code only with minimal explanation. "
            "You may edit one or more files; indicate which file each block goes to."
        )
    else:
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
