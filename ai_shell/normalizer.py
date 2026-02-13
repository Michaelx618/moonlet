from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
import re
from typing import List, Optional, Set, Tuple

from .structural import (
    _has_balanced_braces,
    _is_brace_language_file,
    _is_low_entropy_structural_output,
    _is_parent_under_target,
    _kind_matches_target,
    _python_has_indented_body,
    _python_parses_as_module,
    _repair_python_compact_def,
    build_symbol_index,
    slice_by_byte_range,
)


@dataclass
class NormalizationResult:
    text: str
    error_code: str
    confidence: str
    repairs: List[str] = field(default_factory=list)
    stage: str = ""
    used_markers: bool = False
    salvage_mode: str = ""


_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*")
_MARKER_RE = re.compile(r"BEGIN_SYMBOL\s*(.*?)\s*END_SYMBOL", re.DOTALL)


def _fail(
    stage: str,
    code: str,
    used_markers: bool,
    salvage_mode: str,
    repairs: Optional[List[str]] = None,
) -> NormalizationResult:
    return NormalizationResult(
        text="",
        error_code=code,
        confidence="red",
        repairs=list(repairs or []),
        stage=stage,
        used_markers=used_markers,
        salvage_mode=salvage_mode,
    )


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _is_code_like_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if s.startswith(("#", "//", "/*", "*", "--")):
        return True
    if re.search(
        r"\b(def|class|function|return|if|else|for|while|switch|case|try|catch|throw|const|let|var|fn|func|struct|enum|interface|type|import|from|export)\b",
        s,
    ):
        return True
    if any(tok in s for tok in ("{", "}", "(", ")", ";", "=>", "::", "->", "=")):
        return True
    return False


def _is_junk_line_spam(text: str) -> bool:
    lines = [ln.strip() for ln in _normalize_newlines(text).split("\n") if ln.strip()]
    if len(lines) < 8:
        return False
    tiny = [ln for ln in lines if re.fullmatch(r"[A-Za-z0-9_]{1,2}", ln)]
    if not tiny:
        return False
    top = Counter(tiny).most_common(1)[0][1] if tiny else 0
    return (len(tiny) / len(lines) >= 0.60) and (top / max(1, len(lines)) >= 0.45)


def _strip_line_number_prefix(line: str) -> str:
    text = str(line or "")
    text = re.sub(r"^\s*\d+\s*\|\s*", "", text)
    text = re.sub(r"^\s*\d+\s*:\s*", "", text)
    return text


def _cleanup_text(raw_output: str) -> str:
    text = _normalize_newlines(raw_output)
    text = _FENCE_RE.sub("", text)
    text = text.replace("```", "")

    drop_prefixes = (
        "here is",
        "here's",
        "sure",
        "updated",
        "explanation",
        "response",
        "i updated",
        "the updated",
    )
    keep_lines: List[str] = []
    for line in text.split("\n"):
        cleaned_line = _strip_line_number_prefix(line)
        low = cleaned_line.strip().lower()
        if not low:
            keep_lines.append("")
            continue
        if low == "response:" or low.startswith("response:"):
            continue
        if low.startswith(("diff --git", "index ", "--- a/", "+++ b/", "@@")):
            continue
        if low.startswith("[[[file:") or low.startswith("[[[end]]]"):
            continue
        if any(low.startswith(p) and not _is_code_like_line(cleaned_line) for p in drop_prefixes):
            continue
        keep_lines.append(cleaned_line.rstrip())

    cleaned = "\n".join(keep_lines).strip()
    if _is_junk_line_spam(cleaned):
        return ""
    return cleaned


def _best_code_span(text: str) -> Tuple[str, str]:
    src = _normalize_newlines(text)
    lines = src.split("\n")
    best = ""
    best_score = -1

    cur: List[str] = []
    blank_run = 0

    def flush_block(block: List[str]) -> None:
        nonlocal best, best_score
        if not block:
            return
        nonempty = [ln for ln in block if ln.strip()]
        if not nonempty:
            return
        score = sum(3 if _is_code_like_line(ln) else 1 for ln in nonempty)
        score += min(10, len(nonempty))
        candidate = "\n".join(block).strip()
        if score > best_score or (score == best_score and len(candidate) > len(best)):
            best = candidate
            best_score = score

    for line in lines:
        if line.strip():
            cur.append(line)
            blank_run = 0
            continue
        blank_run += 1
        if cur:
            cur.append(line)
        if blank_run >= 2:
            flush_block(cur)
            cur = []
            blank_run = 0

    flush_block(cur)

    if best:
        return best, "best_span"

    fallback = src.strip()
    if fallback:
        return fallback, "full_text"
    return "", "none"


def _extract_python_target_block(text: str, target_name: str, target_kind: str) -> str:
    if not target_name:
        return ""
    lines = _normalize_newlines(text).split("\n")
    if not lines:
        return ""
    want = str(target_kind or "").strip().lower()
    name = re.escape(target_name)
    if want == "class":
        pats = [re.compile(rf"^\s*class\s+{name}\b")]
    else:
        pats = [
            re.compile(rf"^\s*(?:async\s+def|def)\s+{name}\b"),
            re.compile(rf"^\s*class\s+{name}\b"),
        ]

    def _is_top_level_break(ln: str) -> bool:
        return bool(re.match(r"^\s*(?:async\s+def|def|class)\b", ln or ""))

    hits = [
        i for i, line in enumerate(lines)
        if any(p.match(line or "") for p in pats)
    ]
    if len(hits) != 1:
        return ""
    for i in hits:
        line = lines[i]
        indent = len(re.match(r"^[ \t]*", line or "").group(0))
        end = len(lines)
        for j in range(i + 1, len(lines)):
            ln = lines[j]
            if not ln.strip():
                continue
            cur_indent = len(re.match(r"^[ \t]*", ln).group(0))
            if cur_indent <= indent and _is_top_level_break(ln):
                end = j
                break
            if cur_indent <= indent and not ln.startswith((" ", "\t")) and re.match(r"^\S", ln):
                end = j
                break
        block = "\n".join(lines[i:end]).strip()
        if block:
            return block
    return ""


def _extract_balanced_brace_block(text: str, start_idx: int) -> str:
    src = _normalize_newlines(text)
    if start_idx < 0 or start_idx >= len(src):
        return ""
    brace_idx = src.find("{", start_idx)
    if brace_idx < 0:
        tail = src[start_idx:]
        line_end = tail.find("\n")
        if line_end < 0:
            return tail.strip()
        return tail[:line_end].strip()
    depth = 0
    for i in range(brace_idx, len(src)):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start_idx : i + 1].strip()
    return src[start_idx:].strip()


def _extract_brace_lang_target_block(
    text: str,
    target_name: str,
    target_kind: str,
    focus_file: str,
) -> str:
    if not target_name:
        return ""
    src = _normalize_newlines(text)
    name = re.escape(target_name)
    ext = Path(focus_file or "").suffix.lower()
    patterns: List[re.Pattern[str]] = []
    want = str(target_kind or "").strip().lower()
    if ext in {".js", ".jsx", ".ts", ".tsx"}:
        if want == "class":
            patterns.append(re.compile(rf"(?m)^\s*class\s+{name}\b"))
        else:
            patterns.extend(
                [
                    re.compile(rf"(?m)^\s*(?:export\s+)?function\s+{name}\s*\("),
                    re.compile(rf"(?m)^\s*(?:const|let|var)\s+{name}\s*="),
                    re.compile(rf"(?m)^\s*class\s+{name}\b"),
                ]
            )
    else:
        if want == "class":
            patterns.extend(
                [
                    re.compile(rf"(?m)^\s*class\s+{name}\b"),
                    re.compile(rf"(?m)^\s*struct\s+{name}\b"),
                ]
            )
        else:
            patterns.extend(
                [
                    re.compile(rf"(?m)^\s*[\w\s\*\&:<>,\[\]]+\b{name}\s*\("),
                    re.compile(rf"(?m)^\s*func\s+(?:\([^)]+\)\s*)?{name}\s*\("),
                ]
            )
    hits: List[Tuple[int, int]] = []
    for pat in patterns:
        for m in pat.finditer(src):
            hits.append((m.start(), m.end()))
    unique_starts = sorted({s for s, _e in hits})
    if len(unique_starts) != 1:
        return ""
    start_idx = unique_starts[0]
    for pat in patterns:
        m = pat.search(src, pos=start_idx)
        if not m or m.start() != start_idx:
            continue
        block = _extract_balanced_brace_block(src, m.start())
        if block:
            return block
    return ""


def _extract_target_by_language(
    text: str,
    focus_file: str,
    target_name: str,
    target_kind: str,
) -> str:
    ext = Path(focus_file or "").suffix.lower()
    if ext == ".py":
        return _extract_python_target_block(text, target_name, target_kind)
    if _is_brace_language_file(focus_file):
        return _extract_brace_lang_target_block(
            text=text,
            target_name=target_name,
            target_kind=target_kind,
            focus_file=focus_file,
        )
    return ""


def _extract_with_markers(
    cleaned: str,
    focus_file: str,
    target_name: str,
    target_kind: str,
) -> Tuple[str, bool, str, str]:
    text = cleaned or ""
    begin_count = text.count("BEGIN_SYMBOL")
    end_count = text.count("END_SYMBOL")
    if begin_count == 1 and end_count == 1:
        m = _MARKER_RE.search(text)
        if m:
            inner = (m.group(1) or "").strip()
            if inner:
                return inner, True, "markers", ""

    markerless_text = text.replace("BEGIN_SYMBOL", "").replace("END_SYMBOL", "")
    by_lang = _extract_target_by_language(
        markerless_text,
        focus_file=focus_file,
        target_name=target_name,
        target_kind=target_kind,
    )
    if by_lang:
        return by_lang.strip(), False, "language_extract", ""

    candidate, salvage_mode = _best_code_span(markerless_text)
    if not candidate:
        return "", begin_count > 0 or end_count > 0, salvage_mode, "norm_b_no_candidate"
    return candidate.strip(), False, salvage_mode, ""


def _extract_target_unit(
    candidate: str,
    focus_file: str,
    target_name: str,
    target_kind: str,
) -> Tuple[str, str]:
    symbols = build_symbol_index(focus_file, candidate)
    if not symbols:
        return "", "norm_c_parse_failed"

    top_level = [s for s in symbols if not str(s.parent or "").strip()]
    if not top_level:
        return "", "norm_c_parse_failed"

    # Keep the candidate symbol isolated by byte range whenever possible.
    if len(top_level) == 1:
        sym = top_level[0]
        return slice_by_byte_range(candidate, sym.byte_start, sym.byte_end).strip(), ""

    matches = [
        s
        for s in top_level
        if (not target_name or s.name == target_name)
        and _kind_matches_target(target_kind, s.kind)
    ]
    if not matches:
        return "", "norm_c_symbol_not_found"
    if len(matches) > 1:
        return "", "norm_c_ambiguous_target"

    target = matches[0]
    return slice_by_byte_range(candidate, target.byte_start, target.byte_end).strip(), ""


def _python_header_index(lines: List[str]) -> int:
    pat = re.compile(r"^\s*(?:async\s+def|def|class)\s+[A-Za-z_][A-Za-z0-9_]*")
    for idx, line in enumerate(lines):
        if pat.match(line or ""):
            return idx
    return -1


def _normalize_python_indent(symbol_text: str) -> Tuple[str, bool]:
    text = _normalize_newlines(symbol_text).rstrip("\n")
    lines = text.split("\n")
    hidx = _python_header_index(lines)
    if hidx < 0 or hidx >= len(lines) - 1:
        return text, False

    header = lines[hidx]
    m = re.match(r"^([ \t]*)", header)
    header_ws = str(m.group(1) if m else "")
    base_indent = header_ws + "    "

    body_idxs = [i for i in range(hidx + 1, len(lines)) if lines[i].strip()]
    if not body_idxs:
        return text, False

    old_indents = [len(re.match(r"^[ \t]*", lines[i]).group(0)) for i in body_idxs]
    min_indent = min(old_indents) if old_indents else 0
    desired_min = len(base_indent)
    shift = desired_min - min_indent
    if shift == 0 and _python_has_indented_body(text):
        return text, False

    new_lines = list(lines)
    for i in body_idxs:
        stripped = lines[i].lstrip(" \t")
        ws = re.match(r"^[ \t]*", lines[i]).group(0)
        cur = len(ws)
        new_indent_len = max(desired_min, cur + shift)
        new_lines[i] = (" " * new_indent_len) + stripped

    out = "\n".join(new_lines)
    if _python_parses_as_module(out):
        return out, out != text
    return text, False


def _brace_depth_never_negative(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return True


def _repair_candidate(candidate: str, focus_file: str) -> Tuple[str, List[str], str]:
    text = _normalize_newlines(candidate).strip("\n")
    repairs: List[str] = []
    ext = Path(focus_file or "").suffix.lower()

    if ext == ".py":
        compact = _repair_python_compact_def(text)
        if compact != text:
            text = compact
            repairs.append("python_expand_one_liner")

        reindented, changed = _normalize_python_indent(text)
        if changed:
            text = reindented
            repairs.append("python_reindent_body")

    if _is_brace_language_file(focus_file):
        if not _has_balanced_braces(text):
            opens = text.count("{")
            closes = text.count("}")
            gap = opens - closes
            if gap > 0 and gap <= 3 and _brace_depth_never_negative(text):
                text = text.rstrip() + ("\n" + ("}\n" * gap))
                repairs.append(f"append_{gap}_missing_closing_brace")
            elif gap < 0:
                # Tolerate extra trailing closing braces introduced by formatting noise.
                trimmed = text.rstrip()
                removed = 0
                while removed < min(3, -gap) and trimmed.endswith("}"):
                    trimmed = trimmed[:-1].rstrip()
                    removed += 1
                if removed > 0 and _has_balanced_braces(trimmed):
                    text = trimmed + "\n"
                    repairs.append(f"remove_{removed}_extra_closing_brace")
                else:
                    return text, repairs, "norm_d_risky_repair_required"
            else:
                return text, repairs, "norm_d_risky_repair_required"

    return text, repairs, ""


def _final_scope_check(
    text: str,
    focus_file: str,
    target_name: str,
    target_kind: str,
) -> Tuple[str, str]:
    symbols = build_symbol_index(focus_file, text)
    if not symbols:
        return "", "norm_e_reparse_failed"

    top_level = [s for s in symbols if not str(s.parent or "").strip()]
    if len(top_level) != 1:
        return "", "norm_e_scope_violation"

    top = top_level[0]
    if target_name and top.name != target_name:
        return "", "norm_e_scope_violation"
    if target_kind and not _kind_matches_target(target_kind, top.kind):
        return "", "norm_e_scope_violation"

    if str(target_kind or "").strip().lower() == "class":
        for sym in symbols:
            if sym is top:
                continue
            if not _is_parent_under_target(sym.parent or "", top.name):
                return "", "norm_e_scope_violation"
    elif len(symbols) != 1:
        return "", "norm_e_scope_violation"

    isolated = slice_by_byte_range(text, top.byte_start, top.byte_end).strip()
    if not isolated:
        return "", "norm_e_reparse_failed"
    if not isolated.endswith("\n"):
        isolated += "\n"
    return isolated, ""


def normalize_symbol(
    raw_output: str,
    focus_file: str,
    target_name: str,
    target_kind: str,
    original_symbol_text: str,
    request_text: str,
) -> NormalizationResult:
    del original_symbol_text
    del request_text

    cleaned = _cleanup_text(raw_output)
    if not cleaned.strip():
        return _fail("A", "norm_a_cleanup_empty", used_markers=False, salvage_mode="none")
    if _is_low_entropy_structural_output(cleaned):
        return _fail("A", "norm_a_cleanup_empty", used_markers=False, salvage_mode="none")

    candidate, used_markers, salvage_mode, b_err = _extract_with_markers(
        cleaned=cleaned,
        focus_file=focus_file,
        target_name=target_name,
        target_kind=target_kind,
    )
    if b_err:
        return _fail("B", b_err, used_markers=used_markers, salvage_mode=salvage_mode)

    extracted, c_err = _extract_target_unit(
        candidate,
        focus_file,
        target_name,
        target_kind,
    )

    working = candidate
    if not c_err:
        working = extracted
    elif c_err != "norm_c_parse_failed":
        return _fail("C", c_err, used_markers=used_markers, salvage_mode=salvage_mode)

    repaired, repairs, d_err = _repair_candidate(working, focus_file)
    if d_err:
        return _fail(
            "D",
            d_err,
            used_markers=used_markers,
            salvage_mode=salvage_mode,
            repairs=repairs,
        )

    final_text, e_err = _final_scope_check(
        repaired,
        focus_file,
        target_name,
        target_kind,
    )
    if e_err:
        return _fail(
            "E",
            e_err,
            used_markers=used_markers,
            salvage_mode=salvage_mode,
            repairs=repairs,
        )

    confidence = "yellow" if repairs else "green"
    return NormalizationResult(
        text=final_text,
        error_code="",
        confidence=confidence,
        repairs=repairs,
        stage="E",
        used_markers=used_markers,
        salvage_mode=salvage_mode,
    )
