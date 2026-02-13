"""Pre/post-apply validation, safety checks, and revert logic.

Extracted from agent.py to keep modules focused.
"""

import json
import re
import shlex
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from .files import (
    _norm_rel_path,
    get_root,
    read_single_file_for_context,
    resolve_path,
)
from .intents import has_stub_placeholders
from .utils import dbg
from . import config


# ---------- Replacement key lookup ----------

_REPL_KEYS = ("replacement", "text", "new_text", "content", "code")


def _get_replacement(edit: dict) -> str:
    """Extract replacement text from an edit dict, checking known key names."""
    for k in _REPL_KEYS:
        val = edit.get(k)
        if val:
            return str(val)
    return ""


# ---------- TODO / placeholder detection ----------


def has_todo_markers(content: str) -> bool:
    if not content:
        return False
    return bool(re.search(r"\bTODO\b", content, re.IGNORECASE))


def has_placeholder_conditionals(content: str, target_ext: str) -> bool:
    if not content:
        return False
    if target_ext not in {"c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx"}:
        return False
    patterns = (
        r"if\s*\(\s*/\*.*?TODO",
        r"for\s*\(\s*/\*.*?TODO",
        r"while\s*\(\s*/\*.*?TODO",
        r"return\s*/\*.*?TODO",
        r"=\s*/\*.*?TODO",
        r"\(\s*/\*.*?TODO",
    )
    return any(re.search(p, content, re.IGNORECASE | re.DOTALL) for p in patterns)


def has_todo_markers_in_lines(lines: List[str]) -> bool:
    if not lines:
        return False
    for line in lines:
        if re.search(r"\bTODO\b", line, re.IGNORECASE):
            return True
    return False


def validate_stub_completion(
    original_content: str,
    new_content: str,
    focus_file: str = "",
) -> Optional[str]:
    """Language-agnostic placeholder reduction check."""
    if not has_stub_placeholders(original_content):
        return None
    # Keep this conservative: track only explicit TODO/FIXME/TBD markers.
    old_count = len(re.findall(r"\b(TODO|FIXME|TBD)\b", original_content, re.IGNORECASE))
    new_count = len(re.findall(r"\b(TODO|FIXME|TBD)\b", new_content, re.IGNORECASE))
    if old_count == 0:
        return None
    if new_count >= old_count:
        return "placeholders remain"
    return None


def _extract_anchor_symbols(content: str, target_ext: str = "") -> List[str]:
    if not content:
        return []
    names: List[str] = []
    seen = set()

    def _add(name: str) -> None:
        n = (name or "").strip()
        if not n or n in seen:
            return
        seen.add(n)
        names.append(n)

    try:
        from .tools import extract_symbols_treesitter

        pseudo_path = f"anchor.{target_ext}" if target_ext else "anchor.tmp"
        syms = extract_symbols_treesitter(pseudo_path, content=content) or []
        for sym in syms:
            kind = str(getattr(sym, "kind", "") or "")
            if kind in {"function", "method", "class"}:
                _add(str(getattr(sym, "name", "") or ""))
    except Exception:
        pass

    if not names:
        for m in re.finditer(r"(?m)^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", content):
            _add(m.group(1))
        for m in re.finditer(r"(?m)^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", content):
            _add(m.group(1))
        for m in re.finditer(
            r"(?m)^\s*(?:[A-Za-z_][A-Za-z0-9_\s\*\[\]<>:&,]*?\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{",
            content,
        ):
            name = m.group(1)
            if name not in {"if", "for", "while", "switch", "catch"}:
                _add(name)

    if target_ext in {"c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx"} and re.search(
        r"^\s*int\s+main\s*\(",
        content,
        re.MULTILINE,
    ):
        _add("main")

    return names[:40]


# ---------- Pre-apply validation ----------


def _validate_edits_pre_apply(
    edits: List[Dict[str, object]],
    focus_file: str,
    existing_content: str,
    user_text: str,
) -> Optional[str]:
    """Pre-apply safety check. Returns error string if edits should be rejected, None if OK."""
    if not edits:
        return "EMPTY_EDITS"

    text_lower = (user_text or "").lower()
    is_add = any(k in text_lower for k in ("add ", "insert ", "append ", "extend "))
    has_substantial_edit = False

    for edit in edits:
        replacement = _get_replacement(edit)

        # Reject if replacement contains prompt markers
        prompt_markers = ("FILE CONTEXT:", "USER REQUEST:", "RESPONSE:", "RULES:",
                          "Return ONLY", "INCLUDE:", "LANGUAGE:")
        if any(m in replacement for m in prompt_markers):
            return "PROMPT_MARKERS_IN_REPLACEMENT"

        # For "add" requests, reject if replacement duplicates existing classes
        if is_add and existing_content:
            existing_classes = set(re.findall(r"^class\s+(\w+)", existing_content, re.MULTILINE))
            repl_classes = set(re.findall(r"^class\s+(\w+)", replacement, re.MULTILINE))
            overlap = existing_classes & repl_classes
            if overlap:
                return f"DUPLICATION: class {', '.join(overlap)} already exists"

        if len(replacement.strip()) >= 15:
            has_substantial_edit = True

    # For "add" requests, at least one edit must have substantial code
    if is_add and not has_substantial_edit:
        return "REPLACEMENT_TOO_SHORT"

    return None


def _validate_post_apply(
    focus_file: str,
    original_content: str,
    target_ext: str,
    user_text: str = "",
    strict: bool = True,
    analysis_policy: Optional[dict] = None,
    enforce_anchor_guard: bool = False,
) -> Optional[str]:
    """Post-apply safety check. Returns error string if file should be reverted, None if OK."""
    new_content = read_single_file_for_context(focus_file).get(focus_file, "")
    if not new_content:
        return None  # File was deleted? Shouldn't happen.

    stub_task = has_stub_placeholders(original_content) and any(
        k in (user_text or "").lower()
        for k in ("implement", "complete", "fill", "todo", "stub", "placeholder")
    )

    # Check for suspicious file shrinkage (very conservative, opt-in safety).
    old_len = len(original_content.strip())
    new_len = len(new_content.strip())
    text_lower = (user_text or "").lower()
    rewrite_like = any(
        k in (user_text or "").lower()
        for k in ("rewrite", "refactor", "cleanup", "clean up", "simplify")
    )
    impl_like = any(
        k in text_lower
        for k in ("implement", "complete", "fill", "todo", "stub", "placeholder")
    )
    if (not stub_task) and old_len > 200 and new_len < old_len * 0.4 and not rewrite_like and not impl_like:
        return f"FILE_SHRUNK: {old_len} -> {new_len} chars"

    if enforce_anchor_guard and old_len > 100:
        allow_removal = any(
            k in text_lower for k in ("remove", "delete", "drop", "rewrite", "refactor", "rename")
        )
        if not allow_removal:
            old_anchors = _extract_anchor_symbols(original_content, target_ext=target_ext)
            if old_anchors:
                new_anchors = set(_extract_anchor_symbols(new_content, target_ext=target_ext))
                missing = [name for name in old_anchors if name not in new_anchors]
                if missing:
                    return f"ANCHOR_LOSS: {', '.join(missing[:5])}"

    stub_err = validate_stub_completion(original_content, new_content, focus_file=focus_file)
    if stub_err:
        return f"STUB_FAIL: {stub_err}"

    # Language-specific syntax checks
    if (not config.SKIP_COMPILE_CHECKS) and target_ext in {"py", "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx"}:
        preferred_cmd = ""
        if isinstance(analysis_policy, dict):
            preferred_cmd = str(analysis_policy.get("suggested_validate_cmd") or "").strip()
        ok, err = run_basic_checks([focus_file], strict=strict, preferred_cmd=preferred_cmd or None)
        if not ok:
            return f"SYNTAX_FAIL: {err[:500]}"

    sem_err = _validate_prompt_semantics(user_text or "", new_content)
    if sem_err:
        return f"SEMANTIC_FAIL: {sem_err}"

    return None


def _validate_prompt_semantics(user_text: str, new_content: str) -> Optional[str]:
    """Enforce high-signal prompt semantics when they are explicitly requested."""
    low = (user_text or "").lower()
    code_low = (new_content or "").lower()
    if not low or not code_low:
        return None

    missing: List[str] = []

    if "output.bmp" in low and "output.bmp" not in code_low:
        missing.append("must write result to output.bmp")

    preserve_req = any(
        s in low
        for s in (
            "original file is not changed",
            "keep input file unchanged",
            "do not mutate original",
            "input file unchanged",
        )
    )
    if preserve_req:
        if not re.search(r'fopen\s*\(\s*"output\.bmp"\s*,\s*"wb"', code_low):
            missing.append("must open output.bmp for writing")
        if re.search(r'fopen\s*\(\s*argv\s*\[\s*1\s*\]\s*,\s*"wb"', code_low):
            missing.append("must not overwrite input file")

    metadata_req = all(
        s in low
        for s in ("file size", "image width", "image height", "color depth", "size of image data")
    )
    if metadata_req:
        signals = {
            "file size": ("file size", "bitmap file size"),
            "image width": ("image width", "width in pixels"),
            "image height": ("image height", "height in pixels"),
            "color depth": ("color depth", "bits per pixel", "bit count"),
            "size of image data": ("size of image data", "image data in bytes"),
        }
        for label, variants in signals.items():
            if not any(v in code_low for v in variants):
                missing.append(f"must print metadata field: {label}")

    if ("invert" in low and "pixel" in low) or ("255" in low and "invert" in low):
        if "255 -" not in code_low and "255-" not in code_low:
            missing.append("must invert with 255 - channel_value")

    if "bfoffbits" in low and "bfoffbits" not in code_low:
        missing.append("must use bfOffBits to locate image data offset")

    # When the request is a TASK_CARD, enforce explicit semantic bullets.
    semantic_lines: List[str] = []
    for raw in (user_text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("-"):
            s = s[1:].strip()
        m = re.match(r"^(?:Satisfy semantic requirement|Semantic)\s*:\s*(.+)$", s, re.IGNORECASE)
        if not m:
            continue
        req = m.group(1).strip().rstrip(".")
        if req:
            semantic_lines.append(req.lower())

    if semantic_lines:
        # Keep checks heuristic but stricter than generic prompt-only matching.
        for req in semantic_lines[:6]:
            if "output.bmp" in req:
                if "output.bmp" not in code_low:
                    missing.append("task-card semantic unmet: output.bmp handling")
            if "keep input file unchanged" in req or "preserve source" in req:
                if not re.search(r'fopen\s*\(\s*"output\.bmp"\s*,\s*"wb"', code_low):
                    missing.append("task-card semantic unmet: write transformed data to output file")
                if re.search(r'fopen\s*\(\s*argv\s*\[\s*1\s*\]\s*,\s*"wb"', code_low):
                    missing.append("task-card semantic unmet: input file overwrite detected")
            if "format constraints" in req or "bit depth/compression" in req:
                format_hits = 0
                for sig in (
                    "bibitcount",
                    "bicompression",
                    "bit count",
                    "bits per pixel",
                    "header[28]",
                    "header[30]",
                    "bitmapinfoheader",
                ):
                    if sig in code_low:
                        format_hits += 1
                if format_hits < 1:
                    missing.append("task-card semantic unmet: format constraints not reflected in code")
            if "offset" in req and "image data" in req:
                if "bfoffbits" not in code_low and "header[10]" not in code_low:
                    missing.append("task-card semantic unmet: image data offset handling")
            if "row alignment" in req or "padding" in req:
                if "padding" not in code_low and "row_size" not in code_low and "multiple of 4" not in code_low:
                    missing.append("task-card semantic unmet: row padding/alignment handling")

    if not missing:
        return None
    return "; ".join(missing[:6])


def _validate_diff_pre_apply(
    hunks: list,
    existing_content: str,
    user_text: str,
    intent: str = "",
    focus_file: str = "",
) -> Optional[str]:
    """Pre-apply safety check for unified diff hunks.
    Returns error string if diff should be rejected, None if OK."""
    if not hunks:
        return "EMPTY_DIFF"

    # Only reject on markers that indicate real prompt leakage,
    # NOT our own boundary markers (BEGIN FILE / END FILE)
    prompt_markers = (
        "FILE CONTEXT:", "USER REQUEST:", "RESPONSE:", "RULES:",
        "INCLUDE:", "LANGUAGE:", "SYSTEM/RULES:", "PROJECT MEMORY:",
        "CONTEXT BEFORE:", "CONTEXT AFTER:", "Insert new code into class",
    )

    for hunk in hunks:
        for prefix, content in hunk.lines:
            # Reject if any addition line contains prompt markers
            if prefix == "+" and any(m in content for m in prompt_markers):
                return "PROMPT_MARKERS_IN_DIFF"

    # At least one hunk should add something meaningful
    total_additions = sum(
        1 for h in hunks for p, _ in h.lines if p == "+"
    )
    total_deletions = sum(
        1 for h in hunks for p, _ in h.lines if p == "-"
    )
    if total_additions == 0:
        return "NO_ADDITIONS"
    # Reject diffs that only restate identical +/- lines and produce no net change.
    has_net_delta = False
    for h in hunks:
        adds = [c for p, c in h.lines if p == "+"]
        dels = [c for p, c in h.lines if p == "-"]
        if not adds and not dels:
            continue
        if len(adds) != len(dels):
            has_net_delta = True
            break
        if any(a != d for a, d in zip(adds, dels)):
            has_net_delta = True
            break
    if not has_net_delta:
        return "NO_EFFECT_CHANGES"

    wants_stub_replacement = any(
        k in (user_text or "").lower()
        for k in ("implement", "complete", "fill", "replace", "todo", "placeholder", "stub")
    )
    if intent == "MODIFY_EXISTING" and wants_stub_replacement and has_todo_markers(existing_content):
        deleted_lines = [c for h in hunks for p, c in h.lines if p == "-"]
        if not any(re.search(r"\bTODO\b|placeholder|stub", ln, re.IGNORECASE) for ln in deleted_lines):
            return "NO_TODO_REMOVAL"
    return None


def looks_like_wrong_language(content: str, target_ext: str) -> bool:
    s = (content or "").lstrip()
    if not s:
        return False
    if target_ext == "py":
        java_signals = (
            "public class ",
            "System.out.println",
            "import java.",
            "package ",
            "public static void main",
            "private static ",
            "protected ",
            "#include",
            "using namespace",
            "std::",
        )
        if any(sig in s for sig in java_signals):
            return True
        if re.match(r"^\s*public\s+class\s+\w+", s):
            return True
    return False


def parse_plan(output: str) -> Optional[dict]:
    """Parse a JSON plan from model output.

    Expects {"files": [{"path": ..., "action": ..., "description": ...}, ...]}
    Robust to garbage before/after the JSON.
    """
    raw = (output or "").strip()
    if not raw:
        return None

    # Remove optional markdown fences when the whole payload is fenced.
    raw_before = raw
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    if raw != raw_before:
        dbg("parse_plan: normalization=fences_stripped")

    def _normalize(plan_obj: object) -> Optional[dict]:
        if not isinstance(plan_obj, dict):
            dbg("parse_plan: reason=TOP_LEVEL_NOT_OBJECT")
            return None
        files = plan_obj.get("files")
        # Looser compatibility: accept legacy single-file planner shape.
        if not isinstance(files, list):
            legacy_name = plan_obj.get("name")
            if isinstance(legacy_name, str) and legacy_name.strip():
                edits = plan_obj.get("edits")
                description = "Apply planned edits"
                if isinstance(edits, list) and edits:
                    description = f"Apply {len(edits)} planned edit block(s)"
                dbg("parse_plan: accepted legacy top-level name/edits shape")
                return {
                    "files": [{
                        "path": legacy_name.strip(),
                        "action": "edit",
                        "target": "",
                        "description": description,
                    }]
                }
            dbg("parse_plan: reason=MISSING_FILES_LIST")
            return None
        valid = []
        for entry in files:
            if not isinstance(entry, dict):
                dbg("parse_plan: reason=ENTRY_NOT_OBJECT")
                return None
            path = entry.get("path")
            if not path and isinstance(entry.get("name"), str):
                path = entry.get("name")
            if not isinstance(path, str) or not path.strip():
                dbg("parse_plan: reason=ENTRY_PATH_MISSING")
                return None
            description = entry.get("description", "")
            if not isinstance(description, str) or not description.strip():
                ops = entry.get("operations")
                edits = entry.get("edits")
                if isinstance(ops, list) and ops:
                    description = f"Apply {len(ops)} operation(s)"
                elif isinstance(edits, list) and edits:
                    description = f"Apply {len(edits)} edit block(s)"
                else:
                    description = "Edit file per request"
            valid.append({
                "path": path.strip(),
                "action": entry.get("action", "edit"),
                "target": entry.get("target", ""),
                "description": description,
            })
        # Merge duplicate paths into a single actionable entry.
        merged: Dict[str, dict] = {}
        for e in valid:
            p = str(e.get("path") or "")
            if p not in merged:
                merged[p] = dict(e)
                continue
            prev_desc = str(merged[p].get("description") or "").strip()
            next_desc = str(e.get("description") or "").strip()
            if next_desc and next_desc not in prev_desc:
                merged[p]["description"] = f"{prev_desc}; {next_desc}" if prev_desc else next_desc
            if str(e.get("action") or "").strip().lower() == "delete":
                merged[p]["action"] = "delete"
            elif str(e.get("action") or "").strip().lower() == "create" and merged[p].get("action") != "delete":
                merged[p]["action"] = "create"
        if len(merged) < len(valid):
            dbg(f"parse_plan: normalization=duplicates_merged count={len(valid)-len(merged)}")
        valid = list(merged.values())
        if valid:
            dbg(f"parse_plan: parsed {len(valid)} file entries")
            return {"files": valid}
        dbg("parse_plan: reason=NO_VALID_ENTRIES")
        return None

    # 1) Direct parse first.
    try:
        parsed = json.loads(raw)
        norm = _normalize(parsed)
        if norm:
            return norm
    except Exception:
        pass

    # 1b) Fenced JSON extraction when extra prose surrounds the block.
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", output or "", re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
        try:
            parsed = json.loads(candidate)
            norm = _normalize(parsed)
            if norm:
                return norm
        except Exception:
            pass

    # 2) Extract first balanced JSON object and parse it.
    first = raw.find("{")
    if first >= 0:
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(raw[first:], start=first):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = raw[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        norm = _normalize(parsed)
                        if norm:
                            return norm
                    except Exception:
                        continue

    # 3) Strict mode: do not salvage path-only garbage.
    dbg("parse_plan: failed to parse plan")
    return None


def _is_whole_file_rewrite(hunks: list, existing_content: str) -> bool:
    """Detect diff hunks that effectively rewrite the whole file."""
    if not hunks or not existing_content:
        return False
    file_lines = len(existing_content.splitlines())
    if file_lines == 0:
        return False
    total_additions = sum(1 for h in hunks for p, _ in h.lines if p == "+")
    total_deletions = sum(1 for h in hunks for p, _ in h.lines if p == "-")
    if total_additions >= int(file_lines * 0.8) and total_deletions >= int(file_lines * 0.8):
        return True
    if len(hunks) == 1:
        h = hunks[0]
        if h.old_start == 1 and h.old_count >= max(1, file_lines - 2) and total_deletions > 0:
            return True
    return False


def _diff_within_ranges(hunks: list, ranges: List[Tuple[int, int]]) -> bool:
    """Return True if all hunks overlap with the provided line ranges."""
    if not ranges:
        return True
    for h in hunks:
        if h.old_count > 0:
            h_start = h.old_start
            h_end = h.old_start + h.old_count - 1
        else:
            h_start = h.old_start
            h_end = h.old_start
        if not any(r_start <= h_end and h_start <= r_end for r_start, r_end in ranges):
            return False
    return True


def _has_overlapping_hunks(hunks: list) -> bool:
    """Detect overlapping old-file regions across hunks."""
    if not hunks or len(hunks) <= 1:
        return False
    spans: List[Tuple[int, int]] = []
    for h in hunks:
        start = h.old_start
        count = h.old_count if h.old_count > 0 else 1
        end = start + count - 1
        spans.append((start, end))
    spans.sort(key=lambda s: (s[0], s[1]))
    prev_s, prev_e = spans[0]
    for s, e in spans[1:]:
        if s <= prev_e:
            return True
        prev_s, prev_e = s, e
    return False


def _hunk_span(hunk: object) -> Tuple[int, int]:
    start = int(getattr(hunk, "old_start", 0) or 0)
    if start <= 0:
        start = 1
    count = int(getattr(hunk, "old_count", 0) or 0)
    span = count if count > 0 else 1
    end = start + span - 1
    return start, end


def _hunk_delta_size(hunk: object) -> int:
    lines = list(getattr(hunk, "lines", []) or [])
    return sum(1 for p, _ in lines if p in {"+", "-"})


def _normalize_overlapping_hunks(hunks: list) -> Tuple[list, int]:
    """Drop duplicate/conflicting overlaps by keeping the strongest/latest hunk."""
    if not hunks or len(hunks) <= 1:
        return hunks, 0
    ordered = sorted(
        list(hunks),
        key=lambda h: (
            _hunk_span(h)[0],
            _hunk_span(h)[1],
            int(getattr(h, "new_start", 0) or 0),
        ),
    )
    kept: List[object] = []
    dropped = 0
    for h in ordered:
        if not kept:
            kept.append(h)
            continue
        prev = kept[-1]
        prev_s, prev_e = _hunk_span(prev)
        cur_s, cur_e = _hunk_span(h)
        if cur_s <= prev_e:
            prev_lines = list(getattr(prev, "lines", []) or [])
            cur_lines = list(getattr(h, "lines", []) or [])
            if prev_lines == cur_lines:
                dropped += 1
                continue
            prev_score = _hunk_delta_size(prev)
            cur_score = _hunk_delta_size(h)
            if cur_score >= prev_score:
                kept[-1] = h
            dropped += 1
            continue
        kept.append(h)
    return kept, dropped


def _hunk_has_strong_anchor(hunk: object) -> bool:
    """Require unchanged anchor lines so patch placement is stable."""
    lines = list(getattr(hunk, "lines", []) or [])
    context = [c for p, c in lines if p == " "]
    if len(context) >= 2:
        return True
    for c in context:
        s = str(c or "").strip()
        if not s:
            continue
        if len(s) < 6:
            continue
        if re.fullmatch(r"[\{\}\(\)\[\];,]+", s):
            continue
        if s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        return True
    return False


def _function_spans_for_file(focus_file: str, content: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    if not content:
        return spans
    try:
        from .tools import extract_symbols_treesitter

        syms = extract_symbols_treesitter(focus_file, content=content) or []
        for sym in syms:
            kind = str(getattr(sym, "kind", "") or "")
            if kind not in {"function", "method"}:
                continue
            name = str(getattr(sym, "name", "") or "").strip()
            if not name:
                continue
            start = int(getattr(sym, "line", 1) or 1)
            end = int(getattr(sym, "end_line", start) or start)
            if end < start:
                end = start
            spans.append((name, start, end))
    except Exception:
        spans = []

    if spans:
        return spans[:120]

    # Regex fallback (mostly C/C++/Java-style function starts).
    lines = content.splitlines()
    starts: List[Tuple[str, int]] = []
    pat = re.compile(
        r"^\s*(?:[A-Za-z_][A-Za-z0-9_\s\*:&<>\[\],]*\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{\s*$"
    )
    for i, line in enumerate(lines, start=1):
        m = pat.match(line)
        if not m:
            continue
        name = m.group(1)
        if name in {"if", "for", "while", "switch", "catch"}:
            continue
        starts.append((name, i))
    for idx, (name, start) in enumerate(starts):
        end = starts[idx + 1][1] - 1 if idx + 1 < len(starts) else len(lines)
        if end < start:
            end = start
        spans.append((name, start, end))
    return spans[:120]


def _hunk_touches_functions(
    hunk: object,
    spans: List[Tuple[str, int, int]],
) -> List[str]:
    start = int(getattr(hunk, "old_start", 0) or 0)
    if start <= 0:
        start = 1
    lines = list(getattr(hunk, "lines", []) or [])

    # Attribute function ownership to changed lines rather than full hunk span.
    # This avoids false multi-function hits when unchanged context crosses into
    # neighboring function boundaries.
    touched_lines: List[int] = []
    old_line = start
    has_delta = False
    for prefix, _content in lines:
        if prefix == " ":
            old_line += 1
            continue
        has_delta = True
        if prefix == "-":
            touched_lines.append(max(1, old_line))
            old_line += 1
            continue
        if prefix == "+":
            touched_lines.append(max(1, old_line))
            continue

    # Fallback for malformed/no-delta hunks: use header span behavior.
    if not has_delta or not touched_lines:
        count = int(getattr(hunk, "old_count", 0) or 0)
        span = count if count > 0 else 1
        end = start + span - 1
        touched_lines = list(range(start, end + 1))

    out: List[str] = []
    seen = set()
    for name, s, e in spans:
        for ln in touched_lines:
            if s <= ln <= e and name not in seen:
                seen.add(name)
                out.append(name)
                break
    return out


def _should_enforce_function_hunks(intent: str, user_text: str) -> bool:
    upper_intent = str(intent or "").upper()
    if "STUB" in upper_intent:
        return True
    return bool(re.search(r"\b(todo|stub|implement|fill)\b", (user_text or ""), re.IGNORECASE))


def score_diff_candidate(
    hunks: list,
    existing_content: str,
    user_text: str,
    intent: str,
    focus_file: str,
    target_ext: str,
    analysis_packet: str = "",
) -> Tuple[int, List[str], bool]:
    score = 0
    reasons: List[str] = []
    fatal = False

    if not hunks:
        return -1000, ["EMPTY_DIFF"], True

    added_lines = [c for h in hunks for p, c in h.lines if p == "+"]
    deleted_lines = [c for h in hunks for p, c in h.lines if p == "-"]
    context_lines = [c for h in hunks for p, c in h.lines if p == " "]

    removed_todo = any(re.search(r"\bTODO\b", line, re.IGNORECASE) for line in deleted_lines)
    # TODO removals
    if removed_todo:
        score += 30
        reasons.append("removed_todo")

    # Penalize TODOs near edited regions
    if any(re.search(r"\bTODO\b", line, re.IGNORECASE) for line in context_lines):
        score -= 15
        reasons.append("todo_in_context")

    # Penalize added TODOs
    if any(re.search(r"\bTODO\b", line, re.IGNORECASE) for line in added_lines):
        score -= 25
        reasons.append("todo_added")

    def _is_trivial_add(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        if s in {"{", "}", ";"}:
            return True
        if s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            return True
        return False

    significant_adds = [l for l in added_lines if not _is_trivial_add(l)]
    if len(significant_adds) >= 8:
        if all(re.search(r"\bprintf\s*\(|\bputs\s*\(|\bputchar\s*\(", l) for l in significant_adds):
            score -= 15
            reasons.append("only_prints")
        else:
            score += 20
            reasons.append("non_trivial_adds")
    elif significant_adds:
        if all(re.search(r"\bprintf\s*\(|\bputs\s*\(|\bputchar\s*\(", l) for l in significant_adds):
            score -= 15
            reasons.append("only_prints")

    total_additions = len(added_lines)
    total_deletions = len(deleted_lines)
    file_has_todo = has_todo_markers(existing_content)
    if file_has_todo and total_additions > 0 and not removed_todo:
        score -= 45
        reasons.append("todo_not_replaced")
    if total_deletions == 0 and total_additions >= 20 and len(context_lines) < 2:
        score -= 20
        reasons.append("large_insert_low_context")

    # Analysis packet is advisory-only for scoring.
    # Do not enforce ban symbols / deps modes / unknown identifier heuristics here.
    # Keep scoring focused on structural delta quality and TODO replacement.

    return score, reasons, fatal


def validate_diff_artifact(
    output: str,
    focus_file: str,
    existing_content: str,
    user_text: str,
    intent: str = "",
    prompt_prefill: str = "",
    retrieved_ranges: Optional[List[Tuple[int, int]]] = None,
    keep_all_hunks: bool = False,
) -> Tuple[Optional[list], Optional[str]]:
    """Validate unified diff output for DIRECT_DIFF / RETRIEVE_DIFF routes.

    Returns (hunks, error). If error is None, hunks are ready to apply.
    """
    from .parsing import parse_unified_diff
    from .files import _norm_rel_path

    file_path, hunks = parse_unified_diff(
        output,
        focus_file,
        prompt_prefill=prompt_prefill,
        file_content=existing_content,
        keep_all_hunks=keep_all_hunks,
    )
    if not file_path or not hunks:
        # Distinguish malformed output from syntactically diff-like but no-op patches.
        looks_like_diff = bool(re.search(r"^\s*@@\s+-\d+", output or "", re.MULTILINE))
        if looks_like_diff:
            return None, "NO_OP_DIFF"
        return None, "PARSE_FAIL"
    if _norm_rel_path(file_path) != _norm_rel_path(focus_file):
        return None, "PATH_MISMATCH"

    pre_err = _validate_diff_pre_apply(
        hunks, existing_content, user_text, intent=intent, focus_file=focus_file
    )
    if pre_err:
        return None, pre_err

    if _is_whole_file_rewrite(hunks, existing_content):
        return None, "FULL_REWRITE"
    if _has_overlapping_hunks(hunks):
        normalized_hunks, dropped = _normalize_overlapping_hunks(hunks)
        hunks = normalized_hunks
        if dropped > 0:
            dbg(
                "patch_protocol.overlap_detected=1 "
                f"mode=normalize_then_apply_sequential dropped={dropped}"
            )
        else:
            dbg("patch_protocol.overlap_detected=1 mode=apply_sequential")

    # Require strong anchors for each hunk except pure creation/deletion edge cases.
    for h in hunks:
        lines = list(getattr(h, "lines", []) or [])
        has_delta = any(p in {"+", "-"} for p, _ in lines)
        if not has_delta:
            continue
        old_count = int(getattr(h, "old_count", 0) or 0)
        has_delete = any(p == "-" for p, _ in lines)
        if old_count <= 0 and not has_delete:
            continue
        if not _hunk_has_strong_anchor(h):
            return None, "WEAK_HUNK_ANCHOR"

    file_line_count = len((existing_content or "").splitlines())
    tiny_file_single_hunk = file_line_count < max(
        1, int(config.PATCH_TINY_SINGLE_HUNK_MAX_LINES)
    )
    if tiny_file_single_hunk and len(hunks) != 1:
        dbg(
            "patch_protocol.tiny_multi_hunk=1 "
            f"hunks={len(hunks)} mode=allow_non_overlapping"
        )

    # Stub-like tasks: constrain each hunk to exactly one function.
    if (not tiny_file_single_hunk) and _should_enforce_function_hunks(intent, user_text):
        spans = _function_spans_for_file(focus_file, existing_content)
        if spans:
            for h in hunks:
                touched = _hunk_touches_functions(h, spans)
                if len(touched) == 0:
                    return None, "HUNK_OUTSIDE_FUNCTION"
                if len(touched) > 1:
                    return None, "HUNK_MULTI_FUNCTION"

    if retrieved_ranges:
        allow_outside = any(
            k in (user_text or "").lower()
            for k in ("entire file", "whole file", "full file", "everywhere", "global")
        )
        if not allow_outside and not _diff_within_ranges(hunks, retrieved_ranges):
            return None, "OUTSIDE_RETRIEVED_CONTEXT"

    return hunks, None


def _extract_diff_header_paths(output: str) -> List[str]:
    paths: List[str] = []
    for line in (output or "").splitlines():
        if line.startswith("--- "):
            raw = line[4:].strip()
        elif line.startswith("+++ "):
            raw = line[4:].strip()
        else:
            continue
        if raw.startswith("a/") or raw.startswith("b/"):
            raw = raw[2:]
        if raw == "/dev/null":
            continue
        raw = raw.split("\t")[0].strip()
        if raw:
            paths.append(_norm_rel_path(raw))
    return paths


def _patch_path_allowed(rel_path: str) -> bool:
    if not rel_path:
        return False
    norm = _norm_rel_path(rel_path)
    if not norm:
        return False
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        return False
    if norm == ".." or norm.startswith("../") or "/../" in norm:
        return False
    try:
        resolve_path(norm)
    except Exception:
        return False
    return True


def _normalize_patch_output(output: str) -> str:
    """Normalize model output into a parser-friendly unified diff payload.

    We tolerate markdown fenced wrappers in practice by stripping them before
    strict diff validation.
    """
    text = (output or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return text

    # Remove markdown fences even if they appear inline (for example "```diff").
    # Backticks are presentation-only and should not invalidate an otherwise
    # usable unified diff payload.
    text = re.sub(r"```[A-Za-z0-9_-]*", "", text)
    lines = text.split("\n")
    # Strip leading/trailing fenced wrapper lines.
    if lines and re.match(r"^\s*```[A-Za-z0-9_-]*\s*$", lines[0]):
        lines = lines[1:]
    if lines and re.match(r"^\s*```\s*$", lines[-1]):
        lines = lines[:-1]
    # Remove standalone fence lines anywhere.
    lines = [ln for ln in lines if not re.match(r"^\s*```[A-Za-z0-9_-]*\s*$", ln)]
    text = "\n".join(lines)
    return text.strip()


def _validate_patch_protocol_text(output: str) -> Optional[str]:
    text = output or ""
    if "[[[file:" in text or "[[[end]]]" in text:
        return "FORBIDDEN_TOKEN_FILEBLOCK"
    if config.PATCH_REJECT_BINARY_LIKE:
        binary_signals = (
            "GIT binary patch",
            "Binary files ",
            "\x00",
        )
        if any(sig in text for sig in binary_signals):
            return "BINARY_LIKE_DIFF_REJECTED"

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    first_diff_idx = -1
    for idx, line in enumerate(lines):
        if (
            line.startswith("diff --git ")
            or line.startswith("--- a/")
            or line.startswith("--- ")
        ):
            first_diff_idx = idx
            break
    if first_diff_idx < 0:
        return "MISSING_FILE_HEADERS"

    allowed_meta_prefixes = (
        "diff --git ",
        "index ",
        "--- ",
        "+++ ",
        "@@ ",
        "new file mode ",
        "deleted file mode ",
        "similarity index ",
        "rename from ",
        "rename to ",
    )
    strict_hunk_header = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
    saw_minus = False
    saw_plus = False
    saw_hunk = False
    for line in lines[first_diff_idx:]:
        if not line:
            continue
        if line.startswith("--- "):
            # Accept both git-style (--- a/file) and plain unified (--- file)
            if line.strip() != "--- /dev/null":
                saw_minus = True
            continue
        if line.startswith("+++ "):
            if line.strip() != "+++ /dev/null":
                saw_plus = True
            continue
        if line.startswith("@@"):
            saw_hunk = True
            if config.PATCH_STRICT_PARSE and not strict_hunk_header.match(line):
                return "MALFORMED_HUNK_HEADER"
            continue
        if line.startswith("+") or line.startswith("-") or line.startswith(" "):
            continue
        if line == r"\ No newline at end of file":
            continue
        if any(line.startswith(prefix) for prefix in allowed_meta_prefixes):
            continue
        return "PROSE_OUTSIDE_DIFF"

    if not (saw_minus and saw_plus):
        return "MISSING_FILE_HEADERS"
    if not saw_hunk:
        # Allow pure add/delete payloads only when all payload lines are +/-.
        payload_lines: List[str] = []
        for line in lines[first_diff_idx:]:
            if not line:
                continue
            if any(
                line.startswith(prefix)
                for prefix in allowed_meta_prefixes + (r"\ No newline at end of file",)
            ):
                continue
            payload_lines.append(line)
        if not payload_lines or not all(
            pl.startswith("+") or pl.startswith("-") for pl in payload_lines
        ):
            return "MISSING_HUNKS"
    return None


def validate_patch_protocol_artifact(
    output: str,
    focus_file: str,
    existing_content: str,
    user_text: str,
    intent: str = "",
    prompt_prefill: str = "",
    retrieved_ranges: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[Optional[list], Optional[str]]:
    normalized_output = _normalize_patch_output(output)
    if normalized_output != (output or ""):
        dbg("patch_protocol.normalization=markdown_fences_stripped")
    pre_err = _validate_patch_protocol_text(normalized_output)
    if pre_err:
        return None, pre_err

    header_paths = _extract_diff_header_paths(normalized_output)
    unique_paths = sorted({p for p in header_paths if p})
    if len(unique_paths) > max(1, config.PATCH_MAX_FILES_PER_DIFF):
        return None, "PATCH_TOO_MANY_FILES"
    for path in unique_paths or [_norm_rel_path(focus_file)]:
        if not _patch_path_allowed(path):
            return None, "PATH_NOT_ALLOWED"

    hunks, err = validate_diff_artifact(
        output=normalized_output,
        focus_file=focus_file,
        existing_content=existing_content,
        user_text=user_text,
        intent=intent,
        prompt_prefill=prompt_prefill,
        retrieved_ranges=retrieved_ranges,
        keep_all_hunks=False,
    )
    if err:
        return None, err
    if not hunks:
        return None, "PARSE_FAIL"
    if len(hunks) > max(1, config.PATCH_MAX_HUNKS_PER_FILE):
        return None, "PATCH_TOO_MANY_HUNKS"
    return hunks, None


def validate_plan_json(plan: dict, max_files: int) -> Optional[str]:
    """Validate PLAN_EXECUTE JSON output structure and paths."""
    if not isinstance(plan, dict) or "files" not in plan:
        return "PLAN_INVALID"
    files = plan.get("files")
    if not isinstance(files, list) or not files:
        return "PLAN_EMPTY"
    if len(files) > max_files:
        return "PLAN_TOO_MANY_FILES"

    # Merge duplicate path entries into one actionable item.
    # Some planner outputs emit one entry per function for the same file.
    merged: Dict[str, dict] = {}
    order: List[str] = []
    original_len = len(files)
    for entry in files:
        if not isinstance(entry, dict):
            return "PLAN_ENTRY_INVALID"
        path = entry.get("path")
        if not path or not isinstance(path, str):
            return "PLAN_PATH_MISSING"
        p = path.strip()
        if not p:
            return "PLAN_PATH_MISSING"
        if p not in merged:
            merged[p] = dict(entry)
            order.append(p)
            continue
        prev_desc = str(merged[p].get("description") or "").strip()
        next_desc = str(entry.get("description") or "").strip()
        if next_desc and next_desc not in prev_desc:
            merged[p]["description"] = f"{prev_desc}; {next_desc}" if prev_desc else next_desc
        prev_target = str(merged[p].get("target") or "").strip()
        next_target = str(entry.get("target") or "").strip()
        if next_target and next_target not in prev_target:
            merged[p]["target"] = f"{prev_target}; {next_target}" if prev_target else next_target
        action_new = str(entry.get("action") or "").strip().lower()
        action_prev = str(merged[p].get("action") or "").strip().lower()
        # Keep strongest action when duplicates disagree.
        if action_new == "delete" or (action_new == "create" and action_prev == "edit"):
            merged[p]["action"] = action_new

    files = [merged[p] for p in order]
    if len(files) < original_len:
        dbg(f"validate_plan_json: normalization=duplicates_merged count={original_len-len(files)}")
    plan["files"] = files
    if len(files) > max_files:
        return "PLAN_TOO_MANY_FILES"

    actionable = re.compile(r"\b(add|modify|replace|remove|refactor|implement|fix|create|delete|update|apply|edit)\b", re.IGNORECASE)
    allowed_actions = {"edit", "create", "delete"}
    for entry in files:
        if not isinstance(entry, dict):
            return "PLAN_ENTRY_INVALID"
        path = entry.get("path")
        action = entry.get("action")
        desc = entry.get("description")
        if not path or not isinstance(path, str):
            return "PLAN_PATH_MISSING"
        path = path.strip()
        if not path:
            return "PLAN_PATH_MISSING"
        if action is None or desc is None:
            return "PLAN_FIELDS_MISSING"
        if not isinstance(action, str) or action.strip().lower() not in allowed_actions:
            return "PLAN_ACTION_INVALID"
        if not isinstance(desc, str) or not desc.strip():
            return "PLAN_DESCRIPTION_MISSING"
        if not actionable.search(desc):
            return "PLAN_NON_ACTIONABLE"
        try:
            resolve_path(path)
        except Exception:
            return "PLAN_PATH_OUTSIDE_REPO"
    return None


# ---------- Basic checks ----------


def run_basic_checks(
    touched_files: List[str],
    strict: bool = True,
    preferred_cmd: Optional[str] = None,
    root_override: Optional[str] = None,
) -> Tuple[bool, str]:
    if config.SKIP_COMPILE_CHECKS:
        if config.DEBUG:
            dbg("basic_checks: skipped by config")
        return True, ""

    if root_override:
        from pathlib import Path

        root_path = Path(root_override).expanduser().resolve()
    else:
        root_path = get_root()
    root = str(root_path)
    if preferred_cmd:
        try:
            cmd_text = str(preferred_cmd).strip()
            if cmd_text:
                joined = " ".join(touched_files)
                one = touched_files[0] if touched_files else ""
                cmd_text = cmd_text.replace("{files}", joined).replace("{file}", one)
                cmd = shlex.split(cmd_text)
                if cmd:
                    dbg(f"basic_checks: preferred_cmd={' '.join(cmd)}")
                    res = subprocess.run(
                        cmd,
                        cwd=root,
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    if res.returncode == 0:
                        return True, ""
                    out = (res.stderr or "") + "\n" + (res.stdout or "")
                    low = out.lower()
                    if (
                        "not found" in low
                        or "no such file" in low
                        or "missing script" in low
                        or "unknown command" in low
                    ):
                        dbg("basic_checks: preferred_cmd unavailable, falling back")
                    else:
                        return False, out.strip()[:1000]
        except Exception as exc:
            dbg(f"basic_checks: preferred_cmd exception={exc}")

    py_files = [f for f in touched_files if f.endswith(".py")]
    c_files = [f for f in touched_files if f.endswith(".c")]
    cxx_files = [f for f in touched_files if f.endswith((".cpp", ".cc"))]
    js_files = [f for f in touched_files if f.endswith(".js")]
    ts_files = [f for f in touched_files if f.endswith(".ts")]
    java_files = [f for f in touched_files if f.endswith(".java")]
    go_files = [f for f in touched_files if f.endswith(".go")]
    rust_files = [f for f in touched_files if f.endswith(".rs")]
    rb_files = [f for f in touched_files if f.endswith(".rb")]
    php_files = [f for f in touched_files if f.endswith(".php")]
    sh_files = [f for f in touched_files if f.endswith(".sh")]
    json_files = [f for f in touched_files if f.endswith(".json")]
    yaml_files = [f for f in touched_files if f.endswith((".yaml", ".yml"))]

    def _run_cmd(cmd: List[str], stdin_text: Optional[str] = None) -> Tuple[bool, str]:
        try:
            res = subprocess.run(
                cmd,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                input=stdin_text,
            )
        except Exception as exc:
            return False, str(exc)
        if res.returncode == 0:
            return True, ""
        out = ((res.stderr or "") + "\n" + (res.stdout or "")).strip()
        return False, out[:2000]

    def _which_or_none(tool: str) -> Optional[str]:
        return shutil.which(tool)

    def _run_lsp_checks_first() -> Tuple[bool, str]:
        """Run LSP-style diagnostics before compiler checks.

        For C/C++ this uses clangd's check mode to surface missing headers,
        undeclared identifiers, and semantic diagnostics early.
        """
        c_like_files = [
            f
            for f in touched_files
            if f.endswith((".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"))
        ]
        if not c_like_files:
            return True, ""

        clangd_bin = _which_or_none("clangd")
        if not clangd_bin:
            dbg("basic_checks: lsp check skipped (clangd not found)")
            return True, ""

        for file_path in c_like_files:
            cmd = [
                clangd_bin,
                f"--check={file_path}",
                "--log=error",
                f"--compile-commands-dir={root}",
            ]
            try:
                res = subprocess.run(
                    cmd,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception as exc:
                return False, f"LSP_CHECK_FAIL ({file_path})\n{exc}"
            out = ((res.stderr or "") + "\n" + (res.stdout or "")).strip()
            if res.returncode != 0:
                return False, f"LSP_CHECK_FAIL ({file_path})\n{out[:2000]}"
            # clangd may exit 0 while still reporting semantic diagnostics;
            # scan output for hard-error signals.
            low = out.lower()
            if low and any(
                marker in low
                for marker in (
                    " error:",
                    "failed to",
                    "undeclared identifier",
                    "file not found",
                    "unknown type name",
                    "no member named",
                )
            ):
                return False, f"LSP_CHECK_FAIL ({file_path})\n{out[:2000]}"
        return True, ""

    python_bin = _which_or_none("python3") or sys.executable

    # 1) LSP checks first (as requested), then compiler/tool checks below.
    ok, err = _run_lsp_checks_first()
    if not ok:
        return False, err

    # C (.c)
    for file_path in c_files:
        ok, err = _run_cmd(
            ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", file_path]
        )
        if not ok:
            return False, err

    # C++ (.cpp, .cc)
    for file_path in cxx_files:
        ok, err = _run_cmd(
            ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", file_path]
        )
        if not ok:
            return False, err

    # Python (.py)
    for file_path in py_files:
        ok, err = _run_cmd([python_bin, "-m", "py_compile", file_path])
        if not ok:
            return False, err

    # JavaScript (.js)
    if js_files:
        node_bin = _which_or_none("node")
        if not node_bin:
            return False, "node not found (required for JS syntax checks)"
        for file_path in js_files:
            ok, err = _run_cmd([node_bin, "--check", file_path])
            if not ok:
                return False, err

    # TypeScript (.ts)
    if ts_files:
        npx_bin = _which_or_none("npx")
        if not npx_bin:
            return False, "npx not found (required for TS checks)"
        ok, err = _run_cmd([npx_bin, "tsc", "--noEmit", "--pretty", "false"])
        if not ok:
            return False, err

    # Java (.java)
    if java_files:
        javac_bin = _which_or_none("javac")
        if not javac_bin:
            return False, "javac not found (required for Java checks)"
        for file_path in java_files:
            ok, err = _run_cmd([javac_bin, "-Xlint:all", "-Werror", file_path])
            if not ok:
                return False, err

    # Go (.go)
    if go_files:
        go_bin = _which_or_none("go")
        if not go_bin:
            return False, "go not found (required for Go checks)"
        for file_path in go_files:
            ok, err = _run_cmd([go_bin, "build", file_path])
            if not ok:
                return False, err

    # Rust (.rs)
    if rust_files:
        cargo_bin = _which_or_none("cargo")
        if not cargo_bin:
            return False, "cargo not found (required for Rust checks)"
        ok, err = _run_cmd([cargo_bin, "check"])
        if not ok:
            return False, err

    # Ruby (.rb)
    if rb_files:
        ruby_bin = _which_or_none("ruby")
        if not ruby_bin:
            return False, "ruby not found (required for Ruby checks)"
        for file_path in rb_files:
            ok, err = _run_cmd([ruby_bin, "-c", file_path])
            if not ok:
                return False, err

    # PHP (.php)
    if php_files:
        php_bin = _which_or_none("php")
        if not php_bin:
            return False, "php not found (required for PHP checks)"
        for file_path in php_files:
            ok, err = _run_cmd([php_bin, "-l", file_path])
            if not ok:
                return False, err

    # Bash/Shell (.sh)
    if sh_files:
        bash_bin = _which_or_none("bash")
        if not bash_bin:
            return False, "bash not found (required for shell checks)"
        for file_path in sh_files:
            ok, err = _run_cmd([bash_bin, "-n", file_path])
            if not ok:
                return False, err

    # JSON (.json)
    for file_path in json_files:
        ok, err = _run_cmd([python_bin, "-m", "json.tool", file_path])
        if not ok:
            return False, err

    # YAML (.yaml/.yml) requires PyYAML
    for file_path in yaml_files:
        file_abs = str(root_path / file_path)
        try:
            with open(file_abs, "r", encoding="utf-8", errors="replace") as f:
                yaml_text = f.read()
        except Exception as exc:
            return False, str(exc)
        ok, err = _run_cmd(
            [python_bin, "-c", "import yaml,sys; yaml.safe_load(sys.stdin.read())"],
            stdin_text=yaml_text,
        )
        if not ok:
            return False, err

    return True, ""


def _extract_high_risk_c_warning(text: str) -> str:
    """Return first high-risk C/C++ warning line, or empty string.

    Keep this narrow so we become stricter without blocking on noisy warnings
    like unused params.
    """
    if not text:
        return ""
    patterns = (
        r"warning: format specifies type",
        r"warning: incompatible pointer types",
        r"warning: incompatible integer to pointer conversion",
        r"warning: implicit declaration of function",
        r"warning: call to undeclared function",
        r"warning: use of undeclared identifier",
    )
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if "warning:" not in low:
            continue
        if any(re.search(p, low) for p in patterns):
            return line[:300]
    return ""
