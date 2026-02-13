import json
import re
import hashlib
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Generator, List, Optional, Tuple

from .files import (
    _norm_rel_path,
    apply_unified_diff,
    _is_new_or_empty_file,
    get_root,
    get_include,
    read_single_file_for_context,
)
from .model import backend_name, stream_reply
from . import config, prompt_buffer, state
from .anchors import (
    InsertionAnchor,
    compute_insertion_anchor,
    discover_target_file,
    get_structural_context,
)
from .prompts import (
    _ext,
    build_plan_multi_prompt,
)
from .intents import (
    classify_intent,
    UNKNOWN,
    CLEAR_RANGE,
    DELETE_FILE,
    IMPLEMENT_STUBS,
    extract_target_files,
)
from .agent_analysis import build_compact_analysis, compact_analysis_json
from .task_card import build_task_card
from .router import (
    RequestContext,
    decide_route_with_reason,
    has_assignment_style_text,
    wants_file_rewrite,
    wants_minimal_diff,
    Route,
)
from .validation import (
    _validate_post_apply,
    _validate_prompt_semantics,
    run_basic_checks,
    validate_patch_protocol_artifact,
    validate_plan_json,
    parse_plan,
    has_todo_markers,
    has_placeholder_conditionals,
)
from .utils import dbg, dbg_dump
from .smoke import (
    PLAN_FILENAME as SMOKE_PLAN_FILENAME,
    load_smoke_plan,
    run_smoke,
    build_smoke_report,
    serialize_failures,
    SmokeResult,
)
from .structural import (
    select_target_symbol,
    select_target_symbols,
    build_symbol_index,
    build_packed_context,
    extract_target_snippet,
    normalize_structural_output,
    apply_symbol_replacement,
    validate_structural_candidate,
    validate_replacement_symbol_unit,
    structural_format_retry_rules,
    structural_general_retry_rules,
)
from .normalizer import normalize_symbol
from .fileblock import (
    FileblockRuntime,
    run_fileblock_write,
    run_fileblock_edit,
)


# ---------- Chat (plain) ----------

chat_history: List[Tuple[str, str]] = []

def build_chat_prompt(user_text: str) -> str:
    lines = []
    for user_turn, assistant_turn in chat_history:
        lines.append(f"User: {user_turn}")
        lines.append(f"Assistant: {assistant_turn}")
    lines.append(f"User: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def run_chat(
    user_text: str,
    silent: bool = False,
    focus_file: Optional[str] = None,
    full_context: bool = False,
):
    """Run chat. If silent=True, don't print to stdout (for HTTP server mode)."""
    prompt, _ = prompt_buffer.build_prompt(
        "chat",
        user_text,
        focus_file=focus_file,
        full_context=full_context,
    )
    assistant_reply = stream_reply(
        prompt,
        silent=silent,
        max_new=config.MAX_NEW,
        stop_sequences=["\nUser:", "\nSYSTEM:", "\nCONTEXT:", "\nHISTORY:"],
    )
    chat_history.append((user_text, assistant_reply))
    state.append_chat_turn(user_text, assistant_reply)
    return assistant_reply


# ---------- Agent (Cursor-style) ----------

agent_history: List[Tuple[str, str]] = []
MAX_AGENT_HISTORY = 6  # cap in-memory history to prevent unbounded growth
MAX_AGENT_CHARS = 300  # truncate stored output to avoid prompt bloat

# Route-level temperature targets
TEMP_PLAN = 0.2
TEMP_EXECUTE = 0.2

# Structural KV cache session state:
# one session per (file, packed_context_version, rules_version), reset on tuple change
# and explicitly on new-chat.
_STRUCTURAL_KV_LOCK = threading.Lock()
_STRUCTURAL_KV_EPOCH = 0
_STRUCTURAL_KV_ACTIVE_KEY = ""


def _append_history(user_text: str, output: str) -> None:
    """Append to agent_history with cap and truncation."""
    if config.DISABLE_HISTORY:
        return
    out = (output or "")
    # Drop degenerate low-signal replies from history so they don't pollute
    # subsequent prompts (for example repeated single-character loops).
    stripped = out.strip()
    if _is_model_failure_output(out):
        out = ""
    elif stripped:
        one_char_lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
        if one_char_lines and len(one_char_lines) >= 12 and all(len(ln) <= 2 for ln in one_char_lines):
            uniq = {ln for ln in one_char_lines}
            if len(uniq) <= 2:
                out = ""
        elif len(set(stripped)) <= 2 and len(stripped) >= 24:
            out = ""
    agent_history.append((user_text, out[:MAX_AGENT_CHARS]))
    del agent_history[:-MAX_AGENT_HISTORY]


def _is_model_failure_output(output: str) -> bool:
    s = (output or "").strip()
    if not s:
        return True
    if s.startswith("[Model stalled]") or s.startswith("[Model timeout]"):
        return True
    if s.startswith("[Model error") or s.startswith("Error:"):
        return True
    return False


def _is_collapsed_planner_output(output: str) -> bool:
    s = (output or "").strip()
    if not s:
        return True
    if s in {"```", "```json"}:
        return True
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return True
    short = [ln for ln in lines if len(ln) <= 2]
    if len(lines) >= 10 and len(short) >= int(len(lines) * 0.8):
        uniq = set(short)
        if len(uniq) <= 2:
            return True
    if len(lines) >= 8:
        head = lines[:8]
        if all(ln.lower().startswith("do not ") for ln in head):
            return True
    return False


def _allows_partial_todo_edit(user_text: str) -> bool:
    if not user_text:
        return False
    low = user_text.lower()
    if len(low) > 160:
        return False
    tiny_signals = (
        "comment", "typo", "whitespace", "spacing", "format",
        "rename", "doc", "documentation", "readme",
    )
    big_signals = (
        "implement", "complete", "fill", "replace", "fix", "todo", "placeholder",
    )
    return any(s in low for s in tiny_signals) and not any(s in low for s in big_signals)


def _looks_chat_query(user_text: str) -> bool:
    low = (user_text or "").strip().lower()
    if not low:
        return True
    edit_intents = (
        "implement", "fix", "modify", "update", "edit", "rewrite", "refactor",
        "add", "remove", "replace", "patch", "diff", "fileblock",
        "create file", "write code", "apply",
    )
    if any(tok in low for tok in edit_intents):
        return False
    if len(low) <= 80:
        return True
    if low.endswith("?") or low.startswith(("why ", "what ", "how ", "can ", "is ", "are ")):
        return True
    return False


def _estimated_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _should_activate_task_card(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    return _estimated_token_count(text) > int(config.TASK_CARD_MIN_TOKENS)


def _looks_like_task_card(text: str) -> bool:
    return str(text or "").lstrip().startswith("TASK_CARD")


def _build_retry_base_request(
    sliced_request: str,
    user_text: str,
    focus_file: str,
    file_content: str,
    prefer_task_card: bool = False,
) -> str:
    sliced = (sliced_request or "").strip()
    raw = (user_text or "").strip()
    if _looks_like_task_card(sliced):
        return sliced
    if _looks_like_task_card(raw):
        return raw
    if prefer_task_card:
        tc = build_task_card(
            focus_file,
            raw or sliced,
            file_content or "",
            max_lines=15,
        )
        if tc.strip():
            dbg(f"retry.base_request=task_card_fallback len={len(tc)}")
            return tc
    return sliced or raw








@contextmanager
def _span(trace: List[Dict[str, object]], name: str) -> Generator:
    """Lightweight timing span — appends {"name", "ms"} to trace."""
    t0 = time.time()
    try:
        yield
    finally:
        trace.append({"name": name, "ms": int((time.time() - t0) * 1000)})


def _build_meta(
    prompt_len: int,
    output: str,
    blocks_count: int,
    retried: bool,
    start: float,
    trace: Optional[List[Dict[str, object]]] = None,
    model_calls_used: int = 0,
) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "backend": backend_name(),
        "prompt_len": prompt_len,
        "reply_len": len(output),
        "timeout": output == "[Model timeout]",
        "blocks": blocks_count,
        "duration_ms": int((time.time() - start) * 1000),
        "retried": retried,
        "model_calls_used": model_calls_used,
        "retry_cap": config.MAX_MODEL_CALLS,
        "failure_kind": "none",
        "failure_reason": "",
        "smoke_attempted": False,
        "smoke_passed": None,
        "smoke_failures": [],
        "retry_trigger": "none",
        "requires_user_decision": False,
    }
    if trace is not None:
        meta["trace"] = trace
        if config.DEBUG:
            dbg(f"trace: {trace}")
    return meta


def _resolve_focus_file(user_text: str, focus_file: Optional[str]) -> str:
    """Deterministic focus resolution for single-file flow."""
    if focus_file:
        return _norm_rel_path(focus_file)

    # 1) Explicit file mention in request.
    explicit = extract_target_files(user_text or "")
    if explicit:
        chosen = _norm_rel_path(explicit[0])
        dbg(f"agent: no file selected, explicit target {chosen}")
        return chosen

    dbg("agent: no file selected, focus remains none")
    return ""


def _target_files_for_routing(user_text: str, focus_file: str) -> List[str]:
    """Build validated target files list for routing decisions."""
    files = extract_target_files(user_text)
    ordered: List[str] = []
    seen = set()
    focus_norm = _norm_rel_path(focus_file) if focus_file else ""
    if focus_norm:
        ordered.append(focus_norm)
        seen.add(focus_norm)
    for p in files:
        norm = _norm_rel_path(p)
        if not norm or norm in seen:
            continue
        ordered.append(norm)
        seen.add(norm)
    return ordered


def _validate_no_todos(path: str) -> Optional[str]:
    """Ensure generated file no longer contains TODO/stub placeholders."""
    content = read_single_file_for_context(path).get(path, "")
    if has_todo_markers(content):
        return "STUB_FAIL: placeholders remain"
    return None


def _compute_unified_diff(before: str, after: str, focus_file: str) -> str:
    import difflib

    lines = difflib.unified_diff(
        (before or "").splitlines(keepends=True),
        (after or "").splitlines(keepends=True),
        fromfile=f"a/{focus_file}",
        tofile=f"b/{focus_file}",
        n=3,
    )
    return "".join(lines)


def _diff_stats(diff_text: str) -> Tuple[int, int]:
    adds = 0
    dels = 0
    for ln in (diff_text or "").splitlines():
        if ln.startswith("+++") or ln.startswith("---"):
            continue
        if ln.startswith("+"):
            adds += 1
        elif ln.startswith("-"):
            dels += 1
    return adds, dels


def _history_output_for_storage(
    raw_output: str,
    meta: Optional[Dict[str, object]],
    focus_file: str = "",
) -> str:
    out = str(raw_output or "")
    m = meta or {}
    if out not in {"[Staged file_edit]", "[Applied file_edit]"}:
        return out

    action = "Staged" if out == "[Staged file_edit]" else "Applied"
    target = str(
        m.get("staged_file")
        or m.get("focus_file")
        or focus_file
        or ""
    ).strip()
    diff_text = str(m.get("diff") or "")
    if diff_text.strip():
        adds, dels = _diff_stats(diff_text)
        file_part = f" for {target}" if target else ""
        msg = f"{action} changes{file_part} (+{adds}/-{dels})."
        if out == "[Staged file_edit]":
            msg += " Review and click Accept or Reject."
        return msg

    if out == "[Staged file_edit]":
        if target:
            return f"Staged changes for {target}. Review and click Accept or Reject."
        return "Staged code changes are ready. Review and click Accept or Reject."
    if target:
        return f"Applied code changes to {target}."
    return "Applied code changes."


def _patch_failure_kind_from_rule(rule: str) -> str:
    r = (rule or "").strip().upper()
    if not r:
        return "format"
    if "PATH" in r:
        return "path"
    if "BINARY" in r:
        return "format"
    if "PATCH_TOO_MANY" in r:
        return "format"
    if "SYNTAX" in r:
        return "syntax"
    if "SEMANTIC" in r:
        return "semantic"
    if "PROSE" in r or "FORBIDDEN" in r or "MALFORMED" in r:
        return "format"
    return "format"


def _hunk_debug_text(hunk: object) -> str:
    old_start = int(getattr(hunk, "old_start", 0) or 0)
    old_count = int(getattr(hunk, "old_count", 0) or 0)
    new_start = int(getattr(hunk, "new_start", 0) or 0)
    new_count = int(getattr(hunk, "new_count", 0) or 0)
    lines = list(getattr(hunk, "lines", []) or [])
    out = [f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"]
    for p, c in lines[:120]:
        out.append(f"{p}{c}")
    if len(lines) > 120:
        out.append(f"... ({len(lines)-120} more hunk lines)")
    return "\n".join(out)


def _build_apply_failure_diagnostics(
    focus_file: str,
    original_content: str,
    hunks: list,
    error_text: str,
) -> str:
    lines = (original_content or "").splitlines()
    out: List[str] = [
        "PATCH_APPLY_FAILURE",
        f"file={focus_file}",
        f"error={str(error_text or '').strip()}",
    ]
    ordered = sorted(
        list(hunks or []),
        key=lambda h: (
            int(getattr(h, "old_start", 0) or 0),
            int(getattr(h, "new_start", 0) or 0),
        ),
    )
    for idx, h in enumerate(ordered[:3], start=1):
        old_start = int(getattr(h, "old_start", 1) or 1)
        old_count = int(getattr(h, "old_count", 0) or 0)
        span = old_count if old_count > 0 else 1
        old_end = old_start + span - 1
        ctx_start = max(1, old_start - 6)
        ctx_end = min(len(lines), old_end + 6)
        out.append(f"HUNK_{idx}:")
        out.append(_hunk_debug_text(h))
        out.append(f"CURRENT_LINES_{idx}: {ctx_start}-{ctx_end}")
        if lines and ctx_end >= ctx_start:
            for ln in range(ctx_start, ctx_end + 1):
                out.append(f"{ln}| {lines[ln - 1]}")
    return "\n".join(out).strip()


def _sandbox_apply_diff(
    focus_file: str,
    original_content: str,
    hunks: list,
    preferred_validate_cmd: str = "",
) -> Tuple[bool, str, str, str]:
    """Apply diff in temp root and run syntax checks.

    Returns (ok, failure_reason, candidate_content, candidate_diff)
    """
    candidate_content = ""
    candidate_diff = ""
    with tempfile.TemporaryDirectory(prefix="moonlet_patch_") as tmpdir:
        tmp_root = Path(tmpdir).resolve()
        target = (tmp_root / focus_file).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(original_content or "", encoding="utf-8")
        try:
            apply_unified_diff(focus_file, hunks, root_override=tmp_root)
        except Exception as exc:
            dbg(f"patch_apply.temp_fail reason={exc}")
            diag = _build_apply_failure_diagnostics(
                focus_file=focus_file,
                original_content=original_content,
                hunks=hunks,
                error_text=str(exc),
            )
            dbg_dump("patch_apply_failure_diagnostics", diag)
            return False, diag, "", ""

        candidate_content = target.read_text(encoding="utf-8")
        candidate_diff = _compute_unified_diff(
            original_content or "",
            candidate_content,
            focus_file,
        )
        ok, err = run_basic_checks(
            [focus_file],
            strict=False,
            preferred_cmd=preferred_validate_cmd or None,
            root_override=str(tmp_root),
        )
        if not ok:
            dbg("verify.syntax_fail")
            return False, err or "compile failed", candidate_content, candidate_diff
    return True, "", candidate_content, candidate_diff


def _sandbox_validate_candidate_content(
    focus_file: str,
    candidate_content: str,
    preferred_validate_cmd: str = "",
) -> Tuple[bool, str]:
    """Run compile checks for candidate content in a temp sandbox root."""
    with tempfile.TemporaryDirectory(prefix="moonlet_fileblock_") as tmpdir:
        tmp_root = Path(tmpdir).resolve()
        target = (tmp_root / focus_file).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(candidate_content or "", encoding="utf-8")
        ok, err = run_basic_checks(
            [focus_file],
            strict=False,
            preferred_cmd=preferred_validate_cmd or None,
            root_override=str(tmp_root),
        )
        if not ok:
            dbg("verify.syntax_fail")
            return False, err or "compile failed"
    return True, ""


def _normalize_failure_reason(text: str, max_chars: int = 220) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    return s[:max_chars]


def _extract_failure_from_output(output: str) -> Tuple[str, str]:
    s = (output or "").strip()
    if not s:
        return "none", ""
    for kind in ("SYNTAX_FAIL", "SEMANTIC_FAIL", "WIPE_RISK"):
        marker = f"{kind}:"
        if marker in s:
            reason = s.split(marker, 1)[1].strip()
            mapped = "rewrite_risk" if kind == "WIPE_RISK" else kind.lower().replace("_fail", "")
            return mapped, _normalize_failure_reason(reason)
    if s.startswith("[File edit failed:"):
        return "format", _normalize_failure_reason(s)
    if s.startswith("[No-op file_edit:"):
        return "noop", "model returned no effective file changes"
    return "none", ""


def _build_compile_report(reason: str, max_lines: int = 40) -> str:
    raw = (reason or "").strip()
    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    clipped = lines[: max(1, int(max_lines))]
    out = [
        "COMPILE_REPORT:",
        "- failure: compile_or_validation_failed",
    ]
    if clipped:
        out.append("- snippet:")
        for ln in clipped:
            out.append(f"  - {ln}")
    return "\n".join(out).strip()


def _build_noop_report() -> str:
    return (
        "NOOP_REPORT:\n"
        "- failure: model returned no effective file changes\n"
        "- fix: make at least one concrete code edit in the target file\n"
        "- requirement: do not return identical file content"
    )


def _build_structural_noop_report(pending_issue: str = "") -> str:
    out = [
        "NOOP_REPORT:",
        "- failure: model returned the same target symbol content",
        "- fix: edit only TARGET_SYMBOL and make a concrete, behavior-changing update",
    ]
    pending = _normalize_failure_reason(pending_issue or "", max_chars=260)
    if pending:
        out.append(f"- pending_issue: {pending}")
    return "\n".join(out)


def _looks_single_line_compact_symbol(symbol_text: str, min_chars: int = 240) -> bool:
    text = (symbol_text or "").strip()
    if not text:
        return False
    if "\n" in text:
        return False
    if "{" not in text or "}" not in text:
        return False
    return len(text) >= int(min_chars)


def _build_structural_fail_report(reason: str, target_symbol: str = "") -> str:
    low = str(reason or "").strip().lower()
    repair_lines: List[str] = [
        "Replace existing side-effect statements when adding checks; do not keep unguarded duplicates.",
        "Keep exactly one read/write/assignment operation per target field/step.",
        "Keep braces balanced and preserve the symbol wrapper/shape.",
    ]
    if "replacement_symbol_syntax_invalid" in low:
        repair_lines.insert(0, "Return syntactically valid source code for TARGET_SYMBOL.")
    if "replacement_shape_startswith_mismatch" in low:
        repair_lines.insert(0, "Preserve the target symbol header style and leading keyword token.")
    if "replacement_shape_signature_missing" in low:
        repair_lines.insert(0, "Include a complete signature/header line for TARGET_SYMBOL.")
    if "replacement_shape_brace_mismatch" in low:
        repair_lines.insert(0, "Keep brace-based structure balanced and include the symbol block body.")
    if "replacement_shape_indentation_invalid" in low or "python_one_liner_def_not_allowed" in low:
        repair_lines.insert(0, "Use block formatting with a header line and an indented body.")
    if "replacement_shape_missing_trailing_newline" in low:
        repair_lines.insert(0, "End TARGET_SYMBOL with a trailing newline.")
    if "top_level_symbol_count_collapsed" in low:
        repair_lines.insert(0, "Do not collapse/remove other top-level symbols in the file.")
    if "replacement_symbol_shape_invalid" in low:
        repair_lines.insert(0, "Return exactly one complete symbol definition and nothing else.")
        repair_lines.insert(1, "If compact output is malformed, use normal multi-line formatting.")
    if "missing_symbol_markers" in low:
        repair_lines.insert(0, "Return exactly BEGIN_SYMBOL ... END_SYMBOL with no extra text.")
    if "markerless_output_symbol_mismatch" in low:
        repair_lines.insert(0, "Return only the requested TARGET_SYMBOL (correct name/kind).")
    if "markerless_output_parse_failed" in low:
        repair_lines.insert(0, "Return parseable source for exactly one target symbol.")
    if "degenerate_low_entropy_output" in low:
        repair_lines.insert(0, "Avoid repeated/junk tokens; return a normal symbol body.")
    if "signature" in low:
        repair_lines.insert(0, "Preserve function name, parameter count, and detectable return type.")
    if "brace" in low:
        repair_lines.insert(0, "Do not add/remove unmatched blocks; keep root-level braces stable.")
    if "replacement_outside_target_span" in low:
        repair_lines.insert(0, "Edit only inside TARGET_SYMBOL; do not touch bytes outside the target span.")
    if "symbol_region_corrupted" in low or "unexpected_top_level_symbol" in low:
        repair_lines.insert(0, "Do not introduce extra symbols; keep exactly one target symbol in the edited region.")
    out = [
        "STRUCTURAL_FAIL_REPORT:",
        f"- target: {target_symbol or '<unknown>'}",
        f"- failure: {str(reason or 'structural edit failed').strip()}",
        "- repair:",
    ]
    for line in repair_lines[:4]:
        out.append(f"  - {line}")
    return "\n".join(out).strip()


def _explicit_signature_change_requested(user_text: str, target_name: str = "") -> bool:
    low = (user_text or "").lower()
    if not low:
        return False
    target_low = (target_name or "").strip().lower()
    scoped = low
    if target_low:
        pat = rf"\\b{re.escape(target_low)}\\b"
        if re.search(pat, low):
            scoped = low
    phrases = (
        "change signature",
        "update signature",
        "modify signature",
        "change parameter",
        "update parameter",
        "add parameter",
        "remove parameter",
        "rename parameter",
        "change return type",
        "update return type",
        "make it async",
        "remove async",
        "change function definition",
    )
    return any(p in scoped for p in phrases)


def _build_symbol_subtask_line(user_text: str, symbol_name: str) -> str:
    lines = [ln.strip() for ln in (user_text or "").splitlines() if ln.strip()]
    if not lines:
        return f"Update `{symbol_name}` per request."
    low_name = (symbol_name or "").strip().lower()
    for ln in lines:
        low = ln.lower()
        if low_name and low_name in low:
            return ln[:220]
    for ln in lines:
        low = ln.lower()
        if any(tok in low for tok in ("fix", "update", "change", "add", "remove", "refactor", "implement")):
            return ln[:220]
    return lines[0][:220]


def _structural_kv_tuple_key(focus_file: str) -> str:
    file_key = _norm_rel_path(focus_file or "")
    packed_v = str(getattr(config, "STRUCTURAL_PACKED_CONTEXT_VERSION", "1") or "1")
    rules_v = str(getattr(config, "STRUCTURAL_RULES_VERSION", "1") or "1")
    return f"file={file_key}|packed_v={packed_v}|rules_v={rules_v}"


def reset_structural_kv_cache(reason: str = "manual") -> None:
    global _STRUCTURAL_KV_EPOCH, _STRUCTURAL_KV_ACTIVE_KEY
    with _STRUCTURAL_KV_LOCK:
        _STRUCTURAL_KV_EPOCH += 1
        _STRUCTURAL_KV_ACTIVE_KEY = ""
        epoch = _STRUCTURAL_KV_EPOCH
    dbg(f"structural.kv_reset reason={reason} epoch={epoch}")


def _build_structural_cache_key_base(focus_file: str, request_text: str) -> str:
    del request_text
    tuple_key = _structural_kv_tuple_key(focus_file)
    global _STRUCTURAL_KV_EPOCH, _STRUCTURAL_KV_ACTIVE_KEY
    with _STRUCTURAL_KV_LOCK:
        if tuple_key != _STRUCTURAL_KV_ACTIVE_KEY:
            _STRUCTURAL_KV_EPOCH += 1
            _STRUCTURAL_KV_ACTIVE_KEY = tuple_key
            dbg(
                "structural.kv_reset reason=tuple_changed "
                f"epoch={_STRUCTURAL_KV_EPOCH} tuple={tuple_key}"
            )
        epoch = _STRUCTURAL_KV_EPOCH
    stable = (
        "structural_edit_kv_v2|"
        f"{tuple_key}|"
        f"epoch={epoch}|"
        "contract=minimal|"
        "one_symbol_per_run"
    )
    digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()
    return digest[:24]


def _set_smoke_meta(
    meta: Dict[str, object],
    attempted: bool,
    passed: Optional[bool],
    failures: Optional[List[Dict[str, str]]] = None,
) -> None:
    meta["smoke_attempted"] = bool(attempted)
    meta["smoke_passed"] = passed
    meta["smoke_failures"] = list(failures or [])


def _run_smoke_for_candidate(
    focus_file: str,
    candidate_content: str,
) -> Tuple[bool, Optional[bool], List[Dict[str, str]], str]:
    if not config.SMOKE_ENABLED:
        dbg("smoke.skipped reason=disabled")
        return False, None, [], ""

    root = get_root().resolve()
    plan = load_smoke_plan(root)
    if not plan:
        dbg(f"smoke.skipped reason=plan_missing file={SMOKE_PLAN_FILENAME}")
        return False, None, [], ""

    with tempfile.TemporaryDirectory(prefix="moonlet_smoke_") as tmpdir:
        tmp_root = Path(tmpdir).resolve()
        try:
            shutil.copytree(
                root,
                tmp_root,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", ".ruff_cache", ".venv"),
            )
        except Exception as exc:
            dbg(f"smoke.copy_workspace_fail reason={exc}")
            failure = [{"kind": "BUILD", "message": f"workspace copy failed: {exc}", "command": "", "snippet": ""}]
            result = SmokeResult(passed=False, failures=[])
            report = build_smoke_report(result).replace(
                "unknown smoke failure",
                f"workspace copy failed: {exc}",
            )
            return True, False, failure, report

        target = (tmp_root / focus_file).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(candidate_content or "", encoding="utf-8")

        run_result = run_smoke(plan, str(tmp_root))
        passed = bool(run_result.passed)
        failures = serialize_failures(run_result, max_items=3)
        report = build_smoke_report(run_result, max_lines=80)
        dbg(
            f"smoke.result attempted=1 passed={1 if passed else 0} "
            f"failures={len(failures)}"
        )
        return True, passed, failures, report


def _annotate_failure_meta(result: Dict[str, object], attempt_index: int, attempt_total: int, focus_file: str) -> None:
    meta = (result.get("meta") or {})
    if not isinstance(meta, dict):
        meta = {}
        result["meta"] = meta
    meta["attempt_index"] = attempt_index
    meta["attempt_total"] = attempt_total

    kind = str(meta.get("file_edit_failure_kind") or meta.get("fileblock_failure_kind") or "").strip()
    reason = str(meta.get("failure_reason") or "").strip()
    if not kind or kind == "none" or not reason:
        out_kind, out_reason = _extract_failure_from_output(str(result.get("output") or ""))
        if not kind or kind == "none":
            kind = out_kind
        if not reason:
            reason = out_reason
    if not kind:
        kind = "none"
    meta["failure_kind"] = kind
    meta["failure_reason"] = _normalize_failure_reason(reason)

    non_blocking = bool(meta.get("non_blocking_failure"))
    if kind != "none" and not non_blocking:
        state.append_failure_note(
            mode="file_edit",
            focus_file=focus_file,
            kind=kind,
            summary=meta["failure_reason"],
        )


def _candidate_fullfile_guard_reason(
    original_content: str,
    candidate_content: str,
    target_ext: str,
) -> str:
    """Return non-empty reason when candidate looks like a partial/collapsed rewrite."""
    old = (original_content or "")
    new = (candidate_content or "")
    if not old or not new:
        return ""

    old_lines = len(old.splitlines())
    new_lines = len(new.splitlines())
    c_like = target_ext in {"c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx"}

    # Catch common one-line/minified collapses for medium/large files.
    if old_lines >= 20 and new_lines <= max(8, old_lines // 4):
        return f"collapsed_output_lines old={old_lines} new={new_lines}"

    # For C-like files, dropping all includes is almost always a partial snippet.
    if c_like and re.search(r"(?m)^\s*#\s*include\b", old):
        if not re.search(r"(?m)^\s*#\s*include\b", new):
            return "missing_includes_from_candidate"
        # Hard guard: preserve original local header dependencies.
        old_local = {
            m.strip()
            for m in re.findall(r'(?m)^\s*#\s*include\s*"([^"\n]+)"', old)
            if m.strip()
        }
        if old_local:
            new_local = {
                m.strip()
                for m in re.findall(r'(?m)^\s*#\s*include\s*"([^"\n]+)"', new)
                if m.strip()
            }
            missing_local = sorted(x for x in old_local if x not in new_local)
            if missing_local:
                return "missing_required_local_header: " + ", ".join(missing_local[:4])

    return ""


def _parse_analysis_packet(analysis_packet: str) -> Dict[str, object]:
    if not analysis_packet:
        return {}
    try:
        parsed = json.loads(analysis_packet)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_analysis_for_prompt(
    user_text: str,
    focus_file: str,
    max_chars: int = 700,
) -> Tuple[str, Dict[str, object]]:
    """Build compact analysis JSON string + parsed dict for routing/prompts."""
    try:
        packet = build_compact_analysis(user_text, focus_file, max_chars=max_chars)
        compact = compact_analysis_json(packet, max_chars=max_chars)
        parsed = _parse_analysis_packet(compact)
        return compact, parsed
    except Exception:
        return "", {}


def _analysis_ban_symbols(policy: Dict[str, object]) -> List[str]:
    # ban_symbols is deprecated; keep interface stable but advisory-only.
    return []


def _analysis_touch_points(policy: Dict[str, object]) -> List[Dict[str, object]]:
    if not isinstance(policy, dict):
        return []
    raw = policy.get("touch_points")
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _analysis_has_local_refs(policy: Dict[str, object]) -> bool:
    if not isinstance(policy, dict):
        return False
    for src in policy.get("deps_sources") or []:
        if isinstance(src, dict) and str(src.get("source") or "") in {"include", "inline"}:
            return True
    return False


def _analysis_files_allowed(policy: Dict[str, object], focus_file: str) -> List[str]:
    out: List[str] = []
    seen = set()
    if isinstance(policy, dict):
        raw = policy.get("files_allowed")
        if isinstance(raw, list):
            for item in raw:
                p = _norm_rel_path(str(item or "").strip())
                if not p or p in seen:
                    continue
                seen.add(p)
                out.append(p)
    focus = _norm_rel_path(focus_file)
    if focus and focus not in seen:
        out.insert(0, focus)
    return out[:8]


def _analysis_symbols_allowed(policy: Dict[str, object]) -> List[str]:
    out: List[str] = []
    seen = set()
    if not isinstance(policy, dict):
        return out
    raw = policy.get("symbols_allowed")
    if isinstance(raw, list):
        for item in raw:
            s = str(item or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out[:12]


def _extract_symbol_spans_for_content(focus_file: str, content: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
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
            spans.append((name, start, end))
    except Exception:
        spans = []

    if spans:
        return spans[:40]

    lines = content.splitlines()
    n = len(lines)
    fn_pat = re.compile(
        r"^\s*(?:[A-Za-z_][A-Za-z0-9_\s\*:&<>\[\],]*\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{\s*$"
    )
    i = 1
    while i <= n:
        m = fn_pat.match(lines[i - 1])
        if not m or m.group(1) in {"if", "for", "while", "switch", "catch"}:
            i += 1
            continue
        name = m.group(1)
        depth = 0
        end = i
        j = i
        saw_open = False
        while j <= n:
            l = lines[j - 1]
            depth += l.count("{")
            depth -= l.count("}")
            if "{" in l:
                saw_open = True
            end = j
            if saw_open and depth <= 0 and j > i:
                break
            j += 1
        spans.append((name, i, end))
        i = end + 1
    return spans[:40]


def _changed_line_ranges_from_unified_diff(diff_text: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    if not diff_text:
        return ranges
    for ln in diff_text.splitlines():
        if not ln.startswith("@@"):
            continue
        m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
        if not m:
            continue
        start = int(m.group(1))
        span = int(m.group(2) or "1")
        end = start + max(1, span) - 1
        ranges.append((start, end))
    return ranges


def _symbols_touched_by_diff(
    focus_file: str,
    candidate_content: str,
    candidate_diff: str,
) -> List[str]:
    spans = _extract_symbol_spans_for_content(focus_file, candidate_content)
    changed = _changed_line_ranges_from_unified_diff(candidate_diff)
    if not spans or not changed:
        return []
    touched: List[str] = []
    seen = set()
    for name, s, e in spans:
        hit = False
        for cs, ce in changed:
            if s <= ce and cs <= e:
                hit = True
                break
        if hit and name not in seen:
            seen.add(name)
            touched.append(name)
    return touched


def _diagnostic_line_numbers(text: str) -> List[int]:
    out: List[int] = []
    if not text:
        return out
    # clang/gcc: file:line:col: error...
    for m in re.finditer(r":(\d+):\d+:\s*(?:error|warning):", text):
        out.append(int(m.group(1)))
    # generic L123 references
    for m in re.finditer(r"\bL(\d+)\b", text):
        out.append(int(m.group(1)))
    uniq: List[int] = []
    seen = set()
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq[:20]


def _diagnostic_symbol_hints(text: str) -> List[str]:
    out: List[str] = []
    if not text:
        return out
    pats = [
        r"use of undeclared identifier '([^']+)'",
        r"implicit declaration of function '([^']+)'",
        r"call to undeclared function '([^']+)'",
    ]
    seen = set()
    for pat in pats:
        for m in re.finditer(pat, text):
            name = m.group(1).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out[:12]


def _build_precheck_missing_from_diagnostics(text: str) -> Dict[str, List[str]]:
    if not text:
        return {"types": [], "idents": [], "funcs": []}

    func_names = set()
    for pat in (
        r"implicit declaration of function '([^']+)'",
        r"call to undeclared function '([^']+)'",
    ):
        for m in re.finditer(pat, text):
            name = m.group(1).strip()
            if name:
                func_names.add(name)

    undeclared = set()
    for m in re.finditer(r"use of undeclared identifier '([^']+)'", text):
        name = m.group(1).strip()
        if name:
            undeclared.add(name)
    for m in re.finditer(r"unknown type name '([^']+)'", text):
        name = m.group(1).strip()
        if name:
            undeclared.add(name)

    type_names = set()
    ident_names = set()
    for name in undeclared:
        if name in func_names:
            continue
        if name.endswith("_t") or (name and name[0].isupper()):
            type_names.add(name)
        else:
            ident_names.add(name)

    return {
        "types": sorted(type_names)[:16],
        "idents": sorted(ident_names)[:16],
        "funcs": sorted(func_names)[:16],
    }


def _merge_precheck_missing(
    primary: Optional[Dict[str, List[str]]],
    secondary: Optional[Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"types": [], "idents": [], "funcs": []}
    for key in ("types", "idents", "funcs"):
        seen = set()
        for src in (primary, secondary):
            if not isinstance(src, dict):
                continue
            vals = src.get(key) or []
            if not isinstance(vals, list):
                continue
            for item in vals:
                s = str(item or "").strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                out[key].append(s)
    return out


def _run_precheck_missing_symbols(
    focus_file: str,
    baseline_content: str,
    target_ext: str,
) -> Dict[str, List[str]]:
    c_like = target_ext in {"c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx"}
    if not c_like:
        return {"types": [], "idents": [], "funcs": []}
    abs_path = get_root() / focus_file
    disk_content = read_single_file_for_context(focus_file).get(focus_file, "")
    if target_ext in {"cc", "cpp", "cxx", "hpp", "hh", "hxx"}:
        cmd = ["c++", "-fsyntax-only", "-std=c++17", "-Wall", focus_file]
    else:
        cmd = ["cc", "-fsyntax-only", "-std=c11", "-Wall", focus_file]
    try:
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(baseline_content or "", encoding="utf-8")
        res = subprocess.run(
            cmd,
            cwd=str(get_root()),
            capture_output=True,
            text=True,
            timeout=20,
        )
        if res.returncode == 0:
            return {"types": [], "idents": [], "funcs": []}
        diag = ((res.stderr or "") + "\n" + (res.stdout or "")).strip()
        missing = _build_precheck_missing_from_diagnostics(diag)
        total = len(missing.get("types") or []) + len(missing.get("idents") or []) + len(missing.get("funcs") or [])
        dbg(f"precheck_missing_detected={total}")
        if total:
            dbg_dump("precheck_diagnostics", diag[:1200])
        return missing
    except Exception as exc:
        dbg(f"precheck_missing: skipped ({exc})")
        return {"types": [], "idents": [], "funcs": []}
    finally:
        try:
            abs_path.write_text(disk_content, encoding="utf-8")
        except Exception:
            pass


def _expand_allowed_symbols_from_diagnostics(
    focus_file: str,
    candidate_content: str,
    diagnostics_text: str,
) -> List[str]:
    spans = _extract_symbol_spans_for_content(focus_file, candidate_content)
    if not spans:
        return []
    line_hits = _diagnostic_line_numbers(diagnostics_text)
    hint_names = _diagnostic_symbol_hints(diagnostics_text)
    out: List[str] = []
    seen = set()
    for name, s, e in spans:
        if any(s <= ln <= e for ln in line_hits):
            if name not in seen:
                seen.add(name)
                out.append(name)
    for hint in hint_names:
        for name, _s, _e in spans:
            if hint == name and name not in seen:
                seen.add(name)
                out.append(name)
    return out[:8]


def _validate_allowlisted_symbol_edits(
    focus_file: str,
    candidate_content: str,
    candidate_diff: str,
    analysis_policy: Dict[str, object],
    extra_symbols: Optional[List[str]] = None,
) -> Optional[str]:
    if not config.ENFORCE_ANALYSIS_ALLOWLIST:
        return None
    files_allowed = _analysis_files_allowed(analysis_policy, focus_file)
    focus_norm = _norm_rel_path(focus_file)
    if files_allowed and focus_norm not in files_allowed:
        return f"ALLOWLIST_VIOLATION: file `{focus_norm}` not in files_allowed"
    allowed_syms = set(_analysis_symbols_allowed(analysis_policy))
    for s in (extra_symbols or []):
        if s:
            allowed_syms.add(str(s))
    if not allowed_syms:
        return None
    touched = _symbols_touched_by_diff(focus_file, candidate_content, candidate_diff)
    if not touched:
        return None
    outside = [n for n in touched if n not in allowed_syms]
    if outside:
        return (
            "ALLOWLIST_VIOLATION: touched symbols outside allowlist: "
            + ", ".join(outside[:6])
        )
    return None


def _missing_task_card_symbol_touches(
    task_card_text: str,
    focus_file: str,
    original_content: str,
    candidate_content: str,
) -> List[str]:
    required = prompt_buffer._task_card_required_symbols(task_card_text)
    if not required:
        return []
    candidate_diff = _compute_unified_diff(original_content, candidate_content, focus_file)
    touched = set(_symbols_touched_by_diff(focus_file, candidate_content, candidate_diff))
    return [name for name in required if name not in touched]


def _build_planner_analysis_summary(policy: Dict[str, object], max_chars: int = 400) -> str:
    """Compact planner-safe summary (no bans/deps payload)."""
    if not isinstance(policy, dict) or not policy:
        return ""
    parts: List[str] = []
    summary = str(policy.get("task_summary") or "").strip()
    if summary:
        parts.append(f"summary: {summary[:160]}")
    points = _analysis_touch_points(policy)[:3]
    if points:
        names = []
        for p in points:
            name = str(p.get("symbol") or "").strip()
            if name and name != "<file>":
                names.append(name)
        if names:
            parts.append(f"touch_points: {', '.join(names[:3])}")
    constraints = policy.get("constraints") or policy.get("non_negotiables") or []
    if isinstance(constraints, list) and constraints:
        parts.append(f"constraints: {', '.join(str(x) for x in constraints[:2])}")
    out = "; ".join(parts).strip()
    return out[:max_chars]


def _build_execution_analysis_summary(policy: Dict[str, object], max_lines: int = 4) -> str:
    """Short advisory summary injected into fileblock/diff prompts."""
    if not isinstance(policy, dict) or not policy:
        return ""
    lines: List[str] = []
    summary = str(policy.get("task_summary") or "").strip()
    if summary:
        lines.append(f"summary={summary[:140]}")
    points = _analysis_touch_points(policy)
    if points:
        names = []
        for p in points[:3]:
            n = str(p.get("symbol") or "").strip()
            if n and n != "<file>":
                names.append(n)
        if names:
            lines.append("touch_points=" + ", ".join(names))
    hazards = policy.get("hazards")
    if isinstance(hazards, list):
        hz = [str(h).strip() for h in hazards if str(h).strip()]
        if hz:
            lines.append("hazards=" + "; ".join(hz[:2]))
    build_system = str(policy.get("likely_build_system") or "").strip()
    if build_system:
        lines.append(f"build_system={build_system}")
    validate_cmd = str(policy.get("suggested_validate_cmd") or "").strip()
    if validate_cmd:
        lines.append(f"validate_cmd={validate_cmd[:120]}")
    if not lines:
        return ""
    return "\n".join(lines[:max_lines])


def _build_singlefile_request(
    user_text: str,
    focus_file: str,
    file_content: str,
) -> Dict[str, object]:
    """Build one stable request bundle for single-file execution.

    Task-card flow replaces request slicing in single-file agent mode.
    """
    raw_request = (user_text or "").strip()
    token_est = _estimated_token_count(raw_request)
    if not _should_activate_task_card(raw_request):
        analysis_packet, analysis_policy = _build_analysis_for_prompt(
            raw_request,
            focus_file,
            max_chars=700,
        )
        analysis_summary = _build_execution_analysis_summary(analysis_policy, max_lines=4)
        return {
            "sliced_request": raw_request,
            "task_card": "",
            "analysis_packet": analysis_packet,
            "analysis_policy": analysis_policy,
            "analysis_summary": analysis_summary,
            "context_mode": "focused",
            "context_reason": "below_task_card_token_threshold",
            "task_card_enabled": False,
            "request_token_estimate": token_est,
        }

    task_card = build_task_card(
        focus_file,
        raw_request,
        file_content or "",
        max_lines=15,
    )
    dbg(f"task_card_generated file={focus_file} len={len(task_card)}")
    dbg_dump("task_card", task_card)
    analysis_packet, analysis_policy = _build_analysis_for_prompt(
        task_card,
        focus_file,
        max_chars=700,
    )
    analysis_summary = _build_execution_analysis_summary(analysis_policy, max_lines=4)

    return {
        "sliced_request": task_card,
        "task_card": task_card,
        "analysis_packet": analysis_packet,
        "analysis_policy": analysis_policy,
        "analysis_summary": analysis_summary,
        "context_mode": "full",
        "context_reason": "task_card_full_file",
        "task_card_enabled": True,
        "request_token_estimate": token_est,
    }


def _run_diff_write(
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str = "",
    sliced_request: str = "",
    focus_content_override: Optional[str] = None,
    diagnostics_hint: str = "",
    hard_fail_on_verify_fail: bool = False,
    force_stage_on_verify_fail: bool = False,
) -> Dict[str, object]:
    start = time.time()
    trace: List[Dict[str, object]] = []
    model_calls_used = 0
    target_ext = _ext(focus_file)
    original_content = focus_content_override
    if original_content is None:
        original_content = read_single_file_for_context(focus_file).get(focus_file, "")
    analysis_policy = _parse_analysis_packet(analysis_packet)
    preferred_validate_cmd = str(analysis_policy.get("suggested_validate_cmd") or "").strip()
    request_text = (sliced_request or user_text or "").strip()
    if diagnostics_hint:
        request_text += (
            "\n\nRETRY DIAGNOSTICS:\n"
            f"{diagnostics_hint}\n"
            "Fix these issues with minimal edits."
        )

    def _call(prompt: str, span: str) -> str:
        nonlocal model_calls_used
        model_calls_used += 1
        with _span(trace, span):
            return stream_reply(
                prompt,
                silent=silent,
                temperature=TEMP_EXECUTE,
                max_new=config.DIFF_MAX_NEW,
            )

    prompt, ctx_meta = prompt_buffer.build_prompt(
        "agent",
        request_text,
        focus_file=focus_file,
        full_context=full_context,
        output_contract="diff",
        error_message=diagnostics_hint,
        pre_sliced_request=request_text,
        focus_content_override=focus_content_override,
        analysis_packet=analysis_packet,
    )
    output = _call(prompt, "diff_attempt_1")
    _append_history(user_text, output)
    dbg_dump("diff_output_1", output)

    # Fast-path timeout handling: avoid misclassifying as patch format errors.
    if str(output).strip() == "[Model timeout]":
        meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["reject_rule"] = "MODEL_TIMEOUT"
        meta["file_edit_failure_kind"] = "timeout"
        meta["failure_kind"] = "timeout"
        meta["failure_reason"] = f"model timed out after {config.GEN_TIMEOUT}s"
        return {
            "output": "[File edit failed: MODEL_TIMEOUT]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": True,
            "candidate_content": "",
            "candidate_diff": "",
        }

    intent = classify_intent(request_text, original_content, focus_file)
    hunks, err = validate_patch_protocol_artifact(
        output=output,
        focus_file=focus_file,
        existing_content=original_content,
        user_text=request_text,
        intent=intent,
        prompt_prefill="",
        retrieved_ranges=None,
    )
    if err or not hunks:
        dbg(f"patch_protocol.reject_reason={err or 'PARSE_FAIL'}")
        meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["reject_rule"] = err or "PARSE_FAIL"
        kind = _patch_failure_kind_from_rule(err or "PARSE_FAIL")
        meta["file_edit_failure_kind"] = kind
        meta["failure_kind"] = kind
        meta["failure_reason"] = str(err or "patch validation failed")
        return {
            "output": f"[File edit failed: {err or 'PARSE_FAIL'}]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": output == "[Model timeout]",
            "candidate_content": "",
            "candidate_diff": "",
        }

    dbg(f"patch_protocol.files_count=1")
    dbg(f"patch_protocol.hunks_count={len(hunks)}")
    ok, reason, candidate_content, candidate_diff = _sandbox_apply_diff(
        focus_file=focus_file,
        original_content=original_content,
        hunks=hunks,
        preferred_validate_cmd=preferred_validate_cmd,
    )
    warning_kind = ""
    warning_reason = ""
    if not ok:
        # Non-blocking verify policy: keep candidate so user can inspect/decide.
        # Only hard-fail when patch could not be applied in sandbox.
        if not candidate_content:
            meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["reject_rule"] = "patch_apply_or_verify"
            meta["file_edit_failure_kind"] = "syntax"
            meta["failure_kind"] = "syntax"
            meta["failure_reason"] = _normalize_failure_reason(reason, max_chars=400)
            meta["apply_diagnostics"] = str(reason or "")[:4000]
            if candidate_diff:
                meta["diff"] = candidate_diff
            return {
                "output": f"[File edit failed: SYNTAX_FAIL: {reason[:500]}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": "",
                "candidate_diff": candidate_diff or "",
            }
        if hard_fail_on_verify_fail:
            meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["reject_rule"] = "compile_or_verify_fail"
            meta["file_edit_failure_kind"] = "syntax"
            meta["failure_kind"] = "syntax"
            meta["failure_reason"] = _normalize_failure_reason(reason, max_chars=400)
            meta["compile_report"] = _build_compile_report(reason)
            meta["diff"] = candidate_diff
            return {
                "output": f"[File edit failed: SYNTAX_FAIL: {reason[:500]}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": candidate_content,
                "candidate_diff": candidate_diff,
            }
        dbg("verify.non_blocking_fail")
        warning_kind = "syntax"
        warning_reason = _normalize_failure_reason(reason, max_chars=400)

    collapse_reason = _candidate_fullfile_guard_reason(
        original_content or "",
        candidate_content or "",
        target_ext,
    )
    if collapse_reason:
        meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["reject_rule"] = "wipe_risk"
        meta["file_edit_failure_kind"] = "rewrite_risk"
        meta["failure_kind"] = "rewrite_risk"
        meta["failure_reason"] = _normalize_failure_reason(collapse_reason, max_chars=400)
        if candidate_diff:
            meta["diff"] = candidate_diff
        return {
            "output": f"[File edit failed: WIPE_RISK: {collapse_reason}]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": candidate_content,
            "candidate_diff": candidate_diff,
        }

    semantic_err = _validate_prompt_semantics(request_text, candidate_content)
    if semantic_err:
        dbg("verify.semantic_non_blocking_fail")
        if hard_fail_on_verify_fail:
            meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["reject_rule"] = "semantic_fail"
            meta["file_edit_failure_kind"] = "semantic"
            meta["failure_kind"] = "semantic"
            meta["failure_reason"] = _normalize_failure_reason(semantic_err, max_chars=400)
            meta["diff"] = candidate_diff
            return {
                "output": f"[File edit failed: SEMANTIC_FAIL: {semantic_err[:500]}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": candidate_content,
                "candidate_diff": candidate_diff,
            }
        if not warning_kind:
            warning_kind = "semantic"
        sem_reason = _normalize_failure_reason(semantic_err, max_chars=400)
        warning_reason = (
            f"{warning_reason}; {sem_reason}" if warning_reason else sem_reason
        )

    meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
    meta.update(ctx_meta)
    meta["mode_used"] = Route.FILE_EDIT.value
    meta["route"] = Route.FILE_EDIT.value
    meta["diff"] = candidate_diff
    meta["syntax_error_count"] = 0
    if warning_kind:
        meta["verify_warning"] = True
        meta["non_blocking_failure"] = True
        meta["failure_kind"] = warning_kind
        meta["file_edit_failure_kind"] = warning_kind
        meta["failure_reason"] = warning_reason
    force_stage = bool(force_stage_on_verify_fail and warning_kind)
    if force_stage:
        meta["requires_user_decision"] = True
    if config.STAGE_EDITS or force_stage:
        meta["staged"] = True
        meta["staged_file"] = focus_file
        meta["staged_content"] = candidate_content
        return {
            "output": "[Staged file_edit]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": candidate_content,
            "candidate_diff": candidate_diff,
        }

    try:
        apply_unified_diff(focus_file, hunks)
    except Exception as exc:
        meta["file_edit_failure_kind"] = "syntax"
        meta["failure_kind"] = "syntax"
        meta["failure_reason"] = _normalize_failure_reason(str(exc), max_chars=300)
        return {
            "output": f"[File edit failed: PATCH_APPLY_FAIL: {exc}]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": "",
            "candidate_diff": candidate_diff or "",
        }
    return {
        "output": "[Applied file_edit]",
        "meta": meta,
        "status_prefix": "",
        "blocks_count": 0,
        "retried": False,
        "timeout": False,
        "candidate_content": candidate_content,
        "candidate_diff": candidate_diff,
    }


def _run_diff_edit(
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str = "",
    sliced_request: str = "",
) -> Dict[str, object]:
    attempt_total = 2
    baseline_content = read_single_file_for_context(focus_file).get(focus_file, "")
    first = _run_diff_write(
        user_text=user_text,
        focus_file=focus_file,
        silent=silent,
        full_context=full_context,
        analysis_packet=analysis_packet,
        sliced_request=sliced_request,
        focus_content_override=baseline_content,
        diagnostics_hint="",
        hard_fail_on_verify_fail=True,
        force_stage_on_verify_fail=False,
    )
    _annotate_failure_meta(first, attempt_index=1, attempt_total=attempt_total, focus_file=focus_file)
    first_meta = first.get("meta", {}) or {}
    first_meta["edit_mode"] = "diff"
    retry_trigger = "none"
    retry_hint = ""
    prefer_task_card_retry = False
    prefer_task_card_retry = False
    first_success = str(first.get("output", "")).startswith("[Applied file_edit]") or str(first.get("output", "")).startswith("[Staged file_edit]")
    if first_success:
        attempted, passed, failures, smoke_report = _run_smoke_for_candidate(
            focus_file=focus_file,
            candidate_content=str(first.get("candidate_content") or ""),
        )
        _set_smoke_meta(first_meta, attempted=attempted, passed=passed, failures=failures)
        first_meta["retry_trigger"] = "none"
        if not attempted or bool(passed):
            dbg("retry.success attempt=1")
            return first
        retry_trigger = "smoke"
        retry_hint = smoke_report or "SMOKE_REPORT:\n- failure: smoke test failed"
        dbg("retry.invoked attempt=2 reason=smoke_fail")
    else:
        _set_smoke_meta(first_meta, attempted=False, passed=None, failures=[])
        fail_reason = str(first_meta.get("failure_reason") or "").strip()
        fail_kind = str(first_meta.get("failure_kind") or "").strip()
        reject_rule = str(first_meta.get("reject_rule") or "").strip()
        raw_apply_diag = str(first_meta.get("apply_diagnostics") or "").strip()
        first_timed_out = bool(first_meta.get("timeout"))
        if first_timed_out:
            dbg("retry.skipped reason=timeout_first_attempt")
            return first
        non_retry_rules = {
            "HUNK_MULTI_FUNCTION",
            "HUNK_OUTSIDE_FUNCTION",
            "MODEL_TIMEOUT",
        }
        if reject_rule in non_retry_rules:
            dbg(f"retry.skipped reason=non_retry_rule:{reject_rule}")
            return first
        if not fail_reason:
            fail_kind, fail_reason = _extract_failure_from_output(str(first.get("output") or ""))
        if fail_kind in {"syntax", "semantic"}:
            retry_hint = _build_compile_report(raw_apply_diag or fail_reason)
            retry_trigger = "compile"
        elif fail_kind == "noop":
            retry_hint = _build_noop_report()
            retry_trigger = "noop"
            prefer_task_card_retry = True
        else:
            retry_hint = raw_apply_diag or f"{fail_kind or 'failure'}: {fail_reason or 'unknown'}".strip()
            retry_trigger = "none"
        dbg("retry.invoked attempt=2")

    first_meta["retry_trigger"] = retry_trigger
    retry_request = _build_retry_base_request(
        sliced_request=sliced_request,
        user_text=user_text,
        focus_file=focus_file,
        file_content=baseline_content,
        prefer_task_card=prefer_task_card_retry,
    )
    retry_request += (
        "\n\nRETRY RULES:\n"
        "- Return ONLY a unified diff.\n"
        "- Use one hunk per function, and do not let one hunk span multiple functions.\n"
        "- If multiple functions need edits, output multiple hunks sequentially.\n"
        "- Keep changes minimal and anchored to provided context.\n"
        "- If apply failed, regenerate the failing hunk(s) so context matches RETRY DIAGNOSTICS exactly.\n"
        "- Fix the reported failure precisely.\n"
        "- If diagnostics indicate no-op, make at least one concrete change to target file.\n"
    )
    second = _run_diff_write(
        user_text=user_text,
        focus_file=focus_file,
        silent=silent,
        full_context=full_context,
        analysis_packet=analysis_packet,
        sliced_request=retry_request,
        focus_content_override=baseline_content,
        diagnostics_hint=retry_hint,
        hard_fail_on_verify_fail=False,
        force_stage_on_verify_fail=True,
    )
    _annotate_failure_meta(second, attempt_index=2, attempt_total=attempt_total, focus_file=focus_file)
    second_meta = second.get("meta", {}) or {}
    second_meta["edit_mode"] = "diff"
    _set_smoke_meta(second_meta, attempted=False, passed=None, failures=[])
    second_meta["retry_trigger"] = retry_trigger
    if str(second.get("output", "")).startswith("[Applied file_edit]") or str(second.get("output", "")).startswith("[Staged file_edit]"):
        if bool(second_meta.get("non_blocking_failure")):
            second_meta["requires_user_decision"] = True
        dbg("retry.success attempt=2")
        return second

    dbg("retry.failed attempt=2")
    return second


def _fileblock_runtime() -> FileblockRuntime:
    return FileblockRuntime(
        temp_execute=TEMP_EXECUTE,
        parse_analysis_packet=_parse_analysis_packet,
        append_history=_append_history,
        span=_span,
        build_meta=_build_meta,
        compute_unified_diff=_compute_unified_diff,
        sandbox_validate_candidate_content=_sandbox_validate_candidate_content,
        normalize_failure_reason=_normalize_failure_reason,
        build_compile_report=_build_compile_report,
        candidate_fullfile_guard_reason=_candidate_fullfile_guard_reason,
        extract_failure_from_output=_extract_failure_from_output,
        build_retry_base_request=_build_retry_base_request,
        build_noop_report=_build_noop_report,
        set_smoke_meta=_set_smoke_meta,
        run_smoke_for_candidate=_run_smoke_for_candidate,
        annotate_failure_meta=_annotate_failure_meta,
    )


def _run_fileblock_write(
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str = "",
    sliced_request: str = "",
    focus_content_override: Optional[str] = None,
    diagnostics_hint: str = "",
    hard_fail_on_verify_fail: bool = False,
    force_stage_on_verify_fail: bool = False,
    defer_persist: bool = False,
) -> Dict[str, object]:
    return run_fileblock_write(
        runtime=_fileblock_runtime(),
        user_text=user_text,
        focus_file=focus_file,
        silent=silent,
        full_context=full_context,
        analysis_packet=analysis_packet,
        sliced_request=sliced_request,
        focus_content_override=focus_content_override,
        diagnostics_hint=diagnostics_hint,
        hard_fail_on_verify_fail=hard_fail_on_verify_fail,
        force_stage_on_verify_fail=force_stage_on_verify_fail,
        defer_persist=defer_persist,
    )


def _run_fileblock_edit(
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str = "",
    sliced_request: str = "",
) -> Dict[str, object]:
    return run_fileblock_edit(
        runtime=_fileblock_runtime(),
        user_text=user_text,
        focus_file=focus_file,
        silent=silent,
        full_context=full_context,
        analysis_packet=analysis_packet,
        sliced_request=sliced_request,
    )


def _run_structural_write(
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str = "",
    sliced_request: str = "",
    focus_content_override: Optional[str] = None,
    diagnostics_hint: str = "",
    hard_fail_on_verify_fail: bool = False,
    force_stage_on_verify_fail: bool = False,
    forced_symbol_name: str = "",
    forbidden_symbol_names: Optional[List[str]] = None,
    defer_persist: bool = False,
    execute_temp: Optional[float] = None,
    cache_key_base: str = "",
) -> Dict[str, object]:
    start = time.time()
    trace: List[Dict[str, object]] = []
    model_calls_used = 0
    target_ext = _ext(focus_file)
    allow_invalid_symbol_apply = bool(
        getattr(config, "STRUCTURAL_APPLY_ON_INVALID_SYMBOL", True)
    )
    normalizer_enabled = bool(getattr(config, "NORMALIZER_ENABLED", False))
    optional_verify_enabled = bool(
        getattr(config, "STRUCTURAL_OPTIONAL_VERIFY_ENABLED", False)
    )
    optional_semantic_enabled = bool(
        getattr(config, "STRUCTURAL_OPTIONAL_SEMANTIC_ENABLED", False)
    )
    dbg(f"normalizer.enabled={1 if normalizer_enabled else 0}")
    structural_temp = TEMP_EXECUTE if execute_temp is None else float(execute_temp)
    normalizer_meta: Dict[str, object] = {}
    if normalizer_enabled:
        normalizer_meta = {
            "normalizer_version": "normalizer",
            "normalizer_confidence": "red",
            "normalizer_repairs": [],
            "normalizer_stage": "",
            "normalizer_error_code": "",
        }

    def _attach_normalizer_meta(meta_obj: Dict[str, object]) -> None:
        if normalizer_meta:
            meta_obj.update(normalizer_meta)
    original_content = focus_content_override
    if original_content is None:
        original_content = read_single_file_for_context(focus_file).get(focus_file, "")
    analysis_policy = _parse_analysis_packet(analysis_packet)
    preferred_validate_cmd = str(analysis_policy.get("suggested_validate_cmd") or "").strip()
    request_text = (sliced_request or user_text or "").strip()
    if diagnostics_hint:
        request_text += (
            "\n\nRETRY DIAGNOSTICS:\n"
            f"{diagnostics_hint}\n"
            "Fix these issues with minimal edits."
        )

    forced_name = str(forced_symbol_name or "").strip()
    target = None
    decision_reason = ""
    if forced_name:
        for sym in build_symbol_index(focus_file, original_content or ""):
            if sym.name.lower() == forced_name.lower():
                target = sym
                break
        if target is None:
            decision_reason = "forced_symbol_not_found"
    else:
        decision = select_target_symbol(
            user_text=request_text,
            focus_file=focus_file,
            content=original_content or "",
            analysis_packet=analysis_packet or "",
        )
        if decision.eligible and decision.target:
            target = decision.target
        decision_reason = decision.reason

    if target is None:
        dbg(f"structural.reject_reason={decision_reason or 'ineligible'}")
        meta = _build_meta(0, "", 0, False, start, trace, model_calls_used)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["edit_mode"] = "structural"
        meta["reject_rule"] = "STRUCTURAL_INELIGIBLE"
        meta["failure_kind"] = "structural"
        meta["file_edit_failure_kind"] = "structural"
        meta["failure_reason"] = decision_reason or "structural target not identified"
        meta["structural_fallback_reason"] = decision_reason or "structural target not identified"
        _attach_normalizer_meta(meta)
        return {
            "output": f"[File edit failed: STRUCTURAL_INELIGIBLE: {decision_reason or 'target not identified'}]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": "",
            "candidate_diff": "",
        }

    target_symbol = f"{target.kind} {target.name}".strip()
    snippet = extract_target_snippet(original_content or "", target, padding_lines=1)
    if not snippet:
        dbg("structural.reject_reason=target_snippet_empty")
        meta = _build_meta(0, "", 0, False, start, trace, model_calls_used)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["edit_mode"] = "structural"
        meta["reject_rule"] = "STRUCTURAL_SNIPPET_EMPTY"
        meta["failure_kind"] = "structural"
        meta["file_edit_failure_kind"] = "structural"
        meta["failure_reason"] = "target snippet empty"
        meta["structural_target_symbol"] = target_symbol
        meta["structural_fallback_reason"] = "target snippet empty"
        _attach_normalizer_meta(meta)
        return {
            "output": "[File edit failed: STRUCTURAL_SNIPPET_EMPTY]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": "",
            "candidate_diff": "",
        }

    original_symbol_text = ""
    try:
        original_bytes = (original_content or "").encode("utf-8")
        _s = max(0, int(target.byte_start or 0))
        _e = max(_s, int(target.byte_end or _s))
        if _e > len(original_bytes):
            _e = len(original_bytes)
        original_symbol_text = original_bytes[_s:_e].decode("utf-8", errors="replace")
    except Exception:
        original_symbol_text = ""
    original_signature_line = ""
    for _ln in (original_symbol_text or "").splitlines():
        if _ln.strip():
            original_signature_line = _ln.strip()
            break

    symbol_index = build_symbol_index(focus_file, original_content or "")
    packed_context = build_packed_context(
        target=target,
        content=original_content or "",
        index=symbol_index,
        user_text=request_text,
        focus_file=focus_file,
        max_lines=max(20, int(getattr(config, "STRUCTURAL_PACKED_MAX_LINES", 200))),
        max_bytes=max(512, int(getattr(config, "STRUCTURAL_PACKED_MAX_BYTES", 8192))),
    )
    if packed_context:
        packed_lines = len(packed_context.splitlines())
        packed_bytes = len(packed_context.encode("utf-8"))
        dbg(
            "structural.packed_context "
            f"lines={packed_lines} bytes={packed_bytes} "
            f"target={target.kind}:{target.name}"
        )

    effective_cache_base = str(cache_key_base or "").strip()
    if not effective_cache_base:
        effective_cache_base = _build_structural_cache_key_base(
            focus_file=focus_file,
            request_text=request_text,
        )
    structural_cache_key = ""
    if bool(getattr(config, "STRUCTURAL_KV_CACHE_ENABLED", True)):
        structural_cache_key = f"struct:{effective_cache_base}"
        dbg(f"structural.cache_key={structural_cache_key}")

    compact_single_line_target = _looks_single_line_compact_symbol(original_symbol_text)
    if compact_single_line_target:
        dbg("structural.target_compact_single_line=1")
        request_text += (
            "\n\nFORMAT NOTE:\n"
            "- TARGET_SYMBOL is currently compressed to one line.\n"
            "- Normalize TARGET_SYMBOL to readable multi-line formatting while preserving behavior.\n"
            "- Then apply only the requested minimal fix."
        )

    def _coerce_structural_reply(raw_reply: str) -> str:
        text = (raw_reply or "").replace("\r\n", "\n").replace("\r", "\n")
        cut_idx = -1
        for marker in ("\nEND_SYMBOL", "END_SYMBOL"):
            idx = text.find(marker)
            if idx >= 0 and (cut_idx < 0 or idx < cut_idx):
                cut_idx = idx
        if cut_idx >= 0:
            text = text[:cut_idx]
        stripped = text.strip()
        if not stripped:
            return "BEGIN_SYMBOL\nEND_SYMBOL"
        if "BEGIN_SYMBOL" in stripped:
            if "END_SYMBOL" not in stripped:
                return f"{stripped.rstrip()}\nEND_SYMBOL"
            return stripped
        if "END_SYMBOL" in stripped:
            cleaned = stripped.replace("END_SYMBOL", "").strip()
            return f"BEGIN_SYMBOL\n{cleaned}\nEND_SYMBOL"
        return f"BEGIN_SYMBOL\n{stripped}\nEND_SYMBOL"

    structural_max_new = min(
        int(getattr(config, "STRUCTURAL_MAX_NEW", 256)),
        int(config.DIFF_MAX_NEW),
    )
    structural_stop_sequences = ["\nEND_SYMBOL"]
    dbg("structural.prefix_forced=1")
    dbg(f"structural.stop_count={len(structural_stop_sequences)}")
    dbg(f"stream_reply: structural max_new={structural_max_new}")

    def _call(prompt: str, span: str) -> str:
        nonlocal model_calls_used
        model_calls_used += 1
        with _span(trace, span):
            forced_prompt = f"{prompt.rstrip()}\nBEGIN_SYMBOL\n"
            reply = stream_reply(
                forced_prompt,
                silent=silent,
                temperature=structural_temp,
                max_new=structural_max_new,
                stop_sequences=structural_stop_sequences,
                cache_key=structural_cache_key,
            )
            return _coerce_structural_reply(reply)

    prompt, ctx_meta = prompt_buffer.build_prompt(
        "agent",
        request_text,
        focus_file=focus_file,
        full_context=full_context,
        output_contract="structural",
        structural_target=target.name,
        structural_kind=target.kind,
        structural_line_start=target.line_start,
        structural_line_end=target.line_end,
        structural_byte_start=target.byte_start,
        structural_byte_end=target.byte_end,
        structural_original_symbol=original_symbol_text,
        structural_signature_line=original_signature_line,
        error_message=diagnostics_hint,
        pre_sliced_request=request_text,
        focus_content_override=focus_content_override,
        context_override=packed_context,
        analysis_packet=analysis_packet,
    )
    ctx_meta["structural_prefix_forced"] = True
    ctx_meta["structural_stop_count"] = len(structural_stop_sequences)
    ctx_meta["structural_max_new"] = structural_max_new
    if packed_context:
        ctx_meta["structural_packed_context"] = True
        ctx_meta["structural_packed_context_lines"] = len(packed_context.splitlines())
        ctx_meta["structural_packed_context_bytes"] = len(packed_context.encode("utf-8"))
    if structural_cache_key:
        ctx_meta["structural_cache_key"] = structural_cache_key
    if compact_single_line_target:
        ctx_meta["structural_target_was_single_line_compact"] = True
    output = _call(prompt, "structural_attempt_1")
    _append_history(user_text, output)
    dbg_dump("structural_output_1", output)

    if str(output).strip() == "[Model timeout]":
        meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["edit_mode"] = "structural"
        meta["reject_rule"] = "MODEL_TIMEOUT"
        meta["file_edit_failure_kind"] = "timeout"
        meta["failure_kind"] = "timeout"
        meta["failure_reason"] = f"model timed out after {config.GEN_TIMEOUT}s"
        meta["structural_target_symbol"] = target_symbol
        _attach_normalizer_meta(meta)
        return {
            "output": "[File edit failed: MODEL_TIMEOUT]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": True,
            "candidate_content": "",
            "candidate_diff": "",
        }

    if normalizer_enabled:
        norm = normalize_symbol(
            raw_output=output,
            focus_file=focus_file,
            target_name=target.name,
            target_kind=target.kind,
            original_symbol_text=original_symbol_text,
            request_text=request_text,
        )
        normalizer_meta["normalizer_confidence"] = norm.confidence
        normalizer_meta["normalizer_repairs"] = list(norm.repairs or [])
        normalizer_meta["normalizer_stage"] = norm.stage
        normalizer_meta["normalizer_error_code"] = norm.error_code
        replacement_text = norm.text
        norm_err = norm.error_code
        soft_format_codes = {
            "norm_c_parse_failed",
            "norm_d_risky_repair_required",
            "norm_e_reparse_failed",
        }
        if norm_err in soft_format_codes:
            legacy_text, legacy_err = normalize_structural_output(
                output,
                target_symbol=target.name,
                original_symbol_text=original_symbol_text,
                target_kind=target.kind,
                focus_file=focus_file,
            )
            if not legacy_err and legacy_text:
                replacement_text = legacy_text
                norm_err = ""
                normalizer_meta["normalizer_confidence"] = "yellow"
                normalizer_meta["normalizer_error_code"] = ""
                normalizer_meta["normalizer_repairs"] = list(norm.repairs or []) + [
                    "legacy_normalization_fallback"
                ]
                normalizer_meta["normalizer_stage"] = "E"
                dbg("normalizer.fallback=legacy")
        if norm_err or norm.confidence == "red":
            dbg(f"structural.reject_reason={norm_err or 'normalizer_red'}")
            meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["edit_mode"] = "structural"
            meta["reject_rule"] = "STRUCTURAL_OUTPUT_INVALID"
            meta["file_edit_failure_kind"] = "format"
            meta["failure_kind"] = "format"
            meta["failure_reason"] = norm_err or "normalizer_red"
            meta["structural_target_symbol"] = target_symbol
            meta["structural_fallback_reason"] = norm_err or "normalizer_red"
            _attach_normalizer_meta(meta)
            return {
                "output": f"[File edit failed: STRUCTURAL_OUTPUT_INVALID: {norm_err or 'normalizer_red'}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": "",
                "candidate_diff": "",
            }
    else:
        replacement_text, norm_err = normalize_structural_output(
            output,
            target_symbol=target.name,
            original_symbol_text=original_symbol_text,
            target_kind=target.kind,
            focus_file=focus_file,
        )
    if norm_err:
        dbg(f"structural.reject_reason={norm_err}")
        meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["edit_mode"] = "structural"
        meta["reject_rule"] = "STRUCTURAL_OUTPUT_INVALID"
        meta["file_edit_failure_kind"] = "format"
        meta["failure_kind"] = "format"
        meta["failure_reason"] = norm_err
        meta["structural_target_symbol"] = target_symbol
        meta["structural_fallback_reason"] = norm_err
        _attach_normalizer_meta(meta)
        return {
            "output": f"[File edit failed: STRUCTURAL_OUTPUT_INVALID: {norm_err}]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": "",
            "candidate_diff": "",
        }

    allow_signature_change = _explicit_signature_change_requested(
        request_text,
        target_name=target.name,
    )

    unit_ok, unit_err = validate_replacement_symbol_unit(
        focus_file=focus_file,
        replacement_text=replacement_text,
        target_name=target.name,
        target_kind=target.kind,
        original_symbol_text=original_symbol_text,
        forbidden_symbol_names=list(forbidden_symbol_names or []),
        allow_signature_change=allow_signature_change,
        enforce_shape_guard=not normalizer_enabled,
        normalized_mode=normalizer_enabled,
    )
    warning_kind = ""
    warning_reason = ""
    unit_invalid_symbol = False
    if not unit_ok:
        if not allow_invalid_symbol_apply and not normalizer_enabled:
            dbg(f"structural.reject_reason={unit_err}")
            meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["edit_mode"] = "structural"
            meta["reject_rule"] = "STRUCTURAL_OUTPUT_INVALID"
            meta["file_edit_failure_kind"] = "format"
            meta["failure_kind"] = "format"
            meta["failure_reason"] = unit_err
            meta["structural_target_symbol"] = target_symbol
            meta["structural_fallback_reason"] = unit_err
            _attach_normalizer_meta(meta)
            return {
                "output": f"[File edit failed: STRUCTURAL_OUTPUT_INVALID: {unit_err}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": "",
                "candidate_diff": "",
            }
        unit_invalid_symbol = True
        if normalizer_enabled:
            dbg(f"structural.lenient_continue.normalizer reason={unit_err}")
        dbg(f"structural.lenient_continue reason={unit_err}")
        warning_kind = "format"
        warning_reason = _normalize_failure_reason(unit_err, max_chars=400)

    candidate_content = apply_symbol_replacement(
        original_content or "",
        target,
        replacement_text,
        focus_file=focus_file,
    )
    candidate_diff = _compute_unified_diff(
        original_content or "",
        candidate_content,
        focus_file,
    )
    no_content_change = (candidate_content == (original_content or "")) or (not (candidate_diff or "").strip())
    if no_content_change:
        dbg("structural.noop_change_detected")
        meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
        meta.update(ctx_meta)
        meta["mode_used"] = Route.FILE_EDIT.value
        meta["route"] = Route.FILE_EDIT.value
        meta["edit_mode"] = "structural"
        meta["structural_target_symbol"] = target_symbol
        meta["noop"] = True
        if normalizer_enabled:
            meta["already_up_to_date"] = True
            meta["applied_without_diff"] = True
            _attach_normalizer_meta(meta)
            return {
                "output": "[No-op file_edit: target already up to date]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": candidate_content,
                "candidate_diff": candidate_diff,
            }
        meta["reject_rule"] = "NO_EFFECTIVE_CHANGE"
        meta["failure_kind"] = "noop"
        meta["file_edit_failure_kind"] = "noop"
        meta["failure_reason"] = "model returned no effective file changes"
        meta["structural_fallback_reason"] = "no_effective_change"
        _attach_normalizer_meta(meta)
        return {
            "output": "[No-op file_edit: model returned no effective file changes]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": candidate_content,
            "candidate_diff": candidate_diff,
        }

    if unit_invalid_symbol and allow_invalid_symbol_apply:
        dbg("structural.validate_skipped reason=lenient_invalid_symbol")
    else:
        ok_struct, struct_err = validate_structural_candidate(
            focus_file=focus_file,
            original_content=original_content or "",
            candidate_content=candidate_content,
            target=target,
            allow_signature_change=allow_signature_change,
        )
        if not ok_struct:
            if normalizer_enabled:
                dbg(f"structural.lenient_continue.normalizer reason={struct_err}")
                if not warning_kind:
                    warning_kind = "structural"
                struct_reason = _normalize_failure_reason(struct_err, max_chars=400)
                warning_reason = (
                    f"{warning_reason}; {struct_reason}" if warning_reason else struct_reason
                )
            elif allow_invalid_symbol_apply and struct_err in {
                "candidate_parse_failed",
                "target_symbol_missing_after_replacement",
                "symbol_region_corrupted",
            }:
                dbg(f"structural.lenient_continue reason={struct_err}")
                if not warning_kind:
                    warning_kind = "structural"
                struct_reason = _normalize_failure_reason(struct_err, max_chars=400)
                warning_reason = (
                    f"{warning_reason}; {struct_reason}" if warning_reason else struct_reason
                )
            else:
                dbg(f"structural.reject_reason={struct_err}")
                meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
                meta.update(ctx_meta)
                meta["mode_used"] = Route.FILE_EDIT.value
                meta["route"] = Route.FILE_EDIT.value
                meta["edit_mode"] = "structural"
                meta["reject_rule"] = "STRUCTURAL_VALIDATE_FAIL"
                meta["failure_kind"] = "structural"
                meta["file_edit_failure_kind"] = "structural"
                meta["failure_reason"] = struct_err
                meta["diff"] = candidate_diff
                meta["structural_target_symbol"] = target_symbol
                meta["structural_fallback_reason"] = struct_err
                _attach_normalizer_meta(meta)
                return {
                    "output": f"[File edit failed: STRUCTURAL_VALIDATE_FAIL: {struct_err}]",
                    "meta": meta,
                    "status_prefix": "",
                    "blocks_count": 0,
                    "retried": False,
                    "timeout": False,
                    "candidate_content": candidate_content,
                    "candidate_diff": candidate_diff,
                }
    ok = True
    err = ""
    run_optional_verify = (not normalizer_enabled) or optional_verify_enabled
    if run_optional_verify:
        ok, err = _sandbox_validate_candidate_content(
            focus_file=focus_file,
            candidate_content=candidate_content,
            preferred_validate_cmd=preferred_validate_cmd,
        )
    else:
        dbg("verify.skipped reason=normalizer_optional_disabled")
    if not ok:
        if hard_fail_on_verify_fail and not normalizer_enabled:
            meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["edit_mode"] = "structural"
            meta["reject_rule"] = "compile_or_verify_fail"
            meta["file_edit_failure_kind"] = "syntax"
            meta["failure_kind"] = "syntax"
            meta["failure_reason"] = _normalize_failure_reason(err or "compile failed", max_chars=400)
            meta["compile_report"] = _build_compile_report(err or "compile failed")
            meta["diff"] = candidate_diff
            meta["structural_target_symbol"] = target_symbol
            meta["structural_fallback_reason"] = _normalize_failure_reason(err or "compile failed", max_chars=400)
            _attach_normalizer_meta(meta)
            return {
                "output": f"[File edit failed: SYNTAX_FAIL: {(err or 'compile failed')[:500]}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": candidate_content,
                "candidate_diff": candidate_diff,
            }
        dbg("verify.non_blocking_fail")
        warning_kind = "syntax"
        warning_reason = _normalize_failure_reason(err or "compile failed", max_chars=400)

    collapse_reason = _candidate_fullfile_guard_reason(
        original_content or "",
        candidate_content or "",
        target_ext,
    )
    if collapse_reason:
        if normalizer_enabled:
            dbg(f"structural.lenient_continue.normalizer reason={collapse_reason}")
            if not warning_kind:
                warning_kind = "rewrite_risk"
            collapse_norm = _normalize_failure_reason(collapse_reason, max_chars=400)
            warning_reason = f"{warning_reason}; {collapse_norm}" if warning_reason else collapse_norm
        else:
            meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["edit_mode"] = "structural"
            meta["reject_rule"] = "wipe_risk"
            meta["file_edit_failure_kind"] = "rewrite_risk"
            meta["failure_kind"] = "rewrite_risk"
            meta["failure_reason"] = _normalize_failure_reason(collapse_reason, max_chars=400)
            meta["structural_target_symbol"] = target_symbol
            meta["structural_fallback_reason"] = collapse_reason
            if candidate_diff:
                meta["diff"] = candidate_diff
            _attach_normalizer_meta(meta)
            return {
                "output": f"[File edit failed: WIPE_RISK: {collapse_reason}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": candidate_content,
                "candidate_diff": candidate_diff,
            }

    semantic_err = ""
    run_optional_semantic = (not normalizer_enabled) or optional_semantic_enabled
    if run_optional_semantic:
        semantic_err = _validate_prompt_semantics(request_text, candidate_content)
    else:
        dbg("verify.semantic_skipped reason=normalizer_optional_disabled")
    if semantic_err:
        dbg("verify.semantic_non_blocking_fail")
        if hard_fail_on_verify_fail and not normalizer_enabled:
            meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
            meta.update(ctx_meta)
            meta["mode_used"] = Route.FILE_EDIT.value
            meta["route"] = Route.FILE_EDIT.value
            meta["edit_mode"] = "structural"
            meta["reject_rule"] = "semantic_fail"
            meta["file_edit_failure_kind"] = "semantic"
            meta["failure_kind"] = "semantic"
            meta["failure_reason"] = _normalize_failure_reason(semantic_err, max_chars=400)
            meta["diff"] = candidate_diff
            meta["structural_target_symbol"] = target_symbol
            meta["structural_fallback_reason"] = _normalize_failure_reason(semantic_err, max_chars=400)
            _attach_normalizer_meta(meta)
            return {
                "output": f"[File edit failed: SEMANTIC_FAIL: {semantic_err[:500]}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
                "candidate_content": candidate_content,
                "candidate_diff": candidate_diff,
            }
        if not warning_kind:
            warning_kind = "semantic"
        sem_reason = _normalize_failure_reason(semantic_err, max_chars=400)
        warning_reason = f"{warning_reason}; {sem_reason}" if warning_reason else sem_reason

    meta = _build_meta(len(prompt), output, 0, False, start, trace, model_calls_used)
    meta.update(ctx_meta)
    meta["mode_used"] = Route.FILE_EDIT.value
    meta["route"] = Route.FILE_EDIT.value
    meta["edit_mode"] = "structural"
    meta["structural_target_symbol"] = target_symbol
    meta["diff"] = candidate_diff
    meta["syntax_error_count"] = 0
    _attach_normalizer_meta(meta)
    if warning_kind:
        meta["verify_warning"] = True
        meta["non_blocking_failure"] = True
        meta["failure_kind"] = warning_kind
        meta["file_edit_failure_kind"] = warning_kind
        meta["failure_reason"] = warning_reason

    if defer_persist:
        meta["defer_persist"] = True
        return {
            "output": "[Candidate file_edit]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": candidate_content,
            "candidate_diff": candidate_diff,
        }

    force_stage = bool(force_stage_on_verify_fail and warning_kind)
    if force_stage:
        meta["requires_user_decision"] = True
    if config.STAGE_EDITS or force_stage:
        meta["staged"] = True
        meta["staged_file"] = focus_file
        meta["staged_content"] = candidate_content
        return {
            "output": "[Staged file_edit]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
            "candidate_content": candidate_content,
            "candidate_diff": candidate_diff,
        }

    target_path = get_root() / focus_file
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(candidate_content, encoding="utf-8")
    return {
        "output": "[Applied file_edit]",
        "meta": meta,
        "status_prefix": "",
        "blocks_count": 0,
        "retried": False,
        "timeout": False,
        "candidate_content": candidate_content,
        "candidate_diff": candidate_diff,
    }


def _run_structural_edit(
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str = "",
    sliced_request: str = "",
    forced_symbol_name: str = "",
    forbidden_symbol_names: Optional[List[str]] = None,
    baseline_content_override: Optional[str] = None,
    defer_persist: bool = False,
    cache_key_base: str = "",
) -> Dict[str, object]:
    attempt_total = 2
    allow_invalid_symbol_apply = bool(
        getattr(config, "STRUCTURAL_APPLY_ON_INVALID_SYMBOL", True)
    )
    baseline_content = baseline_content_override
    if baseline_content is None:
        baseline_content = read_single_file_for_context(focus_file).get(focus_file, "")
    effective_cache_base = str(cache_key_base or "").strip()
    if not effective_cache_base:
        effective_cache_base = _build_structural_cache_key_base(
            focus_file=focus_file,
            request_text=(sliced_request or user_text or ""),
        )
    first = _run_structural_write(
        user_text=user_text,
        focus_file=focus_file,
        silent=silent,
        full_context=full_context,
        analysis_packet=analysis_packet,
        sliced_request=sliced_request,
        focus_content_override=baseline_content,
        diagnostics_hint="",
        hard_fail_on_verify_fail=not allow_invalid_symbol_apply,
        force_stage_on_verify_fail=allow_invalid_symbol_apply,
        forced_symbol_name=forced_symbol_name,
        forbidden_symbol_names=forbidden_symbol_names,
        defer_persist=defer_persist,
        cache_key_base=effective_cache_base,
    )
    _annotate_failure_meta(first, attempt_index=1, attempt_total=attempt_total, focus_file=focus_file)
    first_meta = first.get("meta", {}) or {}
    first_meta["edit_mode"] = "structural"
    retry_trigger = "none"
    retry_hint = ""
    retry_failure_reason = ""
    prefer_task_card_retry = False
    first_output = str(first.get("output", ""))
    first_success = (
        first_output.startswith("[Applied file_edit]")
        or first_output.startswith("[Staged file_edit]")
        or (defer_persist and first_output.startswith("[Candidate file_edit]"))
    )
    if first_success:
        if defer_persist:
            dbg("retry.success attempt=1 defer_persist=1")
            return first
        attempted, passed, failures, smoke_report = _run_smoke_for_candidate(
            focus_file=focus_file,
            candidate_content=str(first.get("candidate_content") or ""),
        )
        _set_smoke_meta(first_meta, attempted=attempted, passed=passed, failures=failures)
        first_meta["retry_trigger"] = "none"
        if not attempted or bool(passed):
            dbg("retry.success attempt=1")
            return first
        retry_trigger = "smoke"
        retry_hint = smoke_report or "SMOKE_REPORT:\n- failure: smoke test failed"
        dbg("retry.invoked attempt=2 reason=smoke_fail")
    else:
        _set_smoke_meta(first_meta, attempted=False, passed=None, failures=[])
        fail_reason = str(first_meta.get("failure_reason") or "").strip()
        fail_kind = str(first_meta.get("failure_kind") or "").strip()
        norm_conf = str(first_meta.get("normalizer_confidence") or "").strip().lower()
        norm_err = str(first_meta.get("normalizer_error_code") or "").strip()
        if not fail_reason:
            fail_kind, fail_reason = _extract_failure_from_output(str(first.get("output") or ""))
        retry_failure_reason = fail_reason
        target_symbol = str(first_meta.get("structural_target_symbol") or "").strip()
        explicit_structural_retry = {
            "candidate_parse_failed",
            "symbol_region_corrupted",
            "target_symbol_missing_after_replacement",
            "replacement_outside_target_span",
            "unexpected_top_level_symbol",
        }
        if fail_kind in {"syntax", "semantic"}:
            retry_hint = _build_compile_report(fail_reason)
            retry_trigger = "compile"
        elif fail_kind == "noop":
            first_meta["retry_trigger"] = "none"
            dbg("retry.skipped reason=noop_first_attempt")
            return first
        elif norm_conf == "red" and norm_err:
            retry_hint = ""
            retry_trigger = "format"
            if norm_err:
                retry_failure_reason = norm_err
            dbg("structural.retry.format.count=1")
        elif fail_kind == "structural" and fail_reason in explicit_structural_retry:
            retry_hint = _build_structural_fail_report(fail_reason or "structural edit failed", target_symbol=target_symbol)
            retry_trigger = "structural"
        else:
            first_meta["retry_trigger"] = "none"
            dbg("retry.skipped reason=non_retryable_failure")
            return first
        dbg("retry.invoked attempt=2")

    first_meta["retry_trigger"] = retry_trigger
    if retry_trigger == "format":
        target_symbol = str(first_meta.get("structural_target_symbol") or forced_symbol_name or "").strip()
        fmt_lines = structural_format_retry_rules(
            focus_file=focus_file,
            target_name=target_symbol,
            target_kind="",
        )
        retry_request = (
            "FORMAT ONLY RETRY:\n"
            "- Do not change logic.\n"
            "- Output ONLY the target symbol using the required wrapper.\n"
            f"Target: `{target_symbol or 'target_symbol'}`.\n"
            + "\n".join(fmt_lines)
        ).strip()
        retry_hint = (
            "format_retry_required: "
            + (retry_failure_reason or "follow structural output format exactly")
        )
    else:
        retry_request = _build_retry_base_request(
            sliced_request=sliced_request,
            user_text=user_text,
            focus_file=focus_file,
            file_content=baseline_content,
            prefer_task_card=prefer_task_card_retry,
        )
        retry_request += "\n\n" + "\n".join(structural_general_retry_rules()) + "\n"
    second = _run_structural_write(
        user_text=user_text,
        focus_file=focus_file,
        silent=silent,
        full_context=full_context,
        analysis_packet=analysis_packet,
        sliced_request=retry_request,
        focus_content_override=baseline_content,
        diagnostics_hint=retry_hint,
        hard_fail_on_verify_fail=False,
        force_stage_on_verify_fail=True,
        forced_symbol_name=forced_symbol_name,
        forbidden_symbol_names=forbidden_symbol_names,
        defer_persist=defer_persist,
        execute_temp=(0.4 if retry_trigger == "format" else TEMP_EXECUTE),
        cache_key_base=effective_cache_base,
    )
    _annotate_failure_meta(second, attempt_index=2, attempt_total=attempt_total, focus_file=focus_file)
    second_meta = second.get("meta", {}) or {}
    second_meta["edit_mode"] = "structural"
    _set_smoke_meta(second_meta, attempted=False, passed=None, failures=[])
    second_meta["retry_trigger"] = retry_trigger
    second_output = str(second.get("output", ""))
    second_success = (
        second_output.startswith("[Applied file_edit]")
        or second_output.startswith("[Staged file_edit]")
        or (defer_persist and second_output.startswith("[Candidate file_edit]"))
    )
    if second_success:
        if bool(second_meta.get("non_blocking_failure")):
            second_meta["requires_user_decision"] = True
        if defer_persist:
            dbg("retry.success attempt=2 defer_persist=1")
        dbg("retry.success attempt=2")
        return second
    dbg("retry.failed attempt=2")
    return second


def _run_structural_multi_symbol_edit(
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str,
    sliced_request: str,
    target_names: List[str],
    cache_key_base: str = "",
) -> Dict[str, object]:
    def _finalize_sequence_result(
        result: Dict[str, object],
        baseline: str,
        final_content: str,
        total_steps: int,
        completed_steps: int,
        partial_failure_reason: str = "",
    ) -> Dict[str, object]:
        meta = result.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
            result["meta"] = meta

        aggregate_diff = _compute_unified_diff(baseline or "", final_content or "", focus_file)
        net_changed = bool(final_content != baseline and (aggregate_diff or "").strip())
        meta["focus_file"] = focus_file
        meta["structural_sequence_total"] = total_steps
        meta["structural_sequence_completed"] = completed_steps
        meta["structural_sequence_net_changed"] = net_changed
        if aggregate_diff.strip():
            meta["diff"] = aggregate_diff
        result["candidate_content"] = final_content
        result["candidate_diff"] = aggregate_diff

        if not net_changed:
            return result

        # Sequence-level net changes should not be reported as no-op.
        meta.pop("noop", None)
        meta.pop("already_up_to_date", None)
        meta.pop("applied_without_diff", None)

        if partial_failure_reason:
            meta["structural_sequence_partial_failure"] = True
            meta["verify_warning"] = True
            meta["non_blocking_failure"] = True
            meta["failure_kind"] = "structural"
            meta["file_edit_failure_kind"] = "structural"
            meta["failure_reason"] = _normalize_failure_reason(
                partial_failure_reason,
                max_chars=400,
            )

        out = str(result.get("output") or "")
        if out.startswith("[Applied file_edit]") or out.startswith("[Staged file_edit]"):
            return result

        if config.STAGE_EDITS:
            meta["staged"] = True
            meta["staged_file"] = focus_file
            meta["staged_content"] = final_content
            if partial_failure_reason:
                meta["requires_user_decision"] = True
            result["output"] = "[Staged file_edit]"
            return result

        target_path = get_root() / focus_file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(final_content, encoding="utf-8")
        result["output"] = "[Applied file_edit]"
        return result

    names: List[str] = []
    seen = set()
    for raw in target_names or []:
        n = str(raw or "").strip()
        if not n:
            continue
        low = n.lower()
        if low in seen:
            continue
        seen.add(low)
        names.append(n)
    if not names:
        return _run_structural_edit(
            user_text=user_text,
            focus_file=focus_file,
            silent=silent,
            full_context=full_context,
            analysis_packet=analysis_packet,
            sliced_request=sliced_request,
            cache_key_base=cache_key_base,
        )

    baseline_content = read_single_file_for_context(focus_file).get(focus_file, "")
    current_content = baseline_content
    last_result: Optional[Dict[str, object]] = None
    total = len(names)
    for idx, name in enumerate(names, start=1):
        defer = idx < total
        forbidden = [n for n in names if n.lower() != name.lower()]
        dbg(
            "structural.sequence "
            f"step={idx}/{total} target={name} defer_persist={1 if defer else 0}"
        )
        step_target = None
        for sym in build_symbol_index(focus_file, current_content or ""):
            if sym.name.lower() == name.lower():
                step_target = sym
                break
        subtask_line = _build_symbol_subtask_line((sliced_request or user_text or ""), name)
        kind_hint = str(getattr(step_target, "kind", "") or "symbol").strip() or "symbol"
        step_lines: List[str] = [
            f"TASK: {subtask_line}",
            f"Target: {kind_hint} `{name}`",
            f"Return updated {kind_hint} `{name}` only. No explanation. Keep signature unless explicitly requested.",
        ]
        if step_target is not None:
            target_snippet = extract_target_snippet(
                current_content or "",
                step_target,
                padding_lines=1,
            )
            if target_snippet:
                step_lines.extend(
                    [
                        "TARGET_SNIPPET:",
                        target_snippet,
                    ]
                )
        step_request = "\n".join(step_lines).strip()
        step = _run_structural_edit(
            user_text=user_text,
            focus_file=focus_file,
            silent=silent,
            full_context=full_context,
            analysis_packet=analysis_packet,
            sliced_request=step_request,
            forced_symbol_name=name,
            forbidden_symbol_names=forbidden,
            baseline_content_override=current_content,
            defer_persist=defer,
            cache_key_base=cache_key_base,
        )
        step_meta = step.get("meta", {}) or {}
        step_meta["structural_sequence_total"] = total
        step_meta["structural_sequence_index"] = idx
        step_meta["structural_sequence_target"] = name
        out = str(step.get("output") or "")
        ok = (
            out.startswith("[Applied file_edit]")
            or out.startswith("[Staged file_edit]")
            or out.startswith("[Candidate file_edit]")
        )
        if not ok and out.startswith("[No-op file_edit:"):
            dbg(f"structural.sequence_noop_continue step={idx}/{total} target={name}")
            step_meta["structural_sequence_noop"] = True
            step_meta["structural_sequence_continued"] = True
            current_content = str(step.get("candidate_content") or current_content)
            last_result = step
            continue
        if not ok:
            dbg(f"structural.sequence_failed step={idx}/{total} target={name}")
            if current_content != baseline_content:
                partial_reason = (
                    f"structural sequence stopped at step {idx}/{total} "
                    f"(target `{name}`): {out or 'failed'}"
                )
                dbg(
                    "structural.sequence_partial_apply "
                    f"step={idx}/{total} target={name}"
                )
                return _finalize_sequence_result(
                    result=step,
                    baseline=baseline_content,
                    final_content=current_content,
                    total_steps=total,
                    completed_steps=idx - 1,
                    partial_failure_reason=partial_reason,
                )
            return step
        current_content = str(step.get("candidate_content") or current_content)
        last_result = step

    if last_result is None:
        return _run_structural_edit(
            user_text=user_text,
            focus_file=focus_file,
            silent=silent,
            full_context=full_context,
            analysis_packet=analysis_packet,
            sliced_request=sliced_request,
            cache_key_base=cache_key_base,
        )
    return _finalize_sequence_result(
        result=last_result,
        baseline=baseline_content,
        final_content=current_content,
        total_steps=total,
        completed_steps=total,
    )


def _run_patch_edit(
    user_text: str,
    focus_file: str,
    silent: bool,
    full_context: bool,
    analysis_packet: str = "",
    sliced_request: str = "",
    raw_user_text: str = "",
    allow_full_block: bool = False,
) -> Dict[str, object]:
    baseline_content = read_single_file_for_context(focus_file).get(focus_file, "")
    file_lines = len((baseline_content or "").splitlines())
    raw_text = (raw_user_text or user_text or "").strip()
    is_empty = _is_new_or_empty_file(focus_file)
    explicit_rewrite = wants_file_rewrite(raw_text)
    can_full_block = bool(allow_full_block and (is_empty or explicit_rewrite))

    if can_full_block:
        dbg(
            "route=file_edit mode=fileblock "
            f"lines={file_lines} empty={1 if is_empty else 0} rewrite={1 if explicit_rewrite else 0}"
        )
        result = _run_fileblock_edit(
            user_text=user_text,
            focus_file=focus_file,
            silent=silent,
            full_context=full_context,
            analysis_packet=analysis_packet,
            sliced_request=sliced_request,
        )
    else:
        structural_cache_base = _build_structural_cache_key_base(
            focus_file=focus_file,
            request_text=(sliced_request or user_text or ""),
        )
        structural_targets = select_target_symbols(
            user_text=(sliced_request or user_text or ""),
            focus_file=focus_file,
            content=baseline_content or "",
            analysis_packet=analysis_packet or "",
        )
        structural_decision = select_target_symbol(
            user_text=(sliced_request or user_text or ""),
            focus_file=focus_file,
            content=baseline_content or "",
            analysis_packet=analysis_packet or "",
        )
        if structural_decision.eligible and structural_decision.target:
            dbg(
                "route=file_edit mode=structural "
                f"lines={file_lines} symbol={structural_decision.target.name}"
            )
        else:
            dbg(
                "structural.precheck="
                f"{structural_decision.reason or 'ineligible'}"
            )
            dbg(
                "route=file_edit mode=structural "
                f"lines={file_lines} reason={structural_decision.reason or 'ineligible'}"
            )
        if len(structural_targets) > 1:
            target_names = [t.name for t in structural_targets]
            dbg(
                "route=file_edit mode=structural_multi "
                f"lines={file_lines} symbols={','.join(target_names)}"
            )
            result = _run_structural_multi_symbol_edit(
                user_text=user_text,
                focus_file=focus_file,
                silent=silent,
                full_context=full_context,
                analysis_packet=analysis_packet,
                sliced_request=sliced_request,
                target_names=target_names,
                cache_key_base=structural_cache_base,
            )
            rmeta = result.get("meta", {}) or {}
            rmeta.setdefault("structural_sequence_total", len(target_names))
        else:
            result = _run_structural_edit(
                user_text=user_text,
                focus_file=focus_file,
                silent=silent,
                full_context=full_context,
                analysis_packet=analysis_packet,
                sliced_request=sliced_request,
                cache_key_base=structural_cache_base,
            )
        rmeta = result.get("meta", {}) or {}
        rmeta.setdefault("edit_mode", "structural")
        if not structural_decision.eligible:
            rmeta.setdefault("structural_fallback_reason", structural_decision.reason or "ineligible")

    out = str(result.get("output") or "")
    legacy_requested = bool(
        re.search(r"\[\[\[file:|\bfileblock\b", (user_text or ""), re.IGNORECASE)
    )
    if (
        out.startswith("[File edit failed:")
        and config.PATCH_LEGACY_FILEBLOCK_FALLBACK
        and legacy_requested
    ):
        dbg("patch_protocol.fallback=legacy_stub")
        legacy = _run_legacy_fileblock_stub(
            user_text=user_text,
            focus_file=focus_file,
            base_result=result,
        )
        meta = legacy.get("meta")
        if isinstance(meta, dict):
            meta["legacy_fallback_used"] = True
        return legacy
    return result


def _run_legacy_fileblock_stub(
    user_text: str,
    focus_file: str,
    base_result: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Compatibility stub for removed fileblock path (no execution)."""
    _ = user_text  # keep signature parallel to removed path
    reason = "legacy fileblock path removed; unified diff required"
    reject_rule = "LEGACY_FILEBLOCK_DISABLED"
    meta: Dict[str, object] = {}
    if isinstance(base_result, dict):
        base_meta = base_result.get("meta")
        if isinstance(base_meta, dict):
            meta.update(base_meta)
    meta["reject_rule"] = reject_rule
    meta["file_edit_failure_kind"] = "format"
    meta["failure_kind"] = "format"
    meta["failure_reason"] = reason
    result = {
        "output": f"[File edit failed: {reject_rule}: {reason}]",
        "meta": meta,
        "status_prefix": "",
        "blocks_count": 0,
        "retried": True,
        "timeout": False,
    }
    _annotate_failure_meta(
        result,
        attempt_index=int(meta.get("attempt_index") or 2),
        attempt_total=int(meta.get("attempt_total") or 2),
        focus_file=focus_file,
    )
    return result


def _run_plan_execute(
    user_text: str,
    focus_file: str,
    silent: bool,
    root_analysis_packet: str = "",
    root_analysis_policy: Optional[Dict[str, object]] = None,
    intent_hint: str = "",
) -> Dict[str, object]:
    start = time.time()
    trace: List[Dict[str, object]] = []
    model_calls_used = 0
    files = _target_files_for_routing(user_text, focus_file)
    planner_token_cap = max(250, min(config.PLAN_MAX_NEW, 300))

    if (intent_hint or "").upper() == IMPLEMENT_STUBS and len(files) <= 1 and focus_file:
        dbg("planner_skipped reason=single_file_stub_completion")
        plan = {
            "files": [
                {
                    "path": _norm_rel_path(focus_file),
                    "action": "edit",
                    "description": "Implement requested placeholders with complete working code.",
                    "target": "",
                }
            ]
        }
        output = '{"files":[{"path":"' + _norm_rel_path(focus_file) + '","action":"edit","description":"Implement requested placeholders with complete working code.","target":""}]}'
        err = None
        model_calls_used = 0
    else:
        plan = None
        err = "PLAN_INVALID"

    # Task-card flow: avoid semantic slicing; keep original request text.
    plan_user_text = (user_text or "").strip()
    dbg(f"request_slice_len={len(plan_user_text)}")
    file_matches = _build_plan_file_matches(files, plan_user_text)
    plan_summary = _build_planner_analysis_summary(root_analysis_policy or {}, max_chars=400)
    plan_prompt = build_plan_multi_prompt(
        plan_user_text,
        file_matches,
        focus_file=focus_file,
        analysis_packet=plan_summary,
    )
    if plan_summary:
        dbg(f"analysis_summary_injected=yes len={len(plan_summary)}")
    else:
        dbg("analysis_summary_injected=no")

    def _call(prompt: str, span: str, temp: Optional[float] = None) -> str:
        nonlocal model_calls_used
        model_calls_used += 1
        with _span(trace, span):
            return stream_reply(
                prompt,
                silent=silent,
                temperature=temp,
                max_new=planner_token_cap,
            )

    if plan is None:
        dbg(f"planner_used=yes cap={planner_token_cap} prompt_len={len(plan_prompt)}")
        output = _call(plan_prompt, "plan_attempt_1", temp=TEMP_PLAN)
        _append_history(user_text, output)
        dbg_dump("plan_output_1", output)
        plan = parse_plan(output)
        err = validate_plan_json(plan or {}, config.MAX_PLAN_FILES)
        if err and model_calls_used < 2:
            dbg(f"validate_plan_json: reason={err}")
            if _is_collapsed_planner_output(output):
                retry_prompt = (
                    "Output JSON only. Start with '{' and end with '}'.\n"
                    f"Schema: {{\"files\":[{{\"path\":\"{focus_file}\",\"action\":\"edit\",\"description\":\"<specific change>\",\"target\":\"\"}}]}}\n"
                    "No markdown fences. No prose.\n"
                    f"Request: {plan_user_text[:500]}\n"
                )
            else:
                retry_prompt = (
                    plan_prompt
                    + f"\nERROR: {err}. Output JSON only.\n"
                    + 'Return exactly one entry per file path; do not repeat the same path.\n'
                    + 'Return schema: {"files":[{"path":"<repo-rel-path>","action":"edit","description":"<specific change>","target":""}]}\n'
                    + "No markdown fences. No prose.\n"
                )
            output = _call(retry_prompt, "plan_retry", temp=TEMP_PLAN)
            _append_history(user_text, output)
            dbg_dump("plan_output_retry", output)
            plan = parse_plan(output)
            err = validate_plan_json(plan or {}, config.MAX_PLAN_FILES)
            if err:
                dbg(f"validate_plan_json: reason={err}")

    if err or not plan:
        if err:
            dbg("plan_execute: rejected plan before execution")
        # Planner outputs can fail on long assignment-style prompts.
        # For single-target requests, fall back to a synthetic one-file plan
        # so Route C can continue with execution instead of hard-failing.
        if len(files) == 1 and files[0]:
            dbg("plan_execute: planner invalid; using synthetic single-file plan fallback")
            fallback_desc = "Implement requested changes with complete working code; remove placeholders and TODOs."
            plan = {
                "files": [
                    {
                        "path": files[0],
                        "action": "edit",
                        "description": fallback_desc,
                        "target": "",
                    }
                ]
            }
            err = None
        else:
            meta = _build_meta(len(plan_prompt), output, 0, model_calls_used > 1, start, trace, model_calls_used)
            meta["mode_used"] = Route.PLAN_MULTI.value
            meta["route"] = Route.PLAN_MULTI.value
            return {
                "output": f"[Plan failed: {err or 'PARSE_FAIL'}]",
                "meta": meta,
                "status_prefix": "",
                "blocks_count": 0,
                "retried": model_calls_used > 1,
                "timeout": False,
            }

    results = []
    per_file_diffs: Dict[str, str] = {}
    files_changed: List[str] = []
    for entry in plan["files"]:
        path = _norm_rel_path(entry.get("path", ""))
        if not path:
            continue
        abs_path = get_root() / path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if not abs_path.exists():
            abs_path.write_text("", encoding="utf-8")

        entry_desc = str(entry.get("description", "") or "").strip()
        content = read_single_file_for_context(path).get(path, "")
        card_request = user_text
        if entry_desc:
            card_request = f"{user_text}\n{entry_desc}"
        focused_request = build_task_card(
            path,
            card_request,
            content,
            max_lines=15,
        )
        dbg(f"task_card_generated file={path} len={len(focused_request)}")
        dbg_dump(f"task_card:{path}", focused_request)
        entry_analysis = ""
        entry_analysis_policy: Dict[str, object] = {}
        if (
            root_analysis_packet
            and root_analysis_policy
            and _norm_rel_path(path) == _norm_rel_path(focus_file)
        ):
            entry_analysis = root_analysis_packet
            entry_analysis_policy = dict(root_analysis_policy)
        else:
            entry_analysis, entry_analysis_policy = _build_analysis_for_prompt(
                focused_request,
                path,
                max_chars=700,
            )
        if entry_analysis:
            dbg(f"analysis_compact_len={len(entry_analysis)}")
            dbg(
                "analysis_signals: "
                f"missing={len(_analysis_ban_symbols(entry_analysis_policy))}, "
                f"touch_points={len(_analysis_touch_points(entry_analysis_policy))}, "
                f"local_refs={_analysis_has_local_refs(entry_analysis_policy)}"
            )
            dbg(f"deps_confidence={str(entry_analysis_policy.get('deps_confidence') or 'unknown')}")
        file_lines = len(content.splitlines()) if content else 0
        file_chars = len(content) if content else 0
        ext = _ext(path)
        is_c_like = ext in {"c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx"}
        file_has_todos = has_todo_markers(content)
        syntax_todos = has_placeholder_conditionals(content, ext)
        if _allows_partial_todo_edit(focused_request):
            syntax_todos = False
        sub_intent = classify_intent(focused_request, content, path)
        ctx = RequestContext(
            request_len=len(focused_request or ""),
            file_lines=file_lines,
            file_chars=file_chars,
            target_files=[path],
            intent=sub_intent,
            wants_minimal_diff=wants_minimal_diff(focused_request or ""),
            has_assignment_text=has_assignment_style_text(focused_request or ""),
            file_is_empty=_is_new_or_empty_file(path),
            rewrite_requested=wants_file_rewrite(focused_request or ""),
            file_has_todos=file_has_todos,
            has_syntax_critical_todos=syntax_todos,
            is_c_like=is_c_like,
            analysis_missing_symbols_count=len(_analysis_ban_symbols(entry_analysis_policy)),
            analysis_touch_points_count=len(_analysis_touch_points(entry_analysis_policy)),
            analysis_has_local_refs=_analysis_has_local_refs(entry_analysis_policy),
        )
        sub_route, sub_reason = decide_route_with_reason(ctx)
        # Per-file execution inside PLAN_MULTI should never recurse into PLAN_MULTI.
        if sub_route == Route.PLAN_MULTI:
            sub_route = Route.FILE_EDIT
            sub_reason = f"{sub_reason}->downgraded_to_file_edit"
        dbg(f"sub_route={sub_route.value} reason={sub_reason} file={path}")
        sub_result = _run_patch_edit(
            focused_request,
            path,
            silent,
            full_context=False,
            analysis_packet=entry_analysis,
            sliced_request=focused_request,
            raw_user_text=focused_request,
            allow_full_block=bool(ctx.file_is_empty or ctx.rewrite_requested),
        )
        results.append(f"[{path}] {sub_result['output']}")
        rmeta = sub_result.get("meta", {})
        fdiff = rmeta.get("diff") if isinstance(rmeta, dict) else None
        if isinstance(fdiff, str) and fdiff.strip():
            per_file_diffs[path] = fdiff
            files_changed.append(path)

    meta = _build_meta(len(plan_prompt), output, 0, model_calls_used > 1, start, trace, model_calls_used)
    meta["mode_used"] = Route.PLAN_MULTI.value
    meta["route"] = Route.PLAN_MULTI.value
    meta["files"] = [e.get("path") for e in plan["files"]]
    meta["files_changed"] = files_changed
    meta["per_file_diffs"] = per_file_diffs
    return {
        "output": "\n".join(results) if results else "[No file changes produced]",
        "meta": meta,
        "status_prefix": "",
        "blocks_count": 0,
        "retried": model_calls_used > 1,
        "timeout": False,
    }


# Guard flag to prevent multi-file re-entry (legacy; kept to avoid breakage)
_in_multi_file = False


def _run_agent_core(
    user_text: str,
    focus_file: Optional[str],
    silent: bool,
    full_context: bool,
    want_meta: bool,
) -> Dict[str, object]:
    start = time.time()
    focus_file = _resolve_focus_file(user_text, focus_file)
    history_user_text = user_text

    # Unified behavior: conversational queries can use chat path even in agent mode.
    # Keep this check before intent classification to avoid over-triggering edit routes
    # from broad intent keywords in long context.
    if _looks_chat_query(user_text):
        dbg("agent: mode=chat_in_agent reason=conversational_query_preintent")
        trace: List[Dict[str, object]] = []
        with _span(trace, "chat_in_agent"):
            prompt, ctx_meta = prompt_buffer.build_prompt(
                "chat",
                user_text,
                focus_file=focus_file,
                full_context=False,
            )
            output = stream_reply(
                prompt,
                silent=silent,
                max_new=config.MAX_NEW,
                stop_sequences=["\nUser:", "\nSYSTEM:", "\nCONTEXT:", "\nHISTORY:"],
            )
        _append_history(user_text, output)
        state.append_chat_turn(user_text, output)
        meta = _build_meta(len(prompt), output, 0, False, start, trace, 1)
        meta.update(ctx_meta or {})
        meta["mode_used"] = "chat_in_agent"
        meta["route"] = "chat_in_agent"
        return {
            "output": output,
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": output == "[Model timeout]",
        }

    if not focus_file:
        meta = _build_meta(0, "", 0, False, start)
        meta["mode_used"] = "no_focus_file"
        meta["route"] = "no_focus_file"
        meta["failure_kind"] = "missing_focus_file"
        meta["failure_reason"] = (
            "No file selected. Select a folder and open a file, "
            "or specify a target file path in your request."
        )
        return {
            "output": meta["failure_reason"],
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
        }

    file_content = read_single_file_for_context(focus_file).get(focus_file, "")
    intent = classify_intent(user_text, file_content, focus_file)
    dbg(f"agent: intent={intent} for {focus_file}")

    if intent == DELETE_FILE:
        import os
        abs_path = get_root() / focus_file
        if abs_path.exists():
            os.remove(str(abs_path))
        meta = _build_meta(0, "", 0, False, start)
        meta["mode_used"] = "delete_file"
        return {
            "output": f"[Deleted {focus_file}]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
        }

    if intent == CLEAR_RANGE:
        abs_path = get_root() / focus_file
        if abs_path.exists():
            abs_path.write_text("", encoding="utf-8")
        meta = _build_meta(0, "", 0, False, start)
        meta["mode_used"] = "clear_range"
        return {
            "output": f"[Cleared {focus_file}]",
            "meta": meta,
            "status_prefix": "",
            "blocks_count": 0,
            "retried": False,
            "timeout": False,
        }

    file_lines = len(file_content.splitlines()) if file_content else 0
    file_chars = len(file_content) if file_content else 0
    target_files = _target_files_for_routing(user_text, focus_file)
    analysis_packet = ""
    analysis_policy: Dict[str, object] = {}
    full_context_single = False
    task_card_enabled = False
    if len(target_files) <= 1:
        single_req = _build_singlefile_request(user_text, focus_file, file_content)
        sliced_user_text = str(single_req.get("sliced_request") or "")
        analysis_packet = str(single_req.get("analysis_packet") or "")
        analysis_policy = single_req.get("analysis_policy") or {}
        full_context_single = str(single_req.get("context_mode") or "focused") == "full"
        task_card_enabled = bool(single_req.get("task_card_enabled"))
        if task_card_enabled and sliced_user_text:
            history_user_text = sliced_user_text
        if task_card_enabled:
            dbg("request_slicing: replaced_by_task_card")
        else:
            dbg("request_slicing: bypassed_task_card_threshold")
        dbg(
            "task_card_precondition: "
            f"tokens={int(single_req.get('request_token_estimate') or 0)} "
            f"threshold={int(config.TASK_CARD_MIN_TOKENS)} "
            f"enabled={task_card_enabled}"
        )
        dbg(f"request_slice_len={len(sliced_user_text)}")
        dbg(
            f"context_policy={'full' if full_context_single else 'focused'} "
            f"reason={single_req.get('context_reason', 'default_focused')}"
        )
        if analysis_packet:
            dbg(f"analysis_compact_len={len(analysis_packet)}")
            dbg(
                "analysis_signals: "
                f"missing={len(_analysis_ban_symbols(analysis_policy))}, "
                f"touch_points={len(_analysis_touch_points(analysis_policy))}, "
                f"local_refs={_analysis_has_local_refs(analysis_policy)}"
            )
            dbg(f"deps_confidence={str(analysis_policy.get('deps_confidence') or 'unknown')}")
    else:
        sliced_user_text = (user_text or "").strip()
        dbg("request_slicing: bypassed_multi_file")
        dbg(f"slice_policy=single_pass len={len(sliced_user_text)}")
        dbg(f"request_slice_len={len(sliced_user_text)}")
    ext = _ext(focus_file)
    is_c_like = ext in {"c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx"}
    file_has_todos = has_todo_markers(file_content)
    syntax_todos = has_placeholder_conditionals(file_content, ext)
    if _allows_partial_todo_edit(user_text):
        syntax_todos = False
    route_ctx = RequestContext(
        request_len=len(sliced_user_text or ""),
        file_lines=file_lines,
        file_chars=file_chars,
        target_files=target_files,
        intent=intent,
        wants_minimal_diff=wants_minimal_diff(user_text or ""),
        has_assignment_text=has_assignment_style_text(user_text or ""),
        file_is_empty=_is_new_or_empty_file(focus_file),
        rewrite_requested=wants_file_rewrite(user_text or ""),
        file_has_todos=file_has_todos,
        has_syntax_critical_todos=syntax_todos,
        is_c_like=is_c_like,
        analysis_missing_symbols_count=len(_analysis_ban_symbols(analysis_policy)),
        analysis_touch_points_count=len(_analysis_touch_points(analysis_policy)),
        analysis_has_local_refs=_analysis_has_local_refs(analysis_policy),
    )
    route, route_reason = decide_route_with_reason(route_ctx)
    if len(target_files) <= 1 and route == Route.PLAN_MULTI:
        route = Route.FILE_EDIT
        route_reason = "single_file_forced_file_edit"
    dbg(f"agent: route={route}")
    dbg(f"route_used: {route.value}")
    dbg(f"route_decision_with_analysis={route.value}")
    dbg(f"route={route.value} reason={route_reason}")
    dbg(f"route_chosen={route.value} reason={route_reason}")

    include_paths = set(get_include() or [])
    if route != Route.PLAN_MULTI and not include_paths:
        trimmed_for_discovery = (sliced_user_text or user_text or "")[:400]
        resolved = discover_target_file(trimmed_for_discovery, focus_file)
        if resolved and resolved != focus_file:
            dbg(f"agent: target discovery overrode {focus_file} -> {resolved}")
            focus_file = _norm_rel_path(resolved)
            file_content = read_single_file_for_context(focus_file).get(focus_file, "")
            analysis_packet, analysis_policy = _build_analysis_for_prompt(sliced_user_text, focus_file, max_chars=700)

    if route == Route.PLAN_MULTI:
        result = _run_plan_execute(
            sliced_user_text,
            focus_file,
            silent,
            root_analysis_packet=analysis_packet,
            root_analysis_policy=analysis_policy,
            intent_hint=intent,
        )
    elif route == Route.FILE_EDIT:
        result = _run_patch_edit(
            sliced_user_text,
            focus_file,
            silent,
            full_context=full_context_single,
            analysis_packet=analysis_packet,
            sliced_request=sliced_user_text,
            raw_user_text=user_text,
            allow_full_block=bool(_is_new_or_empty_file(focus_file) or wants_file_rewrite(user_text or "")),
        )
    else:
        # Single-file execution now uses FILE_EDIT only.
        result = _run_patch_edit(
            sliced_user_text,
            focus_file,
            silent,
            full_context=full_context_single,
            analysis_packet=analysis_packet,
            sliced_request=sliced_user_text,
            raw_user_text=user_text,
            allow_full_block=bool(_is_new_or_empty_file(focus_file) or wants_file_rewrite(user_text or "")),
        )
    result["history_user_text"] = history_user_text
    return result


def run_agent(
    user_text: str,
    focus_file: Optional[str] = None,
    silent: bool = False,
    full_context: bool = False,
):
    result = _run_agent_core(user_text, focus_file, silent, full_context, False)
    output = result["output"]
    status_prefix = result["status_prefix"]
    mode_used = result.get("meta", {}).get("mode_used", "agent")
    meta = result.get("meta", {}) or {}
    history_user_text = str(result.get("history_user_text") or user_text)
    history_output = _history_output_for_storage(
        str(output or ""),
        meta if isinstance(meta, dict) else {},
        focus_file or "",
    )
    if mode_used != "chat_in_agent":
        state.append_chat_turn(history_user_text, history_output)
        if str(output or "") in {"[Staged file_edit]", "[Applied file_edit]"}:
            state.append_change_note(history_output)
    return f"{status_prefix}{output}" if status_prefix else output


def run_agent_meta(
    user_text: str,
    focus_file: Optional[str] = None,
    silent: bool = False,
    full_context: bool = False,
):
    result = _run_agent_core(user_text, focus_file, silent, full_context, True)
    output = result["output"]
    status_prefix = result["status_prefix"]
    meta = result["meta"]
    mode_used = meta.get("mode_used", "")
    history_user_text = str(result.get("history_user_text") or user_text)
    history_output = _history_output_for_storage(
        str(output or ""),
        meta if isinstance(meta, dict) else {},
        focus_file or "",
    )
    if mode_used != "chat_in_agent":
        state.append_chat_turn(history_user_text, history_output)
        if str(output or "") in {"[Staged file_edit]", "[Applied file_edit]"}:
            state.append_change_note(history_output)
    return (f"{status_prefix}{output}" if status_prefix else output), meta
