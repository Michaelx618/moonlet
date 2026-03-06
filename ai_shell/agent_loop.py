"""Agent loop: tool-in-the-loop flow. See BORROWED_FROM_CONTINUE.md for design attribution.

Flow:

  1. Tools are sent with the user request (we inject TOOLS_SYSTEM_HINT + optional context).
  2. Model may include tool call(s) in its response.
  3. We execute each tool (built-in: read_file, grep, search_replace, write_file, etc.).
  4. We send tool results back to the model as context ("fed back into the model").
  5. Model responds again (possibly with more tool calls) → repeat from step 3 until the model
     responds with no tool calls (then that reply is the final answer).

How the model sees files:
- We inject workspace path list and optional current/reference file paths only (no full file content).
  The agent is expected to use read_file (and list_files, grep, etc.) to read contents.

Tool calling: we use a completion-style API (single prompt → text reply) and parse function-style
text. For native tool_calls use a backend with USE_CHAT_TOOLS + /v1/chat/completions with tools.
"""

from __future__ import annotations

import difflib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config
from .files import (
    _norm_rel_path,
    get_root,
    is_edit_allowed,
    write_file_text,
)
from .index import get_indexed_files, get_symbols_for_file
from .model import get_session_cache_key, stream_reply_chunks
from .model import chat_completion_with_tools
from .search_replace import apply_search_replace
from .tool_repetition_detector import ToolRepetitionDetector
from .tool_executor import (
    AGENT_TOOLS_JSON,
    TOOLS_SYSTEM_HINT,
    execute_tool_from_kwargs,
    extract_function_style_tool_calls,
    strip_tool_calls_from_output,
    strip_tool_call_code_blocks,
    tool_log,
)
from .utils import dbg

from .file_utils_adapter import generate_diff, is_security_concern

# Junk paths to exclude from context
_REFERENCE_SKIP = (".dsym/", "contents/info.plist", "/spec.txt", "/task.json", "node_modules/", "__pycache__/")

_REPO = Path(__file__).resolve().parent.parent
_PARSE_FAILURE_LOG = _REPO / "tests" / "integration_parse_debug.log"


def _log_parse_failure(round_num: int, reply: str) -> None:
    """Log full model reply when no tool calls were parsed. Useful for debugging tests."""
    try:
        log_path = getattr(config, "DEBUG_LOG_PATH", None) or str(_PARSE_FAILURE_LOG)
        if not log_path:
            log_path = str(_PARSE_FAILURE_LOG)
        with open(log_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[parse_failure] round={round_num} reply_len={len(reply)}\n")
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("--- full model reply ---\n")
            f.write(reply or "(empty)\n")
            f.write("--- end ---\n")
    except Exception:
        pass


def _log_parse_result(round_num: int, reply: str, parsed: list) -> None:
    """Log parse success: reply snippet and what we parsed. For debugging."""
    try:
        log_path = getattr(config, "DEBUG_LOG_PATH", None) or str(_PARSE_FAILURE_LOG)
        if not log_path:
            log_path = str(_PARSE_FAILURE_LOG)
        with open(log_path, "a") as f:
            f.write(f"\n[parse_ok] round={round_num} parsed={[(n, list(k.keys()) if isinstance(k, dict) else []) for n, k in parsed]}\n")
            f.write(f"reply_preview: {repr((reply or '')[:500])}\n")
    except Exception:
        pass


def _log_tool_result(tool: str, path: str = "", result: str = "", added_to_touched: bool = False, exc: str = "") -> None:
    """Log tool execution result for diagnostics (create_new_file, write_file, etc)."""
    try:
        log_path = str(_PARSE_FAILURE_LOG)
        with open(log_path, "a") as f:
            f.write(f"\n[tool_result] {tool} path={path!r} added_to_touched={added_to_touched}\n")
            f.write(f"  result: {(result or '')[:400]}\n")
            if exc:
                f.write(f"  exc: {exc}\n")
    except Exception:
        pass


# --- Conversation state (single representation for both completion and API backends) ---

@dataclass
class Turn:
    """One round: model reply + tool calls + tool results."""
    assistant_content: str
    tool_calls: List[Tuple[str, Dict[str, Any]]]
    tool_results: List[str]
    # API path only: raw tool_calls from chat API (with id, function) for message building
    api_tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ConversationState:
    """Everything the model has seen: initial prompt content + list of turns."""
    initial_content: str
    turns: List[Turn] = field(default_factory=list)
    # For API path: system and user content (derived from same build as initial_content)
    initial_system: str = ""
    initial_user: str = ""


def state_to_prompt(state: ConversationState, next_user_msg: str) -> str:
    """Serialize conversation state to the exact prompt string used by the completion backend.
    Must match the previous inline concatenation so model input is byte-for-byte unchanged.
    """
    out = state.initial_content
    for turn in state.turns:
        combined = "\n\n".join(turn.tool_results)
        out += f"\nAssistant:\n{turn.assistant_content}\n\nTool results:\n{combined}\n\n"
    out += next_user_msg
    return out


def state_to_messages(state: ConversationState) -> List[Dict[str, Any]]:
    """Build chat API message list from conversation state (same structure as API path used before unification)."""
    system = state.initial_system or ""
    user = state.initial_user or ""
    if not system and not user and state.initial_content:
        # Derive from initial_content: system = first paragraph + TOOLS_SYSTEM_HINT, user = rest
        system_msg = getattr(
            config,
            "BASE_AGENT_SYSTEM_MESSAGE",
            "You are a systematic coding agent. Break down problems methodically.",
        )
        prefix = system_msg + "\n\n" + TOOLS_SYSTEM_HINT.strip()
        if state.initial_content.startswith(prefix):
            system = prefix
            user = state.initial_content[len(prefix):]
        else:
            user = state.initial_content
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system or "(none)"},
        {"role": "user", "content": user or "(none)"},
    ]
    for turn in state.turns:
        if turn.api_tool_calls:
            messages.append({
                "role": "assistant",
                "content": turn.assistant_content or "",
                "tool_calls": turn.api_tool_calls,
            })
        else:
            messages.append({"role": "assistant", "content": turn.assistant_content or ""})
        for i, content in enumerate(turn.tool_results):
            tc_id = ""
            if turn.api_tool_calls and i < len(turn.api_tool_calls):
                tc_id = (turn.api_tool_calls[i].get("id") or "")
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": content})
    return messages


def get_reply_completion(
    state: ConversationState,
    next_user_msg: str,
    cache_key: Optional[str],
    max_new: int,
    stop_sequences: List[str],
    on_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    """Completion backend: build prompt from state, stream reply, return full text."""
    if not state.turns:
        prompt = state.initial_content
    else:
        prompt = state_to_prompt(state, next_user_msg)
    round_chunks: List[str] = []
    for token in stream_reply_chunks(
        prompt,
        max_new=max_new,
        stop_sequences=stop_sequences,
        cache_key=cache_key,
    ):
        round_chunks.append(token)
        if on_chunk:
            try:
                on_chunk(token)
            except Exception:
                pass
    return "".join(round_chunks).strip()


def get_reply_api(
    state: ConversationState,
    max_new: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """API backend: build messages from state, call chat_completion_with_tools, return (content, tool_calls)."""
    messages = state_to_messages(state)
    resp = chat_completion_with_tools(
        messages=messages,
        tools=AGENT_TOOLS_JSON,
        max_tokens=max_new,
        temperature=getattr(config, "TEMPERATURE", 0.1),
        top_p=getattr(config, "TOP_P", 0.9),
    )
    choices = resp.get("choices") or []
    if not choices:
        return "", []
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    tool_calls = msg.get("tool_calls") or []
    return content, tool_calls


def _reference_files_section(extra_read_files: Optional[List[str]]) -> str:
    """Reference file paths only. Agent should use read_file to read contents."""
    if not extra_read_files:
        return ""
    root = get_root()
    max_files = max(1, int(getattr(config, "MAX_READ_FILES_IN_PROMPT", 10)))
    filtered = [
        p for p in extra_read_files[: max_files * 2]
        if p and not any(s in (p or "").lower() for s in _REFERENCE_SKIP)
    ][:max_files]
    if not filtered:
        return ""
    paths_only: List[str] = []
    for rel in filtered:
        try:
            norm = _norm_rel_path(rel)
            path = (root / norm).resolve()
            path.relative_to(root.resolve())
            if not path.exists() or not path.is_file():
                continue
            paths_only.append(norm)
        except Exception:
            continue
    if not paths_only:
        return ""
    return "\n\nReference file paths (use read_file to read): " + ", ".join(f'"{p}"' for p in paths_only) + "\n"


def _workspace_paths_section() -> str:
    """Inject actual workspace paths into the prompt so the model uses them instead of inventing paths."""
    try:
        paths = get_indexed_files()
    except Exception:
        return ""
    if not paths:
        return ""
    # Cap so prompt doesn't blow up
    paths = paths[:30]
    return (
        "\n\nWorkspace paths (use these exact paths for read_file/write_file/edit_existing_file; do not create new top-level directories):\n"
        + ", ".join(paths)
    )


def _focus_file_section(focus_file: Optional[str]) -> str:
    """Current file path only. Agent should use read_file to read contents."""
    if not focus_file:
        return ""
    root = get_root()
    try:
        rel = _norm_rel_path(focus_file)
        path = (root / rel).resolve()
        path.relative_to(root.resolve())
        if not path.exists() or not path.is_file():
            return ""
        return (
            f"\n\nCurrent file path (use in tools): \"{rel}\". Use read_file to read its contents.\n"
        )
    except Exception:
        return ""


def _code_context_section(
    focus_file: Optional[str],
    extra_read_files: Optional[List[str]],
) -> str:
    """@Code context: symbols (functions, classes) with snippets for focus_file and extra_read_files."""
    files_to_scan: List[str] = []
    if focus_file:
        files_to_scan.append(_norm_rel_path(focus_file))
    for p in (extra_read_files or [])[:8]:
        norm = _norm_rel_path(p)
        if norm and norm not in files_to_scan:
            files_to_scan.append(norm)
    if not files_to_scan:
        return ""
    root = get_root()
    parts: List[str] = []
    for rel in files_to_scan:
        try:
            path = (root / rel).resolve()
            path.relative_to(root.resolve())
            if not path.exists() or not path.is_file():
                continue
            content = path.read_text()
            lines = content.splitlines(keepends=True)
            syms = get_symbols_for_file(rel)
            if not syms:
                continue
            block = [f"@Code ({rel}):"]
            for s in syms[:20]:
                name = s.get("name", "?")
                kind = s.get("kind", "symbol")
                line1 = max(1, int(s.get("line", 1)))
                line2 = max(line1, int(s.get("end_line", line1)))
                snippet_lines = lines[line1 - 1 : line2] if line1 <= len(lines) else []
                snippet = "".join(snippet_lines).strip()
                # No truncation
                block.append(f"  {kind} {name} (L{line1}-{line2}):\n{snippet}")
            parts.append("\n".join(block))
        except Exception:
            continue
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


def _folder_context_section(context_folders: Optional[List[str]]) -> str:
    """@Folder context: list files (and optionally content) for each folder."""
    if not context_folders:
        return ""
    try:
        all_indexed = get_indexed_files()
    except Exception:
        return ""
    root = get_root()
    parts: List[str] = []
    max_files_per_folder = int(getattr(config, "MAX_FOLDER_CONTEXT_FILES", 15))
    for folder in context_folders[:5]:
        folder_norm = _norm_rel_path(folder).rstrip("/")
        prefix = folder_norm + "/" if folder_norm else ""
        in_folder = [f for f in all_indexed if f == folder_norm or f.startswith(prefix)]
        in_folder = in_folder[:max_files_per_folder]
        if not in_folder:
            parts.append(f"@Folder ({folder_norm or '.'}): (no indexed files)")
            continue
        # List filenames only; agent should use read_file for contents
        parts.append(f"@Folder ({folder_norm or '.'}): " + ", ".join(in_folder))
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


def _build_prompt(
    spec: str,
    focus_file: Optional[str],
    mode: str,
    extra_read_files: Optional[List[str]] = None,
    context_folders: Optional[List[str]] = None,
) -> str:
    instruction = (spec or "").strip()
    system_msg = getattr(
        config,
        "BASE_AGENT_SYSTEM_MESSAGE",
        "You are a systematic coding agent. Break down problems methodically.",
    )
    # Agent mode: all tools available; tools sent with user request
    return (
        f"{system_msg}\n\n"
        f"{TOOLS_SYSTEM_HINT.strip()}\n\n"
        f"{_workspace_paths_section()}"
        f"{_code_context_section(focus_file, extra_read_files)}"
        f"{_folder_context_section(context_folders)}"
        f"{_reference_files_section(extra_read_files)}"
        f"User request:\n{instruction}"
        f"{_focus_file_section(focus_file)}"
    )


def _run_one_tool(
    name: str,
    kwargs: Dict[str, str],
    per_file_staged: Dict[str, str],
    per_file_before: Dict[str, str],
    touched: List[str],
) -> str:
    """Run one tool; track edits for search_replace and write_file. Return result string."""
    name = (name or "").strip().lower()

    if name == "view_diff":
        path = (kwargs.get("path") or "").strip().strip("'\"")
        if not path:
            return "[view_diff] Error: path required"
        norm = _norm_rel_path(path)
        old_c = per_file_before.get(norm, "")
        new_c = per_file_staged.get(norm, "")
        if not old_c and not new_c:
            return f"[view_diff] No edits for {norm} in this session."
        old_lines = (old_c or "").splitlines(keepends=True)
        new_lines = (new_c or "").splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                old_lines, new_lines,
                fromfile=norm, tofile=norm, lineterm="", n=3,
            )
        )
        if not diff.strip():
            return f"[view_diff] No changes for {norm}."
        return f"[view_diff] {norm}:\n{diff}"

    if name == "search_replace":
        old_str = (kwargs.get("old_string") or "").strip()
        new_str = kwargs.get("new_string", "")
        path_str = (kwargs.get("path") or kwargs.get("filepath") or "").strip().strip("'\"")
        # Replace all when model omits replace_all or passes true; replace first only when model writes false
        replace_all = kwargs.get("replace_all") not in (False, "false", "0", "no")
        if not old_str or not path_str:
            return "[search_replace] Error: old_string and path required"
        edit = {"old_string": old_str, "new_string": new_str, "path": path_str, "replace_all": replace_all}
        if config.DEBUG_AGENT_LOG:
            def _dlog(loc: str, msg: str, data: dict, hid: str) -> None:
                try:
                    with open(config.DEBUG_LOG_PATH, "a") as _f:
                        _f.write(json.dumps({"location": loc, "message": msg, "data": data, "timestamp": int(time.time() * 1000)}) + "\n")
                except Exception:
                    pass
        else:
            _dlog = lambda _loc, _msg, _data, _hid: None
        _touched_before = len(touched)
        ok, msg, meta = apply_search_replace(
            edit, allowed_edit_files=None, root=get_root(), stage_only=False
        )
        _dlog("agent_loop._run_one_tool:search_replace_after", "after", {"ok": ok, "msg_preview": (msg or "")[:120], "meta_path": (meta.get("path", "").strip() if meta else ""), "touched_before": _touched_before}, "H4")
        if ok and meta:
            path = meta.get("path", "").strip()
            if path and path not in touched:
                touched.append(path)
            # Only update before/staged when we actually changed content (preserve diff for UI).
            # When "Already applied", old_content == new_content; don't overwrite or we lose the diff.
            if path:
                old_c = meta.get("old_content", "")
                new_c = meta.get("new_content", "")
                if old_c != new_c or path not in touched:
                    per_file_before[path] = old_c
                    per_file_staged[path] = new_c
            tool_log(f"search_replace -> {path or path_str}")
        _dlog("agent_loop._run_one_tool:search_replace_touched", "touched_after", {"touched_after": len(touched)}, "H4")
        dbg(
            f"agent_loop search_replace result path={path_str!r} replace_all={replace_all} "
            f"ok={ok} msg={msg!r}"
        )
        # Return exact success/fail message (no prefix)
        return msg

    if name == "write_file":
        path_str = (kwargs.get("path") or "").strip().strip("'\"")
        content = kwargs.get("content")
        if content is None:
            content = ""
        if not path_str:
            return "[write_file] Error: path required"
        root = get_root()
        try:
            rel_path = _norm_rel_path(path_str)
            abs_path = (root / rel_path).resolve()
            abs_path.relative_to(root.resolve())
        except Exception:
            return f"[write_file] Error: path outside root: {path_str}"
        if not is_edit_allowed(rel_path, allow_new=True):
            return (
                f"[write_file] Edit not allowed for {rel_path} "
                "(file not in imported/allow list; add this file to imported files to edit it)"
            )
        if is_security_concern(abs_path):
            return f"[write_file] Security concern for {rel_path}"
        old_content = abs_path.read_text() if abs_path.exists() else ""
        write_file_text(rel_path, str(content))
        if rel_path not in touched:
            touched.append(rel_path)
        per_file_before[rel_path] = old_content
        per_file_staged[rel_path] = str(content)
        tool_log(f"write_file -> {rel_path}")
        return f"[write_file] Wrote {rel_path}"

    if name == "create_new_file":
        result = execute_tool_from_kwargs(name, kwargs)
        path_str = (kwargs.get("path") or kwargs.get("filepath") or "").strip().strip('"\'')
        success = isinstance(result, str) and "Created" in result and "Error" not in result[:30]
        _log_tool_result("create_new_file", path=path_str, result=result, added_to_touched=success)
        if success:
            path_str = (kwargs.get("path") or kwargs.get("filepath") or "").strip().strip("'\"")
            if path_str:
                try:
                    rel_path = _norm_rel_path(path_str)
                    if rel_path not in touched:
                        touched.append(rel_path)
                    content = kwargs.get("content") or kwargs.get("contents") or ""
                    per_file_before[rel_path] = ""
                    per_file_staged[rel_path] = str(content)
                    tool_log(f"create_new_file -> {rel_path}")
                except Exception as e:
                    _log_tool_result("create_new_file", path=path_str, result=result, added_to_touched=False, exc=str(e))
        return result

    return execute_tool_from_kwargs(name, kwargs)


# Map API tool names to internal executor names.
# edit_existing_file has its own route (path + changes). search_replace = find-and-replace.
# single_find_and_replace is a deprecated alias for search_replace.
_API_TOOL_NAME_MAP = {
    "view_subdirectory": "list_files",
    "grep_search": "grep",
    "edit_existing_file": "edit_existing_file",
    "search_replace": "search_replace",
    "single_find_and_replace": "search_replace",
    "create_new_file": "create_new_file",
    "run_terminal_command": "run_terminal_cmd",
}

_MUTATING_TOOLS = frozenset({
    "search_replace",
    "edit_existing_file",
    "multi_edit",
    "write_file",
    "create_new_file",
})


def _tool_call_signature(name: str, args: Dict[str, Any]) -> str:
    payload = {
        "name": (name or "").strip().lower(),
        "args": args or {},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _parse_api_tool_calls(tool_calls_api: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Convert API tool_calls to list of (name, args) for shared execute path."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    for tc in tool_calls_api or []:
        name = (tc.get("function", {}).get("name") or "").strip().lower()
        args_str = tc.get("function", {}).get("arguments") or "{}"
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except Exception:
            args = {}
        args = {str(k).lower(): (v if isinstance(v, str) else str(v) if v is not None else "") for k, v in args.items()}
        if name == "read_file" and "filepath" in args and "path" not in args:
            args["path"] = args["filepath"]
        if name == "edit_existing_file" and "filepath" in args and "path" not in args:
            args["path"] = args["filepath"]
        out.append((name, args))
    return out


def _execute_tool_calls(
    func_calls: List[Tuple[str, Dict[str, Any]]],
    round_num: int,
    per_file_staged: Dict[str, str],
    per_file_before: Dict[str, str],
    touched: List[str],
    applied_edits: set[str],
    repetition_detector: ToolRepetitionDetector,
    action_feed: List[Dict[str, Any]],
    on_action: Optional[Callable[[Dict[str, Any]], None]],
) -> Tuple[List[str], List[str]]:
    """Execute all tool calls; return (results, results_for_turn). Caller may truncate and prepend 'Already edited'."""
    results: List[str] = []
    for tool_name, kw in func_calls:
        internal_name = _API_TOOL_NAME_MAP.get(tool_name, tool_name)
        action = {
            "type": "tool_call",
            "tool": tool_name,
            "args": dict(kw or {}),
            "round": round_num,
        }
        action_feed.append(action)
        if on_action:
            try:
                on_action(action)
            except Exception:
                pass
        tool_args = kw or {}
        sig = _tool_call_signature(internal_name, tool_args)
        repetition_check = repetition_detector.check(internal_name, tool_args)
        if not repetition_check.allow_execution:
            result = repetition_check.message or (
                f"Tool call repetition limit reached for {internal_name}. Please try a different approach."
            )
            dbg(f"agent_loop: repetition guard blocked tool={internal_name} sig={sig[:120]}")
        elif internal_name in _MUTATING_TOOLS and sig in applied_edits:
            result = "Already applied. Do not call this edit again; do the next task or respond with a brief summary and no further tool calls."
            dbg(f"agent_loop: duplicate edit blocked tool={internal_name}")
        else:
            result = _run_one_tool(
                internal_name,
                tool_args,
                per_file_staged=per_file_staged,
                per_file_before=per_file_before,
                touched=touched,
            )
            if internal_name in _MUTATING_TOOLS and isinstance(result, str):
                if internal_name == "search_replace" and result.startswith("Successfully edited "):
                    applied_edits.add(sig)
                elif internal_name == "write_file" and "Wrote" in result and "Error" not in result[:30]:
                    applied_edits.add(sig)
                elif internal_name == "multi_edit" and not result.startswith("[multi_edit] Error"):
                    applied_edits.add(sig)
                elif internal_name == "edit_existing_file" and "Error" not in result[:80]:
                    applied_edits.add(sig)
                elif internal_name == "create_new_file" and "Created" in result:
                    applied_edits.add(sig)
        preview = (result[:220] + "...") if isinstance(result, str) and len(result) > 220 else result
        dbg(
            f"agent_loop tool_result round={round_num} tool={internal_name} "
            f"len={len(result) if isinstance(result, str) else 0} preview={preview!r}"
        )
        results.append(result)
    return results, list(results)


def _build_meta(
    touched: List[str],
    per_file_before: Dict[str, str],
    per_file_staged: Dict[str, str],
    action_feed: List[Dict[str, Any]],
    duration_ms: int,
    output: str,
    mode_used: str = "agent",
    ok: bool = True,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Build meta dict shared by completion and API paths."""
    per_file_diffs: Dict[str, str] = {}
    for path in touched:
        old_c = per_file_before.get(path, "")
        new_c = per_file_staged.get(path, "")
        old_lines = (old_c or "").splitlines(keepends=True)
        new_lines = (new_c or "").splitlines(keepends=True)
        per_file_diffs[path] = "".join(
            difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path, lineterm="", n=3)
        )
    if summary is None:
        summary = f"Applied edits for: {', '.join(touched)}" if touched else (output or "No tool calls.")
    return {
        "mode_used": mode_used,
        "ok": ok,
        "summary": summary,
        "touched": touched,
        "files_changed": touched,
        "per_file_diffs": per_file_diffs,
        "per_file_before": per_file_before,
        "per_file_staged": per_file_staged,
        "per_file_after": dict(per_file_staged),
        "staged": bool(per_file_staged),
        "applied_directly": True,
        "agent_actions": action_feed,
        "duration_ms": duration_ms,
        "output": output[:2000],
    }


def run_agent(
    spec: str,
    focus_file: Optional[str] = None,
    mode: str = "agent",
    silent: bool = True,
    max_rounds: Optional[int] = None,
    on_action: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    extra_read_files: Optional[List[str]] = None,
    context_folders: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run agent tool loop. Returns (output_str, meta).

    on_action: optional callback called for each tool invocation with
      {"type": "tool_call", "tool": name, "args": kwargs, "round": n}.
    on_chunk: optional callback called for each streamed token/chunk of model output.
    extra_read_files: optional paths to inject as Reference files.
    context_folders: optional folder paths for @Folder context (list + content).
    """
    del silent
    # 0 = no cap; > 0 = safety limit
    _max = max_rounds if max_rounds is not None else int(getattr(config, "MAX_TOOL_ROUNDS", 0))
    rounds = _max if _max > 0 else 0  # 0 means no cap

    use_chat_tools = getattr(config, "USE_CHAT_TOOLS", False)
    initial_content = _build_prompt(
        spec,
        focus_file=focus_file,
        mode=mode,
        extra_read_files=extra_read_files,
        context_folders=context_folders,
    )
    # For API path, state_to_messages uses system + user (same structure as _run_agent_api_tools: system only, no TOOLS in system)
    system_msg = getattr(
        config,
        "BASE_AGENT_SYSTEM_MESSAGE",
        "You are a systematic coding agent. Break down problems methodically.",
    )
    if use_chat_tools:
        initial_system = system_msg
        initial_user = (
            f"{_workspace_paths_section()}"
            f"{_code_context_section(focus_file, extra_read_files)}"
            f"{_folder_context_section(context_folders)}"
            f"{_reference_files_section(extra_read_files)}"
            f"User request:\n{(spec or '').strip()}"
            f"{_focus_file_section(focus_file)}"
        )
    else:
        initial_system = ""
        initial_user = ""
    state = ConversationState(
        initial_content=initial_content,
        turns=[],
        initial_system=initial_system,
        initial_user=initial_user,
    )
    cache_key = get_session_cache_key()
    stop_sequences = ["\n\nUser:", "\n\nSYSTEM:", "\n\nCONTEXT:", "\n\nHISTORY:"]
    max_new = int(getattr(config, "MAX_NEW", 2048))
    READ_ONLY_TOOLS = frozenset({
        "read_file", "list_files", "grep", "glob_file_search", "codebase_search",
        "view_repo_map", "view_diff",
    })
    READ_ONLY_TOOLS_API = frozenset({
        "read_file", "list_files", "grep", "view_subdirectory", "grep_search",
        "glob_search", "codebase_search", "view_repo_map", "view_diff", "symbols",
    })

    per_file_staged: Dict[str, str] = {}
    per_file_before: Dict[str, str] = {}
    touched: List[str] = []
    applied_edits: set[str] = set()
    repetition_detector = ToolRepetitionDetector(getattr(config, "TOOL_REPETITION_LIMIT", 3))
    action_feed: List[Dict[str, Any]] = []
    final_reply = ""
    start = time.time()
    nudge_sent = False
    round_num = 0
    next_user_msg = "User: Proceed. Use the tool results above.\nAssistant:"

    while True:
        round_num += 1
        if rounds > 0 and round_num > rounds:
            dbg(f"agent_loop: reached cap rounds={rounds}, stopping")
            break

        # --- Get model reply (backend-specific) ---
        if use_chat_tools:
            try:
                reply, tool_calls_api = get_reply_api(state, max_new=max_new)
            except Exception as e:
                dbg(f"agent_loop api round {round_num} error: {e}")
                output = (final_reply or str(e)).strip()
                meta = _build_meta(
                    touched=touched,
                    per_file_before=per_file_before,
                    per_file_staged=per_file_staged,
                    action_feed=action_feed,
                    duration_ms=int((time.time() - start) * 1000),
                    output=output,
                    mode_used="agent_api_tools",
                    ok=False,
                    summary=str(e),
                )
                return output, meta
            # Parse API tool_calls to (name, args) for shared execute path
            func_calls = _parse_api_tool_calls(tool_calls_api)
        else:
            print(f"[agent] round={round_num} starting stream_reply_chunks", file=sys.stderr)
            sys.stderr.flush()
            first_token = True
            def _on_chunk(t: str) -> None:
                nonlocal first_token
                if first_token:
                    print(f"[agent] first token: {repr(t[:80])}{'...' if len(t) > 80 else ''}", file=sys.stderr)
                    sys.stderr.flush()
                    first_token = False
                if on_chunk:
                    try:
                        on_chunk(t)
                    except Exception:
                        pass
            reply = get_reply_completion(
                state,
                next_user_msg,
                cache_key=cache_key,
                max_new=max_new,
                stop_sequences=stop_sequences,
                on_chunk=_on_chunk,
            )
            func_calls = extract_function_style_tool_calls(reply)
            tool_calls_api = []  # completion path has no API tool_calls

        # Log attempted (parsed) tool calls to debug log and stderr for diagnostics
        if func_calls:
            tool_log(
                f"attempted tool calls ({len(func_calls)}): {[(n, list(k.keys()) if isinstance(k, dict) else []) for n, k in func_calls]}"
            )
            _log_parse_result(round_num, reply, parsed=func_calls)
        else:
            tool_log("attempted tool calls (0): none parsed from reply")
            _log_parse_failure(round_num, reply)

        final_reply = reply

        if not func_calls:
            dbg(f"agent_loop: round={round_num} model reply (no tool calls): {repr(reply[:500])}")
            if not touched and not nudge_sent and not use_chat_tools:
                nudge_sent = True
                state.turns.append(Turn(
                    assistant_content=reply,
                    tool_calls=[],
                    tool_results=[],
                    api_tool_calls=None,
                ))
                next_user_msg = (
                    "User: You have not made any file edits yet. Use search_replace or edit_existing_file or write_file to apply the requested code changes. Output your tool call(s) now.\nAssistant:"
                )
                dbg("agent_loop: no tool calls and no edits yet, sending nudge")
                continue
            break

        # --- Execute tools (shared) ---
        results, results_for_turn = _execute_tool_calls(
            func_calls=func_calls,
            round_num=round_num,
            per_file_staged=per_file_staged,
            per_file_before=per_file_before,
            touched=touched,
            applied_edits=applied_edits,
            repetition_detector=repetition_detector,
            action_feed=action_feed,
            on_action=on_action,
        )
        # Prepend "Already edited" to first result when needed (both backends)
        if touched and results_for_turn:
            results_for_turn[0] = (
                f"[Already edited this session: {', '.join(touched)}. Do not repeat.]\n\n" + results_for_turn[0]
            )
            results[0] = results_for_turn[0]

        max_chars = int(getattr(config, "MAX_TOOL_RESULT_CHARS", 0) or 0)
        combined = "\n\n".join(results_for_turn)
        if max_chars > 0 and len(combined) > max_chars:
            combined = combined[:max_chars] + "\n\n...[truncated]"
            results_for_turn = [combined]
        state.turns.append(Turn(
            assistant_content=reply,
            tool_calls=func_calls,
            tool_results=results_for_turn,
            api_tool_calls=tool_calls_api if use_chat_tools else None,
        ))

        # Next round: compute next_user_msg (completion path) and/or state is updated for API
        next_user_msg = "User: Proceed. Use the tool results above.\nAssistant:"
        if touched:
            next_user_msg = f"User: Files already edited this session: {', '.join(touched)}. Do not repeat these edits.\n\n" + next_user_msg
        read_only_tools = READ_ONLY_TOOLS_API if use_chat_tools else READ_ONLY_TOOLS
        if not touched and round_num >= 2 and all(
            _API_TOOL_NAME_MAP.get(t[0], t[0]) in read_only_tools for t in func_calls
        ):
            next_user_msg = (
                "User: You have only been reading files so far. If the user asked for a code change or fix, "
                "output search_replace(...) or edit_existing_file(...) or write_file(...) now. Use the tool results above. Do not only read more files.\nAssistant:"
            )
            dbg("agent_loop: read-only round with no edits yet, nudging to edit")
        dbg(f"agent_loop: round={round_num} tools={len(func_calls)} touched={len(touched)}")

    output = strip_tool_calls_from_output(final_reply).strip()
    output = strip_tool_call_code_blocks(output).strip()
    duration_ms = int((time.time() - start) * 1000)

    per_file_diffs: Dict[str, str] = {}
    for path in touched:
        old_c = per_file_before.get(path, "")
        new_c = per_file_staged.get(path, "")
        if generate_diff:
            try:
                per_file_diffs[path] = generate_diff(old_c, new_c, path)
            except Exception:
                pass
        if path not in per_file_diffs:
            old_lines = (old_c or "").splitlines(keepends=True)
            new_lines = (new_c or "").splitlines(keepends=True)
            per_file_diffs[path] = "".join(
                difflib.unified_diff(
                    old_lines, new_lines,
                    fromfile=path, tofile=path, lineterm="", n=3,
                )
            )

    summary = (
        f"Applied edits for: {', '.join(touched)}"
        if touched
        else (output or "No tool calls.")
    )
    mode_used = "agent_api_tools" if use_chat_tools else "agent"
    meta = _build_meta(
        touched=touched,
        per_file_before=per_file_before,
        per_file_staged=per_file_staged,
        action_feed=action_feed,
        duration_ms=duration_ms,
        output=output,
        mode_used=mode_used,
        ok=True,
        summary=summary,
    )
    # Override per_file_diffs if we used generate_diff (completion path)
    if not use_chat_tools and per_file_diffs:
        meta["per_file_diffs"] = per_file_diffs
    # Prefer model's explanation when present; fall back to summary
    return (output.strip() or summary), meta
