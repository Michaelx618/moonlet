"""Agent loop: tool-in-the-loop flow (design borrowed from Continue; see BORROWED_FROM_CONTINUE.md).

Flow:

  1. Tools are sent with the user request (we inject TOOLS_SYSTEM_HINT + optional context).
  2. Model may include tool call(s) in its response.
  3. We execute each tool (built-in: read_file, grep, search_replace, write_file, etc.).
  4. We send tool results back to the model as context (same as Continue: "fed back into the model").
  5. Model responds again (possibly with more tool calls) → repeat from step 3 until the model
     responds with no tool calls (then that reply is the final answer).

How the model sees files (us vs Continue):
- Continue: the model discovers files by calling tools. It gets tool schemas + user message;
  it has no built-in list of workspace paths. It calls view_subdirectory / list_dir / read_file
  to see what exists and then edit_existing_file / create_new_file with the paths it found.
- Us: we inject a workspace path list into the prompt (_workspace_paths_section from the index)
  so the model sees real paths (e.g. checkpasswd.c, validate.c) up front and is told not to
  create new top-level dirs. It can still call list_files(path=".") to see more. We also
  inject optional Reference files (extra_read_files) and Current file (focus_file) content.
  So we give path list + optional file contents; Continue relies on the model calling tools
  to discover paths. If we didn't inject paths and the model never called list_files, it
  could "guess" a path (e.g. w7/checkpasswd.c) and create a new folder by mistake.

Why it works in Continue but not always here:
- Continue uses either (a) native tool calling (OpenAI/Anthropic/etc. return structured
  tool_calls in the API response, no free-text parsing) or (b) "system message tools":
  tools described in XML in the system message, model returns XML tool calls, they parse XML.
- We use a completion-style API (single prompt → text reply) and parse function-style
  text like search_replace(old_string="...", path="..."). If the model outputs a different
  format (e.g. code block, XML, or prose), we don't detect tool calls. For full parity,
  use a backend with native tool support (USE_CHAT_TOOLS + /v1/chat/completions with tools)
  or add XML tool parsing to match Continue's fallback path.
"""

from __future__ import annotations

import difflib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config
from .files import (
    _norm_rel_path,
    get_root,
    is_edit_allowed,
    read_single_file_for_context,
    write_file_text,
)
from .index import get_indexed_files, get_symbols_for_file
from .model import get_session_cache_key, stream_reply_chunks
from .model import chat_completion_with_tools
from .search_replace import apply_search_replace
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

try:
    from file_utils import generate_diff, is_security_concern
except ImportError:
    generate_diff = None

    def is_security_concern(*args, **kwargs):
        return False


# Junk paths to exclude from context
_REFERENCE_SKIP = (".dsym/", "contents/info.plist", "/spec.txt", "/task.json", "node_modules/", "__pycache__/")


def _reference_files_section(extra_read_files: Optional[List[str]]) -> str:
    """Build Reference files block from extra_read_files. No truncation (same policy as llama.cpp)."""
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
    parts: List[str] = []
    for rel in filtered:
        try:
            norm = _norm_rel_path(rel)
            path = (root / norm).resolve()
            path.relative_to(root.resolve())
            if not path.exists() or not path.is_file():
                continue
            content_map = read_single_file_for_context(norm)
            content = content_map.get(norm) or ""
            parts.append(f"--- {norm} ---\n{content}\n(Use path \"{norm}\" in tool calls.)")
        except Exception:
            continue
    if not parts:
        return ""
    return "\n\nReference files (priority context):\n" + "\n\n".join(parts)


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
    """Current file content. No truncation (same policy as llama.cpp)."""
    if not focus_file:
        return ""
    root = get_root()
    try:
        rel = _norm_rel_path(focus_file)
        path = (root / rel).resolve()
        path.relative_to(root.resolve())
        if not path.exists() or not path.is_file():
            return ""
        content = path.read_text()
        return (
            f"\n\nCurrent file (path for tools: \"{rel}\"):\n---\n{content}\n---\n"
            f"(Use path \"{rel}\" in edit_existing_file/write_file; do not create a new directory.)"
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
                # No truncation (match Continue)
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
        lines = [f"@Folder ({folder_norm or '.'}):"] + [f"  {f}" for f in in_folder]
        # Optionally add full content of first few files (no truncation, same policy as llama.cpp)
        for f in in_folder[:3]:
            try:
                path = (root / f).resolve()
                if path.exists() and path.is_file():
                    content = path.read_text()
                    lines.append(f"\n--- {f} ---\n{content}")
            except Exception:
                pass
        parts.append("\n".join(lines))
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
    # Continue: "Agent mode: All tools available for making changes"; tools sent with user request
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
        if not old_str or not path_str:
            return "[search_replace] Error: old_string and path required"
        edit = {"old_string": old_str, "new_string": new_str, "path": path_str}
        # #region agent log
        import json as _json
        import time as _time
        def _dlog(loc: str, msg: str, data: dict, hid: str) -> None:
            try:
                with open("/Users/michael/moonlet/.cursor/debug-c7e239.log", "a") as _f:
                    _f.write(_json.dumps({"sessionId": "c7e239", "hypothesisId": hid, "location": loc, "message": msg, "data": data, "timestamp": int(_time.time() * 1000)}) + "\n")
            except Exception:
                pass
        _touched_before = len(touched)
        # #endregion
        ok, msg, meta = apply_search_replace(
            edit, allowed_edit_files=None, root=get_root(), stage_only=False
        )
        # #region agent log
        _meta_path = meta.get("path", "").strip() if meta else ""
        _dlog("agent_loop._run_one_tool:search_replace_after", "after", {"ok": ok, "msg_preview": (msg or "")[:120], "meta_path": _meta_path, "touched_before": _touched_before}, "H4")
        # #endregion
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
        # #region agent log
        _dlog("agent_loop._run_one_tool:search_replace_touched", "touched_after", {"touched_after": len(touched)}, "H4")
        # #endregion
        # Continue-style: return exact success/fail message (no prefix)
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

    return execute_tool_from_kwargs(name, kwargs)


# Map API tool names to internal executor names. edit_existing_file = path+changes only; search_replace = find-and-replace.
_API_TOOL_NAME_MAP = {
    "view_subdirectory": "list_files",
    "grep_search": "grep",
    "edit_existing_file": "search_replace",  # overridden when args has "changes" -> edit_existing_file
    "search_replace": "search_replace",
    "single_find_and_replace": "search_replace",
    "create_new_file": "create_new_file",
    "run_terminal_command": "run_terminal_cmd",
}


def _run_agent_api_tools(
    spec: str,
    focus_file: Optional[str],
    mode: str,
    extra_read_files: Optional[List[str]],
    context_folders: Optional[List[str]],
    max_rounds: int,  # 0 = no cap (Continue-style)
    on_action: Optional[Callable[[Dict[str, Any]], None]],
) -> Tuple[str, Dict[str, Any]]:
    """Run agent using chat_completion_with_tools (native API tool calls)."""
    system_msg = getattr(
        config,
        "BASE_AGENT_SYSTEM_MESSAGE",
        "You are a systematic coding agent. Break down problems methodically.",
    )
    user_content = (
        f"{_workspace_paths_section()}"
        f"{_code_context_section(focus_file, extra_read_files)}"
        f"{_folder_context_section(context_folders)}"
        f"{_reference_files_section(extra_read_files)}"
        f"User request:\n{(spec or '').strip()}"
        f"{_focus_file_section(focus_file)}"
    )
    # System message aligned with Continue (no extra append; tool descriptions carry edit guidance)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    per_file_staged: Dict[str, str] = {}
    per_file_before: Dict[str, str] = {}
    touched: List[str] = []
    action_feed: List[Dict[str, Any]] = []
    final_content = ""
    start = time.time()
    max_new = int(getattr(config, "MAX_NEW", 2048))
    READ_ONLY = frozenset({"read_file", "list_files", "grep", "view_subdirectory", "grep_search", "glob_search", "codebase_search", "view_repo_map", "view_diff", "symbols"})

    round_num = 0
    while True:
        round_num += 1
        if max_rounds > 0 and round_num > max_rounds:
            dbg(f"agent_loop api_tools: reached cap max_rounds={max_rounds}, stopping")
            break
        try:
            resp = chat_completion_with_tools(
                messages=messages,
                tools=AGENT_TOOLS_JSON,
                max_tokens=max_new,
                temperature=getattr(config, "TEMPERATURE", 0.1),
                top_p=getattr(config, "TOP_P", 0.9),
            )
        except Exception as e:
            dbg(f"agent_loop api_tools round {round_num} error: {e}")
            output = (final_content or str(e)).strip()
            meta = {
                "mode_used": "agent_api_tools",
                "ok": False,
                "summary": str(e),
                "touched": touched,
                "files_changed": touched,
                "per_file_diffs": {},
                "per_file_before": per_file_before,
                "per_file_staged": per_file_staged,
                "per_file_after": dict(per_file_staged),
                "staged": bool(per_file_staged),
                "applied_directly": True,
                "agent_actions": action_feed,
                "duration_ms": int((time.time() - start) * 1000),
                "output": output[:2000],
            }
            return output, meta

        choices = resp.get("choices") or []
        if not choices:
            break
        msg = choices[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        final_content = content
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            break

        tool_results: List[Dict[str, Any]] = []
        for tc in tool_calls:
            name = (tc.get("function", {}).get("name") or "").strip().lower()
            args_str = tc.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except Exception:
                args = {}
            args = {str(k).lower(): (v if isinstance(v, str) else str(v) if v is not None else "") for k, v in args.items()}
            if name == "read_file" and "filepath" in args and "path" not in args:
                args["path"] = args["filepath"]
            if name == "edit_existing_file":
                if "filepath" in args and "path" not in args:
                    args["path"] = args["filepath"]
                if args.get("changes") not in (None, ""):
                    internal_name = "edit_existing_file"
                else:
                    internal_name = _API_TOOL_NAME_MAP.get(name, name)
            else:
                internal_name = _API_TOOL_NAME_MAP.get(name, name)
            action = {"type": "tool_call", "tool": name, "args": args, "round": round_num}
            action_feed.append(action)
            if on_action:
                try:
                    on_action(action)
                except Exception:
                    pass
            result = _run_one_tool(
                internal_name,
                args,
                per_file_staged=per_file_staged,
                per_file_before=per_file_before,
                touched=touched,
            )
            # Debug: log read_file result so we can verify model sees post-edit content
            if internal_name == "read_file" and result and not result.strip().startswith("["):
                path_for_log = (args.get("filepath") or args.get("path") or "").strip().strip("'\"")
                dbg(
                    f"read_file round={round_num} path={path_for_log!r} len={len(result)} "
                    f"has_clamp_int={('clamp_int' in result)}"
                )
            tc_id = tc.get("id") or ""
            tool_results.append({"role": "tool", "tool_call_id": tc_id, "content": result})

        if touched and tool_results:
            tool_results[0]["content"] = (
                f"[Already edited this session: {', '.join(touched)}. Do not repeat.]\n\n" + tool_results[0]["content"]
            )
        messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})
        for tr in tool_results:
            messages.append(tr)
        dbg(f"agent_loop api_tools round={round_num} tools={len(tool_calls)} touched={len(touched)}")

    output = final_content.strip()
    output = strip_tool_call_code_blocks(output).strip()
    duration_ms = int((time.time() - start) * 1000)
    per_file_diffs: Dict[str, str] = {}
    for path in touched:
        old_c = per_file_before.get(path, "")
        new_c = per_file_staged.get(path, "")
        old_lines = (old_c or "").splitlines(keepends=True)
        new_lines = (new_c or "").splitlines(keepends=True)
        per_file_diffs[path] = "".join(
            difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path, lineterm="", n=3)
        )
    summary = f"Applied edits for: {', '.join(touched)}" if touched else (output or "No tool calls.")
    meta: Dict[str, Any] = {
        "mode_used": "agent_api_tools",
        "ok": True,
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
    # Prefer model's explanation (Continue-style) when present; fall back to summary
    return (output.strip() or summary), meta


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
    extra_read_files: optional paths to inject as Reference files (Continue-style context).
    context_folders: optional folder paths for @Folder context (list + content).
    """
    del silent
    # 0 = no cap (Continue-style); > 0 = safety limit
    _max = max_rounds if max_rounds is not None else int(getattr(config, "MAX_TOOL_ROUNDS", 0))
    rounds = _max if _max > 0 else 0  # 0 means no cap

    if getattr(config, "USE_CHAT_TOOLS", False):
        return _run_agent_api_tools(
            spec,
            focus_file=focus_file,
            mode=mode,
            extra_read_files=extra_read_files,
            context_folders=context_folders,
            max_rounds=rounds,
            on_action=on_action,
        )

    prompt = _build_prompt(
        spec,
        focus_file=focus_file,
        mode=mode,
        extra_read_files=extra_read_files,
        context_folders=context_folders,
    )
    cache_key = get_session_cache_key()
    # Use double-newline so we don't cut off mid-reply (e.g. ".\nUser" in prose)
    stop_sequences = ["\n\nUser:", "\n\nSYSTEM:", "\n\nCONTEXT:", "\n\nHISTORY:"]
    max_new = int(getattr(config, "MAX_NEW", 2048))
    READ_ONLY_TOOLS = frozenset({
        "read_file", "list_files", "grep", "glob_file_search", "codebase_search",
        "view_repo_map", "view_diff",
    })

    per_file_staged: Dict[str, str] = {}
    per_file_before: Dict[str, str] = {}
    touched: List[str] = []
    action_feed: List[Dict[str, Any]] = []
    final_reply = ""
    start = time.time()
    nudge_sent = False
    round_num = 0

    while True:
        round_num += 1
        if rounds > 0 and round_num > rounds:
            dbg(f"agent_loop: reached cap rounds={rounds}, stopping")
            break
        # Step 2: get model response (may contain tool calls)
        round_chunks: List[str] = []
        print(f"[agent] round={round_num} starting stream_reply_chunks prompt_len={len(prompt)}", file=sys.stderr)
        sys.stderr.flush()
        first_token = True
        for token in stream_reply_chunks(
            prompt,
            max_new=max_new,
            stop_sequences=stop_sequences,
            cache_key=cache_key,
        ):
            if first_token:
                print(f"[agent] first token: {repr(token[:80])}{'...' if len(token) > 80 else ''}", file=sys.stderr)
                sys.stderr.flush()
                first_token = False
            round_chunks.append(token)
            if on_chunk:
                try:
                    on_chunk(token)
                except Exception:
                    pass
        reply = "".join(round_chunks).strip()
        final_reply = reply

        func_calls = extract_function_style_tool_calls(reply)
        if not func_calls:
            # Log what the model actually said so we can fix parsing or prompt
            dbg(f"agent_loop: round={round_num} model reply (no tool calls parsed): {repr(reply[:500])}")
            # Continue-style: when model responds with no tool calls, that's the final answer; we stop.
            # Pragmatic addition: if user asked for edits and we have none yet, nudge once.
            if not touched and not nudge_sent:
                nudge_sent = True
                prompt = (
                    f"{prompt}\nAssistant:\n{reply}\n\n"
                    "User: You have not made any file edits yet. Use search_replace or edit_existing_file or write_file to apply the requested code changes. Output your tool call(s) now.\nAssistant:"
                )
                dbg("agent_loop: no tool calls and no edits yet, sending nudge")
                continue
            break

        # Step 3: execute each tool; Step 4: collect results to send back to model
        results: List[str] = []
        for tool_name, kw in func_calls:
            if tool_name == "edit_existing_file" and (kw or {}).get("changes") is not None:
                internal_name = "edit_existing_file"
            else:
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
            result = _run_one_tool(
                internal_name,
                kw,
                per_file_staged=per_file_staged,
                per_file_before=per_file_before,
                touched=touched,
            )
            results.append(result)

        combined = "\n\n".join(results)
        max_chars = int(getattr(config, "MAX_TOOL_RESULT_CHARS", 0) or 0)
        if max_chars > 0 and len(combined) > max_chars:
            combined = combined[:max_chars] + "\n\n...[truncated]"

        # Step 4/5: send tool results back to model (Continue: "fed back into the model as context")
        next_user_msg = "User: Continue. Use the tool results above.\nAssistant:"
        if touched:
            next_user_msg = f"User: Files already edited this session: {', '.join(touched)}. Do not repeat these edits.\n\n" + next_user_msg
        # If we still have no edits and this round was only read-only, nudge to output edits next
        if not touched and round_num >= 2 and all(_API_TOOL_NAME_MAP.get(t[0], t[0]) in READ_ONLY_TOOLS for t in func_calls):
            next_user_msg = (
                "User: You have only been reading files so far. If the user asked for a code change or fix, "
                "output search_replace(...) or edit_existing_file(...) or write_file(...) now. Use the tool results above. Do not only read more files.\nAssistant:"
            )
            dbg("agent_loop: read-only round with no edits yet, nudging to edit")
        prompt = (
            f"{prompt}\nAssistant:\n{reply}\n\nTool results:\n{combined}\n\n"
            f"{next_user_msg}"
        )
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
    meta: Dict[str, Any] = {
        "mode_used": "agent",
        "ok": True,
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
    # Prefer model's explanation (Continue-style) when present; fall back to summary
    return (output.strip() or summary), meta
