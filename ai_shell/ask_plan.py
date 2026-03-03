"""Ask and Plan routes: read-only tool loops with separate prompts. Do not modify agent_loop."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config
from .agent_loop import (
    ConversationState,
    Turn,
    get_reply_completion,
    state_to_prompt,
    _workspace_paths_section,
    _reference_files_section,
    _focus_file_section,
    _code_context_section,
    _folder_context_section,
)
from .tool_executor import (
    ASK_TOOLS_HINT,
    PLAN_TOOLS_HINT,
    WRITE_TOOL_NAMES,
    execute_tool_from_kwargs,
    extract_function_style_tool_calls,
    strip_tool_calls_from_output,
    strip_tool_call_code_blocks,
)
from .model import get_session_cache_key
from .utils import dbg


def _build_ask_prompt(
    spec: str,
    focus_file: Optional[str],
    extra_read_files: Optional[List[str]] = None,
    context_folders: Optional[List[str]] = None,
) -> str:
    """Build prompt for Ask mode: read-only tools, answer in text."""
    instruction = (spec or "").strip()
    system_msg = getattr(
        config,
        "BASE_AGENT_SYSTEM_MESSAGE",
        "You are a systematic coding agent. Break down problems methodically.",
    )
    return (
        f"{system_msg}\n\n"
        f"{ASK_TOOLS_HINT.strip()}\n\n"
        f"{_workspace_paths_section()}"
        f"{_code_context_section(focus_file, extra_read_files)}"
        f"{_folder_context_section(context_folders)}"
        f"{_reference_files_section(extra_read_files)}"
        f"User request:\n{instruction}"
        f"{_focus_file_section(focus_file)}"
    )


def _build_plan_prompt(
    spec: str,
    focus_file: Optional[str],
    extra_read_files: Optional[List[str]] = None,
    context_folders: Optional[List[str]] = None,
) -> str:
    """Build prompt for Plan mode: read-only tools, explore and produce a plan."""
    instruction = (spec or "").strip()
    system_msg = getattr(
        config,
        "BASE_AGENT_SYSTEM_MESSAGE",
        "You are a systematic coding agent. Break down problems methodically.",
    )
    return (
        f"{system_msg}\n\n"
        f"{PLAN_TOOLS_HINT.strip()}\n\n"
        f"{_workspace_paths_section()}"
        f"{_code_context_section(focus_file, extra_read_files)}"
        f"{_folder_context_section(context_folders)}"
        f"{_reference_files_section(extra_read_files)}"
        f"User request:\n{instruction}"
        f"{_focus_file_section(focus_file)}"
    )


def _run_read_only_loop(
    initial_content: str,
    mode_used: str,
    reject_message: str,
    on_action: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    max_rounds: int = 20,
) -> Tuple[str, Dict[str, Any]]:
    """Shared loop for ask/plan: completion backend, execute only read-only tools."""
    state = ConversationState(initial_content=initial_content, turns=[], initial_system="", initial_user="")
    cache_key = get_session_cache_key()
    stop_sequences = ["\n\nUser:", "\n\nSYSTEM:", "\n\nCONTEXT:", "\n\nHISTORY:"]
    max_new = int(getattr(config, "MAX_NEW", 2048))
    start = time.time()
    next_user_msg = "User: Proceed. Use the tool results above.\nAssistant:"
    round_num = 0

    while True:
        round_num += 1
        if max_rounds > 0 and round_num > max_rounds:
            dbg(f"ask_plan: reached cap rounds={max_rounds}, stopping")
            break

        first_token = True

        def _on_chunk(t: str) -> None:
            nonlocal first_token
            if first_token:
                print(f"[{mode_used}] round={round_num} first token", file=sys.stderr)
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
        if not func_calls:
            break

        results_for_turn: List[str] = []
        for name, kwargs in func_calls:
            name_lower = (name or "").strip().lower()
            if on_action:
                try:
                    on_action({"type": "tool_call", "tool": name_lower, "args": kwargs or {}, "round": round_num})
                except Exception:
                    pass
            if name_lower in WRITE_TOOL_NAMES:
                results_for_turn.append(reject_message)
            else:
                try:
                    result = execute_tool_from_kwargs(name_lower, kwargs or {})
                    results_for_turn.append(result or "")
                except Exception as e:
                    results_for_turn.append(f"[{name_lower}] Error: {e}")

        combined = "\n\n".join(results_for_turn)
        state.turns.append(
            Turn(assistant_content=reply, tool_calls=func_calls, tool_results=results_for_turn, api_tool_calls=None)
        )
        next_user_msg = "User: Proceed. Use the tool results above.\nAssistant:"

    output = strip_tool_calls_from_output(reply).strip()
    output = strip_tool_call_code_blocks(output).strip()
    duration_ms = int((time.time() - start) * 1000)
    meta = {
        "mode_used": mode_used,
        "ok": True,
        "summary": output[:500] if output else "Done.",
        "touched": [],
        "files_changed": [],
        "per_file_diffs": {},
        "per_file_before": {},
        "per_file_staged": {},
        "per_file_after": {},
        "staged": False,
        "applied_directly": True,
        "agent_actions": [],
        "duration_ms": duration_ms,
        "output": (output or "")[:2000],
    }
    return (output or "").strip(), meta


def run_ask(
    text: str,
    focus_file: Optional[str] = None,
    silent: bool = True,
    on_action: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    extra_read_files: Optional[List[str]] = None,
    context_folders: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Ask route: read-only tools, answer in text. Returns (output, meta)."""
    del silent
    initial_content = _build_ask_prompt(text, focus_file, extra_read_files, context_folders)
    return _run_read_only_loop(
        initial_content=initial_content,
        mode_used="ask",
        reject_message="Ask mode: no file edits. Use Agent mode to edit.",
        on_action=on_action,
        on_chunk=on_chunk,
    )


def run_plan(
    text: str,
    focus_file: Optional[str] = None,
    silent: bool = True,
    on_action: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    extra_read_files: Optional[List[str]] = None,
    context_folders: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Plan route: read-only tools, explore and produce a plan. Returns (output, meta)."""
    del silent
    initial_content = _build_plan_prompt(text, focus_file, extra_read_files, context_folders)
    return _run_read_only_loop(
        initial_content=initial_content,
        mode_used="plan",
        reject_message="Plan mode: file edits are disabled. Use Agent mode to apply changes.",
        on_action=on_action,
        on_chunk=on_chunk,
    )
