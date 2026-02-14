"""Agent and chat entry points. Freedom mode only."""

from typing import Dict, List, Optional, Tuple

from . import config, prompt_buffer, state
from .model import stream_reply
from .utils import dbg


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


# ---------- Agent (freedom mode only) ----------

agent_history: List[Tuple[str, str]] = []


def reset_structural_kv_cache(reason: str = "manual") -> None:
    """No-op; structural mode removed."""
    pass


def _diff_stats(diff_text: str) -> Tuple[int, int]:
    adds = dels = 0
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
    meta: Optional[Dict],
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


def _run_agent_core(
    user_text: str,
    focus_file: Optional[str],
    silent: bool,
    full_context: bool,
    want_meta: bool,
) -> Dict:
    from .agent_tools import run_freedom_edit

    dbg("agent: freedom mode")
    result = run_freedom_edit(user_text or "", focus_file=focus_file, silent=silent)
    result["history_user_text"] = user_text
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
    meta = result.get("meta", {}) or {}
    mode_used = meta.get("mode_used", "agent")
    history_user_text = str(result.get("history_user_text") or user_text)
    edited_file = meta.get("staged_file") or meta.get("focus_file") or (meta.get("files_changed") or [None])[0] or ""
    history_output = _history_output_for_storage(
        str(output or ""),
        meta if isinstance(meta, dict) else {},
        edited_file,
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
    edited_file = meta.get("staged_file") or meta.get("focus_file") or (meta.get("files_changed") or [None])[0] or ""
    history_output = _history_output_for_storage(
        str(output or ""),
        meta if isinstance(meta, dict) else {},
        edited_file,
    )
    if mode_used != "chat_in_agent":
        state.append_chat_turn(history_user_text, history_output)
        if str(output or "") in {"[Staged file_edit]", "[Applied file_edit]"}:
            state.append_change_note(history_output)
    return (f"{status_prefix}{output}" if status_prefix else output), meta
