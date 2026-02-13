import json
import os
import tempfile
import time
from typing import Dict, List, Tuple

from . import config


def _state_path() -> str:
    return os.path.expanduser(config.STATE_PATH)


def load_state() -> Dict:
    """Load state from disk or return empty dict."""
    path = _state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except Exception:
        return {}


def save_state(state: Dict) -> None:
    """Persist state to disk atomically."""
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".moonlet_state.", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=True, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def get_project_summary() -> str:
    """Return stored project summary or empty string."""
    state = load_state()
    return str(state.get("project_summary") or "")


def set_project_summary(summary: str) -> None:
    """Store project summary."""
    state = load_state()
    state["project_summary"] = summary or ""
    save_state(state)


def append_change_note(note: str) -> None:
    """Append a change note, keeping most recent."""
    if not note:
        return
    state = load_state()
    notes = list(state.get("change_notes") or [])
    notes.append(note)
    state["change_notes"] = notes[-MAX_CHANGE_NOTES:]
    save_state(state)


def get_change_notes() -> List[str]:
    """Return recent change notes."""
    state = load_state()
    return list(state.get("change_notes") or [])


MAX_CHAT_HISTORY = 7
MAX_CHANGE_NOTES = 2
MAX_FAILURE_HISTORY = 7

def append_chat_turn(user_text: str, assistant_text: str) -> None:
    """Append a chat turn to history."""
    if config.DISABLE_HISTORY:
        return
    if not user_text and not assistant_text:
        return
    state = load_state()
    history = list(state.get("chat_history") or [])
    history.append({"user": user_text or "", "assistant": assistant_text or ""})
    state["chat_history"] = history[-MAX_CHAT_HISTORY:]
    save_state(state)


def get_recent_chat(limit: int = MAX_CHAT_HISTORY) -> List[Tuple[str, str]]:
    """Return recent chat turns as (user, assistant)."""
    if config.DISABLE_HISTORY:
        return []
    state = load_state()
    history = list(state.get("chat_history") or [])
    tail = history[-limit:]
    return [(entry.get("user", ""), entry.get("assistant", "")) for entry in tail]


def clear_chat_session() -> None:
    """Clear persisted chat/failure/change memory for a fresh conversation."""
    state = load_state()
    state["chat_history"] = []
    state["failure_history"] = []
    state["change_notes"] = []
    save_state(state)


def append_failure_note(
    mode: str,
    focus_file: str,
    kind: str,
    summary: str,
) -> None:
    """Persist compact failure memory for prompt-time guidance."""
    if config.DISABLE_HISTORY:
        return
    s = " ".join((summary or "").split()).strip()
    if not s:
        return
    state = load_state()
    notes = list(state.get("failure_history") or [])

    # Deduplicate nearby repeats.
    key = f"{mode}|{focus_file}|{kind}|{s}".strip()
    if notes:
        last = notes[-1]
        last_key = f"{last.get('mode','')}|{last.get('focus_file','')}|{last.get('kind','')}|{last.get('summary','')}"
        if key == last_key:
            return

    notes.append(
        {
            "mode": (mode or "")[:32],
            "focus_file": (focus_file or "")[:180],
            "kind": (kind or "")[:32],
            "summary": s,
            "ts": int(time.time()),
        }
    )
    state["failure_history"] = notes[-MAX_FAILURE_HISTORY:]
    save_state(state)


def get_recent_failures(limit: int = 2) -> List[Dict[str, str]]:
    """Return recent failure notes in ascending timestamp order."""
    if config.DISABLE_HISTORY:
        return []
    state = load_state()
    notes = list(state.get("failure_history") or [])
    tail = notes[-max(1, int(limit)) :]
    out: List[Dict[str, str]] = []
    for n in tail:
        out.append(
            {
                "mode": str(n.get("mode") or ""),
                "focus_file": str(n.get("focus_file") or ""),
                "kind": str(n.get("kind") or ""),
                "summary": str(n.get("summary") or ""),
                "ts": str(n.get("ts") or ""),
            }
        )
    return out
