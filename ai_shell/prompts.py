"""Prompt helpers used by runtime paths."""

import re
from pathlib import Path
from typing import List

AGENT_SYSTEM_PROMPT = (
    "You are a code-editing assistant. Enforce TASK_CARD rules when a task card is present; "
    "otherwise follow the user request. Keep edits minimal, preserve unrelated code, "
    "keep the file buildable/syntactically valid. "
    "Before editing, consider: edge cases, error handling, and consistency with existing code. "
    "Apply the same analytical rigor you would in a code review."
).strip()

STRUCTURAL_MINIMALITY_PROMPT = (
    "IMPORTANT:\n"
    "- Make the smallest possible change.\n"
    "- Add validation and error handling when the request implies robustness (e.g., implement, complete, fix, make robust) or when the code clearly needs it. Otherwise keep changes minimal.\n"
    "- Do NOT duplicate existing logic.\n"
    "- Modify only what is necessary to satisfy the request."
)


def _ext(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")


def _language_name_for_ext(ext: str) -> str:
    return {
        "py": "Python",
        "js": "JavaScript",
        "ts": "TypeScript",
        "tsx": "TypeScript/TSX",
        "jsx": "JavaScript/JSX",
        "java": "Java",
        "kt": "Kotlin",
        "go": "Go",
        "rs": "Rust",
        "c": "C",
        "cpp": "C++",
        "cc": "C++",
        "h": "C/C++ header",
        "hpp": "C++ header",
        "json": "JSON",
        "yaml": "YAML",
        "yml": "YAML",
        "md": "Markdown",
        "sh": "Shell",
        "zsh": "Shell",
        "css": "CSS",
    }.get(ext, ext.upper() or "Text")


def _slice_request_for_focus(user_text: str, focus_file: str, max_chars: int = 1000) -> str:
    """Compact request text for planning."""
    text = (user_text or "").strip()
    if not text:
        return ""

    focus = str(focus_file or "").lower().strip()
    stem = Path(focus_file).stem.lower() if focus_file else ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text[:max_chars]

    action_re = re.compile(r"\b(implement|fix|modify|update|replace|complete|fill|remove|add|refactor|create|delete)\b", re.IGNORECASE)
    constraint_re = re.compile(r"\b(must|should|required|preserve|avoid|never|only|exactly)\b", re.IGNORECASE)

    scored: List[tuple[int, str]] = []
    seen = set()
    for ln in lines:
        key = ln.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        score = 0
        low = key
        if action_re.search(low):
            score += 4
        if constraint_re.search(low):
            score += 3
        if focus and focus in low:
            score += 4
        elif stem and stem in low:
            score += 2
        if score <= 0:
            continue
        scored.append((score, ln))

    selected = [ln for _s, ln in sorted(scored, key=lambda x: -x[0])[:18]]
    if not selected:
        selected = lines[:8]

    out_lines: List[str] = []
    used = 0
    for ln in selected:
        extra = len(ln) + (1 if out_lines else 0)
        if used + extra > max_chars:
            break
        out_lines.append(ln)
        used += extra
    out = "\n".join(out_lines).strip()
    return out if out else text[:max_chars]


def build_plan_multi_prompt(
    user_text: str,
    file_matches: list,
    focus_file: str = "",
    analysis_packet: str = "",
) -> str:
    """Build planner prompt that requests a strict JSON file plan."""
    request_text = _slice_request_for_focus(user_text, focus_file or "", max_chars=1000)

    file_sections = []
    for fm in file_matches:
        path = str(getattr(fm, "path", "") or "").strip()
        if not path:
            continue
        defs = getattr(fm, "definitions", None) or []
        defs_str = ", ".join(str(x) for x in defs[:10]) if defs else "none"
        section = f"- {path} (defines: {defs_str})"
        snippet = str(getattr(fm, "snippet", "") or "").strip()
        if snippet:
            snip_lines = snippet.splitlines()[:8]
            snip = "\n".join(f"    {l}" for l in snip_lines)
            section += f"\n  Excerpt:\n{snip}"
        file_sections.append(section)

    files_block = "\n".join(file_sections)
    analysis_block = ""
    if analysis_packet:
        analysis_block = (
            "ANALYSIS (JSON):\n"
            f"{analysis_packet}\n"
            "Do NOT copy the ANALYSIS JSON into your plan output.\n\n"
        )

    return (
        "You are a code-editing planner. Output JSON plan only.\n\n"
        f"{analysis_block}"
        f"FILES:\n{files_block}\n\n"
        f"REQUEST: {request_text}\n\n"
        "RULES:\n"
        "- JSON only. No markdown fences.\n"
        "- Return exactly one entry per file path; do not repeat the same path.\n"
        "- action must be one of: edit, create, delete.\n"
        "OUTPUT FORMAT (JSON only):\n"
        "{\"files\":[{\"path\":\"<repo-rel-path>\",\"action\":\"edit\",\"description\":\"<specific change>\",\"target\":\"\"}]}\n"
    )
