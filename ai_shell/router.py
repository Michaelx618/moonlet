from dataclasses import dataclass
from enum import Enum
from typing import List
import re
import os


class Route(str, Enum):
    # Canonical 3-route model
    FILE_EDIT = "file_edit"
    PLAN_MULTI = "plan_multi"
    # Backward-compatible aliases (keep old names stable for callers/logs).
    PLAN_EXECUTE = "plan_multi"
    FILEBLOCK_EDIT = "file_edit"
    FILEBLOCK_WRITE = "file_edit"


STUB_PLAN_THRESHOLD = int(os.getenv("MOONLET_STUB_PLAN_THRESHOLD", "800"))


@dataclass
class RequestContext:
    request_len: int
    file_lines: int
    file_chars: int
    target_files: List[str]
    intent: str
    wants_minimal_diff: bool
    has_assignment_text: bool
    file_is_empty: bool
    rewrite_requested: bool
    file_has_todos: bool
    has_syntax_critical_todos: bool
    is_c_like: bool
    analysis_missing_symbols_count: int = 0
    analysis_touch_points_count: int = 0
    analysis_has_local_refs: bool = False

    @property
    def single_file(self) -> bool:
        return len(self.target_files) <= 1


def decide_route_with_reason(ctx: RequestContext) -> tuple[Route, str]:
    # 1) Multi-file edits: plan/execute path.
    if len(ctx.target_files) > 1:
        return Route.PLAN_MULTI, "multi_file_explicit"

    # 2) Empty/new or explicit rewrite requests: file edit.
    if ctx.file_is_empty or ctx.rewrite_requested:
        return Route.FILE_EDIT, "empty_or_rewrite"

    # 3) Default single-file route is FILE_EDIT.
    if ctx.single_file:
        if (ctx.intent or "").upper() == "IMPLEMENT_STUBS":
            return Route.FILE_EDIT, "single_file_stubs"
        return Route.FILE_EDIT, "single_file_default"

    # Fallback (should be unreachable with current context shape).
    return Route.FILE_EDIT, "default_file_edit"


def decide_route(ctx: RequestContext) -> Route:
    route, _ = decide_route_with_reason(ctx)
    return route


def wants_file_rewrite(user_text: str) -> bool:
    if not user_text:
        return False
    low = user_text.lower()
    patterns = (
        r"\brewrite (the )?(entire|whole|full) (file|module)\b",
        r"\breplace (the )?(entire|whole|full) (file|module)\b",
        r"\brefactor (the )?(entire|whole|full) (file|module)\b",
        r"\brewrite the file\b",
        r"\bfull file rewrite\b",
        r"\boverwrite the file\b",
    )
    return any(re.search(p, low) for p in patterns)


def has_assignment_style_text(user_text: str) -> bool:
    if not user_text:
        return False
    lowered = user_text.lower()
    markers = (
        "task 1", "task 2", "assignment", "lab", "problem set",
        "download", "due", "points", "grading", "requirements",
        "starter code", "what to submit", "our goal this week", "week's lab",
    )
    return any(m in lowered for m in markers)


def wants_minimal_diff(user_text: str) -> bool:
    if not user_text:
        return False
    low = user_text.lower()
    patterns = (
        r"\bminimal diff\b",
        r"\bsmall diff\b",
        r"\bdiff only\b",
        r"\bpatch only\b",
        r"\bline[- ]?edits?\b",
        r"\bdo not rewrite\b",
        r"\bkeep file structure\b",
    )
    return any(re.search(p, low) for p in patterns)
