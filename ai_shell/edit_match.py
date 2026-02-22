"""Continue-style search match strategies for find-and-replace.

Ported from core/edit/searchAndReplace/findSearchMatch.ts.
Strategies: exact, trimmed, flexible_whitespace, case_insensitive, whitespace_ignored.
"""

import re
from typing import List, Optional, Tuple


def _exact_match(file_content: str, search_content: str) -> Optional[Tuple[int, int]]:
    """Exact string match. Returns (start, end) or None."""
    idx = file_content.find(search_content)
    if idx == -1:
        return None
    return (idx, idx + len(search_content))


def _trimmed_match(file_content: str, search_content: str) -> Optional[Tuple[int, int]]:
    """Match search_content after stripping leading/trailing whitespace."""
    trimmed = search_content.strip()
    if not trimmed:
        return None
    idx = file_content.find(trimmed)
    if idx == -1:
        return None
    return (idx, idx + len(trimmed))


def _case_insensitive_match(file_content: str, search_content: str) -> Optional[Tuple[int, int]]:
    """Case-insensitive match. Returns (start, end) in original file_content."""
    lower_file = file_content.lower()
    lower_search = search_content.lower()
    idx = lower_file.find(lower_search)
    if idx == -1:
        return None
    return (idx, idx + len(search_content))


def _flexible_whitespace_match(file_content: str, search_content: str) -> Optional[Tuple[int, int]]:
    """Match with optional whitespace between tokens. Accepts 'incomplete' strings:
    e.g. model sends 'clamp(' and file has 'clamp (' or 'clamp  (' - still matches.
    """
    trimmed = search_content.strip()
    if not trimmed:
        return None
    # Allow optional whitespace between non-whitespace tokens
    tokens = re.split(r"\s+", trimmed)
    # For single token (e.g. 'clamp('), allow \\s* before punctuation so 'clamp (' matches
    if len(tokens) == 1:
        pattern = re.escape(tokens[0])
        for punct in ("(", ")", ",", ";", ":", "[", "]", "{", "}"):
            pattern = pattern.replace(re.escape(punct), r"\s*" + re.escape(punct))
    else:
        pattern = r"\s*".join(re.escape(t) for t in tokens)
    m = re.search(pattern, file_content)
    if m is None:
        return None
    return (m.start(), m.end())


def _whitespace_ignored_match(file_content: str, search_content: str) -> Optional[Tuple[int, int]]:
    """Match ignoring all whitespace (spaces, tabs, newlines). Maps back to original positions."""
    stripped_file = "".join(c for c in file_content if not c.isspace())
    stripped_search = "".join(c for c in search_content if not c.isspace())
    if not stripped_search:
        return None
    idx = stripped_file.find(stripped_search)
    if idx == -1:
        return None
    # Map stripped index back to original: count non-whitespace chars to get start
    orig_start = -1
    non_ws = 0
    for i, c in enumerate(file_content):
        if not c.isspace():
            if non_ws == idx:
                orig_start = i
                break
            non_ws += 1
    if orig_start == -1:
        return None
    # Find end: orig_start + length of search in original (same non-ws count)
    need = len(stripped_search)
    non_ws = 0
    orig_end = orig_start
    for i in range(orig_start, len(file_content)):
        if not file_content[i].isspace():
            non_ws += 1
            if non_ws == need:
                orig_end = i + 1
                break
        orig_end = i + 1
    return (orig_start, orig_end)


# Order of strategies (same as Continue; flexible_whitespace added for incomplete/model strings)
_MATCH_STRATEGIES: List[Tuple[str, object]] = [
    ("exact", _exact_match),
    ("trimmed", _trimmed_match),
    ("flexible_whitespace", _flexible_whitespace_match),
    ("case_insensitive", _case_insensitive_match),
    ("whitespace_ignored", _whitespace_ignored_match),
]


def find_search_match(
    file_content: str,
    search_content: str,
) -> Optional[Tuple[int, int, str]]:
    """Find a single match. Returns (start, end, strategy_name) or None.

    Tries strategies in order: exact, trimmed, case_insensitive, whitespace_ignored.
    Empty search_content matches at position 0 (like Continue).
    """
    trimmed = search_content.strip()
    if trimmed == "":
        return (0, 0, "emptySearch")
    for name, strategy in _MATCH_STRATEGIES:
        result = strategy(file_content, search_content)
        if result is not None:
            return (result[0], result[1], name)
    return None


def find_search_matches(
    file_content: str,
    search_content: str,
) -> List[Tuple[int, int]]:
    """Find all non-overlapping matches. Returns [(start, end), ...].

    Uses the same strategies as find_search_match; applied iteratively
    for each occurrence. Empty search returns [(0, 0)] (Continue behavior).
    """
    if search_content.strip() == "":
        return [(0, 0)]
    matches: List[Tuple[int, int]] = []
    remaining = file_content
    offset = 0
    while remaining:
        m = find_search_match(remaining, search_content)
        if m is None:
            break
        start, end, _ = m
        abs_start = offset + start
        abs_end = offset + end
        if matches and abs_start <= matches[-1][0]:
            break  # prevent infinite loop
        matches.append((abs_start, abs_end))
        offset = abs_end
        remaining = file_content[offset:]
    return matches
