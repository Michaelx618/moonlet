"""Relevance pipeline: find files to edit from user request (AI-like search-first approach).

Tiers:
  1. Explicit file mention (extract_target_files)
  2. Symbol discovery (CamelCase + snake_case via discover_target_file)
  3. Keyword grep + optional semantic search
  4. TODO intent (grep TODO/FIXME when user says implement/complete)
  5. Fallback (open file, or first source file in project)
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from . import config
from .anchors import discover_target_file
from .files import _norm_rel_path, get_root, is_allowed_file, read_single_file_for_context
from .intents import extract_target_files
from .tools import grep_search
from .utils import dbg


_SOURCE_EXTS = frozenset(
    {"c", "h", "cc", "cpp", "cxx", "hpp", "py", "js", "jsx", "ts", "tsx", "go", "rs", "java"}
)
_BUILD_FILE_NAMES = frozenset(
    {
        "makefile", "gnumakefile", "cmakelists.txt", "dockerfile", "docker-compose.yml",
        "docker-compose.yaml", "justfile", "gemfile", "rakefile", "procfile", "brewfile",
    }
)
_KEYWORD_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "this", "that", "all", "you", "your",
        "new", "class", "function", "method", "file", "code", "make", "fix", "edit",
        "update", "implement", "complete", "add", "change", "modify", "create",
        "task", "lab", "submit", "download", "please", "ensure", "note", "see",
    }
)


def _tier1_explicit(user_text: str) -> List[str]:
    """Tier 1: Explicit file mentions."""
    return extract_target_files(user_text or "")


def _tier1_5_parent_child(user_text: str) -> List[str]:
    """Tier 1.5: When user says parent and/or child, return parentcreates.c, childcreates.c."""
    low = (user_text or "").lower()
    if "parent" not in low and "child" not in low:
        return []
    root = get_root()
    out: List[str] = []
    for pattern in ("parentcreates.c", "childcreates.c"):
        for p in root.rglob(pattern):
            try:
                rel = str(p.relative_to(root))
                if rel not in out and is_allowed_file(p):
                    out.append(_norm_rel_path(rel))
                break
            except ValueError:
                continue
    return out


def _tier2_symbol(user_text: str, open_file: Optional[str]) -> List[str]:
    """Tier 2: Symbol discovery (CamelCase + snake_case)."""
    trimmed = (user_text or "")[:500]
    if not trimmed:
        return []
    resolved = discover_target_file(trimmed, open_file)
    if resolved:
        return [_norm_rel_path(resolved)]
    return []


def _tier2_5_build_intent(user_text: str) -> List[str]:
    """Tier 2.5: Build intent — when user says make/build/compile, discover Makefile."""
    low = (user_text or "").lower()
    if not any(k in low for k in ("make", "build", "compile", "makefile", "target")):
        return []
    root = get_root()
    for fname in ("Makefile", "makefile", "GNUmakefile"):
        candidate = root / fname
        if candidate.exists() and candidate.is_file() and is_allowed_file(candidate):
            return [_norm_rel_path(str(candidate))]
    for candidate in root.rglob("CMakeLists.txt"):
        try:
            rel = str(candidate.relative_to(root))
            if is_allowed_file(rel):
                return [_norm_rel_path(rel)]
        except ValueError:
            continue
    return []


def _extract_keywords(text: str) -> List[str]:
    """Extract significant words (3+ chars, not stopwords) for grep."""
    words = re.findall(r"\b[a-zA-Z_]\w{2,}\b", (text or "").lower())
    return [w for w in words if w not in _KEYWORD_STOPWORDS][:8]


def _list_editable_files(max_files: int = 80) -> List[str]:
    """List editable files in repo (for semantic search candidates)."""
    root = get_root()
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if len(found) >= max_files:
            break
        dirnames[:] = [d for d in dirnames if d not in config.IGNORE_DIRS and not d.startswith(".")]
        for fname in sorted(filenames):
            try:
                rel = str(Path(dirpath, fname).relative_to(root))
            except ValueError:
                continue
            if _is_editable_file(rel):
                found.append(_norm_rel_path(rel))
    return found


def _get_file_summaries(paths: List[str], max_chars: int = 500) -> List[Tuple[str, str]]:
    """Build (path, content_preview) for embedding."""
    summaries: List[Tuple[str, str]] = []
    for path in paths:
        content = read_single_file_for_context(path).get(path, "")
        if content is None:
            content = ""
        preview = (content or "")[:max_chars]
        summaries.append((path, preview))
    return summaries


def _tier3_semantic(user_text: str, candidate_files: List[str]) -> List[str]:
    """Tier 3b: Semantic search on candidates. Returns [] if disabled or fails."""
    if not candidate_files:
        return []
    if not getattr(config, "SEMANTIC_SEARCH_ENABLED", False):
        return []
    try:
        from .semantic import semantic_search_files

        top_k = getattr(config, "SEMANTIC_TOP_K", 5)
        summaries = _get_file_summaries(candidate_files)
        if not summaries:
            return []
        result = semantic_search_files(user_text, summaries, top_k=top_k)
        if result:
            dbg(f"semantic: found {len(result)} files")
        return result
    except Exception as exc:
        dbg(f"semantic tier failed: {exc}")
        return []


def _tier3_keyword_grep(user_text: str) -> List[str]:
    """Tier 3a: Keyword grep — rank files by match count."""
    keywords = _extract_keywords(user_text)
    if not keywords:
        return []
    file_scores: dict = {}
    for kw in keywords[:5]:
        result = grep_search(kw, max_results=50, literal=True)
        for m in result.matches:
            f = _norm_rel_path(m.file)
            if not _is_editable_file(f):
                continue
            file_scores[f] = file_scores.get(f, 0) + 1
    if not file_scores:
        return []
    ranked = sorted(file_scores.items(), key=lambda x: -x[1])
    return [f for f, _ in ranked[:5]]


def _is_editable_file(path: str) -> bool:
    """Include source files and build/config files (Makefile, Dockerfile, etc.)."""
    p = Path(path)
    name = p.name.lower()
    ext = p.suffix.lower().lstrip(".")
    if ext in _SOURCE_EXTS:
        return is_allowed_file(p)
    if name in _BUILD_FILE_NAMES:
        return is_allowed_file(p)
    return False


def _tier4_todo_intent(user_text: str) -> List[str]:
    """Tier 4: TODO intent — grep TODO/FIXME when user wants to implement."""
    low = (user_text or "").lower()
    if not any(k in low for k in ("implement", "todo", "stub", "complete", "fill")):
        return []
    result = grep_search("TODO", max_results=30, literal=True)
    if not result.matches:
        result = grep_search("FIXME", max_results=30, literal=True)
    file_counts: dict = {}
    for m in result.matches:
        f = _norm_rel_path(m.file)
        if _is_editable_file(f):
            file_counts[f] = file_counts.get(f, 0) + 1
    if not file_counts:
        return []
    ranked = sorted(file_counts.items(), key=lambda x: -x[1])
    return [f for f, _ in ranked[:5]]


def _tier5_fallback(open_file: Optional[str]) -> List[str]:
    """Tier 5: Open file or first source file in project."""
    if open_file:
        norm = _norm_rel_path(open_file)
        root = get_root()
        candidate = (root / norm).resolve()
        if candidate.exists() and candidate.is_file() and is_allowed_file(candidate):
            return [norm]
    # First source file in project (bounded walk)
    root = get_root()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in config.IGNORE_DIRS and not d.startswith(".")]
        for fname in sorted(filenames):
            try:
                rel = str(Path(dirpath, fname).relative_to(root))
            except ValueError:
                continue
            if _is_editable_file(rel):
                return [_norm_rel_path(rel)]
    return []


def find_relevant_files(
    user_text: str,
    open_file: Optional[str] = None,
) -> List[str]:
    """Find relevant files to edit using a 5-tier relevance pipeline.

    Returns the first non-empty result from:
      1. Explicit file mentions (extract_target_files)
      2. Symbol discovery (CamelCase + snake_case)
      3. Keyword grep (extract words, rank by matches)
      4. TODO intent (grep TODO/FIXME)
      5. Fallback (open file or first source file)
    """
    text = user_text or ""
    # Tier 1
    t1 = _tier1_explicit(text)
    if t1:
        dbg(f"relevance: tier1 explicit -> {t1}")
        return t1
    # Tier 1.5: parent/child -> parentcreates.c, childcreates.c
    t1_5 = _tier1_5_parent_child(text)
    if t1_5:
        dbg(f"relevance: tier1.5 parent_child -> {t1_5}")
        return t1_5
    # Tier 2
    t2 = _tier2_symbol(text, open_file)
    if t2:
        dbg(f"relevance: tier2 symbol -> {t2}")
        return t2
    # Tier 2.5: build intent (make, build, compile)
    t2_5 = _tier2_5_build_intent(text)
    if t2_5:
        dbg(f"relevance: tier2.5 build_intent -> {t2_5}")
        return t2_5
    # Tier 3: keyword grep + optional semantic search
    t3_keyword = _tier3_keyword_grep(text)
    if getattr(config, "SEMANTIC_SEARCH_ENABLED", False):
        candidates = t3_keyword if t3_keyword else _list_editable_files(max_files=50)
        t3_semantic = _tier3_semantic(text, candidates)
        if t3_semantic:
            dbg(f"relevance: tier3 semantic -> {t3_semantic}")
            return t3_semantic
    if t3_keyword:
        dbg(f"relevance: tier3 keyword -> {t3_keyword}")
        return t3_keyword
    # Tier 4
    t4 = _tier4_todo_intent(text)
    if t4:
        dbg(f"relevance: tier4 todo -> {t4}")
        return t4
    # Tier 5
    t5 = _tier5_fallback(open_file)
    if t5:
        dbg(f"relevance: tier5 fallback -> {t5}")
        return t5
    return []
