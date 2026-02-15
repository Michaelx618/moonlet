"""File index for the coding agent.

Indexes imported/root files when root or include changes.
Provides a stable list of files the agent can access via tools (grep, symbols, read).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from . import config
from .files import (
    ROOT_PATH,
    _norm_rel_path,
    get_include,
    get_root,
    is_allowed_file,
)
from .utils import dbg


# Cached list of indexed file paths (relative to root)
_indexed_files: List[str] = []
# Optional: symbol cache per file (path -> list of Symbol-like dicts)
_symbol_cache: Dict[str, List[dict]] = {}
_symbol_cache_max_files: int = 50  # cap cache size


def _list_editable_files(max_files: int = 200) -> List[str]:
    """List editable files in repo, respecting include filter."""
    root = get_root()
    include = get_include()
    files: List[str] = []
    seen: Set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in config.IGNORE_DIRS and not d.startswith(".")
        ]
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            try:
                full = Path(dirpath, fname).resolve()
                rel = str(full.relative_to(root.resolve())).replace("\\", "/")
            except ValueError:
                continue
            if not is_allowed_file(rel):
                continue
            if include:
                # Match exact path or path under an included directory
                matched = rel in include
                if not matched:
                    for p in include:
                        prefix = p.rstrip("/")
                        if prefix and (rel == prefix or rel.startswith(prefix + "/")):
                            matched = True
                            break
                if not matched:
                    continue
            norm = _norm_rel_path(rel)
            if norm in seen:
                continue
            seen.add(norm)
            files.append(norm)
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break

    return sorted(files)


def rebuild_index() -> List[str]:
    """Rebuild the file index. Call when root or include changes."""
    global _indexed_files, _symbol_cache
    _indexed_files = _list_editable_files()
    _symbol_cache.clear()
    dbg(f"index: rebuilt, {len(_indexed_files)} files")
    return _indexed_files


def get_indexed_files() -> List[str]:
    """Return the list of indexed files. Rebuilds if empty."""
    global _indexed_files
    if not _indexed_files:
        rebuild_index()
    return list(_indexed_files)


def invalidate_index() -> None:
    """Invalidate the index (e.g. before root/include change)."""
    global _indexed_files, _symbol_cache
    _indexed_files = []
    _symbol_cache.clear()


def get_symbols_for_file(rel_path: str) -> List[dict]:
    """Get tree-sitter symbols for a file. Uses cache when available."""
    from .tools import extract_symbols_treesitter

    rel_path = _norm_rel_path(rel_path)
    if rel_path in _symbol_cache:
        return _symbol_cache[rel_path]

    syms = extract_symbols_treesitter(rel_path, root=get_root())
    result = [
        {
            "name": s.name,
            "kind": s.kind,
            "line": s.line,
            "end_line": s.end_line,
        }
        for s in syms
    ]

    # Cache if under limit
    if len(_symbol_cache) < _symbol_cache_max_files:
        _symbol_cache[rel_path] = result

    return result
