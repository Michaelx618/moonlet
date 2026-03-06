"""File index for the coding agent.

Indexes imported/root files when root or include changes.
Provides a stable list of files the agent can access via tools (grep, symbols, read).
"""

import os
import sys
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


def walk_editable_files(
    root: Path,
    include: Optional[List[str]],
    max_files: int,
    *,
    require_include: bool = True,
) -> List[str]:
    """Canonical walk: list relative paths under root, respecting include filter.
    Used by index, relevance, and indexing/codebase_indexer."""
    if not root.exists() or not root.is_dir():
        return []
    if require_include and not include:
        return []
    files: List[str] = []
    seen: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
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
            if include and not getattr(config, "DISABLE_INDEX", False):
                matched = rel in include
                if not matched:
                    for p in include:
                        prefix = (p or "").rstrip("/")
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
                return sorted(files)
        if len(files) >= max_files:
            break
    return sorted(files)


def _list_editable_files(max_files: int = 200) -> List[str]:
    """List editable files in repo, respecting include filter."""
    root = get_root()
    include = get_include()
    if getattr(config, "INDEX_REQUIRES_IMPORT", True) and not include:
        return []
    return walk_editable_files(root, include, max_files, require_include=False)


def rebuild_index() -> List[str]:
    """Rebuild the file index. Call when root or include changes."""
    global _indexed_files, _symbol_cache
    try:
        root = get_root()
        include = get_include()
        _indexed_files = _list_editable_files()
        _symbol_cache.clear()
        # Log so user can see why 200 vs 4: first build = full repo root; later = after set_include/set_root (e.g. task folder)
        if getattr(config, "INDEX_REQUIRES_IMPORT", True) and not include:
            include_info = " include=none (index idle until import)"
        elif getattr(config, "DISABLE_INDEX", False):
            include_info = " include=disabled"
        else:
            include_info = f" include={len(include)} path(s)"
        dbg(f"index: rebuilt, {len(_indexed_files)} files (cap=200) root={root}{include_info}")
        return _indexed_files
    except Exception as e:
        _indexed_files = []
        _symbol_cache.clear()
        print(f"[index: rebuild failed — {e}]", file=sys.stderr)
        return _indexed_files


def get_indexed_files() -> List[str]:
    """Return the list of indexed files. Rebuilds if empty."""
    global _indexed_files
    if not _indexed_files:
        rebuild_index()
    return list(_indexed_files)


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
