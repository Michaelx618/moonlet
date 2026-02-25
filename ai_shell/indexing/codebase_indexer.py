"""Orchestrate refresh (get file stats, get_compute_delete_add_remove, run indexes)."""

import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

from . import refresh_index as ref
from .chunk_index import ChunkCodebaseIndex
from .code_snippets_index import CodeSnippetsCodebaseIndex
from .embeddings_index import EmbeddingsCodebaseIndex
from .full_text_search_index import FullTextSearchCodebaseIndex
from .types import IndexTag, PathAndCacheKey, RefreshIndexResults


def _get_file_stats(paths: List[str], root: Path) -> dict:
    """Return path -> { last_modified: ms, size: bytes } for each path that exists."""
    out = {}
    for p in paths:
        full = root / p
        try:
            if full.is_file():
                st = full.stat()
                out[p] = {"last_modified": st.st_mtime * 1000, "size": st.st_size}
        except Exception:
            pass
    return out


def _walk_directory(root: Path, include: Optional[List[str]], max_files: int) -> List[str]:
    """List relative file paths under root, respecting include filter (same semantics as index._list_editable_files)."""
    from .. import config
    from ..files import is_allowed_file, _norm_rel_path

    files: List[str] = []
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in getattr(config, "IGNORE_DIRS", set()) and not d.startswith(".")]
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
                return sorted(files)
    return sorted(files)


def _get_branch(_directory: str) -> str:
    """Return git branch for directory, or 'main'."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_directory,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "main"


def get_indexes_to_build(
    read_file: Callable[[str], str],
    root: Optional[Path] = None,
    enable_chunks: Optional[bool] = None,
    enable_code_snippets: Optional[bool] = None,
    enable_fts: Optional[bool] = None,
    enable_embeddings: Optional[bool] = None,
) -> List[Any]:
    """Return list of index instances to build. When enable_* is None, reads from config."""
    from .. import config
    if enable_chunks is None:
        enable_chunks = getattr(config, "INDEX_ENABLE_CHUNKS", True)
    if enable_code_snippets is None:
        enable_code_snippets = getattr(config, "INDEX_ENABLE_CODE_SNIPPETS", True)
    if enable_fts is None:
        enable_fts = getattr(config, "INDEX_ENABLE_FTS", True)
    if enable_embeddings is None:
        enable_embeddings = getattr(config, "INDEX_ENABLE_EMBEDDINGS", False)

    chunk_max_lines = int(getattr(config, "INDEX_CHUNK_MAX_LINES", 300))
    indexes: List[Any] = []
    if enable_chunks:
        indexes.append(ChunkCodebaseIndex(read_file=read_file, max_lines=chunk_max_lines))
    if enable_code_snippets:
        indexes.append(CodeSnippetsCodebaseIndex(read_file=read_file, root=root))
    if enable_fts:
        indexes.append(FullTextSearchCodebaseIndex(read_file=read_file))
    if enable_embeddings:
        indexes.append(EmbeddingsCodebaseIndex(read_file=read_file))
    return indexes


def refresh_codebase_index(
    directory: str,
    include_paths: Optional[List[str]] = None,
    max_files: int = 500,
    files_per_batch: int = 200,
) -> Iterator[dict]:
    """Refresh all indexes for a directory. Yields progress dicts (progress, desc, status)."""
    from ..files import get_root, read_file_text

    root = get_root() if directory in (".", "") else Path(directory).resolve()
    if not root.exists() or not root.is_dir():
        yield {"progress": 1, "desc": "Invalid root", "status": "done"}
        return

    branch = _get_branch(str(root))
    tag = IndexTag(directory=str(root), branch=branch, artifact_id="")  # per-index tag has its own artifact_id

    # Discover files
    files = _walk_directory(root, include_paths, max_files)
    if not files:
        yield {"progress": 1, "desc": "No files to index", "status": "done"}
        return

    def read_file(rel_path: str) -> str:
        return read_file_text(rel_path)

    stats = _get_file_stats(files, root)
    indexes = get_indexes_to_build(read_file=read_file, root=root)

    for index in indexes:
        tag = IndexTag(directory=str(root), branch=branch, artifact_id=index.artifact_id)
        try:
            results, last_updated, mark_complete = ref.get_compute_delete_add_remove(tag, stats, read_file)
        except Exception as e:
            yield {"progress": 0, "desc": f"Planning failed: {e}", "status": "failed"}
            continue

        total_ops = len(results.compute) + len(results.del_) + len(results.add_tag) + len(results.remove_tag)
        if total_ops == 0:
            # Just update lastUpdated in catalog for unchanged files
            from .types import IndexResultType
            for path, cache_key in last_updated:
                mark_complete([(path, cache_key)], IndexResultType.UPDATE_LAST_UPDATED)
            continue

        # Batch (same as Continue: slice each list by files_per_batch)
        batch_size = files_per_batch
        pos = 0
        while pos < len(results.compute) or pos < len(results.del_) or pos < len(results.add_tag) or pos < len(results.remove_tag):
            batch = RefreshIndexResults(
                compute=results.compute[pos: pos + batch_size],
                del_=results.del_[pos: pos + batch_size],
                add_tag=results.add_tag[pos: pos + batch_size],
                remove_tag=results.remove_tag[pos: pos + batch_size],
            )
            for update in index.update(tag, batch, mark_complete, None):
                yield update
            pos += batch_size

    yield {"progress": 1, "desc": "Indexing complete", "status": "done"}


def refresh_index_file(rel_path: str) -> Iterator[dict]:
    """Re-index a single file (e.g. after save). Yields progress dicts."""
    from ..files import get_root, read_file_text

    root = get_root()
    if not root or not root.exists():
        yield {"progress": 1, "desc": "No root", "status": "done"}
        return

    rel_path = (rel_path or "").strip().replace("\\", "/")
    if not rel_path:
        yield {"progress": 1, "desc": "No path", "status": "done"}
        return

    full = root / rel_path
    if not full.is_file():
        yield {"progress": 1, "desc": "Not a file", "status": "done"}
        return

    def read_file(p: str) -> str:
        return read_file_text(p)

    stats = _get_file_stats([rel_path], root)
    if not stats:
        yield {"progress": 1, "desc": "File not found", "status": "done"}
        return

    branch = _get_branch(str(root))
    indexes = get_indexes_to_build(read_file=read_file, root=root)

    paths_limit = {rel_path}
    for index in indexes:
        tag = IndexTag(directory=str(root), branch=branch, artifact_id=index.artifact_id)
        try:
            results, last_updated, mark_complete = ref.get_compute_delete_add_remove(tag, stats, read_file, paths_limit=paths_limit)
        except Exception as e:
            yield {"progress": 0, "desc": f"Planning failed: {e}", "status": "failed"}
            continue
        from .types import IndexResultType
        if not (results.compute or results.del_ or results.add_tag or results.remove_tag):
            for path, cache_key in last_updated:
                mark_complete([(path, cache_key)], IndexResultType.UPDATE_LAST_UPDATED)
            continue
        for update in index.update(tag, results, mark_complete, None):
            yield update

    yield {"progress": 1, "desc": "File indexed", "status": "done"}


def refresh_index_simple() -> List[str]:
    """One-shot refresh for current root/include (like rebuild_index but with catalog).
    Returns list of indexed paths (from catalog after refresh).
    """
    from ..files import get_include, get_root

    root = get_root()
    include = get_include()
    include_list = list(include) if include else None
    paths = []
    for update in refresh_codebase_index(str(root), include_paths=include_list):
        pass
    # Return paths from tag_catalog for the default tag (any artifact)
    db = ref.get_db()
    cur = db.execute(
        "SELECT DISTINCT path FROM tag_catalog WHERE dir = ?",
        (str(root),),
    )
    paths = [r[0] for r in cur.fetchall()]
    return sorted(paths)
