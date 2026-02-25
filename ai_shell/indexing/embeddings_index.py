"""Embeddings index: embed chunks and store in LanceDB for semantic codebase search."""

import os
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from .chunk_index import get_chunks_for_path
from .refresh_index import get_db
from .types import IndexResultType, IndexTag, PathAndCacheKey, RefreshIndexResults
from .utils import tag_to_string

ARTIFACT_ID = "embeddings"


def _get_lancedb_path() -> Path:
    path = os.getenv("MOONLET_LANCEDB_PATH", "")
    if path:
        return Path(path)
    try:
        from ..files import get_root
        root = get_root()
        if root:
            return root / ".moonlet" / "lancedb"
    except Exception:
        pass
    return Path(os.path.expanduser("~")) / ".moonlet_lancedb"


def _get_embedding_model():
    """Return embed model from semantic module, or None."""
    try:
        from ..semantic import embed_text, _get_embedding_model as get_model
        if get_model() is None:
            return None
        return embed_text
    except Exception:
        return None


def _table_name(tag: IndexTag) -> str:
    """Safe table name from tag (dir hash + branch + artifact)."""
    import hashlib
    h = hashlib.sha1(f"{tag.directory}|{tag.branch}".encode()).hexdigest()[:12]
    return f"emb_{h}"


class EmbeddingsCodebaseIndex:
    """Index chunk contents as embeddings in LanceDB for semantic search."""

    artifact_id = ARTIFACT_ID

    def __init__(self, read_file: Callable[[str], str]):
        self.read_file = read_file
        self._db = None
        self._embed_fn = None

    def _ensure_db(self):
        if self._db is None:
            try:
                import lancedb
            except ImportError:
                return None
            path = _get_lancedb_path()
            path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(path))
        return self._db

    def _ensure_embed(self):
        if self._embed_fn is None:
            self._embed_fn = _get_embedding_model()
        return self._embed_fn

    def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: Callable,
        repo_name: Optional[str],
    ) -> Iterator[dict]:
        db = self._ensure_db()
        embed_fn = self._ensure_embed()
        if db is None or embed_fn is None:
            for (path, cache_key) in results.compute:
                mark_complete([(path, cache_key)], IndexResultType.COMPUTE)
            for item in results.add_tag:
                mark_complete([item], IndexResultType.ADD_TAG)
            for item in results.remove_tag:
                mark_complete([item], IndexResultType.REMOVE_TAG)
            for item in results.del_:
                mark_complete([item], IndexResultType.DELETE)
            yield {"progress": 1, "desc": "Embeddings skipped (no lancedb or embed model)", "status": "indexing"}
            return

        table_name = _table_name(tag)
        tag_str = tag_to_string(tag)

        # Compute: get chunks from chunks table, embed, add to LanceDB
        for i, (path, cache_key) in enumerate(results.compute):
            chunks = get_chunks_for_path(path, cache_key)
            if not chunks:
                mark_complete([(path, cache_key)], IndexResultType.COMPUTE)
                continue
            rows = []
            for c in chunks:
                vec = embed_fn(c["content"])
                if vec is None:
                    continue
                rows.append({
                    "vector": vec,
                    "path": path,
                    "start_line": c["startLine"],
                    "end_line": c["endLine"],
                    "content": (c["content"] or "")[:2000],
                    "tag": tag_str,
                })
            if rows:
                try:
                    if table_name not in db.table_names():
                        db.create_table(table_name, rows)
                    else:
                        db.open_table(table_name).add(rows)
                except Exception:
                    pass
            mark_complete([(path, cache_key)], IndexResultType.COMPUTE)
            yield {"progress": (i + 1) / len(results.compute), "desc": f"Embedding {path}", "status": "indexing"}

        for item in results.add_tag:
            mark_complete([item], IndexResultType.ADD_TAG)
            yield {"progress": 1, "desc": "Add tag", "status": "indexing"}
        for item in results.remove_tag:
            mark_complete([item], IndexResultType.REMOVE_TAG)
            yield {"progress": 1, "desc": "Remove tag", "status": "indexing"}
        for path, cache_key in results.del_:
            try:
                t = db.open_table(table_name)
                t.delete(f"path = '{path.replace(chr(39), chr(39)+chr(39))}'")
            except Exception:
                pass
            mark_complete([(path, cache_key)], IndexResultType.DELETE)
            yield {"progress": 1, "desc": f"Removing {path}", "status": "indexing"}


def query_embeddings(
    query: str,
    tag: IndexTag,
    n: int = 10,
    filter_paths: Optional[List[str]] = None,
) -> List[dict]:
    """Semantic search: embed query, search LanceDB, return [{ path, start_line, end_line, content }, ...]."""
    try:
        import lancedb
    except ImportError:
        return []
    embed_fn = _get_embedding_model()
    if embed_fn is None:
        return []
    vec = embed_fn(query)
    if vec is None:
        return []
    path = _get_lancedb_path()
    if not path.exists():
        return []
    db = lancedb.connect(str(path))
    table_name = _table_name(tag)
    if table_name not in db.table_names():
        return []
    tbl = db.open_table(table_name)
    results = tbl.search(vec).limit(n * 2)
    try:
        rows = results.to_list()
    except Exception:
        rows = []
    out = []
    seen = set()
    for r in rows:
        path_key = (r.get("path"), r.get("start_line"), r.get("end_line"))
        if path_key in seen:
            continue
        seen.add(path_key)
        if filter_paths and r.get("path") not in filter_paths:
            continue
        out.append({
            "path": r.get("path"),
            "start_line": r.get("start_line"),
            "end_line": r.get("end_line"),
            "content": r.get("content", ""),
        })
        if len(out) >= n:
            break
    return out
