"""Structural chunk index: split files into chunks (by line range) for FTS/embeddings."""

from typing import Callable, Iterator, List, Optional

from .refresh_index import get_db
from .types import IndexResultType, IndexTag, PathAndCacheKey, RefreshIndexResults
from .utils import tag_to_string

ARTIFACT_ID = "chunks"
DEFAULT_CHUNK_MAX_LINES = 300


def _create_tables(conn) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cacheKey TEXT NOT NULL,
            path TEXT NOT NULL,
            idx INTEGER NOT NULL,
            startLine INTEGER NOT NULL,
            endLine INTEGER NOT NULL,
            content TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunk_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT NOT NULL,
            chunkId INTEGER NOT NULL,
            FOREIGN KEY (chunkId) REFERENCES chunks (id),
            UNIQUE (tag, chunkId)
        );
    """)


def _chunk_file(content: str, max_lines: int) -> List[dict]:
    lines = (content or "").splitlines(keepends=True)
    if not lines:
        return [{"startLine": 1, "endLine": 1, "content": ""}]
    out = []
    start = 0
    idx = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        chunk_content = "".join(lines[start:end])
        out.append({"idx": idx, "startLine": start + 1, "endLine": end, "content": chunk_content})
        idx += 1
        start = end
    return out


class ChunkCodebaseIndex:
    artifact_id = ARTIFACT_ID

    def __init__(self, read_file: Callable[[str], str], max_lines: int = DEFAULT_CHUNK_MAX_LINES):
        self.read_file = read_file
        self.max_lines = max_lines

    def update(self, tag: IndexTag, results: RefreshIndexResults, mark_complete: Callable, repo_name: Optional[str]) -> Iterator[dict]:
        db = get_db()
        _create_tables(db)
        tag_str = tag_to_string(tag)
        for i, (path, cache_key) in enumerate(results.compute):
            try:
                content = self.read_file(path)
            except Exception:
                content = ""
            chunks = _chunk_file(content, self.max_lines)
            for c in chunks:
                cur = db.execute(
                    """INSERT INTO chunks (cacheKey, path, idx, startLine, endLine, content) VALUES (?, ?, ?, ?, ?, ?)""",
                    (cache_key, path, c["idx"], c["startLine"], c["endLine"], c["content"]),
                )
                db.execute("INSERT OR IGNORE INTO chunk_tags (chunkId, tag) VALUES (?, ?)", (cur.lastrowid, tag_str))
            db.commit()
            yield {"progress": (i + 1) / len(results.compute), "desc": f"Chunking {path}", "status": "indexing"}
            mark_complete([(path, cache_key)], IndexResultType.COMPUTE)
        for path, cache_key in results.add_tag:
            cur = db.execute("SELECT id FROM chunks WHERE path = ? AND cacheKey = ?", (path, cache_key))
            for (cid,) in cur.fetchall():
                db.execute("INSERT OR IGNORE INTO chunk_tags (chunkId, tag) VALUES (?, ?)", (cid, tag_str))
            db.commit()
            mark_complete([(path, cache_key)], IndexResultType.ADD_TAG)
            yield {"progress": 1, "desc": f"Add tag {path}", "status": "indexing"}
        for path, cache_key in results.remove_tag:
            db.execute(
                """DELETE FROM chunk_tags WHERE tag = ? AND chunkId IN (SELECT id FROM chunks WHERE cacheKey = ? AND path = ?)""",
                (tag_str, cache_key, path),
            )
            db.commit()
            mark_complete([(path, cache_key)], IndexResultType.REMOVE_TAG)
            yield {"progress": 1, "desc": f"Remove tag {path}", "status": "indexing"}
        for path, cache_key in results.del_:
            cur = db.execute("SELECT id FROM chunks WHERE cacheKey = ? AND path = ?", (cache_key, path))
            ids = [r[0] for r in cur.fetchall()]
            if ids:
                placeholders = ",".join("?" * len(ids))
                db.execute(f"DELETE FROM chunk_tags WHERE chunkId IN ({placeholders})", ids)
                db.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
            db.commit()
            mark_complete([(path, cache_key)], IndexResultType.DELETE)
            yield {"progress": 1, "desc": f"Removing {path}", "status": "indexing"}


def get_chunks_for_path(path: str, cache_key: str) -> List[dict]:
    db = get_db()
    _create_tables(db)
    cur = db.execute("SELECT id, path, idx, startLine, endLine, content FROM chunks WHERE path = ? AND cacheKey = ? ORDER BY idx", (path, cache_key))
    return [{"id": r[0], "path": r[1], "index": r[2], "startLine": r[3], "endLine": r[4], "content": r[5]} for r in cur.fetchall()]
