"""Full-text search index: SQLite FTS5 over file contents."""

from typing import Callable, Iterator, List, Optional

from .refresh_index import get_db
from .types import IndexResultType, IndexTag, PathAndCacheKey, RefreshIndexResults
from .utils import tag_to_string

ARTIFACT_ID = "sqliteFts"


def _create_tables(conn) -> None:
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(path, content, tokenize='trigram');
        CREATE TABLE IF NOT EXISTS fts_metadata (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            cacheKey TEXT NOT NULL,
            FOREIGN KEY (id) REFERENCES fts(rowid)
        );
    """)


class FullTextSearchCodebaseIndex:
    artifact_id = ARTIFACT_ID

    def __init__(self, read_file: Callable[[str], str]):
        self.read_file = read_file

    def update(self, tag: IndexTag, results: RefreshIndexResults, mark_complete: Callable, repo_name: Optional[str]) -> Iterator[dict]:
        db = get_db()
        _create_tables(db)
        tag_str = tag_to_string(tag)
        for i, (path, cache_key) in enumerate(results.compute):
            try:
                content = self.read_file(path)
            except Exception:
                content = ""
            db.execute("INSERT INTO fts (path, content) VALUES (?, ?)", (path, content))
            rowid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            db.execute("INSERT OR REPLACE INTO fts_metadata (id, path, cacheKey) VALUES (?, ?, ?)", (rowid, path, cache_key))
            db.commit()
            yield {"progress": (i + 1) / len(results.compute), "desc": f"FTS {path}", "status": "indexing"}
            mark_complete([(path, cache_key)], IndexResultType.COMPUTE)
        for path, cache_key in results.add_tag:
            mark_complete([(path, cache_key)], IndexResultType.ADD_TAG)
            yield {"progress": 1, "desc": f"Add tag {path}", "status": "indexing"}
        for path, cache_key in results.remove_tag:
            mark_complete([(path, cache_key)], IndexResultType.REMOVE_TAG)
            yield {"progress": 1, "desc": f"Remove tag {path}", "status": "indexing"}
        for path, cache_key in results.del_:
            cur = db.execute("SELECT id FROM fts_metadata WHERE path = ? AND cacheKey = ?", (path, cache_key))
            rows = cur.fetchall()
            for (rid,) in rows:
                db.execute("DELETE FROM fts WHERE rowid = ?", (rid,))
            db.execute("DELETE FROM fts_metadata WHERE path = ? AND cacheKey = ?", (path, cache_key))
            db.commit()
            mark_complete([(path, cache_key)], IndexResultType.DELETE)
            yield {"progress": 1, "desc": f"Removing {path}", "status": "indexing"}


def retrieve_fts(text: str, tag: IndexTag, n: int = 20, filter_paths: Optional[List[str]] = None) -> List[dict]:
    db = get_db()
    _create_tables(db)
    if filter_paths:
        placeholders = ",".join("?" * len(filter_paths))
        query = "SELECT fts.path, fts.content FROM fts JOIN fts_metadata ON fts.rowid = fts_metadata.id WHERE fts MATCH ? AND fts_metadata.path IN (" + placeholders + ") LIMIT ?"
        params = [text.replace('"', ' ') + "*", *filter_paths, n]
    else:
        query = "SELECT path, content FROM fts WHERE fts MATCH ? LIMIT ?"
        params = [text.replace('"', ' ') + "*", n]
    try:
        cur = db.execute(query, params)
        rows = cur.fetchall()
        return [{"path": r[0], "content": r[1]} for r in rows]
    except Exception:
        return []
