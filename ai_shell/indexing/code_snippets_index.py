"""Code snippets index: tree-sitter symbols per file in SQLite."""

from pathlib import Path
from typing import Callable, Iterator, List, Optional

from .refresh_index import get_db
from .types import IndexResultType, IndexTag, PathAndCacheKey, RefreshIndexResults
from .utils import tag_to_string

ARTIFACT_ID = "codeSnippets"


def _create_tables(conn) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS code_snippets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            cacheKey TEXT NOT NULL,
            content TEXT NOT NULL,
            title TEXT NOT NULL,
            signature TEXT,
            startLine INTEGER NOT NULL,
            endLine INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS code_snippets_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT NOT NULL,
            snippetId INTEGER NOT NULL,
            FOREIGN KEY (snippetId) REFERENCES code_snippets (id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_code_snippets_tags_snippet_tag
            ON code_snippets_tags (snippetId, tag);
    """)


def _get_snippets_for_file(filepath: str, contents: str, root: Path) -> List[dict]:
    from ..tools import extract_symbols_treesitter
    symbols = extract_symbols_treesitter(filepath, content=contents, root=root)
    lines = (contents or "").splitlines()
    out = []
    for s in symbols:
        start = max(0, s.line - 1)
        end = min(len(lines), s.end_line)
        content = "\n".join(lines[start:end]) if lines else ""
        out.append({
            "title": s.name,
            "content": content,
            "signature": f"{s.kind} {s.name}",
            "startLine": s.line,
            "endLine": s.end_line,
        })
    return out


class CodeSnippetsCodebaseIndex:
    artifact_id = ARTIFACT_ID

    def __init__(self, read_file: Callable[[str], str], root: Optional[Path] = None):
        self.read_file = read_file
        self._root = root

    def _root_path(self) -> Path:
        if self._root is not None:
            return self._root
        from ..files import get_root
        return get_root()

    def update(self, tag: IndexTag, results: RefreshIndexResults, mark_complete: Callable, repo_name: Optional[str]) -> Iterator[dict]:
        db = get_db()
        _create_tables(db)
        tag_str = tag_to_string(tag)
        root = self._root_path()
        for i, (path, cache_key) in enumerate(results.compute):
            snippets = []
            try:
                content = self.read_file(path)
                snippets = _get_snippets_for_file(path, content, root)
            except Exception:
                pass
            for snip in snippets:
                cur = db.execute(
                    """INSERT INTO code_snippets (path, cacheKey, content, title, signature, startLine, endLine)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (path, cache_key, snip["content"], snip["title"], snip.get("signature") or "", snip["startLine"], snip["endLine"]),
                )
                db.execute("INSERT INTO code_snippets_tags (snippetId, tag) VALUES (?, ?)", (cur.lastrowid, tag_str))
            db.commit()
            yield {"progress": (i + 1) / len(results.compute), "desc": f"Indexing {path}", "status": "indexing"}
            mark_complete([(path, cache_key)], IndexResultType.COMPUTE)
        for path, cache_key in results.del_:
            cur = db.execute("SELECT id FROM code_snippets WHERE path = ? AND cacheKey = ?", (path, cache_key))
            rows = cur.fetchall()
            if rows:
                ids = [r[0] for r in rows]
                placeholders = ",".join("?" * len(ids))
                db.execute(f"DELETE FROM code_snippets_tags WHERE snippetId IN ({placeholders})", ids)
                db.execute(f"DELETE FROM code_snippets WHERE id IN ({placeholders})", ids)
            db.commit()
            mark_complete([(path, cache_key)], IndexResultType.DELETE)
            yield {"progress": 1, "desc": f"Removing {path}", "status": "indexing"}
        for path, cache_key in results.add_tag:
            cur = db.execute("SELECT id FROM code_snippets WHERE path = ? AND cacheKey = ?", (path, cache_key))
            existing_ids = [r[0] for r in cur.fetchall()]
            if not existing_ids:
                snippets = []
                try:
                    content = self.read_file(path)
                    snippets = _get_snippets_for_file(path, content, root)
                except Exception:
                    pass
                for snip in snippets:
                    cur2 = db.execute(
                        """INSERT INTO code_snippets (path, cacheKey, content, title, signature, startLine, endLine)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (path, cache_key, snip["content"], snip["title"], snip.get("signature") or "", snip["startLine"], snip["endLine"]),
                    )
                    db.execute("INSERT OR IGNORE INTO code_snippets_tags (snippetId, tag) VALUES (?, ?)", (cur2.lastrowid, tag_str))
            else:
                for sid in existing_ids:
                    db.execute("INSERT OR IGNORE INTO code_snippets_tags (snippetId, tag) VALUES (?, ?)", (sid, tag_str))
            db.commit()
            mark_complete([(path, cache_key)], IndexResultType.ADD_TAG)
            yield {"progress": 1, "desc": f"Adding tag {path}", "status": "indexing"}
        for path, cache_key in results.remove_tag:
            cur = db.execute("SELECT id FROM code_snippets WHERE cacheKey = ? AND path = ?", (cache_key, path))
            rows = cur.fetchall()
            if rows:
                ids = [r[0] for r in rows]
                placeholders = ",".join("?" * len(ids))
                db.execute(f"DELETE FROM code_snippets_tags WHERE tag = ? AND snippetId IN ({placeholders})", [tag_str] + ids)
            db.commit()
            mark_complete([(path, cache_key)], IndexResultType.REMOVE_TAG)
            yield {"progress": 1, "desc": f"Removing tag {path}", "status": "indexing"}


def get_all_snippets_for_tag(tag: IndexTag) -> List[dict]:
    db = get_db()
    _create_tables(db)
    tag_str = tag_to_string(tag)
    cur = db.execute(
        """SELECT cs.id, cs.path, cs.title FROM code_snippets cs
           JOIN code_snippets_tags cst ON cs.id = cst.snippetId WHERE cst.tag = ?""",
        (tag_str,),
    )
    rows = cur.fetchall()
    return [{"id": str(r[0]), "title": r[2], "description": r[1], "path": r[1]} for r in rows]
