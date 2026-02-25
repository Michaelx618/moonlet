"""SQLite catalog, global cache, add/remove/compute for incremental indexing."""

import os
import sqlite3
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

from .types import IndexResultType, IndexTag, PathAndCacheKey, RefreshIndexResults
from .utils import calculate_hash

FileStatsMap = dict
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


def _get_index_sqlite_path() -> str:
    path = os.getenv("MOONLET_INDEX_SQLITE_PATH", "")
    if path:
        return path
    try:
        from ..files import get_root
        root = get_root()
        if root:
            d = root / ".moonlet"
            d.mkdir(parents=True, exist_ok=True)
            return str(d / "index.sqlite")
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), ".moonlet_index.sqlite")


_db: Optional[sqlite3.Connection] = None
_index_sqlite_path: str = ""


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS tag_catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dir TEXT NOT NULL,
            branch TEXT NOT NULL,
            artifactId TEXT NOT NULL,
            path TEXT NOT NULL,
            cacheKey TEXT NOT NULL,
            lastUpdated INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS global_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cacheKey TEXT NOT NULL,
            dir TEXT NOT NULL,
            branch TEXT NOT NULL,
            artifactId TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS indexing_lock (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            locked INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            dirs TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_tag_catalog_unique
            ON tag_catalog(dir, branch, artifactId, path, cacheKey);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_global_cache_unique
            ON global_cache(cacheKey, dir, branch, artifactId);
    """)


def get_db() -> sqlite3.Connection:
    global _db, _index_sqlite_path
    _index_sqlite_path = _get_index_sqlite_path()
    if _db is not None and os.path.exists(_index_sqlite_path):
        return _db
    _db = sqlite3.connect(_index_sqlite_path, timeout=15)
    _db.execute("PRAGMA busy_timeout = 3000")
    _create_tables(_db)
    return _db


def get_saved_items_for_tag(tag: IndexTag) -> List[Tuple[str, str, int]]:
    db = get_db()
    cur = db.execute(
        "SELECT path, cacheKey, lastUpdated FROM tag_catalog WHERE dir = ? AND branch = ? AND artifactId = ?",
        (tag.directory, tag.branch, tag.artifact_id),
    )
    return list(cur.fetchall())


def get_add_remove_for_tag(
    tag: IndexTag,
    current_files: FileStatsMap,
    read_file: Callable[[str], str],
    paths_limit: Optional[Set[str]] = None,
) -> Tuple[
    List[PathAndCacheKey],
    List[PathAndCacheKey],
    List[PathAndCacheKey],
    Callable[[List[PathAndCacheKey], IndexResultType], None],
]:
    import time
    db = get_db()
    new_ts = int(time.time() * 1000)
    files = {p: s for p, s in current_files.items() if (s.get("size") or 0) <= MAX_FILE_SIZE_BYTES}
    if paths_limit is not None:
        files = {p: s for p, s in files.items() if p in paths_limit}
    saved = get_saved_items_for_tag(tag)
    if paths_limit is not None:
        saved = [(p, ck, lu) for (p, ck, lu) in saved if p in paths_limit]
    path_groups: dict = {}
    for path, cache_key, last_updated in saved:
        if path not in path_groups:
            path_groups[path] = {"latest": (last_updated, cache_key), "all_versions": [cache_key]}
        else:
            path_groups[path]["all_versions"].append(cache_key)
            if last_updated > path_groups[path]["latest"][0]:
                path_groups[path]["latest"] = (last_updated, cache_key)
    update_new_version: List[PathAndCacheKey] = []
    update_old_version: List[PathAndCacheKey] = []
    remove: List[PathAndCacheKey] = []
    update_last_updated: List[PathAndCacheKey] = []
    for path, group in list(path_groups.items()):
        latest_ts, latest_key = group["latest"]
        all_versions = group["all_versions"]
        if path not in files:
            for ck in all_versions:
                remove.append((path, ck))
        else:
            file_mtime = files[path].get("last_modified") or 0
            if latest_ts < file_mtime:
                content = read_file(path)
                new_hash = calculate_hash(content)
                if latest_key != new_hash:
                    update_new_version.append((path, new_hash))
                    for ck in all_versions:
                        update_old_version.append((path, ck))
                else:
                    update_last_updated.append((path, latest_key))
                    for ck in all_versions:
                        if ck != latest_key:
                            update_old_version.append((path, ck))
            else:
                update_last_updated.append((path, latest_key))
                for ck in all_versions:
                    if ck != latest_key:
                        update_old_version.append((path, ck))
        files.pop(path, None)
    add: List[PathAndCacheKey] = []
    for path in files:
        try:
            content = read_file(path)
            add.append((path, calculate_hash(content)))
        except Exception:
            pass
    def mark_complete(items: List[PathAndCacheKey], result_type: IndexResultType) -> None:
        for path, cache_key in items:
            if result_type == IndexResultType.COMPUTE or result_type == IndexResultType.ADD_TAG:
                db.execute(
                    "REPLACE INTO tag_catalog (path, cacheKey, lastUpdated, dir, branch, artifactId) VALUES (?, ?, ?, ?, ?, ?)",
                    (path, cache_key, new_ts, tag.directory, tag.branch, tag.artifact_id),
                )
            elif result_type == IndexResultType.DELETE or result_type == IndexResultType.REMOVE_TAG:
                db.execute(
                    "DELETE FROM tag_catalog WHERE cacheKey = ? AND path = ? AND dir = ? AND branch = ? AND artifactId = ?",
                    (cache_key, path, tag.directory, tag.branch, tag.artifact_id),
                )
            elif result_type == IndexResultType.UPDATE_LAST_UPDATED:
                db.execute(
                    "UPDATE tag_catalog SET cacheKey = ?, lastUpdated = ? WHERE path = ? AND dir = ? AND branch = ? AND artifactId = ?",
                    (cache_key, new_ts, path, tag.directory, tag.branch, tag.artifact_id),
                )
        db.commit()
    return (
        add + [(p, ck) for p, ck in update_new_version],
        remove + [(p, ck) for p, ck in update_old_version],
        update_last_updated,
        mark_complete,
    )


def get_tags_from_global_cache(cache_key: str, artifact_id: str) -> List[Tuple[str, str, str]]:
    db = get_db()
    cur = db.execute(
        "SELECT dir, branch, artifactId FROM global_cache WHERE cacheKey = ? AND artifactId = ?",
        (cache_key, artifact_id),
    )
    return list(cur.fetchall())


def get_compute_delete_add_remove(
    tag: IndexTag,
    current_files: FileStatsMap,
    read_file: Callable[[str], str],
    repo_name: Optional[str] = None,
    paths_limit: Optional[Set[str]] = None,
) -> Tuple[RefreshIndexResults, List[PathAndCacheKey], Callable]:
    add, remove, last_updated, mark_complete = get_add_remove_for_tag(tag, current_files, read_file, paths_limit=paths_limit)
    compute: List[PathAndCacheKey] = []
    del_list: List[PathAndCacheKey] = []
    add_tag: List[PathAndCacheKey] = []
    remove_tag: List[PathAndCacheKey] = []
    for path, cache_key in add:
        existing = get_tags_from_global_cache(cache_key, tag.artifact_id)
        if existing:
            add_tag.append((path, cache_key))
        else:
            compute.append((path, cache_key))
    for path, cache_key in remove:
        existing = get_tags_from_global_cache(cache_key, tag.artifact_id)
        if len(existing) > 1:
            remove_tag.append((path, cache_key))
        else:
            del_list.append((path, cache_key))
    results = RefreshIndexResults(compute=compute, del_=del_list, add_tag=add_tag, remove_tag=remove_tag)
    def mark_complete_with_global(items: List[PathAndCacheKey], result_type: IndexResultType) -> None:
        mark_complete(items, result_type)
        db = get_db()
        if result_type == IndexResultType.COMPUTE or result_type == IndexResultType.ADD_TAG:
            for path, cache_key in items:
                db.execute(
                    "REPLACE INTO global_cache (cacheKey, dir, branch, artifactId) VALUES (?, ?, ?, ?)",
                    (cache_key, tag.directory, tag.branch, tag.artifact_id),
                )
        elif result_type == IndexResultType.DELETE or result_type == IndexResultType.REMOVE_TAG:
            for path, cache_key in items:
                db.execute(
                    "DELETE FROM global_cache WHERE cacheKey = ? AND dir = ? AND branch = ? AND artifactId = ?",
                    (cache_key, tag.directory, tag.branch, tag.artifact_id),
                )
        db.commit()
    return results, last_updated, mark_complete_with_global


def index_lock_is_locked() -> Optional[Tuple[bool, int, str]]:
    db = get_db()
    cur = db.execute("SELECT locked, timestamp, dirs FROM indexing_lock WHERE locked = 1 LIMIT 1")
    row = cur.fetchone()
    return (bool(row[0]), row[1], row[2]) if row else None


def index_lock_lock(dirs: str) -> None:
    db = get_db()
    db.execute("INSERT INTO indexing_lock (locked, timestamp, dirs) VALUES (1, ?, ?)", (int(__import__("time").time() * 1000), dirs))
    db.commit()


def index_lock_update_timestamp() -> None:
    db = get_db()
    db.execute("UPDATE indexing_lock SET timestamp = ? WHERE locked = 1", (int(__import__("time").time() * 1000),))
    db.commit()


def index_lock_unlock() -> None:
    db = get_db()
    db.execute("DELETE FROM indexing_lock WHERE locked = 1")
    db.commit()
