"""Indexing utils: tag string, content hash."""

import hashlib


def tag_to_string(tag) -> str:
    """Single string for (dir, branch, artifact_id) for use as tag in DB."""
    return f"{tag.directory}|{tag.branch}|{tag.artifact_id}"


def calculate_hash(file_contents: str) -> str:
    """Content-addressable cache key (SHA-256 of file contents)."""
    h = hashlib.sha256()
    h.update(file_contents.encode("utf-8", errors="replace"))
    return h.hexdigest()
