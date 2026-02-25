"""Indexing types: IndexTag, PathAndCacheKey, RefreshIndexResults, etc."""

from enum import Enum
from typing import Any, Callable, Iterator, List, Optional, Protocol, Tuple

PathAndCacheKey = Tuple[str, str]  # (path, cache_key)


class IndexResultType(str, Enum):
    COMPUTE = "compute"
    DELETE = "del"
    ADD_TAG = "addTag"
    REMOVE_TAG = "removeTag"
    UPDATE_LAST_UPDATED = "updateLastUpdated"


class IndexTag:
    def __init__(self, directory: str, branch: str, artifact_id: str):
        self.directory = directory
        self.branch = branch
        self.artifact_id = artifact_id

    def __repr__(self) -> str:
        return f"IndexTag({self.directory!r}, {self.branch!r}, {self.artifact_id!r})"


class RefreshIndexResults:
    """The four lists from get_compute_delete_add_remove."""

    def __init__(
        self,
        compute: Optional[List[PathAndCacheKey]] = None,
        del_: Optional[List[PathAndCacheKey]] = None,
        add_tag: Optional[List[PathAndCacheKey]] = None,
        remove_tag: Optional[List[PathAndCacheKey]] = None,
    ):
        self.compute = compute or []
        self.del_ = del_ or []
        self.add_tag = add_tag or []
        self.remove_tag = remove_tag or []


MarkCompleteCallback = Callable[[List[PathAndCacheKey], IndexResultType], Any]


class CodebaseIndex(Protocol):
    artifact_id: str

    def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: MarkCompleteCallback,
        repo_name: Optional[str],
    ) -> Iterator[dict]:
        ...
