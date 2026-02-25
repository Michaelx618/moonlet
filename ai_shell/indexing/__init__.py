"""Content-addressed indexing: catalog, incremental refresh, code snippets, chunks, FTS, embeddings."""

from .chunk_index import ChunkCodebaseIndex, get_chunks_for_path
from .codebase_indexer import (
    refresh_codebase_index,
    refresh_index_file,
    refresh_index_simple,
    get_indexes_to_build,
)
from .code_snippets_index import (
    CodeSnippetsCodebaseIndex,
    get_all_snippets_for_tag,
)
from .embeddings_index import EmbeddingsCodebaseIndex, query_embeddings
from .full_text_search_index import (
    FullTextSearchCodebaseIndex,
    retrieve_fts,
)
from .refresh_index import (
    get_db,
    get_compute_delete_add_remove,
    get_add_remove_for_tag,
    index_lock_lock,
    index_lock_unlock,
    index_lock_is_locked,
)
from .types import (
    IndexTag,
    IndexResultType,
    PathAndCacheKey,
    RefreshIndexResults,
)
from .utils import tag_to_string, calculate_hash

__all__ = [
    "ChunkCodebaseIndex",
    "get_chunks_for_path",
    "refresh_codebase_index",
    "refresh_index_file",
    "refresh_index_simple",
    "get_indexes_to_build",
    "CodeSnippetsCodebaseIndex",
    "get_all_snippets_for_tag",
    "EmbeddingsCodebaseIndex",
    "query_embeddings",
    "FullTextSearchCodebaseIndex",
    "retrieve_fts",
    "get_db",
    "get_compute_delete_add_remove",
    "get_add_remove_for_tag",
    "index_lock_lock",
    "index_lock_unlock",
    "index_lock_is_locked",
    "IndexTag",
    "IndexResultType",
    "PathAndCacheKey",
    "RefreshIndexResults",
    "tag_to_string",
    "calculate_hash",
]
