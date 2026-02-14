"""Semantic search via embeddings (optional: requires sentence-transformers).

When SC2_SEMANTIC_SEARCH=1 and sentence-transformers is installed,
find files by meaning, not just keywords.
"""

from typing import List, Optional, Tuple

from . import config
from .utils import dbg

_embedding_model = None


def _get_embedding_model():
    """Lazy-load SentenceTransformer; return None if import or load fails."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    if not getattr(config, "SEMANTIC_SEARCH_ENABLED", False):
        return None
    try:
        from sentence_transformers import SentenceTransformer

        model_name = getattr(config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer(model_name)
        dbg(f"semantic: loaded embedding model {model_name}")
        return _embedding_model
    except ImportError:
        dbg("semantic: sentence_transformers not installed, skipping")
        return None
    except Exception as exc:
        dbg(f"semantic: model load failed: {exc}")
        return None


def embed_text(text: str) -> Optional[List[float]]:
    """Embed single string. Returns None if model unavailable."""
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()
    except Exception as exc:
        dbg(f"semantic: embed failed: {exc}")
        return None


def embed_texts(texts: List[str]):
    """Batch embed. Returns numpy array or None if model unavailable."""
    model = _get_embedding_model()
    if model is None or not texts:
        return None
    try:
        return model.encode(texts, convert_to_numpy=True)
    except Exception as exc:
        dbg(f"semantic: batch embed failed: {exc}")
        return None


def semantic_search_files(
    query: str,
    file_summaries: List[Tuple[str, str]],
    top_k: int = 5,
) -> List[str]:
    """Find top_k files by semantic similarity to query.

    file_summaries: [(path, content_preview), ...]
    Returns list of paths ordered by relevance (highest first).
    """
    if not file_summaries:
        return []
    model = _get_embedding_model()
    if model is None:
        return []
    try:
        import numpy as np

        docs = [f"{path}\n{preview}" for path, preview in file_summaries]
        paths = [path for path, _ in file_summaries]
        query_vec = model.encode(query, convert_to_numpy=True)
        doc_vecs = model.encode(docs, convert_to_numpy=True)
        # Cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
        scores = np.dot(doc_norms, query_norm)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [paths[i] for i in top_indices]
    except Exception as exc:
        dbg(f"semantic: search failed: {exc}")
        return []
