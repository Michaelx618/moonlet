from pathlib import Path

from ai_shell.rag_store import RagStore


def test_rag_store_build_and_query(tmp_path: Path):
    root = tmp_path
    (root / "a.py").write_text("def add(x, y):\n    return x + y\n")
    (root / "b.py").write_text("def subtract(x, y):\n    return x - y\n")
    store = RagStore(root=root)
    # Build index with explicit paths
    n = store.build(["a.py", "b.py"], force=True)
    assert n >= 2
    hits = store.query("add numbers", top_k=2)
    assert hits
    assert any(h.path == "a.py" for h in hits)

