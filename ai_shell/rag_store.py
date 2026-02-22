"""Lightweight persistent hybrid retrieval for local repos."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .files import get_root, read_single_file_for_context


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_]\w{2,}", (text or "").lower())


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


@dataclass
class RagHit:
    path: str
    score: float


class RagStore:
    def __init__(self, root: Path | None = None):
        self.root = (root or get_root()).resolve()
        self.path = self.root / ".moonlet_rag_index.json"

    def build(self, paths: List[str], force: bool = True) -> int:
        data: Dict[str, Dict[str, object]] = {"files": {}}
        if not force and self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except Exception:
                data = {"files": {}}
        files = data.setdefault("files", {})
        for rel in paths:
            text = read_single_file_for_context(rel).get(rel, "")
            if not text:
                continue
            preview = text[:3000]
            toks = _tokens(preview)
            files[rel] = {"preview": preview, "tokens": toks[:500]}
        self.path.write_text(json.dumps(data, ensure_ascii=True))
        return len(files)

    def query(self, query_text: str, top_k: int = 5) -> List[RagHit]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return []
        files = data.get("files") or {}
        q_toks = _tokens(query_text)
        scored: List[Tuple[str, float]] = []
        for rel, meta in files.items():
            toks = list((meta or {}).get("tokens") or [])
            preview = str((meta or {}).get("preview") or "")
            kw_score = _jaccard(q_toks, toks)
            text_score = 0.0
            low = preview.lower()
            for t in set(q_toks[:12]):
                if t in low:
                    text_score += 0.03
            score = kw_score + text_score
            if score > 0:
                scored.append((str(rel), float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [RagHit(path=p, score=s) for p, s in scored[: max(1, int(top_k))]]

