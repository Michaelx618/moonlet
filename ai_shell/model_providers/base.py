"""Provider protocol for model runtimes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class ModelProvider(Protocol):
    kind: str
    name: str

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout_s: int,
        stop: Optional[List[str]] = None,
        cache_key: str = "",
    ) -> str:
        ...

    def status(self) -> Dict[str, Any]:
        ...

    def clear_cache(self) -> None:
        ...

