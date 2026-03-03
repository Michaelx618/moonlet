"""Adapters that normalize existing model backends to a provider interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MLXProviderAdapter:
    kind = "mlx"
    name = "mlx-lm"

    def __init__(self, backend: Any):
        self.backend = backend

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
        _ = cache_key
        return self.backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            stop=stop,
        )

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind}

    def clear_cache(self) -> None:
        return None

