"""Adapters that normalize existing model backends to a provider interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ServerProviderAdapter:
    kind = "gguf_server"
    name = "llama.cpp-server"

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
        return self.backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            stop=stop,
            cache_key=cache_key,
        )

    def status(self) -> Dict[str, Any]:
        proc = getattr(self.backend, "proc", None)
        return {
            "name": self.name,
            "kind": self.kind,
            "base_url": getattr(self.backend, "base_url", ""),
            "healthy": bool(self.backend._is_healthy(timeout=0.25)),
            "port_open": bool(self.backend._port_open()),
            "pid": int(proc.pid) if (proc and proc.poll() is None) else None,
        }

    def clear_cache(self) -> None:
        if hasattr(self.backend, "clear_kv_cache"):
            self.backend.clear_kv_cache()


class CliProviderAdapter:
    kind = "gguf_cli"
    name = "llama.cpp-cli"

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
        _ = stop, cache_key
        return self.backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
        )

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind}

    def clear_cache(self) -> None:
        return None


class PythonProviderAdapter:
    kind = "gguf_python"
    name = "llama.cpp-python"

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
        _ = timeout_s, cache_key
        chunks = self.backend(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop if stop else None,
            stream=False,
        )
        if isinstance(chunks, dict):
            choices = chunks.get("choices") or []
            if choices and isinstance(choices[0], dict):
                return str(choices[0].get("text") or "")
        return ""

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind}

    def clear_cache(self) -> None:
        return None

