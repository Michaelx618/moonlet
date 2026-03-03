import hashlib
import json
import os
import re
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import config
from .model_roles import resolve_role_config
from .model_providers import MLXProviderAdapter
from .utils import dbg


class _MLXBackend:
    """MLX (Apple Silicon) backend using mlx_lm. Model id = HuggingFace id or local path."""

    def __init__(self, model_id: str):
        from mlx_lm import load
        self.model_id = model_id
        self._model, self._tokenizer = load(model_id)

    def format_prompt(self, user_content: str) -> str:
        """Format user text with the tokenizer's chat template (required for correct generation)."""
        tokenizer = self._tokenizer
        if getattr(tokenizer, "has_chat_template", False) or getattr(
            getattr(tokenizer, "_tokenizer", None), "chat_template", None
        ):
            try:
                messages = [{"role": "user", "content": user_content}]
                out = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                if isinstance(out, str):
                    return out
                # tokenize=True returns list of ids; decode for string prompt
                if hasattr(tokenizer, "decode"):
                    return tokenizer.decode(out, skip_special_tokens=False)
            except Exception:
                pass
        return user_content

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout_s: int,
        stop: Optional[List[str]] = None,
    ) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
        kwargs = {"sampler": sampler, "verbose": False}
        if getattr(config, "MLX_MAX_KV_SIZE", 0) > 0:
            kwargs["max_kv_size"] = config.MLX_MAX_KV_SIZE
        out = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max(1, int(max_tokens)),
            **kwargs,
        )
        if not isinstance(out, str):
            out = str(out or "")
        if stop:
            for s in stop:
                if s and s in out:
                    idx = out.find(s)
                    if idx >= 0:
                        out = out[:idx]
        return out.strip()

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ):
        """Yield tokens as they are generated so the UI can show progress."""
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler
        # mlx_lm 0.30+ generate_step() does not accept temperature/top_p; use sampler
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
        kwargs = {"sampler": sampler}
        if getattr(config, "MLX_MAX_KV_SIZE", 0) > 0:
            kwargs["max_kv_size"] = config.MLX_MAX_KV_SIZE
        accumulated = []
        for resp in mlx_stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max(1, int(max_tokens)),
            **kwargs,
        ):
            token = getattr(resp, "text", resp)
            if not isinstance(token, str):
                token = str(token or "")
            accumulated.append(token)
            # Stop on exact stop token (don't yield it)
            if stop:
                for s in stop:
                    if s and (token == s or token.strip() == s):
                        return
                full = "".join(accumulated)
                for s in stop:
                    if s and s in full:
                        return  # stop before yielding token that completed the sequence
            yield token


_BACKEND_NAME = "unknown"
_BACKEND_KIND = "unknown"
_PROVIDER: Optional[object] = None


def load_model() -> Tuple[Optional[object], object]:
    """Load model backend: MLX only (SC2_MLX_MODEL or SC2_MLX_MODEL_PATH)."""
    global _BACKEND_NAME, _BACKEND_KIND, _PROVIDER
    if not (config.MLX_MODEL or getattr(config, "MLX_MODEL_PATH", None)):
        raise SystemExit("Set SC2_MLX_MODEL or SC2_MLX_MODEL_PATH (GGUF/llama.cpp backends have been removed).")
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError("MLX backend requires: pip install mlx mlx-lm") from exc
    model_id = getattr(config, "MLX_MODEL_PATH", None) or config.MLX_MODEL
    if not model_id:
        raise RuntimeError("Set SC2_MLX_MODEL or SC2_MLX_MODEL_PATH")
    print(f"[Loading MLX model: {model_id}]", file=sys.stderr)
    sys.stderr.flush()
    print("[  Loading weights…]", file=sys.stderr)
    sys.stderr.flush()
    backend = _MLXBackend(model_id)
    _BACKEND_NAME = "mlx"
    _BACKEND_KIND = "mlx"
    _PROVIDER = MLXProviderAdapter(backend)
    print("[  MLX backend ready]", file=sys.stderr)
    sys.stderr.flush()
    return None, backend


# Lazy load for MLX: bind port first, then load model in background.
_model_ready = threading.Event()
_load_lock = threading.Lock()
_loading = False

if config.MLX_MODEL or getattr(config, "MLX_MODEL_PATH", None):
    _BACKEND_NAME = "mlx"
    _BACKEND_KIND = "mlx"
    _PROVIDER = None
    tokenizer = None
    model = None  # set by ensure_model_loaded()
else:
    tokenizer = None
    model = None
    raise SystemExit("Set SC2_MLX_MODEL or SC2_MLX_MODEL_PATH (GGUF/llama.cpp backends have been removed).")

# Model access is not thread-safe (binding or subprocess backend).
MODEL_LOCK = threading.Lock()


def ensure_model_loaded() -> None:
    """Load MLX model once (no-op if already loaded or not using MLX)."""
    global model, tokenizer, _PROVIDER, _loading
    if _BACKEND_KIND != "mlx":
        return
    with _load_lock:
        if model is not None:
            return
        if _loading:
            pass  # another thread is loading; we'll wait below
        else:
            _loading = True
            try:
                _tok, _m = load_model()
                tokenizer = _tok
                model = _m
            finally:
                _loading = False
                _model_ready.set()
            return
    _model_ready.wait()


def get_model():
    """Return the model backend; for MLX this triggers load if not yet loaded."""
    ensure_model_loaded()
    if _BACKEND_KIND == "mlx" and model is None:
        raise RuntimeError("MLX model failed to load; check server logs or runtime-debug.log")
    return model


def backend_name() -> str:
    return _BACKEND_NAME


def chat_completion_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    max_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """Chat completions with tools (native tool_calls). Not supported with MLX backend; use completion path (USE_CHAT_TOOLS=0)."""
    raise RuntimeError(
        "chat_completion_with_tools is not available (GGUF/llama.cpp backends removed). "
        "Use completion mode (USE_CHAT_TOOLS=0) so the agent uses stream_reply and parses tool calls from text."
    )


def clear_kv_cache(model: Any = None) -> None:
    """Clear backend cache if supported. Call on app shutdown."""
    _ = model
    if _PROVIDER and hasattr(_PROVIDER, "clear_cache"):
        _PROVIDER.clear_cache()
    else:
        dbg("clear_kv_cache: no clear_cache on provider")


def backend_status() -> Dict[str, Any]:
    info: Dict[str, Any] = {"backend": _BACKEND_NAME, "kind": _BACKEND_KIND}
    if _BACKEND_KIND == "mlx" and model is None:
        info["status"] = "loading"
        return info
    if _PROVIDER and hasattr(_PROVIDER, "status"):
        try:
            info["provider"] = _PROVIDER.status()
        except Exception:
            pass
    return info


def _apply_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> Tuple[str, str]:
    out = text or ""
    reason = "eos"
    stops = [s for s in (stop_sequences or []) if s]
    if "[[[end]]]" not in stops:
        stops.append("[[[end]]]")
    cut_at: Optional[int] = None
    for marker in stops:
        idx = out.find(marker)
        if idx >= 0 and (cut_at is None or idx < cut_at):
            cut_at = idx
    if cut_at is not None:
        out = out[:cut_at]
        reason = "native_stop"
    return out, reason


def _stream_reply_mlx(
    prompt: str,
    label: str,
    silent: bool,
    max_new: int,
    stop_sequences: Optional[List[str]],
    temperature: Optional[float],
    cache_key: str = "",
) -> str:
    _ = cache_key
    prompt, stop_sequences = _wrap_chatml(prompt, stop_sequences)
    dbg(f"stream_reply: mlx prompt_len={len(prompt)}")
    capped_max_new = max_new if max_new > 0 else config.MAX_NEW
    temp = config.TEMPERATURE if temperature is None else temperature
    start_time = time.time()
    if not silent:
        sys.stdout.write(label)
        sys.stdout.flush()
    try:
        with MODEL_LOCK:
            reply_raw = get_model().generate(
                prompt=prompt,
                max_tokens=capped_max_new,
                temperature=temp,
                top_p=config.TOP_P,
                timeout_s=config.GEN_TIMEOUT,
                stop=stop_sequences,
            )
    except Exception as exc:
        reply = f"[Model error: {exc}]"
        stop_reason = "error"
    else:
        reply, stop_reason = _apply_stop_sequences(reply_raw, stop_sequences)
        reply = reply.strip()
    if not silent:
        sys.stdout.write(reply)
        sys.stdout.write("\n")
        sys.stdout.flush()
    gen_ms = int((time.time() - start_time) * 1000)
    dbg(f"stream_reply: mlx reply_len={len(reply)} gen_ms={gen_ms} stop_reason={stop_reason}")
    return reply


def _wrap_chatml(prompt: str, stop_sequences: Optional[List[str]]) -> Tuple[str, List[str]]:
    """Wrap a raw prompt in chatml tags for chat-finetuned models.

    Returns (wrapped_prompt, updated_stop_sequences).
    """
    if not getattr(config, "USE_CHATML_WRAP", False):
        return prompt, list(stop_sequences or [])
    if "<|im_start|>" in prompt:
        return prompt, list(stop_sequences or [])
    wrapped = (
        "<|im_start|>system\n"
        "You are a coding assistant. Output only the requested code or tool calls. No explanations.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    stops = list(stop_sequences or [])
    if "<|im_end|>" not in stops:
        stops.append("<|im_end|>")
    # Prevent model from repeating assistant turn header (infinite loop)
    if "<|im_start|>" not in stops:
        stops.append("<|im_start|>")
    return wrapped, stops


def stream_reply(
    prompt: str,
    label: str = "> ",
    silent: bool = False,
    max_new: int = 0,
    stop_sequences: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    cache_key: str = "",
    role: str = "chat",
) -> str:
    """Stream model output while returning full text."""
    role_cfg = resolve_role_config(role)
    if max_new <= 0:
        max_new = int(role_cfg.max_new)
    if temperature is None:
        temperature = float(role_cfg.temperature)
    if not stop_sequences:
        stop_sequences = list(role_cfg.stop_sequences)
    if _BACKEND_KIND == "mlx":
        return _stream_reply_mlx(
            prompt,
            label,
            silent,
            max_new,
            stop_sequences,
            temperature,
            cache_key,
        )
    raise RuntimeError("Only MLX backend is supported (GGUF/llama.cpp removed).")


def get_session_cache_key() -> str:
    """Shared cache key for agent and chat — same slot so both modes reuse KV cache.
    Based on root + include so all requests in the same session share one slot."""
    try:
        from .files import get_include, get_root
        root = str(get_root() or "")
        include = get_include() or []
        parts = [root, "|".join(sorted(str(p) for p in include))]
        raw = "\0".join(parts)
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        if getattr(config, "DEBUG_KV_CACHE", False):
            slot = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 32
            inc_label = "disabled" if getattr(config, "DISABLE_INDEX", False) else f"{len(include)} files"
            dbg(f"kv_cache: session_key={key} slot={slot} root={root[:40]}... include={inc_label}")
        return key
    except Exception:
        return ""


def stream_reply_chunks(
    prompt: str,
    max_new: int = 0,
    stop_sequences: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    cache_key: str = "",
    role: str = "chat",
) -> Iterator[str]:
    """Yield model output chunks for streaming to clients."""
    role_cfg = resolve_role_config(role)
    if max_new <= 0:
        max_new = int(role_cfg.max_new)
    if temperature is None:
        temperature = float(role_cfg.temperature)
    if not stop_sequences:
        stop_sequences = list(role_cfg.stop_sequences)
    capped_max_new = max_new if max_new > 0 else config.MAX_NEW
    temp = config.TEMPERATURE if temperature is None else temperature

    if _BACKEND_KIND == "mlx":
        # Use same ChatML wrap as GGUF when disabled, so prompt structure matches and model is less likely to hallucinate edits.
        if getattr(config, "MLX_USE_CHAT_TEMPLATE", False):
            backend = get_model()
            prompt = backend.format_prompt(prompt)
            stops = list(stop_sequences or [])
            if "<|im_end|>" not in stops:
                stops.append("<|im_end|>")
            if "<|im_start|>" not in stops:
                stops.append("<|im_start|>")
            stop_sequences = stops
        else:
            prompt, stop_sequences = _wrap_chatml(prompt, stop_sequences)

    if _BACKEND_KIND == "mlx":
        try:
            print(f"[model] MLX stream_generate starting prompt_len={len(prompt)} max_tokens={capped_max_new}", file=sys.stderr)
            sys.stderr.flush()
            first = True
            with MODEL_LOCK:
                for token in get_model().stream_generate(
                    prompt=prompt,
                    max_tokens=capped_max_new,
                    temperature=temp,
                    top_p=config.TOP_P,
                    stop=stop_sequences,
                ):
                    # Yield every token including "" - MLX often yields "" for partial segments
                    if first and token:
                        print(f"[model] MLX first token: {repr(token[:80])}{'...' if len(token) > 80 else ''}", file=sys.stderr)
                        sys.stderr.flush()
                        first = False
                    yield token
        except Exception as exc:
            print(f"[model] MLX error: {exc}", file=sys.stderr)
            sys.stderr.flush()
            yield f"[Model error: {exc}]"
        return

    raise RuntimeError("Only MLX backend is supported (GGUF/llama.cpp removed).")
