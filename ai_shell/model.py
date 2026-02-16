import atexit
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from . import config
from .utils import dbg


class _LlamaServerBackend:
    def __init__(
        self,
        model_path: str,
        server_bin: str,
        host: str,
        port: int,
        ctx: int,
        threads: int,
        gpu_layers: int,
    ):
        self.model_path = model_path
        self.server_bin = server_bin
        self.host = host
        self.port = int(port)
        self.ctx = max(512, int(ctx or 4096))
        self.threads = max(1, int(threads or 1))
        self.gpu_layers = int(gpu_layers)
        self.proc: Optional[subprocess.Popen] = None
        self._start_lock = threading.Lock()
        self.base_url = f"http://{self.host}:{self.port}"
        atexit.register(self.stop)

    def stop(self) -> None:
        proc = self.proc
        if not proc:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        except Exception:
            pass
        self.proc = None

    def _is_healthy(self, timeout: float = 1.0) -> bool:
        try:
            req = urllib_request.Request(
                self.base_url + "/health",
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            with urllib_request.urlopen(req, timeout=max(0.05, float(timeout))) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _port_open(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.5):
                return True
        except Exception:
            return False

    def _is_port_bindable(self, port: int) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, int(port)))
            return True
        except Exception:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass

    def ensure_started(self) -> None:
        with self._start_lock:
            if self.proc and self.proc.poll() is None and self._is_healthy():
                return

            if self._is_healthy():
                # Existing server already running (possibly from previous launch).
                self.proc = None
                return

            chosen_port = self.port
            if self._port_open() and not self._is_healthy():
                # Something else is bound; pick the next available local port.
                for cand in range(self.port + 1, self.port + 40):
                    if self._is_port_bindable(cand):
                        chosen_port = cand
                        break
                self.port = chosen_port
                self.base_url = f"http://{self.host}:{self.port}"
                dbg(f"llama-server port busy; switched to {self.port}")

            cmd = [
                self.server_bin,
                "--model",
                self.model_path,
                "--host",
                self.host,
                "--port",
                str(chosen_port),
                "--ctx-size",
                str(self.ctx),
                "--threads",
                str(self.threads),
            ]
            if self.gpu_layers is not None:
                cmd.extend(["--n-gpu-layers", str(self.gpu_layers)])
            cmd.extend(["--no-warmup"])

            dbg(f"llama-server start: {' '.join(cmd)}")
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=False,
            )

            deadline = time.time() + max(15, int(config.LLAMA_SERVER_START_TIMEOUT))
            while time.time() < deadline:
                if self.proc and self.proc.poll() is not None:
                    raise RuntimeError("llama-server exited during startup")
                if self._is_healthy() or self._port_open():
                    # Give HTTP endpoint one more moment after port binds.
                    time.sleep(0.15)
                    if self._is_healthy():
                        return
                time.sleep(0.15)
            raise RuntimeError("llama-server did not become ready in time")

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
        self.ensure_started()
        payload = {
            "prompt": prompt,
            "n_predict": max(1, int(max_tokens)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        if bool(getattr(config, "STRUCTURAL_KV_CACHE_ENABLED", True)):
            payload["cache_prompt"] = True
        if cache_key:
            slots = max(1, int(getattr(config, "LLAMA_SERVER_CACHE_SLOTS", 32)))
            slot = int(hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:8], 16) % slots
            payload["id_slot"] = int(slot)
            dbg(f"llama-server cache slot={slot} key={cache_key[:24]}")
            if getattr(config, "DEBUG_KV_CACHE", False):
                dbg(f"kv_cache: request slot={slot} prompt_len={len(prompt)} cache_prompt=True")
        stop_list = [s for s in (stop or []) if s]
        if stop_list:
            payload["stop"] = stop_list

        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.base_url + "/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=max(1, int(timeout_s))) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"llama-server HTTP {getattr(exc, 'code', '?')}: {detail[:300]}")
        except Exception as exc:
            raise RuntimeError(f"llama-server request failed: {exc}")

        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"llama-server returned invalid JSON: {exc}")

        if getattr(config, "DEBUG_KV_CACHE", False) and isinstance(obj, dict):
            cache_tok = obj.get("cache_prompt_tokens") or obj.get("prompt_tokens")
            timings = obj.get("timings") or {}
            dbg(f"kv_cache: response cache_prompt_tokens={cache_tok} timings={timings}")

        if isinstance(obj, dict):
            if isinstance(obj.get("content"), str):
                return obj.get("content", "")
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    text = c0.get("text")
                    if isinstance(text, str):
                        return text
                    msg = c0.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg.get("content", "")
        raise RuntimeError("llama-server response missing completion content")


class _CliBackend:
    def __init__(self, model_path: str, ctx: int, threads: int, cli_bin: str):
        self.model_path = model_path
        self.ctx = max(512, int(ctx or 4096))
        self.threads = max(1, int(threads or 1))
        self.cli_bin = cli_bin

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout_s: int,
    ) -> str:
        cmd = [
            self.cli_bin,
            "-m",
            self.model_path,
            "--device",
            "none",
            "-no-cnv",
            "--simple-io",
            "--no-display-prompt",
            "--no-warmup",
            "--color",
            "off",
            "-lv",
            "0",
            "-t",
            str(self.threads),
            "-c",
            str(self.ctx),
            "-n",
            str(max(1, int(max_tokens))),
            "--temp",
            f"{float(temperature):.3f}",
            "--top-p",
            f"{float(top_p):.3f}",
            "-p",
            prompt,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)),
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(err or f"{os.path.basename(self.cli_bin)} exited {proc.returncode}")
        return (proc.stdout or "").strip()


_BACKEND_NAME = "unknown"
_BACKEND_KIND = "unknown"


def _find_llama_server_binary() -> Optional[str]:
    env_bin = str(config.LLAMA_SERVER_BIN or os.getenv("SC2_LLAMA_SERVER_BIN", "")).strip()
    repo_bin = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "tools",
            "llama.cpp",
            "bin",
            "llama-server",
        )
    )
    candidates = [
        env_bin,
        repo_bin,
        "/opt/homebrew/opt/llama.cpp/bin/llama-server",
        shutil.which("llama-server") or "",
    ]
    for cand in candidates:
        if cand and os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _find_cli_binary() -> Optional[str]:
    env_bin = str(os.getenv("SC2_LLAMA_CLI", "")).strip()
    repo_bin = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "tools",
            "llama.cpp",
            "bin",
            "llama-completion",
        )
    )
    repo_cli = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "tools",
            "llama.cpp",
            "bin",
            "llama-cli",
        )
    )
    candidates = [
        env_bin,
        repo_bin,
        repo_cli,
        "/opt/homebrew/opt/llama.cpp/bin/llama-completion",
        shutil.which("llama-completion") or "",
        shutil.which("llama-cli") or "",
    ]
    for cand in candidates:
        if not cand:
            continue
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def load_model() -> Tuple[Optional[object], object]:
    """Load GGUF backend (llama-server, python binding, or CLI fallback)."""
    global _BACKEND_NAME, _BACKEND_KIND
    if not config.GGUF_PATH:
        raise SystemExit(
            "Set SC2_GGUF to a .gguf file path (and install llama-cpp-python)."
        )

    model_name = os.path.basename(config.GGUF_PATH)
    use_llama_server = bool(config.LLAMA_SERVER_ENABLED)
    server_bin = _find_llama_server_binary()
    cli_bin = _find_cli_binary()
    prefer_cli = os.getenv("SC2_PREFER_CLI", "1").lower() not in {"0", "false", "no"}

    if use_llama_server and server_bin:
        print(f"[Loading GGUF model via llama-server: {model_name}]", file=sys.stderr)
        print(f"[  path: {config.GGUF_PATH}]", file=sys.stderr)
        print(f"[  llama-server: {server_bin}]", file=sys.stderr)
        print(f"[  gpu_layers: {config.GGUF_GPU_LAYERS}]", file=sys.stderr)
        print(
            f"[  endpoint: http://{config.LLAMA_SERVER_HOST}:{config.LLAMA_SERVER_PORT}]",
            file=sys.stderr,
        )
        _BACKEND_NAME = "gguf-server"
        _BACKEND_KIND = "gguf_server"
        backend = _LlamaServerBackend(
            model_path=config.GGUF_PATH,
            server_bin=server_bin,
            host=config.LLAMA_SERVER_HOST,
            port=config.LLAMA_SERVER_PORT,
            ctx=config.GGUF_CTX,
            threads=config.GGUF_THREADS,
            gpu_layers=config.GGUF_GPU_LAYERS,
        )
        # Start once at boot so model remains loaded and ready for requests.
        backend.ensure_started()
        print(f"[  llama-server ready: {backend.base_url}]", file=sys.stderr)
        return None, backend

    if prefer_cli and cli_bin:
        print(f"[Loading GGUF model via llama.cpp CLI: {model_name}]", file=sys.stderr)
        print(f"[  path: {config.GGUF_PATH}]", file=sys.stderr)
        print(f"[  cli: {cli_bin}]", file=sys.stderr)
        _BACKEND_NAME = "gguf-cli"
        _BACKEND_KIND = "gguf_cli"
        return None, _CliBackend(config.GGUF_PATH, config.GGUF_CTX, config.GGUF_THREADS, cli_bin)

    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:
        if prefer_cli and cli_bin:
            print(
                "[llama_cpp unavailable; falling back to llama.cpp CLI backend]",
                file=sys.stderr,
            )
            _BACKEND_NAME = "gguf-cli"
            _BACKEND_KIND = "gguf_cli"
            return None, _CliBackend(config.GGUF_PATH, config.GGUF_CTX, config.GGUF_THREADS, cli_bin)
        raise SystemExit("Install with: pip install llama-cpp-python") from exc

    print(f"[Loading GGUF model: {model_name}]", file=sys.stderr)
    print(f"[  path: {config.GGUF_PATH}]", file=sys.stderr)
    llm = Llama(
        model_path=config.GGUF_PATH,
        n_ctx=config.GGUF_CTX,
        n_threads=config.GGUF_THREADS,
        n_gpu_layers=config.GGUF_GPU_LAYERS,
        flash_attn=True,
        n_batch=512,
        verbose=False,
    )
    print(
        f"[Model loaded: {model_name}, ctx={config.GGUF_CTX}, threads={config.GGUF_THREADS}, gpu_layers={config.GGUF_GPU_LAYERS}]",
        file=sys.stderr,
    )
    _BACKEND_NAME = "gguf"
    _BACKEND_KIND = "gguf_python"
    return None, llm


tokenizer, model = load_model()

# Model access is not thread-safe (binding or subprocess backend).
MODEL_LOCK = threading.Lock()


def backend_name() -> str:
    return _BACKEND_NAME


def backend_status() -> Dict[str, Any]:
    info: Dict[str, Any] = {"backend": _BACKEND_NAME, "kind": _BACKEND_KIND}
    if _BACKEND_KIND == "gguf_server" and isinstance(model, _LlamaServerBackend):
        proc = model.proc
        info["llama_server"] = {
            "base_url": model.base_url,
            "healthy": bool(model._is_healthy(timeout=0.25)),
            "port_open": bool(model._port_open()),
            "pid": int(proc.pid) if (proc and proc.poll() is None) else None,
        }
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


def _stream_reply_cli(
    prompt: str,
    label: str,
    silent: bool,
    max_new: int,
    stop_sequences: Optional[List[str]],
    temperature: Optional[float],
    cache_key: str = "",
) -> str:
    _ = cache_key
    dbg(f"stream_reply: gguf-cli prompt_len={len(prompt)}")
    capped_max_new = max_new if max_new > 0 else config.MAX_NEW
    dbg(f"stream_reply: gguf-cli max_new={capped_max_new}")
    temp = config.TEMPERATURE if temperature is None else temperature
    head = prompt[:300].replace("\n", "\\n")
    tail = prompt[-300:].replace("\n", "\\n")
    dbg(f"stream_reply: prompt_head={head}")
    dbg(f"stream_reply: prompt_tail={tail}")

    start_time = time.time()
    if not silent:
        sys.stdout.write(label)
        sys.stdout.flush()

    try:
        with MODEL_LOCK:
            reply_raw = model.generate(
                prompt=prompt,
                max_tokens=capped_max_new,
                temperature=temp,
                top_p=config.TOP_P,
                timeout_s=config.GEN_TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        dbg(f"stream_reply: gguf-cli timeout after {config.GEN_TIMEOUT}s")
        reply = "[Model timeout]"
        stop_reason = "timeout"
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
    out_tok_est = len(reply.split())
    total_tok_est = len(prompt.split()) + out_tok_est
    gen_tok_per_sec = out_tok_est / max(gen_ms / 1000, 0.001)
    total_tok_per_sec = total_tok_est / max(gen_ms / 1000, 0.001)
    dbg(
        f"stream_reply: gguf-cli reply_len={len(reply)} gen_ms={gen_ms} "
        f"out_tok~={out_tok_est} total_tok~{total_tok_est} "
        f"gen_tok/s={gen_tok_per_sec:.1f} total_tok/s={total_tok_per_sec:.1f} "
        f"stop_reason={stop_reason}"
    )
    if reply.startswith("Error:") or reply.startswith("[Model error"):
        dbg(f"stream_reply: output looks like error: {reply[:100]}")
    return reply


def _stream_reply_server(
    prompt: str,
    label: str,
    silent: bool,
    max_new: int,
    stop_sequences: Optional[List[str]],
    temperature: Optional[float],
    cache_key: str = "",
) -> str:
    dbg(f"stream_reply: gguf-server prompt_len={len(prompt)}")
    capped_max_new = max_new if max_new > 0 else config.MAX_NEW
    dbg(f"stream_reply: gguf-server max_new={capped_max_new}")
    temp = config.TEMPERATURE if temperature is None else temperature
    head = prompt[:300].replace("\n", "\\n")
    tail = prompt[-300:].replace("\n", "\\n")
    dbg(f"stream_reply: prompt_head={head}")
    dbg(f"stream_reply: prompt_tail={tail}")

    start_time = time.time()
    if not silent:
        sys.stdout.write(label)
        sys.stdout.flush()

    try:
        with MODEL_LOCK:
            reply_raw = model.generate(
                prompt=prompt,
                max_tokens=capped_max_new,
                temperature=temp,
                top_p=config.TOP_P,
                timeout_s=config.GEN_TIMEOUT,
                stop=stop_sequences,
                cache_key=cache_key,
            )
    except Exception as exc:
        msg = str(exc)
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            dbg(f"stream_reply: gguf-server timeout after {config.GEN_TIMEOUT}s")
            reply = "[Model timeout]"
            stop_reason = "timeout"
        else:
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
    out_tok_est = len(reply.split())
    total_tok_est = len(prompt.split()) + out_tok_est
    gen_tok_per_sec = out_tok_est / max(gen_ms / 1000, 0.001)
    total_tok_per_sec = total_tok_est / max(gen_ms / 1000, 0.001)
    dbg(
        f"stream_reply: gguf-server reply_len={len(reply)} gen_ms={gen_ms} "
        f"out_tok~={out_tok_est} total_tok~{total_tok_est} "
        f"gen_tok/s={gen_tok_per_sec:.1f} total_tok/s={total_tok_per_sec:.1f} "
        f"stop_reason={stop_reason}"
    )
    return reply


def stream_reply(
    prompt: str,
    label: str = "> ",
    silent: bool = False,
    max_new: int = 0,
    stop_sequences: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    cache_key: str = "",
) -> str:
    """Stream model output while returning full text."""
    if _BACKEND_KIND == "gguf_server":
        return _stream_reply_server(
            prompt,
            label,
            silent,
            max_new,
            stop_sequences,
            temperature,
            cache_key,
        )
    if _BACKEND_KIND == "gguf_cli":
        return _stream_reply_cli(
            prompt,
            label,
            silent,
            max_new,
            stop_sequences,
            temperature,
            cache_key,
        )

    dbg(f"stream_reply: gguf prompt_len={len(prompt)}")
    capped_max_new = max_new if max_new > 0 else config.MAX_NEW
    dbg(f"stream_reply: gguf max_new={capped_max_new}")
    temp = config.TEMPERATURE if temperature is None else temperature
    head = prompt[:300].replace("\n", "\\n")
    tail = prompt[-300:].replace("\n", "\\n")
    dbg(f"stream_reply: prompt_head={head}")
    dbg(f"stream_reply: prompt_tail={tail}")

    tokens: List[str] = []
    start_time = time.time()
    if not silent:
        sys.stdout.write(label)
    stop_reason = "unknown"
    saw_finish_reason: Optional[str] = None
    accumulated = ""

    native_stop = list(stop_sequences or [])
    native_stop.append("[[[end]]]")

    with MODEL_LOCK:
        try:
            # type: ignore[attr-defined]
            for chunk in model(
                prompt,
                max_tokens=capped_max_new,
                temperature=temp,
                top_p=config.TOP_P,
                stop=native_stop,
                stream=True,
            ):
                choice = chunk["choices"][0]
                token = choice.get("text", "")
                tokens.append(token)
                accumulated += token
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    saw_finish_reason = str(finish_reason)
                    if finish_reason == "stop":
                        stop_reason = "native_stop"
                    elif finish_reason not in {"length"}:
                        stop_reason = str(finish_reason)
                if not silent:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                if "[[[end]]]" in accumulated:
                    dbg("stream_reply: early stop on [[[end]]] marker")
                    stop_reason = "native_stop"
                    break
                if (
                    len(tokens) >= max(1, config.REP_MIN_TOKENS)
                    and len(accumulated) >= max(1, config.REP_MIN_CHARS)
                ):
                    window = max(10, config.REP_WINDOW)
                    tail_toks = tokens[-window:]
                    norm = []
                    for t in tail_toks:
                        s = (t or "").strip().lower()
                        if not s:
                            continue
                        if re.fullmatch(r"[\W_]+", s):
                            continue
                        norm.append(s)
                    if len(norm) >= 10:
                        freq = {}
                        for s in norm:
                            freq[s] = freq.get(s, 0) + 1
                        top = max(freq.values()) if freq else 0
                        ratio = top / max(1, len(norm))
                        if ratio >= 0.90:
                            dbg(
                                f"stream_reply: repetition loop detected after {len(tokens)} tokens "
                                f"(ratio={ratio:.2f}), stopping"
                            )
                            stop_reason = "repetition"
                            break
                elapsed = time.time() - start_time
                if elapsed > config.GEN_TIMEOUT:
                    dbg(f"stream_reply: gguf timeout after {config.GEN_TIMEOUT}s")
                    stop_reason = "timeout"
                    break
                if elapsed > 20 and len(tokens) < 10:
                    dbg(
                        f"stream_reply: stall detected ({len(tokens)} tokens in {int(elapsed)}s), aborting"
                    )
                    tokens.clear()
                    tokens.append("[Model stalled]")
                    stop_reason = "stall"
                    break
        except Exception as exc:
            error_msg = f"[Model error: {exc}]"
            if not silent:
                sys.stdout.write(f"\n{error_msg}\n")
            return error_msg

    if not silent:
        sys.stdout.write("\n")
    reply = "".join(tokens).strip()
    gen_ms = int((time.time() - start_time) * 1000)
    total_tokens = len(prompt.split()) + len(tokens)
    gen_tok_per_sec = len(tokens) / max(gen_ms / 1000, 0.001)
    total_tok_per_sec = total_tokens / max(gen_ms / 1000, 0.001)
    if stop_reason == "unknown":
        if saw_finish_reason == "length":
            if len(tokens) >= capped_max_new:
                stop_reason = "max_tokens"
            else:
                stop_reason = "model_length"
        else:
            stop_reason = "max_tokens" if len(tokens) >= capped_max_new else "eos"
    if not reply and time.time() - start_time > config.GEN_TIMEOUT:
        reply = "[Model timeout]"
    dbg(
        f"stream_reply: gguf reply_len={len(reply)} gen_ms={gen_ms} "
        f"out_tok={len(tokens)} total_tok~{total_tokens} "
        f"gen_tok/s={gen_tok_per_sec:.1f} total_tok/s={total_tok_per_sec:.1f} "
        f"stop_reason={stop_reason}"
    )
    if reply.startswith("Error:") or reply.startswith("[Model error"):
        dbg(f"stream_reply: output looks like error: {reply[:100]}")
    return reply


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
            slot = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % max(1, int(getattr(config, "LLAMA_SERVER_CACHE_SLOTS", 32)))
            dbg(f"kv_cache: session_key={key} slot={slot} root={root[:40]}... include={len(include)} files")
        return key
    except Exception:
        return ""


def stream_reply_chunks(
    prompt: str,
    max_new: int = 0,
    stop_sequences: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    cache_key: str = "",
) -> Iterator[str]:
    """Yield model output chunks for streaming to clients."""
    capped_max_new = max_new if max_new > 0 else config.MAX_NEW
    temp = config.TEMPERATURE if temperature is None else temperature

    if _BACKEND_KIND == "gguf_server":
        try:
            with MODEL_LOCK:
                raw = model.generate(
                    prompt=prompt,
                    max_tokens=capped_max_new,
                    temperature=temp,
                    top_p=config.TOP_P,
                    timeout_s=config.GEN_TIMEOUT,
                    stop=stop_sequences,
                    cache_key=cache_key,
                )
        except Exception as exc:
            yield f"[Model error: {exc}]"
            return
        clipped, _reason = _apply_stop_sequences(raw, stop_sequences)
        text = clipped.strip()
        chunk_size = 96
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
        return

    if _BACKEND_KIND == "gguf_cli":
        try:
            with MODEL_LOCK:
                raw = model.generate(
                    prompt=prompt,
                    max_tokens=capped_max_new,
                    temperature=temp,
                    top_p=config.TOP_P,
                    timeout_s=config.GEN_TIMEOUT,
                )
        except subprocess.TimeoutExpired:
            yield "[Model timeout]"
            return
        except Exception as exc:
            yield f"[Model error: {exc}]"
            return
        clipped, _reason = _apply_stop_sequences(raw, stop_sequences)
        text = clipped.strip()
        chunk_size = 96
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
        return

    start_time = time.time()
    native_stop = list(stop_sequences or [])

    with MODEL_LOCK:
        # type: ignore[attr-defined]
        for chunk in model(
            prompt,
            max_tokens=capped_max_new,
            temperature=temp,
            top_p=config.TOP_P,
            stop=native_stop if native_stop else None,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            yield token
            if time.time() - start_time > config.GEN_TIMEOUT:
                break
