import base64
import json
import mimetypes
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from . import config
from . import agent
from . import prompt_buffer, state
from .agent import run_agent_meta
from .files import (
    get_include,
    get_root,
    list_repo_files,
    read_file_text,
    resolve_path,
    set_include,
    set_root,
    _norm_rel_path,
    is_binary_file,
    write_file_text,
)
from .model import backend_name, backend_status, stream_reply_chunks
from .utils import dbg

try:
    from file_utils import (
        generate_diff,
        is_security_concern,
        validate_file_path,
    )
except ImportError:
    def generate_diff(*args, **kwargs):
        return ""

    def is_security_concern(*args, **kwargs):
        return False

    def validate_file_path(*args, **kwargs):
        raise NotImplementedError("file_utils not available")


class APIServer(BaseHTTPRequestHandler):
    def _attach_path_debug(self, meta: Dict, focus_file: Optional[str]):
        meta["root"] = str(get_root())
        meta["focus_file"] = focus_file
        meta["include"] = get_include()
        if not focus_file:
            return
        try:
            meta["abs_path"] = str(resolve_path(focus_file))
        except Exception as exc:
            meta["abs_path"] = f"[error] {exc}"

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Requested-With",
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

    def _json(self, status: int, body: Dict):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _sse(self):
        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _sse_send(self, data: str, event: Optional[str] = None):
        if event:
            self.wfile.write(f"event: {event}\n".encode("utf-8"))
        for line in (data or "").splitlines() or [""]:
            self.wfile.write(f"data: {line}\n".encode("utf-8"))
        self.wfile.write(b"\n")
        try:
            self.wfile.flush()
        except Exception:
            pass

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()
        return

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("content-length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return

        if parsed.path == "/stream":
            mode = data.get("mode", "agent")
            text = data.get("text") or ""
            focus_file = data.get("focus_file")
            file_path = data.get("file_path")
            if not text.strip():
                self._json(400, {"error": "text is required"})
                return
            self._sse()
            sent_done = False
            try:
                if file_path:
                    target = validate_file_path(
                        file_path, get_root(), must_exist=False
                    )
                    focus_file = str(target.relative_to(get_root()))
                elif focus_file:
                    target = validate_file_path(
                        focus_file, get_root(), must_exist=False
                    )
                    focus_file = str(target.relative_to(get_root()))

                if mode == "chat":
                    chat_max_new = config.MAX_NEW
                    prompt, _ = prompt_buffer.build_prompt(
                        "chat",
                        text,
                        focus_file=focus_file,
                        # Chat should use reduced context path to keep latency low.
                        # The prompt builder still includes file-aware context.
                        full_context=False,
                    )
                    output_chunks = []
                    start = time.time()
                    chat_buffered = (
                        config.BUFFER_OUTPUT
                        and (
                            bool(file_path)
                            or len(prompt) >= config.CHAT_BUFFER_PROMPT_CHARS
                            or len(text) >= config.CHAT_BUFFER_INPUT_CHARS
                        )
                    )
                    chat_stop = ["\nUser:", "\nSYSTEM:", "\nCONTEXT:", "\nHISTORY:"]
                    if chat_buffered:
                        last_status = start
                        for token in stream_reply_chunks(
                            prompt,
                            max_new=chat_max_new,
                            stop_sequences=chat_stop,
                        ):
                            output_chunks.append(token)
                            now = time.time()
                            if now - last_status >= 0.5:
                                self._sse_send("working", event="status")
                                last_status = now
                    else:
                        for token in stream_reply_chunks(
                            prompt,
                            max_new=chat_max_new,
                            stop_sequences=chat_stop,
                        ):
                            output_chunks.append(token)
                            self._sse_send(token, event="chunk")
                    output = "".join(output_chunks).strip()
                    agent.chat_history.append((text, output))
                    state.append_chat_turn(text, output)
                    meta = {
                        "backend": backend_name(),
                        "prompt_len": len(prompt),
                        "reply_len": len(output),
                        "timeout": output == "[Model timeout]",
                        "duration_ms": int((time.time() - start) * 1000),
                        "buffer_mode": "adaptive_buffered" if chat_buffered else "direct",
                        "truncated": False,
                        "chat_max_new": chat_max_new,
                    }
                    if chat_buffered:
                        self._sse_send(output, event="chunk")
                else:
                    output, meta = run_agent_meta(
                        text,
                        focus_file=focus_file,
                        silent=True,
                        full_context=bool(file_path),
                    )
                    meta["buffer_mode"] = "direct"
                    self._attach_path_debug(meta, focus_file)
                    self._sse_send(output, event="chunk")

                if focus_file:
                    meta["focus_file"] = focus_file
                self._sse_send(json.dumps(meta), event="meta")
            except Exception as exc:
                self._sse_send(str(exc), event="error")
            finally:
                if not sent_done:
                    self._sse_send("[DONE]", event="done")
                    sent_done = True
            return

        # /file (write/upload)
        if parsed.path == "/file":
            rel_path = data.get("path") or ""
            content = data.get("content")
            if not rel_path:
                self._json(400, {"error": "path is required"})
                return
            if content is None:
                self._json(400, {"error": "content is required"})
                return
            if is_binary_file(rel_path):
                self._json(
                    415,
                    {"error": "binary file writes are not supported"},
                )
                return
            try:
                target = resolve_path(rel_path)
                if is_security_concern(target):
                    self._json(
                        403,
                        {"error": "Security concern: cannot write to system paths"},
                    )
                    return

                # Generate diff if file exists
                diff = None
                if target.exists():
                    old_content = read_file_text(rel_path)
                    if old_content != content:
                        diff = generate_diff(old_content, content, str(target))

                dbg(f"POST /file write path={rel_path} bytes={len(content)}")
                write_file_text(rel_path, content)
                response = {"status": "ok", "path": rel_path}
                if diff:
                    response["diff"] = diff
                self._json(200, response)
            except Exception as exc:  # pragma: no cover
                self._json(500, {"error": str(exc)})
            return

        if parsed.path == "/run":
            cmd = (data.get("cmd") or "").strip()
            cwd_rel = (data.get("cwd") or "").strip()
            timeout_s = data.get("timeout", 20)
            if not cmd:
                self._json(400, {"error": "cmd is required"})
                return
            try:
                timeout_val = int(timeout_s)
            except Exception:
                timeout_val = 20
            timeout_val = max(1, min(timeout_val, 60))
            try:
                cwd_path = get_root()
                if cwd_rel:
                    cwd_target = resolve_path(cwd_rel)
                    if not cwd_target.exists():
                        self._json(400, {"error": f"cwd not found: {cwd_rel}"})
                        return
                    if not cwd_target.is_dir():
                        self._json(400, {"error": f"cwd is not a directory: {cwd_rel}"})
                        return
                    cwd_path = cwd_target
                dbg(f"POST /run cmd={cmd!r} cwd={str(cwd_path)} timeout={timeout_val}s")
                res = subprocess.run(
                    cmd,
                    cwd=str(cwd_path),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_val,
                )
                out = res.stdout or ""
                err = res.stderr or ""
                # Keep payloads bounded for UI.
                max_chars = 20000
                if len(out) > max_chars:
                    out = out[:max_chars] + "\n...[stdout truncated]"
                if len(err) > max_chars:
                    err = err[:max_chars] + "\n...[stderr truncated]"
                self._json(
                    200,
                    {
                        "status": "ok",
                        "cmd": cmd,
                        "cwd": str(cwd_path),
                        "code": res.returncode,
                        "stdout": out,
                        "stderr": err,
                    },
                )
            except subprocess.TimeoutExpired as exc:
                out = exc.stdout or ""
                err = exc.stderr or ""
                self._json(
                    200,
                    {
                        "status": "timeout",
                        "cmd": cmd,
                        "cwd": str(cwd_path),
                        "code": 124,
                        "stdout": out,
                        "stderr": (err + "\nCommand timed out.").strip(),
                    },
                )
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return
        if parsed.path == "/include":
            paths = data.get("paths")
            if paths is None:
                self._json(400, {"error": "paths required"})
                return
            try:
                set_include(paths)
                self._json(200, {"include": get_include()})
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        if parsed.path == "/history/clear":
            try:
                state.clear_chat_session()
                # Clear in-memory chat history used by some code paths.
                try:
                    agent.chat_history.clear()
                except Exception:
                    pass
                try:
                    agent.agent_history.clear()
                except Exception:
                    pass
                try:
                    agent.reset_structural_kv_cache(reason="new_chat")
                except Exception:
                    pass
                self._json(200, {"status": "ok"})
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return
        if parsed.path == "/root":
            new_root = data.get("path")
            if not new_root:
                self._json(400, {"error": "path is required"})
                return
            try:
                root = set_root(new_root)
                # Ensure files are re-listed internally if needed
                list_repo_files()  # Warm up cache if there is one

                self._json(200, {"root": str(root)})
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return

        self._json(404, {"error": "unknown endpoint"})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            body = {"status": "ok", "backend": backend_name()}
            try:
                extra = backend_status()
                if isinstance(extra, dict):
                    body.update(extra)
            except Exception:
                pass
            self._json(200, body)
            return
        if parsed.path == "/files":
            try:
                qs = parse_qs(parsed.query)
                requested_root = qs.get("root", [""])[0]
                current_root = str(get_root())
                if requested_root and requested_root != current_root:
                    self._json(
                        409,
                        {"error": "root mismatch", "root": current_root},
                    )
                    return
                files = list_repo_files()
                self._json(200, {"files": files})
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return
        if parsed.path == "/file":
            qs = parse_qs(parsed.query)
            rel_path = qs.get("path", [""])[0]
            if not rel_path:
                self._json(400, {"error": "path query param required"})
                return
            try:
                if is_binary_file(rel_path):
                    rel_norm = _norm_rel_path(rel_path)
                    include = get_include()
                    if include and rel_norm not in include:
                        self._json(403, {"error": "file not in selected set"})
                        return
                    target = resolve_path(rel_path)
                    if not target.exists():
                        self._json(404, {"error": "not found"})
                        return
                    data = target.read_bytes()
                    b64 = base64.b64encode(data).decode("ascii")
                    mime, _ = mimetypes.guess_type(str(target))
                    self._json(
                        200,
                        {
                            "path": rel_path,
                            "content": b64,
                            "binary": True,
                            "encoding": "base64",
                            "mime": mime or "application/octet-stream",
                        },
                    )
                else:
                    content = read_file_text(rel_path)
                    self._json(200, {"path": rel_path, "content": content})
            except FileNotFoundError:
                self._json(404, {"error": "not found"})
            except Exception as exc:  # pragma: no cover
                self._json(500, {"error": str(exc)})
            return
        if parsed.path == "/root":
            self._json(200, {"root": str(get_root())})
            return
        if parsed.path == "/include":
            self._json(200, {"include": get_include()})
            return
        if parsed.path == "/history":
            qs = parse_qs(parsed.query)
            default_limit = int(getattr(state, "MAX_CHAT_HISTORY", 7))
            try:
                limit = int((qs.get("limit", [str(default_limit)])[0] or str(default_limit)))
            except Exception:
                limit = default_limit
            limit = max(1, min(limit, default_limit))
            pairs = state.get_recent_chat(limit)
            history = [{"user": u or "", "assistant": a or ""} for u, a in pairs]
            self._json(200, {"history": history})
            return

        self._json(404, {"error": "unknown endpoint"})


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", config.SERVER_PORT), APIServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"HTTP server listening on http://127.0.0.1:{config.SERVER_PORT}")
