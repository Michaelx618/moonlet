import base64
import copy
import json
import sys
import mimetypes
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from . import config
from . import state
from .relevance import find_relevant_files
from .verify import run_verify
from .agent_loop import run_agent
from .files import (
    delete_file,
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
from .model import backend_name, backend_status, clear_kv_cache
from .tool_executor import tool_log
from .utils import dbg

import difflib

try:
    from file_utils import (
        generate_diff,
        is_security_concern,
        validate_file_path,
    )
except ImportError:
    def generate_diff(
        old_content: str,
        new_content: str,
        filepath: str,
        context_lines: int = 3,
    ) -> str:
        old_lines = (old_content or "").splitlines(keepends=True)
        new_lines = (new_content or "").splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=str(filepath),
            tofile=str(filepath),
            lineterm="",
            n=context_lines,
        )
        return "".join(diff)

    def is_security_concern(*args, **kwargs):
        return False

    def validate_file_path(*args, **kwargs):
        raise NotImplementedError("file_utils not available")


_PATCH_PROPOSALS: Dict[str, Dict] = {}
_PATCH_PROPOSAL_ORDER: list[str] = []
_LAST_APPLIED: Dict[str, Dict] = {}


def _store_patch_proposal(meta: Dict) -> str:
    pid = str(uuid.uuid4())[:12]
    payload = {
        "id": pid,
        "ts": int(time.time()),
        "per_file_staged": dict(meta.get("per_file_staged") or {}),
        "per_file_before": dict(meta.get("per_file_before") or {}),
        "per_file_diffs": dict(meta.get("per_file_diffs") or {}),
        "touched": list(meta.get("touched") or []),
    }
    _PATCH_PROPOSALS[pid] = payload
    _PATCH_PROPOSAL_ORDER.append(pid)
    if len(_PATCH_PROPOSAL_ORDER) > 30:
        old = _PATCH_PROPOSAL_ORDER.pop(0)
        _PATCH_PROPOSALS.pop(old, None)
    return pid


def _apply_patch_proposal(pid: str) -> Dict:
    entry = dict(_PATCH_PROPOSALS.get(pid) or {})
    if not entry:
        raise KeyError("proposal_not_found")
    staged = dict(entry.get("per_file_staged") or {})
    before = dict(entry.get("per_file_before") or {})
    touched = list(entry.get("touched") or [])
    applied = []
    for fpath in touched:
        content = staged.get(fpath)
        if content is None:
            continue
        write_file_text(fpath, content)
        applied.append(fpath)
    _LAST_APPLIED["last"] = {
        "id": pid,
        "per_file_before": before,
        "per_file_after": staged,
        "touched": applied,
        "ts": int(time.time()),
    }
    return {"proposal_id": pid, "applied": applied}


def _undo_last_applied() -> Dict:
    last = dict(_LAST_APPLIED.get("last") or {})
    if not last:
        raise KeyError("no_applied_patch")
    before = dict(last.get("per_file_before") or {})
    restored = []
    for fpath, content in before.items():
        write_file_text(fpath, content)
        restored.append(fpath)
    _LAST_APPLIED.clear()
    return {"restored": restored, "proposal_id": last.get("id")}


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
            mode = (data.get("mode") or "agent").strip().lower()
            text = data.get("text") or ""
            focus_file = data.get("focus_file")
            file_path = data.get("file_path")
            last_error = data.get("last_error") or ""
            previous_patches = data.get("previous_patches") or []
            iteration = int(data.get("iteration") or 0)
            if not text.strip() and mode != "repair":
                self._json(400, {"error": "text is required"})
                return
            if mode == "repair" and not text.strip():
                self._json(400, {"error": "text (original spec) is required for repair"})
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

                # Agent mode: when no file open, discover target from request text
                if mode == "agent" and not (focus_file or "").strip():
                    discovered = find_relevant_files(text, open_file=None)
                    if discovered:
                        focus_file = discovered[0]
                        dbg(f"agent: discovered focus_file from request: {focus_file}")

                if getattr(config, "DISABLE_FOCUS_FILE", False):
                    focus_file = None

                if mode in ("agent", "chat", "repair"):
                    run_text = text if mode != "repair" else f"{text}\n\nRepair target:\n{last_error}"
                    extra_read = [
                        str(p).strip() for p in (data.get("extra_read_files") or [])
                        if str(p).strip()
                    ]
                    context_folders = [
                        str(p).strip() for p in (data.get("context_folders") or data.get("extra_folders") or [])
                        if str(p).strip()
                    ] or None
                    def on_action(action: Dict) -> None:
                        self._sse_send(json.dumps(action), event="action")
                        if action.get("type") == "tool_call":
                            from .utils import tool_call_log
                            tool_call_log(action.get("tool", ""), action.get("args") or {})

                    output, meta = run_agent(
                        run_text,
                        focus_file=focus_file,
                        mode=mode,
                        silent=True,
                        on_action=on_action,
                        extra_read_files=extra_read or None,
                        context_folders=context_folders,
                    )
                    meta["buffer_mode"] = "direct"
                    meta["output"] = output
                    meta["explanation"] = output
                    self._attach_path_debug(meta, focus_file)
                    self._sse_send(output, event="chunk")
                    if mode != "repair":
                        state.append_chat_turn(text, output)
                    elif meta.get("touched"):
                        state.append_chat_turn(text, output)
                        state.append_change_note(f"Repair staged for: {', '.join(meta['touched'])}")

                if focus_file:
                    meta["focus_file"] = focus_file
                approval_mode = bool(getattr(config, "APPROVAL_MODE", True))
                auto_apply = bool(getattr(config, "AUTO_APPLY_ON_SUCCESS", False))
                if approval_mode and meta.get("per_file_staged") and not auto_apply:
                    proposal_id = _store_patch_proposal(meta)
                    meta["proposal_id"] = proposal_id
                    meta["requires_approval"] = True
                    meta["applied_directly"] = False
                    self._sse_send(
                        f"Patch proposal ready ({proposal_id}). Approve to apply changes.",
                        event="chunk",
                    )
                self._sse_send(json.dumps(meta), event="meta")

                # Write to disk AFTER sending meta so frontend gets diff highlights first.
                # Auto-apply runs when approval mode is disabled or explicit auto-apply is enabled.
                if (not approval_mode or auto_apply) and meta.get("per_file_staged"):
                    touched = meta.get("touched") or meta.get("files_changed") or []
                    for fpath, content in meta["per_file_staged"].items():
                        if content is not None:
                            write_file_text(fpath, content)
                            dbg(f"server: wrote {fpath} to disk (applied_directly)")

                    # Verify + retry loop: compile/test, feed errors back to model
                    skip_checks = getattr(config, "SKIP_COMPILE_CHECKS", False)
                    max_retries = getattr(config, "MAX_VERIFY_RETRIES", 2)
                    if touched and not skip_checks:
                        for retry_i in range(max_retries):
                            self._sse_send("Verifying...", event="chunk")
                            exit_code, _stdout, _stderr, first_error = run_verify(
                                staged_paths=list(touched),
                            )
                            if exit_code == 0:
                                dbg(f"server: verify passed (attempt {retry_i + 1})")
                                self._sse_send("Build OK.", event="chunk")
                                break
                            dbg(f"server: verify failed (attempt {retry_i + 1}), error: {first_error[:200]}")
                            self._sse_send(
                                f"Build failed (retry {retry_i + 1}/{max_retries}):\n{first_error[:300]}",
                                event="chunk",
                            )
                            # Feed error back to main agent (same entrypoint as normal requests)
                            retry_text = (
                                f"ERROR after your edit:\n{first_error}\n\n"
                                f"Fix the error in the code. Original request:\n{text}"
                            )
                            _, retry_meta = run_agent(
                                retry_text,
                                focus_file=focus_file or (touched[0] if touched else None),
                                mode="repair",
                                silent=True,
                                extra_read_files=list(touched),
                            )
                            retry_staged = retry_meta.get("per_file_staged") or {}
                            retry_touched = retry_meta.get("touched") or []
                            if not retry_touched:
                                dbg("server: verify retry produced no edits, stopping")
                                break
                            # Write retry edits to disk
                            for fpath, content in retry_staged.items():
                                if content is not None:
                                    write_file_text(fpath, content)
                                    dbg(f"server: wrote {fpath} to disk (retry {retry_i + 1})")
                            # Update meta with retry results for frontend
                            meta["per_file_diffs"] = retry_meta.get("per_file_diffs", meta.get("per_file_diffs", {}))
                            meta["per_file_staged"] = retry_staged
                            meta["per_file_before"] = retry_meta.get("per_file_before", meta.get("per_file_before", {}))
                            touched = retry_touched
                            # Send updated meta so frontend refreshes diff highlights
                            self._sse_send(json.dumps(meta), event="meta")
                        else:
                            # All retries exhausted, still failing
                            dbg(f"server: verify failed after {max_retries} retries")
                            self._sse_send(
                                f"Build still failing after {max_retries} retries.",
                                event="chunk",
                            )
            except Exception as exc:
                self._sse_send(str(exc), event="error")
            finally:
                if not sent_done:
                    self._sse_send("[DONE]", event="done")
                    sent_done = True
            return

        if parsed.path == "/v2/execute":
            text = (data.get("text") or "").strip()
            focus_file = (data.get("focus_file") or "").strip()
            if not text:
                self._json(400, {"error": "text is required"})
                return
            try:
                if getattr(config, "USE_LEGACY_PIPELINE", False):
                    output, meta = run_pipeline(text, silent=True, focus_file=focus_file or None)
                    meta["output"] = output
                    self._json(200, meta)
                else:
                    extra_read = [
                        str(p).strip() for p in (data.get("extra_read_files") or [])
                        if str(p).strip()
                    ]
                    context_folders = [
                        str(p).strip() for p in (data.get("context_folders") or data.get("extra_folders") or [])
                        if str(p).strip()
                    ] or None
                    output, meta = run_agent(
                        text,
                        focus_file=focus_file or None,
                        mode="agent",
                        silent=True,
                        extra_read_files=extra_read or None,
                        context_folders=context_folders,
                    )
                    meta["output"] = output
                    self._json(200, meta)
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return

        if parsed.path == "/proposal/accept":
            proposal_id = str(data.get("proposal_id") or "").strip()
            if not proposal_id:
                self._json(400, {"error": "proposal_id is required"})
                return
            try:
                out = _apply_patch_proposal(proposal_id)
                self._json(200, {"ok": True, **out})
            except KeyError:
                self._json(404, {"error": "proposal_not_found", "proposal_id": proposal_id})
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return

        if parsed.path == "/proposal/reject":
            proposal_id = str(data.get("proposal_id") or "").strip()
            if not proposal_id:
                self._json(400, {"error": "proposal_id is required"})
                return
            existed = bool(_PATCH_PROPOSALS.pop(proposal_id, None))
            self._json(200, {"ok": True, "proposal_id": proposal_id, "rejected": existed})
            return

        if parsed.path == "/proposal/undo":
            try:
                out = _undo_last_applied()
                self._json(200, {"ok": True, **out})
            except KeyError:
                self._json(404, {"error": "no_applied_patch"})
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return

        # /file (write/upload or delete)
        if parsed.path == "/file":
            rel_path = data.get("path") or ""
            content = data.get("content")
            if not rel_path:
                self._json(400, {"error": "path is required"})
                return
            # content is None or null: delete file
            if content is None:
                try:
                    target = resolve_path(rel_path)
                    if is_security_concern(target):
                        self._json(
                            403,
                            {"error": "Security concern: cannot delete system paths"},
                        )
                        return
                    delete_file(rel_path)
                    self._json(200, {"status": "deleted", "path": rel_path})
                except FileNotFoundError:
                    self._json(404, {"error": "not found"})
                except PermissionError as exc:
                    self._json(403, {"error": str(exc)})
                except Exception as exc:  # pragma: no cover
                    self._json(500, {"error": str(exc)})
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
                tool_log(f"run_terminal_cmd (sandbox /run): cmd={cmd!r} cwd={str(cwd_path)} timeout={timeout_val}s")
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
                tool_log(f"run_terminal_cmd (sandbox /run): exit={res.returncode} stdout={len(out)} chars stderr={len(err)} chars")
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
                tool_log(f"run_terminal_cmd (sandbox /run): TIMEOUT after {timeout_val}s cmd={cmd!r}")
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

        if parsed.path == "/verify":
            try:
                paths = data.get("paths") or []
                exit_code, stdout, stderr, first_error = run_verify(
                    get_root(),
                    staged_paths=paths if isinstance(paths, list) else [],
                )
                max_chars = 20000
                if len(stdout) > max_chars:
                    stdout = stdout[:max_chars] + "\n...[stdout truncated]"
                if len(stderr) > max_chars:
                    stderr = stderr[:max_chars] + "\n...[stderr truncated]"
                self._json(
                    200,
                    {
                        "ok": exit_code == 0,
                        "exit_code": exit_code,
                        "stdout": stdout,
                        "stderr": stderr,
                        "first_error": first_error,
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
            paths = paths if isinstance(paths, list) else []
            try:
                set_include(paths)
                self._json(200, {"include": get_include()})
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f"[/include error] {exc}\n{tb}", file=sys.stderr)
                try:
                    with open("moonlet_import_error.txt", "w") as f:
                        f.write(f"{exc}\n\n{tb}")
                except Exception:
                    pass
                err_msg = f"{exc}\n\n{tb}"
                self._json(400, {"error": err_msg})
            return
        if parsed.path == "/history/clear":
            try:
                state.clear_chat_session()
                self._json(200, {"status": "ok"})
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return
        if parsed.path == "/root":
            new_root = (data.get("path") or "").strip()
            if not new_root:
                self._json(400, {"error": "path is required"})
                return
            try:
                root = set_root(new_root)
                # Ensure files are re-listed internally if needed
                list_repo_files()  # Warm up cache if there is one

                self._json(200, {"root": str(root)})
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f"[/root error] {exc}\n{tb}", file=sys.stderr)
                try:
                    with open("moonlet_import_error.txt", "w") as f:
                        f.write(f"{exc}\n\n{tb}")
                except Exception:
                    pass
                err_msg = f"{exc}\n\n{tb}"
                self._json(400, {"error": err_msg})
            return

        self._json(404, {"error": "unknown endpoint"})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/clear-cache":
            try:
                from . import model as _model_mod
                model_obj = getattr(_model_mod, "model", None)
                clear_kv_cache(model_obj)
                self._json(200, {"status": "ok", "message": "KV cache cleared"})
            except Exception as e:
                self._json(500, {"error": str(e)})
            return

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
                import traceback
                tb = traceback.format_exc()
                print(f"[/files error] {exc}\n{tb}", file=sys.stderr)
                try:
                    with open("moonlet_import_error.txt", "w") as f:
                        f.write(f"{exc}\n\n{tb}")
                except Exception:
                    pass
                err_msg = f"{exc}\n\n{tb}"
                self._json(500, {"error": err_msg})
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
