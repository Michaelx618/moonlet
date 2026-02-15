"""Agent mode: chat-style prompt + code extraction and apply."""

import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .files import get_root
from .model import stream_reply
from .output_parser import parse_flexible_output, strip_code_blocks_for_display
from .prompt_buffer import build_prompt
from .relevance import find_relevant_files
from . import state
from .tool_executor import (
    TOOLS_SYSTEM_HINT,
    execute_tool,
    extract_tool_calls,
    strip_tool_calls_from_output,
    tool_log,
)
from .utils import dbg, dbg_dump


def _resolve_candidate_paths(paths: List[str], root: Path) -> List[str]:
    """Resolve each path to its actual full path in the repo (e.g. checkpasswd.c -> w7starter/checkpasswd.c)."""
    if not paths:
        return []
    resolved: List[str] = []
    seen: set = set()
    for p in paths:
        norm = p.replace("\\", "/").strip()
        if not norm:
            continue
        if "/" in norm:
            target = root / norm
            if target.exists() and target.is_file():
                rel = str(target.relative_to(root)).replace("\\", "/")
                if rel not in seen:
                    resolved.append(rel)
                    seen.add(rel)
                continue
        # Basename or path not found: rglob to find actual path in repo
        base = norm.split("/")[-1]
        for found in root.rglob(base):
            try:
                rel = str(found.relative_to(root)).replace("\\", "/")
                if rel not in seen and found.is_file():
                    resolved.append(rel)
                    seen.add(rel)
                    break
            except ValueError:
                continue
    return resolved or paths


def run_freedom_edit(
    user_text: str,
    focus_file: Optional[str] = None,
    silent: bool = False,
) -> Dict:
    """Run agent mode: same prompt as chat, then extract code and apply to files.
    Model decides which file(s) to edit; open file is ignored for context."""
    start = time.time()
    root = get_root()

    # 1. Discover relevant files from request (no open-file bias)
    # Do NOT fall back to focus_file (open file) — use only what relevance finds.
    # Otherwise lang-only blocks get assigned to the open file for all blocks.
    candidate_paths = find_relevant_files(user_text or "", open_file=None)

    # When relevance is disabled, candidate_paths is empty. Use imported files (index) as candidates
    # so the parser can assign ```c blocks to the right file, and FILES: shows what the user imported.
    if not candidate_paths:
        from .files import get_include
        from .index import get_indexed_files
        if get_include():
            candidate_paths = get_indexed_files()

    # 2. Build prompt: no file content — model uses tools (read, grep, symbols) to fetch what it needs
    # Resolve paths so model sees actual repo paths (e.g. w7starter/checkpasswd.c) not basenames
    resolved_paths = _resolve_candidate_paths(candidate_paths, root)
    context_override = ""
    if resolved_paths:
        context_override = "FILES: " + ", ".join(resolved_paths)
    else:
        # No files selected: tell model to discover files via tools before editing
        context_override = (
            "No files selected. Use [[[list_files]]] to see available files, "
            "or [[[grep:pattern]]] to find the file mentioned in the user's request. "
            "Then [[[read:path]]] to read it before producing your diff."
        )
    prompt, _ = build_prompt(
        "chat_for_agent",
        user_text,
        focus_file=None,
        full_context=True,
        context_override=context_override,
        system_append=TOOLS_SYSTEM_HINT,
    )

    # 3. Tool loop: let agent call grep/symbols/read/list_files, then produce final output
    reply = ""
    max_rounds = max(1, int(getattr(config, "MAX_TOOL_ROUNDS", 3)))
    for _round in range(max_rounds):
        reply = stream_reply(
            prompt,
            label="> " if not silent else "",
            silent=silent,
            max_new=config.MAX_NEW,
            stop_sequences=["\nUser:", "\nSYSTEM:", "\nCONTEXT:", "\nHISTORY:"],
        )
        dbg_dump("freedom_edit_reply", reply or "")

        tool_calls = extract_tool_calls(reply or "")
        if not tool_calls:
            break

        tool_log(f"round {_round + 1}: model requested {len(tool_calls)} tool(s)")
        dbg(f"agent: tool round {_round + 1}, {len(tool_calls)} calls")
        results = []
        for name, arg in tool_calls:
            out = execute_tool(name, arg)
            results.append(out)
            dbg(f"agent: {name}({arg[:40]}...) -> {len(out)} chars")

        combined = "\n\n".join(results)
        max_chars = getattr(config, "MAX_TOOL_RESULT_CHARS", 0) or 0
        if max_chars > 0 and len(combined) > max_chars:
            combined = combined[:max_chars] + "\n\n...[truncated]"
            tool_log(f"round {_round + 1} done: truncated to {max_chars} chars")
        else:
            tool_log(f"round {_round + 1} done: feeding {len(combined)} chars back to model")
        prompt = (
            f"{prompt}\n{reply}\n\nTool results:\n{combined}\n\n"
            "User: Continue. Use the tool results above. Produce your final answer (code blocks or diff if editing).\n"
            "Assistant:"
        )

    # 3. Parse reply for code blocks / diff (use resolved paths for path resolution)
    kind, diff_data, blocks = parse_flexible_output(reply or "", candidate_paths=resolved_paths or candidate_paths)

    # Resolve paths and fix common model errors (e.g. /n -> \n in C strings)
    # Use agent-indicated path; resolve basenames via rglob so w6/parentcreates.c matches.
    if blocks and kind == "blocks":
        resolved_blocks = []
        for path, content in blocks:
            # Fix /n -> \n (model often writes literal /n instead of escape)
            if "/n" in content and ("#include" in content or "printf" in content or "fprintf" in content):
                content = content.replace('/n"', '\\n"').replace("/n'", "\\n'")
            base = path.split("/")[-1]
            if "/" not in path or path == base:
                # Prefer rglob to find actual file (e.g. w6/parentcreates.c); else candidates
                found = None
                for p in root.rglob(base):
                    try:
                        rel = str(p.relative_to(root))
                        if rel.endswith(base) and p.is_file():
                            found = rel
                            break
                    except ValueError:
                        continue
                if found:
                    path = found
                else:
                    for c in candidate_paths:
                        if c.endswith("/" + base) or c == base:
                            path = c
                            break
            resolved_blocks.append((path, content))
        blocks = resolved_blocks

    touched: List[str] = []
    skipped: List[Dict] = []
    per_file_diffs: Dict[str, str] = {}
    per_file_staged: Dict[str, str] = {}
    per_file_before: Dict[str, str] = {}

    if kind == "blocks" and blocks:
        from .files import apply_blocks_with_report

        # Always stage; user Accept writes
        touched, skipped, per_file_diffs, per_file_staged, per_file_before = apply_blocks_with_report(
            blocks, show_diff=True, dry_run=True
        )
    elif kind == "diff" and diff_data:
        from .files import apply_unified_diff, generate_diff, is_edit_allowed, resolve_path

        path, hunks = diff_data[0], diff_data[1]
        is_delete_diff = diff_data[2] if len(diff_data) > 2 else False

        def _try_apply(apply_path: str):
            target = resolve_path(apply_path)
            old_content = target.read_text() if target.exists() else ""
            # Always dry_run: stage first, user Accept writes. Compare before/after for green.
            new_content = apply_unified_diff(apply_path, hunks, dry_run=True)
            return target, old_content, new_content, apply_path

        # If model used wrong path (e.g. w7/ from assignment) but file exists elsewhere, use it
        apply_path = path
        target = resolve_path(path)
        if not target.exists():
            base = path.split("/")[-1]
            for rp in (resolved_paths or candidate_paths or []):
                cand = root / rp.replace("\\", "/")
                if (rp.endswith("/" + base) or rp == base) and cand.exists() and cand.is_file() and is_edit_allowed(rp):
                    apply_path = rp
                    dbg(f"freedom_edit: diff path {path} not found, using existing {apply_path}")
                    break
            if apply_path == path:
                for p in root.rglob(base):
                    try:
                        rel = str(p.relative_to(root)).replace("\\", "/")
                        if rel.endswith(base) and p.is_file() and is_edit_allowed(rel):
                            apply_path = rel
                            dbg(f"freedom_edit: diff path {path} not found, using existing {apply_path}")
                            break
                    except ValueError:
                        continue

        # Reject diffs targeting app source or paths outside include filter
        target = resolve_path(apply_path)
        allow_new = not (target.exists() and target.is_file())
        if not is_edit_allowed(apply_path, allow_new=allow_new):
            raise PermissionError(
                f"Cannot edit {apply_path}: outside allowed set (imported files only; app source is protected)"
            )

        # New-file diff (--- /dev/null) applied to existing file: replace content, don't insert
        is_new_file_diff = all(
            getattr(h, "old_start", 1) == 0 and getattr(h, "old_count", 1) == 0
            for h in hunks
        )
        target_exists = target.exists() and target.is_file()

        try:
            if is_delete_diff and target_exists:
                # Delete diff (+++ /dev/null): stage deletion
                old_content = target.read_text()
                touched.append(apply_path)
                try:
                    per_file_diffs[apply_path] = generate_diff(old_content, "", str(target))
                except Exception:
                    pass
                per_file_staged[apply_path] = None  # Sentinel: delete on Accept
                per_file_before[apply_path] = old_content
            elif is_new_file_diff and target_exists:
                # Extract new content from + lines and replace file
                new_lines = []
                for h in hunks:
                    for prefix, line in getattr(h, "lines", []):
                        if prefix == "+":
                            new_lines.append(line)
                new_content = "\n".join(new_lines) + ("\n" if new_lines else "")
                old_content = target.read_text()
                # Always stage; user Accept writes
                touched.append(apply_path)
                try:
                    per_file_diffs[apply_path] = generate_diff(old_content, new_content, str(target))
                except Exception:
                    pass
                per_file_staged[apply_path] = new_content
                per_file_before[apply_path] = old_content
            else:
                target, old_content, new_content, applied_path = _try_apply(apply_path)
                touched.append(applied_path)
                try:
                    per_file_diffs[applied_path] = generate_diff(old_content, new_content, str(target))
                except Exception:
                    pass
                per_file_staged[applied_path] = new_content
                per_file_before[applied_path] = old_content
        except FileNotFoundError:
            # Model may say w7/checkpasswd.c but file is at checkpasswd.c — resolve via rglob/candidates
            base = path.split("/")[-1]
            resolved_path = None
            for p in root.rglob(base):
                try:
                    rel = str(p.relative_to(root)).replace("\\", "/")
                    if rel.endswith(base) and p.is_file() and is_edit_allowed(rel):
                        resolved_path = rel
                        break
                except ValueError:
                    continue
            if not resolved_path and (resolved_paths or candidate_paths):
                for c in (resolved_paths or candidate_paths):
                    if (c.endswith("/" + base) or c == base) and is_edit_allowed(c):
                        resolved_path = c
                        break
            if resolved_path:
                try:
                    target, old_content, new_content, applied_path = _try_apply(resolved_path)
                    touched.append(applied_path)
                    try:
                        per_file_diffs[applied_path] = generate_diff(old_content, new_content, str(target))
                    except Exception:
                        pass
                    per_file_staged[applied_path] = new_content
                    per_file_before[applied_path] = old_content
                except Exception as e:
                    dbg(f"freedom_edit: diff apply failed (retry): {e}")
                    raise
            else:
                raise
        except Exception as e:
            dbg(f"freedom_edit: diff apply failed: {e}")
            err_msg = str(e)
            return {
                "output": strip_code_blocks_for_display(reply or "") or f"Diff apply failed: {err_msg}",
                "meta": {
                    "mode_used": "freedom",
                    "touched": [],
                    "skipped": [],
                    "error": err_msg,
                    "failure_kind": "apply",
                    "failure_reason": err_msg,
                    "elapsed_s": round(time.time() - start, 2),
                },
                "status_prefix": "",
                "blocks_count": 0,
                "retried": False,
                "timeout": False,
            }

    elapsed = time.time() - start
    meta = {
        "mode_used": "freedom",
        "touched": touched,
        "skipped": skipped,
        "elapsed_s": round(elapsed, 2),
    }
    # Stage for UI: green diff + Accept/Reject buttons
    if touched:
        meta["files_changed"] = touched
        meta["per_file_diffs"] = per_file_diffs
        meta["per_file_staged"] = per_file_staged
        meta["per_file_before"] = per_file_before
        meta["staged"] = bool(per_file_staged)
        if per_file_staged:
            meta["staged_file"] = list(per_file_staged.keys())[0]
            meta["staged_content"] = per_file_staged.get(meta["staged_file"], "")
        # Single-file: also set top-level diff/focus_file so frontend can show Accept/Reject
        if len(touched) == 1 and per_file_diffs:
            meta["diff"] = per_file_diffs.get(touched[0], "")
            meta["focus_file"] = meta.get("focus_file") or touched[0]

    # 5. Display only explanation; code was extracted and applied/staged
    #    Strip tool calls from reply for clean display
    reply_for_display = strip_tool_calls_from_output(reply or "")
    display_output = strip_code_blocks_for_display(reply_for_display)
    action = "Staged for"  # Always stage; user Accept writes
    if touched and not display_output.strip():
        display_output = f"{action}: {', '.join(touched)}"
    elif touched and display_output.strip():
        display_output = f"{display_output.strip()}\n\n{action}: {', '.join(touched)}"

    # 6. Append to history (explanation only, for consistent panel display)
    state.append_chat_turn(user_text or "", display_output)

    return {
        "output": display_output,
        "meta": meta,
        "status_prefix": "",
        "blocks_count": len(touched),
        "retried": False,
        "timeout": False,
    }
