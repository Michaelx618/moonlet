"""Agent mode: chat-style prompt + code extraction and apply."""

import time
from typing import Dict, List, Optional

from . import config
from .files import get_root
from .model import stream_reply
from .output_parser import parse_flexible_output, strip_code_blocks_for_display
from .prompt_buffer import build_prompt, build_freedom_context
from .relevance import find_relevant_files
from . import state
from .utils import dbg, dbg_dump


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

    # 2. Build prompt with multi-file context (tree-sitter + grep for snippets)
    context_override = build_freedom_context(user_text or "", candidate_paths) if candidate_paths else ""
    prompt, _ = build_prompt(
        "chat_for_agent",
        user_text,
        focus_file=None,
        full_context=True,
        context_override=context_override,
    )

    # 2. Stream reply (same params as chat)
    reply = stream_reply(
        prompt,
        label="> " if not silent else "",
        silent=silent,
        max_new=config.MAX_NEW,
        stop_sequences=["\nUser:", "\nSYSTEM:", "\nCONTEXT:", "\nHISTORY:"],
    )
    dbg_dump("freedom_edit_reply", reply or "")

    # 3. Parse reply for code blocks / diff (candidate_paths from step 1)
    kind, diff_data, blocks = parse_flexible_output(reply or "", candidate_paths=candidate_paths)

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

    if kind == "blocks" and blocks:
        from .files import apply_blocks_with_report

        # When STAGE_EDITS: don't write to disk; stage for Accept/Reject
        touched, skipped, per_file_diffs, per_file_staged = apply_blocks_with_report(
            blocks, show_diff=True, dry_run=config.STAGE_EDITS
        )
    elif kind == "diff" and diff_data:
        from .files import apply_unified_diff, generate_diff, resolve_path

        path, hunks = diff_data
        try:
            target = resolve_path(path)
            old_content = target.read_text() if target.exists() else ""
            if config.STAGE_EDITS:
                new_content = apply_unified_diff(path, hunks, dry_run=True)
            else:
                apply_unified_diff(path, hunks, dry_run=False)
                new_content = target.read_text()
            touched.append(path)
            try:
                per_file_diffs[path] = generate_diff(old_content, new_content, str(target))
            except Exception:
                pass
            per_file_staged[path] = new_content
        except Exception as e:
            dbg(f"freedom_edit: diff apply failed: {e}")
            return {
                "output": strip_code_blocks_for_display(reply or "") or f"Diff apply failed: {e}",
                "meta": {
                    "mode_used": "freedom",
                    "touched": [],
                    "skipped": [],
                    "error": str(e),
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
        meta["staged"] = bool(per_file_staged)
        if per_file_staged:
            meta["staged_file"] = list(per_file_staged.keys())[0]
            meta["staged_content"] = per_file_staged.get(meta["staged_file"], "")

    # 5. Display only explanation; code was extracted and applied/staged
    display_output = strip_code_blocks_for_display(reply or "")
    action = "Staged for" if config.STAGE_EDITS else "Applied to"
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
