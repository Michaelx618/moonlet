import os
import sys
import time

from . import config


def dbg(message: str):
    if not config.DEBUG:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[debug] [{ts} pid={os.getpid()}] {message}"
    print(line, file=sys.stderr)
    try:
        with open(config.DEBUG_LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def dbg_chat(message: str):
    """Log when SC2_DEBUG_CHAT=1. Use for chat prompt/context debugging."""
    if not getattr(config, "DEBUG_CHAT", False):
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[chat_debug] [{ts} pid={os.getpid()}] {message}"
    print(line, file=sys.stderr)
    try:
        with open(config.DEBUG_LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def dbg_dump(label: str, text: str):
    """Dump debug output. Truncated by default; full dump when SC2_DEBUG_VERBOSE=true."""
    if not config.DEBUG:
        return
    try:
        content = text or ""
        with open(config.DEBUG_LOG_PATH, "a") as f:
            if config.DEBUG_DUMP_VERBOSE:
                # Full dump
                f.write(f"\n[debug_dump] {label}\n")
                f.write(content + "\n")
            else:
                # Truncated: header + first N non-empty lines / max chars
                max_lines = config.DEBUG_DUMP_MAX_LINES
                max_chars = config.DEBUG_DUMP_MAX_CHARS
                lines = [ln for ln in content.splitlines() if ln.strip()]
                preview_lines = lines[:max_lines]
                preview = "\n".join(preview_lines)
                if len(preview) > max_chars:
                    preview = preview[:max_chars]
                truncated = len(lines) > max_lines or len(content) > max_chars
                f.write(
                    f"\n[debug_dump] {label} (len={len(content)})"
                    f"{' …(truncated)' if truncated else ''}\n"
                )
                f.write(preview + "\n")
    except Exception:
        pass
