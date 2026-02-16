"""Agent-callable tools: grep, symbols, read_file, list_files.

The coding agent can invoke these via a simple text protocol in its output:
  [[[grep:pattern]]]
  [[[grep:pattern:glob]]]
  [[[symbols:path]]]
  [[[read:path]]]
  [[[list_files]]]

Results are appended to the conversation so the agent can use them.
"""

import re
import sys
from typing import List, Optional, Tuple

from . import config
from .files import _path_in_include, get_include, get_root, read_single_file_for_context
from .index import get_indexed_files, get_symbols_for_file
from .tools import grep_search
from .utils import dbg


def _tool_log(msg: str) -> None:
    """Log tool activity to stderr when DEBUG_TOOLS is on."""
    if getattr(config, "DEBUG_TOOLS", True):
        print(f"[tool] {msg}", file=sys.stderr)


def tool_log(msg: str) -> None:
    """Public alias for agent to log tool-related activity."""
    _tool_log(msg)


TOOL_PATTERN = re.compile(
    r"\[\[\[(grep|symbols|read|list_files)(?::([^\]]*))?\]\]\]",
    re.IGNORECASE,
)

TOOLS_SYSTEM_HINT = """You have access to these tools. Output a tool call on its own line when you need it; the result will be provided.

CONTEXT includes the content of all FILES (see --- path --- blocks). Edit based on what you see — do NOT guess or hallucinate. If you need another file not in CONTEXT, use [[[read:path]]]. If no FILES are listed, use [[[list_files]]] or [[[grep:name]]] first, then [[[read:path]]].

Output code with a brief explanation. Indicate which file the code goes to.

- [[[grep:pattern]]] — Search for text across indexed files (ripgrep). Use literal pattern.
- [[[grep:pattern:*.ext]]] — Same, but limit to files matching glob (e.g. *.c, *.py).
- [[[symbols:path]]] — Get functions/classes in a file (tree-sitter). path = relative path like w6/main.c
- [[[read:path]]] — Read full content of a file.
- [[[list_files]]] — List all indexed files in the project."""

# Chat mode: answer questions, use tools to read/list — do NOT instruct to produce diffs
CHAT_TOOLS_HINT = """You have access to these tools. Output a tool call on its own line when you need it; the result will be provided.

CONTEXT already includes the content of all imported files (see --- path --- blocks). Use that to answer. If you need to search or re-read: [[[grep:pattern]]], [[[read:path]]], [[[list_files]]]. Do NOT ask the user to provide file content — it is in CONTEXT or use the read tool. Answer in plain text. Do NOT produce diffs or code edits unless the user explicitly asks you to change code.

- [[[grep:pattern]]] — Search for text across indexed files.
- [[[read:path]]] — Read full content of a file.
- [[[list_files]]] — List all indexed files in the project."""


def execute_tool(name: str, arg: str) -> str:
    """Execute a tool and return its result as a string."""
    name = (name or "").strip().lower()
    arg = (arg or "").strip()

    _tool_log(f"call: {name}({arg!r})")

    if name == "grep":
        if not get_include():
            return "[grep] No files imported. Import files first to search."
        parts = arg.split(":", 1)
        pattern = (parts[0] or "").strip()
        glob = (parts[1] or "").strip() if len(parts) > 1 else ""
        if not pattern:
            return "[grep] Error: pattern required"
        try:
            result = grep_search(
                pattern,
                root=get_root(),
                max_results=25,
                literal=True,
                file_glob=glob if glob else "",
            )
        except Exception as e:
            return f"[grep] Error: {e}"
        if not result.matches:
            _tool_log(f"  -> no matches")
            return f"[grep] No matches for {pattern!r}"
        lines = []
        for m in result.matches[:25]:
            lines.append(f"  {m.file}:{m.line}: {m.text.strip()}")
        if result.truncated or len(result.matches) > 25:
            lines.append(f"  ... ({len(result.matches)} total, showing first 25)")
        _tool_log(f"  -> {len(result.matches)} matches")
        return "[grep] Results:\n" + "\n".join(lines)

    if name == "symbols":
        path = arg.strip()
        if not path:
            return "[symbols] Error: path required"
        if not get_include():
            return "[symbols] No files imported. Import files first."
        if not _path_in_include(path):
            return f"[symbols] File not in imported set: {path}"
        try:
            syms = get_symbols_for_file(path)
        except Exception as e:
            return f"[symbols] Error: {e}"
        if not syms:
            _tool_log(f"  -> no symbols")
            return f"[symbols] No symbols found in {path} (or file not indexed)"
        lines = [f"  {s['kind']} {s['name']} (L{s['line']}-{s['end_line']})" for s in syms[:40]]
        if len(syms) > 40:
            lines.append(f"  ... ({len(syms)} total)")
        _tool_log(f"  -> {len(syms)} symbols")
        return f"[symbols] {path}:\n" + "\n".join(lines)

    if name == "read":
        path = arg.strip()
        if not path:
            return "[read] Error: path required"
        if not get_include():
            return "[read] No files imported. Import files first to read file content."
        if not _path_in_include(path):
            return f"[read] File not in imported set: {path}"
        try:
            content_map = read_single_file_for_context(path)
            content = content_map.get(path, "")
        except Exception as e:
            return f"[read] Error: {e}"
        if content is None or content == "":
            _tool_log(f"  -> empty/not found")
            return f"[read] File empty or not found: {path}"
        _tool_log(f"  -> {len(content)} chars")
        return f"[read] {path}:\n{content}"

    if name == "list_files":
        if not get_include():
            return "[list_files] No files imported. Import files first."
        try:
            files = get_indexed_files()
        except Exception as e:
            return f"[list_files] Error: {e}"
        if not files:
            _tool_log(f"  -> no files indexed")
            return "[list_files] No files indexed"
        _tool_log(f"  -> {len(files)} files")
        return "[list_files] Indexed files:\n" + "\n".join(f"  {f}" for f in files[:25])

    return f"[tool] Unknown tool: {name}"


def extract_tool_calls(text: str) -> List[Tuple[str, str]]:
    """Extract tool calls from model output. Returns [(tool_name, arg), ...]."""
    calls = []
    for m in TOOL_PATTERN.finditer(text):
        name = (m.group(1) or "").strip()
        arg = (m.group(2) or "").strip()
        if name:
            calls.append((name, arg))
    return calls


def strip_tool_calls_from_output(text: str) -> str:
    """Remove tool call lines from output for display."""
    return TOOL_PATTERN.sub("", text).strip()
