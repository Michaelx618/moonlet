"""Agent-callable tools: read_file, list_files, grep, edit_existing_file, write_file, run_terminal_cmd.

Supports:
1. Function-style (primary): read_file(path="..."), edit_existing_file(old_string="...", new_string="...", path="...")
2. XML-style (Continue fallback): <tool_call>{"name":"...","arguments":{...}}</tool_call>
3. Legacy bracket: [[[read:path]]], [[[grep:pattern]]], etc.
(search_replace is not in the schema; model uses edit_existing_file/single_find_and_replace; search_replace code path kept.)

Results are appended to the conversation so the agent can use them.
"""

import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .files import (
    _norm_rel_path,
    _path_in_include,
    get_include,
    get_root,
    is_edit_allowed,
    read_single_file_for_context,
    write_file_text,
)
from .index import get_indexed_files, get_symbols_for_file
from .search_replace import (
    _parse_python_string,
    _unescape_string,
    apply_single_multi_edit,
    execute_search_replace,
)
from .tools import grep_search
from .utils import dbg

try:
    from file_utils import is_security_concern
except ImportError:
    def is_security_concern(*args, **kwargs):
        return False


def _parse_function_kwargs(text: str, start: int, keys: Tuple[str, ...]) -> Tuple[Optional[Dict[str, str]], int]:
    """Parse key="value" kwargs from text starting at start. Returns (kwargs_dict, end_pos)."""
    pos = start
    kwargs: Dict[str, str] = {}
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(text):
            break
        if text[pos] == ")":
            return kwargs, pos + 1
        key_match = re.match(r"([a-zA-Z_]\w*)\s*=\s*", text[pos:])
        if not key_match:
            pos += 1
            continue
        key = key_match.group(1).lower()
        pos += key_match.end()
        if pos >= len(text) or key not in {k.lower() for k in keys}:
            continue
        if text[pos] == '"':
            triple = text[pos : pos + 3] == '"""'
            quote = '"""' if triple else '"'
            val, end_pos = _parse_python_string(text, pos, quote[0])
            if val is not None and not triple:
                val = _unescape_string(val)
            if val is not None:
                kwargs[key] = val
            pos = end_pos
        else:
            break
    return kwargs, pos


def _extract_xml_tool_calls(text: str) -> List[Tuple[str, Dict[str, str]]]:
    """Extract tool calls from XML-style blocks (Continue fallback format). Returns [(name, kwargs), ...]."""
    if not text or not isinstance(text, str):
        return []
    results: List[Tuple[str, Dict[str, str]]] = []
    # <tool_call>{"name": "read_file", "arguments": {"filepath": "x"}}</tool_call> or similar
    for m in re.finditer(r"<tool_call>\s*(\{[^<]*\})\s*</tool_call>", text, re.IGNORECASE | re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            name = (obj.get("name") or "").strip().lower()
            args = obj.get("arguments") or obj.get("args") or {}
            if not name or name not in {t.lower() for t in FUNCTION_STYLE_TOOLS}:
                continue
            kwargs = {str(k).lower(): str(v) if v is not None else "" for k, v in args.items()}
            canonical = name
            if name == "view_subdirectory":
                canonical = "list_files"
            elif name == "grep_search":
                canonical = "grep"
            elif name == "glob_search":
                canonical = "glob_file_search"
            elif name == "codebase_tool":
                canonical = "codebase_search"
            elif name == "edit_existing_file":
                canonical = "search_replace"
            elif name == "create_new_file":
                canonical = "write_file"
            elif name == "run_terminal_command":
                canonical = "run_terminal_cmd"
            results.append((canonical, kwargs))
        except Exception:
            continue
    return results


def _extract_write_file_from_code_block(text: str) -> Optional[Tuple[str, str]]:
    """If model output is a single fenced code block with a filename (e.g. ```c checkpasswd.c), return (path, content)."""
    if not text or not isinstance(text, str):
        return None
    # ```lang filename or ```filename, then newline, then content until ```
    m = re.search(r"^```\s*\w*\s*([^\s\n]+)\s*\n(.*?)```", text.strip(), re.DOTALL)
    if not m:
        return None
    path_candidate = (m.group(1) or "").strip().strip('"\'')
    content = (m.group(2) or "").strip()
    if not path_candidate or not content:
        return None
    # Must look like a file path (has a dot or is a known ext)
    if "." in path_candidate or len(path_candidate) > 2:
        return (path_candidate, content)
    return None


def extract_function_style_tool_calls(text: str) -> List[Tuple[str, Dict[str, str]]]:
    """Extract function-style or XML-style tool calls from model output. Returns [(tool_name, kwargs), ...].
    Also parses inside markdown code blocks (e.g. ```python ... edit_existing_file(...) ... ```)."""
    if not text or not isinstance(text, str):
        return []
    results: List[Tuple[str, Dict[str, str]]] = []
    tool_names = "|".join(re.escape(t) for t in FUNCTION_STYLE_TOOLS)
    pattern = re.compile(rf"\b({tool_names})\s*\(", re.IGNORECASE)

    def extract_from_chunk(chunk: str) -> None:
        for m in pattern.finditer(chunk):
            name = (m.group(1) or "").strip().lower()
            if name == "read_file":
                keys = ("path", "filepath")
            elif name in ("list_files", "view_subdirectory"):
                keys = ("path",)
            elif name in ("grep", "grep_search"):
                keys = ("pattern", "path")
            elif name in ("glob_file_search", "glob_search"):
                keys = ("glob", "path")
            elif name in ("codebase_search", "codebase_tool"):
                keys = ("query",)
            elif name == "view_repo_map":
                keys = ()
            elif name == "view_diff":
                keys = ("path",)
            elif name in ("search_replace", "single_find_and_replace"):
                keys = ("old_string", "new_string", "path", "replace_all")
            elif name == "edit_existing_file":
                keys = ("filepath", "path", "changes", "old_string", "new_string")
            elif name == "multi_edit":
                keys = ("path", "filepath", "edits")
            elif name in ("write_file", "create_new_file"):
                keys = ("path", "content")
            elif name in ("run_terminal_cmd", "run_terminal_command"):
                keys = ("command",)
            else:
                continue
            kwargs, _ = _parse_function_kwargs(chunk, m.end(), keys)
            if kwargs or name == "view_repo_map":
                canonical = name
                if name == "view_subdirectory":
                    canonical = "list_files"
                elif name == "grep_search":
                    canonical = "grep"
                elif name == "glob_search":
                    canonical = "glob_file_search"
                elif name == "codebase_tool":
                    canonical = "codebase_search"
                elif name == "edit_existing_file":
                    canonical = "edit_existing_file" if kwargs.get("changes") is not None else "search_replace"
                elif name == "single_find_and_replace":
                    canonical = "search_replace"
                elif name == "create_new_file":
                    canonical = "create_new_file"
                elif name == "run_terminal_command":
                    canonical = "run_terminal_cmd"
                results.append((name, kwargs))

    extract_from_chunk(text)
    if not results:
        for code_block in re.finditer(r"```[^\n]*\n(.*?)```", text, re.DOTALL):
            extract_from_chunk(code_block.group(1))
            if results:
                break
    if not results:
        results = _extract_xml_tool_calls(text)
    # If still none, treat fenced code block with filename as write_file (model output ```c file.c)
    if not results:
        code_block = _extract_write_file_from_code_block(text)
        if code_block:
            path, content = code_block
            results.append(("write_file", {"path": path, "content": content}))
    return results


# Placeholder patterns that indicate partial/stub content (would overwrite real code)
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\.\.\.\s*existing\s+code\s*\.\.\.", re.IGNORECASE),
    re.compile(r"#\s*\.\.\.\s*existing", re.IGNORECASE),
    re.compile(r"//\s*\.\.\.\s*existing", re.IGNORECASE),
)


def _content_has_placeholders(content: str) -> bool:
    """True if content looks like a stub with placeholders instead of full file content."""
    if not content or not isinstance(content, str):
        return False
    for pat in _PLACEHOLDER_PATTERNS:
        if pat.search(content):
            return True
    return False


def _tool_log(msg: str) -> None:
    """Log tool activity to stderr and debug log."""
    if getattr(config, "DEBUG_TOOLS", True):
        print(f"[tool] {msg}", file=sys.stderr)
    dbg(f"tool: {msg}")
    # Always write tool calls to debug log (even when DEBUG=0) for traceability
    log_path = getattr(config, "DEBUG_LOG_PATH", None)
    if log_path:
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a") as f:
                f.write(f"[tool] [{ts} pid={os.getpid()}] {msg}\n")
        except Exception:
            pass


def _tool_wrote_log(rel_path: str, content_len: Optional[int] = None, edits_count: Optional[int] = None) -> None:
    """Write a single [tool_wrote] line to the debug log so each round's edits are visible."""
    log_path = getattr(config, "DEBUG_LOG_PATH", None)
    if not log_path:
        return
    parts = [f"path={rel_path}"]
    if content_len is not None:
        parts.append(f"content_len={content_len}")
    if edits_count is not None:
        parts.append(f"edits={edits_count}")
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[tool_wrote] [{ts} pid={os.getpid()}] {' '.join(parts)}\n")
    except Exception:
        pass


def tool_log(msg: str) -> None:
    """Public alias for agent to log tool-related activity."""
    _tool_log(msg)


TOOL_PATTERN = re.compile(
    r"\[\[\[(grep|symbols|read|list_files)(?::([^\]]*))?\]\]\]",
    re.IGNORECASE,
)

# Function-style tool names: our names + Continue aliases (all 4 edit forms)
FUNCTION_STYLE_TOOLS = (
    "read_file",
    "list_files",
    "view_subdirectory",
    "grep",
    "grep_search",
    "glob_file_search",
    "glob_search",
    "codebase_search",
    "codebase_tool",
    "view_repo_map",
    "view_diff",
    "search_replace",
    "edit_existing_file",
    "single_find_and_replace",
    "multi_edit",
    "write_file",
    "create_new_file",
    "run_terminal_cmd",
    "run_terminal_command",
)

# Tool schema: Continue-style tool names only. search_replace is disabled for the model (not in schema)
# but remains in code; edit_existing_file and single_find_and_replace map to it internally.
TOOLS_SCHEMA = """Available tools. Call using function-style: tool_name(arg="value", ...).

Read-only:
- read_file: read_file(filepath="path to file")
- view_subdirectory / list_files: list_files(path="directory path, optional")
- grep_search / grep: grep_search(pattern="...", path="optional scope")
- glob_search / glob_file_search: glob_file_search(glob="*.py", path="optional dir")
- codebase_tool / codebase_search: codebase_search(query="natural language query")
- view_repo_map: no args
- view_diff: view_diff(path="file path") — diff after edits in this session

Read-write (Continue names):
- edit_existing_file: edit_existing_file(path="file", old_string="...", new_string="...") or edit_existing_file(path="file", changes="full/partial content")
- single_find_and_replace: single_find_and_replace(path="file", old_string="...", new_string="...", replace_all=true)
- multi_edit: multi_edit(path="file", edits=[{old_string, new_string}, ...])
- create_new_file / write_file: write_file(path="file", content="...")
- run_terminal_command: run_terminal_command(command="shell command")"""

TOOLS_SYSTEM_HINT = """You have access to tools provided as a JSON-style schema. Use function-style calls: tool_name(arg="value", ...).

""" + TOOLS_SCHEMA.strip() + """

Current file (if any) may be included below as "Current file".
- Use read-only tools (read_file, grep_search, list_files, etc.) to gather context if needed.
- When the user asks to change, fix, or add code: you must output one or more edit_existing_file(...) or single_find_and_replace(...) or write_file(...) calls. Reading files alone does not fulfill an edit request. Output your edit tool call(s) in the same format as above.
- For editing existing files: use old_string and new_string (edit_existing_file or single_find_and_replace). Do NOT write full-file content with placeholders like \"... existing code ...\" — that will overwrite and erase real code.
- After making edits, give a brief explanation of what you changed (one or two sentences)."""

# Chat mode: answer questions, use tools to read/list — do NOT instruct to produce diffs
CHAT_TOOLS_HINT = """You have access to these tools. Use the function-style format when you need them:

- read_file(path="src/main.py") — Read a file
- list_files(path="api/") — List files in directory
- grep(pattern="pattern", path=".") — Search across files
- run_terminal_cmd(command="pytest") — Run shell command

CONTEXT already includes imported files. Use tools if you need more. Answer in plain text. Do NOT produce diffs or code edits unless the user explicitly asks."""


# Full tools as JSON schema for API tool calls (/v1/chat/completions with tools)
AGENT_TOOLS_JSON: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file. Use filepath for the path.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string", "description": "Path relative to project root"}},
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_subdirectory",
            "description": "List files in a directory (or root if path omitted).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for a pattern across files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Scope path or .", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_search",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string", "description": "e.g. *.py"},
                    "path": {"type": "string", "default": "."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "codebase_search",
            "description": "Natural language search over the codebase.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_repo_map",
            "description": "Show top-level structure of the repo (grouped by directory).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_diff",
            "description": "Show diff for a file after edits in this session.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_existing_file",
            "description": "Edit file: use 'changes' for full/partial content, or old_string/new_string for find-and-replace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "filepath": {"type": "string"},
                    "changes": {"type": "string", "description": "Full or partial file content (with placeholders like // ... existing code ...)"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "single_find_and_replace",
            "description": "Replace old_string with new_string in a file. Use replace_all to replace every occurrence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": "Apply multiple find-and-replace edits to one file in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "filepath": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                                "replace_all": {"type": "boolean"},
                            },
                            "required": ["old_string", "new_string"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_new_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Run a shell command from the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "symbols",
            "description": "List symbols (functions, classes) in a file. Use for @Code-style context.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


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
        return content

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


def execute_tool_from_kwargs(name: str, kwargs: Dict[str, str]) -> str:
    """Execute a tool from function-style kwargs. Returns result string."""
    name = (name or "").strip().lower()
    _tool_log(f"call: {name}({kwargs})")

    if name == "read_file":
        path = (kwargs.get("filepath") or kwargs.get("path") or "").strip().strip('"\'')
        if not path:
            return "[read_file] Error: path (or filepath) required"
        return execute_tool("read", path)

    if name == "list_files":
        path_prefix = (kwargs.get("path") or ".").strip().strip('"\'').rstrip("/")
        if not get_include():
            return "[list_files] No files imported. Import files first."
        try:
            all_files = get_indexed_files()
        except Exception as e:
            return f"[list_files] Error: {e}"
        files = all_files
        if path_prefix and path_prefix != ".":
            files = [f for f in all_files if f == path_prefix or f.startswith(path_prefix + "/")]
        if not files:
            _tool_log(f"  -> no files under {path_prefix!r}")
            # Hint so the model doesn't loop: show actual top-level paths
            tops = list(dict.fromkeys((f.split("/")[0] for f in all_files)))[:15]
            hint = f" Try list_files(path=\".\") to list all. Top-level in index: {', '.join(tops)}."
            return f"[list_files] No files under {path_prefix!r}.{hint}"
        _tool_log(f"  -> {len(files)} files")
        return "[list_files] Files:\n" + "\n".join(f"  {f}" for f in files[:25])

    if name == "grep":
        pattern = (kwargs.get("pattern") or "").strip().strip('"\'')
        path_arg = (kwargs.get("path") or ".").strip().strip('"\'')
        if not pattern:
            return "[grep] Error: pattern required"
        if not get_include():
            return "[grep] No files imported. Import files first to search."
        include_paths = None
        if path_arg and path_arg != ".":
            all_files = get_indexed_files()
            include_paths = [f for f in all_files if f == path_arg or f.startswith(path_arg.rstrip("/") + "/")]
        try:
            result = grep_search(
                pattern,
                root=get_root(),
                max_results=25,
                literal=True,
                include_paths=include_paths,
            )
        except Exception as e:
            return f"[grep] Error: {e}"
        if not result.matches:
            _tool_log(f"  -> no matches")
            return f"[grep] No matches for {pattern!r}"
        lines = [f"  {m.file}:{m.line}: {m.text.strip()}" for m in result.matches[:25]]
        if result.truncated or len(result.matches) > 25:
            lines.append(f"  ... ({len(result.matches)} total, showing first 25)")
        _tool_log(f"  -> {len(result.matches)} matches")
        return "[grep] Results:\n" + "\n".join(lines)

    if name == "glob_file_search":
        glob_pat = (kwargs.get("glob") or "*").strip().strip('"\'')
        path_arg = (kwargs.get("path") or ".").strip().strip('"\'').rstrip("/")
        if not get_include():
            return "[glob_file_search] No files imported. Import files first."
        root = get_root()
        try:
            base = (root / path_arg).resolve() if path_arg and path_arg != "." else root
            if not base.exists():
                return f"[glob_file_search] Path not found: {path_arg}"
            if base.is_file():
                return "[glob_file_search] Path must be a directory."
            matches: List[str] = []
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    rel = str(p.relative_to(root)).replace("\\", "/")
                except ValueError:
                    continue
                if not fnmatch.fnmatch(p.name, glob_pat) and not fnmatch.fnmatch(rel, glob_pat):
                    continue
                indexed = get_indexed_files()
                if indexed and rel not in indexed and not any(rel == x or rel.startswith(x.rstrip("/") + "/") for x in indexed):
                    continue
                matches.append(rel)
            matches = sorted(set(matches))[:50]
            _tool_log(f"  -> {len(matches)} matches")
            if not matches:
                return f"[glob_file_search] No files matching {glob_pat!r} under {path_arg or '.'}"
            return "[glob_file_search] Files:\n" + "\n".join(f"  {m}" for m in matches)
        except Exception as e:
            return f"[glob_file_search] Error: {e}"

    if name == "codebase_search":
        query = (kwargs.get("query") or "").strip().strip('"\'')
        if not query:
            return "[codebase_search] Error: query required"
        if not get_include():
            return "[codebase_search] No files imported. Import files first."
        try:
            from .relevance import find_relevant_files
            paths = find_relevant_files(query, open_file=None)
            if not paths:
                # Fallback: grep for query words and return files
                result = grep_search(
                    query[:100],
                    root=get_root(),
                    max_results=30,
                    literal=True,
                    include_paths=get_indexed_files(),
                )
                paths = list(dict.fromkeys(m.file for m in result.matches))[:15]
            if not paths:
                _tool_log("  -> no results")
                return "[codebase_search] No relevant files found."
            _tool_log(f"  -> {len(paths)} files")
            return "[codebase_search] Relevant files:\n" + "\n".join(f"  {p}" for p in paths[:20])
        except Exception as e:
            return f"[codebase_search] Error: {e}"

    if name == "view_repo_map":
        if not get_include():
            return "[view_repo_map] No files imported. Import files first."
        try:
            files = get_indexed_files()
            if not files:
                return "[view_repo_map] No files indexed."
            # Group by top-level path (first segment or ".")
            groups: Dict[str, List[str]] = {}
            for f in files:
                parts = f.split("/")
                top = parts[0] if len(parts) > 1 else "."
                if top not in groups:
                    groups[top] = []
                groups[top].append(f)
            lines = []
            for top in sorted(groups.keys(), key=lambda x: (x == ".", x)):
                count = len(groups[top])
                preview = ", ".join(groups[top][:5])
                if count > 5:
                    preview += f" ... (+{count - 5} more)"
                lines.append(f"  {top}/ ({count} files): {preview}")
            _tool_log(f"  -> {len(groups)} top-level dirs")
            return "[view_repo_map]\n" + "\n".join(lines)
        except Exception as e:
            return f"[view_repo_map] Error: {e}"

    if name == "view_diff":
        path = (kwargs.get("path") or "").strip().strip('"\'')
        if not path:
            return "[view_diff] Error: path required"
        return "[view_diff] No diff available for that file in this session. (Diffs are shown after edits in agent mode.)"

    if name == "symbols":
        path = (kwargs.get("path") or "").strip().strip('"\'')
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

    # search_replace: disabled in schema (model uses edit_existing_file / single_find_and_replace); kept for implementation
    if name == "search_replace":
        old_str = (kwargs.get("old_string") or "").strip()
        new_str = kwargs.get("new_string", "")
        path_str = (kwargs.get("path") or kwargs.get("filepath") or "").strip().strip('"\'')
        replace_all = kwargs.get("replace_all") in (True, "true", "1", "yes")
        if not old_str or not path_str:
            return "[search_replace] Error: old_string and path required"
        edit = {"old_string": old_str, "new_string": new_str, "path": path_str, "replace_all": replace_all}
        ok, msg, meta = execute_search_replace(
            edit, allowed_edit_files=None, root=get_root(), stage_only=False
        )
        if ok:
            _tool_log(f"  -> {msg}")
            _tool_wrote_log(path_str, edits_count=1)
            return f"[search_replace] {msg}"
        _tool_log(f"  -> FAILED: {msg}")
        return f"[search_replace] {msg}"

    if name == "multi_edit":
        path_str = (kwargs.get("path") or kwargs.get("filepath") or "").strip().strip('"\'')
        edits_raw = kwargs.get("edits")
        if not path_str:
            return "[multi_edit] Error: path required"
        if edits_raw is None:
            return "[multi_edit] Error: edits array required"
        if isinstance(edits_raw, str):
            try:
                edits_list = json.loads(edits_raw)
            except json.JSONDecodeError:
                return "[multi_edit] Error: edits must be a JSON array"
        elif isinstance(edits_raw, list):
            edits_list = edits_raw
        else:
            return "[multi_edit] Error: edits must be an array"
        edits = []
        for item in edits_list:
            if not isinstance(item, dict):
                continue
            o = (item.get("old_string") or "").strip()
            n = (item.get("new_string") or "").strip()
            if o == n:
                continue
            edits.append({"old_string": o, "new_string": n, "replace_all": item.get("replace_all") is True})
        if not edits:
            return "[multi_edit] Error: no valid edits in array"
        ok, msg, _ = apply_single_multi_edit(
            path_str, edits, allowed_edit_files=None, root=get_root(), stage_only=False
        )
        if ok:
            _tool_log(f"  -> {msg}")
            _tool_wrote_log(path_str, edits_count=len(edits))
            return f"[multi_edit] {msg}"
        return f"[multi_edit] {msg}"

    if name == "edit_existing_file":
        # Continue form: filepath + changes (full or partial content)
        path_str = (kwargs.get("filepath") or kwargs.get("path") or "").strip().strip('"\'')
        changes = kwargs.get("changes")
        if not path_str:
            return "[edit_existing_file] Error: filepath required"
        if changes is None:
            return "[edit_existing_file] Error: changes required (or use search_replace for old_string/new_string)"
        root = get_root()
        try:
            rel_path = _norm_rel_path(path_str)
            abs_path = (root / rel_path).resolve()
            abs_path.relative_to(root.resolve())
        except Exception:
            return f"[edit_existing_file] Error: path outside root: {path_str}"
        if not is_edit_allowed(rel_path, allow_new=False):
            return (
                f"[edit_existing_file] Edit not allowed for {rel_path} "
                "(file not in imported/allow list; add this file to imported files to edit it)"
            )
        if is_security_concern(abs_path):
            return f"[edit_existing_file] Security concern for {rel_path}"
        if not abs_path.exists() or not abs_path.is_file():
            return f"[edit_existing_file] File not found: {rel_path}"
        if _content_has_placeholders(str(changes)):
            return (
                "[edit_existing_file] Rejected: content contains placeholders like \"... existing code ...\". "
                "Use old_string and new_string for targeted edits instead of full-file changes with placeholders."
            )
        write_file_text(rel_path, str(changes))
        _tool_log(f"  -> wrote {rel_path} (changes)")
        _tool_wrote_log(rel_path, content_len=len(str(changes)))
        return f"[edit_existing_file] Applied changes to {rel_path}"

    if name == "create_new_file":
        path = (kwargs.get("path") or kwargs.get("filepath") or "").strip().strip('"\'')
        content = kwargs.get("content") or kwargs.get("contents") or ""
        if not path:
            return "[create_new_file] Error: path required"
        root = get_root()
        try:
            rel_path = _norm_rel_path(path)
            abs_path = (root / rel_path).resolve()
            abs_path.relative_to(root.resolve())
        except Exception:
            return f"[create_new_file] Error: path outside root: {path}"
        if abs_path.exists() and abs_path.is_file():
            return f"[create_new_file] File {rel_path} already exists. Use the edit tool to edit this file."
        if not is_edit_allowed(rel_path, allow_new=True):
            return (
                f"[create_new_file] Edit not allowed for {rel_path} "
                "(file not in imported/allow list; add this file to imported files to edit it)"
            )
        if is_security_concern(abs_path):
            return f"[create_new_file] Security concern for {rel_path}"
        try:
            write_file_text(rel_path, str(content))
            _tool_log(f"  -> created {rel_path}")
            _tool_wrote_log(rel_path, content_len=len(str(content)))
            return f"[create_new_file] Created {rel_path}"
        except Exception as e:
            return f"[create_new_file] Error: {e}"

    if name == "write_file":
        path = (kwargs.get("path") or "").strip().strip('"\'')
        content = kwargs.get("content")
        if content is None:
            content = ""
        if not path:
            return "[write_file] Error: path required"
        root = get_root()
        try:
            rel_path = _norm_rel_path(path)
            abs_path = (root / rel_path).resolve()
            abs_path.relative_to(root.resolve())
        except Exception:
            return f"[write_file] Error: path outside root: {path}"
        # Prevent creating new top-level dirs (e.g. w7/) when workspace has files at root
        indexed = get_indexed_files()
        if indexed and "/" in rel_path:
            top = rel_path.split("/", 1)[0]
            has_under = any(p == top or p.startswith(top + "/") for p in indexed)
            if not has_under:
                base = rel_path.split("/")[-1]
                at_root = [p for p in indexed if p == base or p.endswith("/" + base)]
                if at_root:
                    return f"[write_file] Do not create directory {top}/. Use workspace path: {at_root[0]}"
        if not is_edit_allowed(rel_path, allow_new=True):
            return (
                f"[write_file] Edit not allowed for {rel_path} "
                "(file not in imported/allow list; add this file to imported files to edit it)"
            )
        if is_security_concern(abs_path):
            return f"[write_file] Security concern for {rel_path}"
        if abs_path.exists() and abs_path.is_file() and _content_has_placeholders(str(content)):
            return (
                "[write_file] Rejected: content contains placeholders like \"... existing code ...\". "
                "Do not overwrite an existing file with partial content. Use edit_existing_file(path=..., old_string=\"...\", new_string=\"...\") or single_find_and_replace for targeted edits."
            )
        try:
            write_file_text(rel_path, content)
            _tool_log(f"  -> wrote {rel_path}")
            _tool_wrote_log(rel_path, content_len=len(content))
            return f"[write_file] Wrote {rel_path}"
        except Exception as e:
            return f"[write_file] Error: {e}"

    if name == "run_terminal_cmd":
        cmd = (kwargs.get("command") or "").strip().strip('"\'')
        if not cmd:
            return "[run_terminal_cmd] Error: command required"
        timeout_val = min(60, max(1, int(getattr(config, "VERIFY_TIMEOUT", 60))))
        try:
            res = subprocess.run(
                cmd,
                cwd=str(get_root()),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_val,
            )
            out = res.stdout or ""
            err = res.stderr or ""
            max_chars = 20000
            if len(out) > max_chars:
                out = out[:max_chars] + "\n...[stdout truncated]"
            if len(err) > max_chars:
                err = err[:max_chars] + "\n...[stderr truncated]"
            _tool_log(f"  -> exit {res.returncode}")
            return f"[run_terminal_cmd] exit={res.returncode}\nstdout:\n{out}\nstderr:\n{err}"
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "")[:5000]
            err = ((exc.stderr or "") + "\nCommand timed out.")[:5000]
            return f"[run_terminal_cmd] timeout\nstdout:\n{out}\nstderr:\n{err}"
        except Exception as e:
            return f"[run_terminal_cmd] Error: {e}"

    return f"[tool] Unknown tool: {name}"


def strip_function_style_tool_calls(text: str) -> str:
    """Remove function-style tool calls from output for display."""
    for name in FUNCTION_STYLE_TOOLS:
        pattern = re.compile(rf"\b{re.escape(name)}\s*\([^)]*(?:\([^)]*\)[^)]*)*\)", re.IGNORECASE | re.DOTALL)
        text = pattern.sub("", text)
    return text.strip()


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


def strip_tool_call_code_blocks(text: str) -> str:
    """Remove markdown code blocks that contain tool calls so the agent panel shows only the explanation."""
    if not text or not isinstance(text, str):
        return text
    tool_names = "|".join(re.escape(t) for t in FUNCTION_STYLE_TOOLS)
    # Match ```[optional lang/path line]\n(content)``` - allow anything on opening line (e.g. ```python src/math_utils.py)
    block_pattern = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
    tool_call_in_block = re.compile(rf"\b({tool_names})\s*\(", re.IGNORECASE)

    def repl(m):
        content = m.group(1) or ""
        if tool_call_in_block.search(content):
            return "\n\n"
        return m.group(0)

    return block_pattern.sub(repl, text).strip()
